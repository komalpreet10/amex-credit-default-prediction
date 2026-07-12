from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from amex_default.config import CATEGORICAL_FEATURES, DATE_COL, ID_COL
from amex_default.features_spark import build_customer_features
from gcp.config import (
    BQ_LOCATION,
    CHANGED_CUSTOMERS_TABLE,
    CUSTOMER_FEATURES_TABLE,
    PROJECT_ID,
    SELECTED_FEATURES_URI,
    STATEMENT_HISTORY_TABLE,
)

LOGGER = logging.getLogger(__name__)

RECENT_STATS = ("mean", "std", "min", "max", "last")
FULL_STATS = (*RECENT_STATS, "median", "first")
DIFF_STATS = ("mean", "std", "min", "max")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=PROJECT_ID)
    parser.add_argument("--bq-location", default=BQ_LOCATION)
    parser.add_argument("--raw-table", default=STATEMENT_HISTORY_TABLE)
    parser.add_argument("--changed-customers-table", default=CHANGED_CUSTOMERS_TABLE)
    parser.add_argument("--feature-table", default=CUSTOMER_FEATURES_TABLE)
    parser.add_argument(
        "--staging-table", default=f"{CUSTOMER_FEATURES_TABLE}_refresh_staging"
    )
    parser.add_argument("--selected-features-uri", default=SELECTED_FEATURES_URI)
    parser.add_argument("--statement-cycle", default=None)
    parser.add_argument("--cycle-column", default="statement_cycle")
    return parser.parse_args()


def quote_identifier(name: str) -> str:
    return f"`{name.replace('`', '')}`"


def load_json_from_gcs(spark: SparkSession, uri: str) -> Any:
    payload = spark.sparkContext.wholeTextFiles(uri).values().first()
    return json.loads(payload)


def strip_engineered_suffix(feature_name: str) -> str | None:
    for suffix in ("_last_minus_mean",):
        if feature_name.endswith(suffix):
            return feature_name[: -len(suffix)]

    for stat in DIFF_STATS:
        suffix = f"_diff_{stat}"
        if feature_name.endswith(suffix):
            return feature_name[: -len(suffix)]

    for window in ("3m", "6m"):
        for stat in RECENT_STATS:
            suffix = f"_{stat}_{window}"
            if feature_name.endswith(suffix):
                return feature_name[: -len(suffix)]

    for stat in FULL_STATS:
        suffix = f"_{stat}"
        if feature_name.endswith(suffix):
            return feature_name[: -len(suffix)]

    return None


def required_raw_columns(selected_features: Iterable[str]) -> list[str]:
    raw_features = []
    seen = {ID_COL, DATE_COL}
    for feature in selected_features:
        raw_feature = strip_engineered_suffix(feature)
        if raw_feature and raw_feature not in seen:
            raw_features.append(raw_feature)
            seen.add(raw_feature)
    return [ID_COL, DATE_COL, *raw_features]


def validate_selected_features(features: Any) -> list[str]:
    if not isinstance(features, list) or not features:
        raise ValueError("Selected feature list must be a non-empty JSON list.")
    return [str(feature) for feature in features]


def read_changed_customers(spark: SparkSession, args: argparse.Namespace) -> DataFrame:
    changed = (
        spark.read.format("bigquery")
        .option("table", args.changed_customers_table)
        .load()
    )
    if args.statement_cycle:
        if args.cycle_column not in changed.columns:
            raise ValueError(
                f"Cycle column {args.cycle_column!r} is missing from "
                f"{args.changed_customers_table}."
            )
        changed = changed.where(F.col(args.cycle_column) == F.lit(args.statement_cycle))
    return changed.select(ID_COL).where(F.col(ID_COL).isNotNull()).distinct()


def read_statement_history(
    spark: SparkSession,
    raw_table: str,
    raw_columns: list[str],
    changed_customers: DataFrame,
) -> DataFrame:
    statements = spark.read.format("bigquery").option("table", raw_table).load()
    missing = [column for column in raw_columns if column not in statements.columns]
    if missing:
        raise ValueError(f"Raw statement table is missing columns: {missing}")
    return statements.select(*raw_columns).join(
        changed_customers, on=ID_COL, how="inner"
    )


def select_model_features(
    features: DataFrame, selected_features: list[str]
) -> DataFrame:
    missing = [
        feature for feature in selected_features if feature not in features.columns
    ]
    for feature in missing:
        features = features.withColumn(feature, F.lit(0.0))
    return features.select(ID_COL, *selected_features)


def write_staging_table(features: DataFrame, table: str) -> None:
    (features.write.format("bigquery").option("table", table).mode("overwrite").save())


def merge_staging_to_feature_table(
    project: str,
    location: str,
    staging_table: str,
    feature_table: str,
    selected_features: list[str],
) -> None:
    from google.cloud import bigquery

    client = bigquery.Client(project=project, location=location)
    target = quote_identifier(feature_table)
    source = quote_identifier(staging_table)
    columns = [ID_COL, *selected_features]
    update_clause = ",\n        ".join(
        f"T.{quote_identifier(column)} = S.{quote_identifier(column)}"
        for column in selected_features
    )
    insert_columns = ", ".join(quote_identifier(column) for column in columns)
    insert_values = ", ".join(f"S.{quote_identifier(column)}" for column in columns)
    query = f"""
    MERGE `{feature_table}` T
    USING `{staging_table}` S
    ON T.{quote_identifier(ID_COL)} = S.{quote_identifier(ID_COL)}
    WHEN MATCHED THEN UPDATE SET
        {update_clause}
    WHEN NOT MATCHED THEN INSERT ({insert_columns})
    VALUES ({insert_values})
    """
    LOGGER.info("Merging %s into %s", source, target)
    client.query(query, location=location).result()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    spark = SparkSession.builder.appName("amex-refresh-selected-features").getOrCreate()

    try:
        selected_features = validate_selected_features(
            load_json_from_gcs(spark, args.selected_features_uri)
        )
        raw_columns = required_raw_columns(selected_features)
        LOGGER.info(
            "Refreshing %d selected features from %d raw columns",
            len(selected_features),
            len(raw_columns),
        )

        changed_customers = read_changed_customers(spark, args)
        statements = read_statement_history(
            spark,
            raw_table=args.raw_table,
            raw_columns=raw_columns,
            changed_customers=changed_customers,
        )

        categorical = [
            column for column in CATEGORICAL_FEATURES if column in raw_columns
        ]
        for column in categorical:
            statements = statements.withColumn(column, F.col(column).cast("string"))

        refreshed = select_model_features(
            build_customer_features(statements, categorical_features=categorical),
            selected_features,
        ).persist()

        write_staging_table(refreshed, args.staging_table)
        merge_staging_to_feature_table(
            project=args.project,
            location=args.bq_location,
            staging_table=args.staging_table,
            feature_table=args.feature_table,
            selected_features=selected_features,
        )
        LOGGER.info("Completed selected-feature refresh")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
