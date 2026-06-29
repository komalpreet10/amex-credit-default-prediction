from __future__ import annotations

import argparse
import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, FloatType

from amex_default.config import CATEGORICAL_FEATURES, DATE_COL, ID_COL

INPUT = "gs://amex-credit-risk-ml-data/raw/train_data.csv"
OUTPUT = "gs://amex-credit-risk-ml-data/processed/v1/train_preprocessed/"

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT)
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--missing-threshold", type=float, default=80.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def preprocess(df: DataFrame, missing_threshold: float) -> tuple[DataFrame, list[str]]:
    required = {ID_COL, DATE_COL}
    missing_required = required - set(df.columns)
    if missing_required:
        raise ValueError(f"Missing required columns: {sorted(missing_required)}")

    df = df.withColumn(DATE_COL, F.to_date(DATE_COL))

    total_rows = df.count()
    if total_rows == 0:
        raise ValueError("Input dataset is empty.")

    missing_exprs = []
    for field in df.schema.fields:
        condition = F.col(field.name).isNull()
        if isinstance(field.dataType, (DoubleType, FloatType)):
            condition = condition | F.isnan(F.col(field.name))

        missing_exprs.append(
            (
                F.sum(F.when(condition, 1).otherwise(0))
                / F.lit(total_rows)
                * F.lit(100.0)
            ).alias(field.name)
        )

    missing_pct = df.agg(*missing_exprs).first().asDict()

    columns_to_drop = [
        column
        for column, pct in missing_pct.items()
        if column not in required and pct > missing_threshold
    ]

    df = df.drop(*columns_to_drop)

    for column in CATEGORICAL_FEATURES:
        if column in df.columns:
            df = df.withColumn(column, F.col(column).cast("string"))

    return df, columns_to_drop


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    args = parse_args()
    if not 0 <= args.missing_threshold <= 100:
        raise ValueError("--missing-threshold must be between 0 and 100.")

    spark = SparkSession.builder.appName("amex-preprocess").getOrCreate()

    try:
        LOGGER.info("Reading data from %s", args.input)
        df = spark.read.csv(args.input, header=True, inferSchema=True)

        LOGGER.info("Preprocessing data")
        preprocessed, dropped = preprocess(df, args.missing_threshold)

        LOGGER.info("Dropped %d columns", len(dropped))
        if dropped:
            LOGGER.info("Dropped columns: %s", dropped)

        mode = "overwrite" if args.overwrite else "errorifexists"

        LOGGER.info("Writing output to %s", args.output)
        preprocessed.write.parquet(
            args.output,
            mode=mode,
            compression="snappy",
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
