from collections.abc import Sequence

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from amex_default.config import CATEGORICAL_FEATURES, DATE_COL, ID_COL, TARGET_COL


def infer_continuous_features(
    df: DataFrame,
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
) -> list[str]:
    excluded = {ID_COL, TARGET_COL, DATE_COL, *categorical_features}
    numeric_types = ("float", "double", "decimal")

    return [
        field.name
        for field in df.schema.fields
        if field.name not in excluded
        and field.dataType.simpleString().startswith(numeric_types)
    ]


def build_customer_features(
    df: DataFrame,
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
) -> DataFrame:
    if ID_COL not in df.columns:
        raise ValueError(f"Missing required column: {ID_COL}")
    if DATE_COL not in df.columns:
        raise ValueError(f"Missing required column: {DATE_COL}")

    df = df.withColumn(DATE_COL, F.to_date(F.col(DATE_COL)))

    categorical = [c for c in categorical_features if c in df.columns]
    continuous = infer_continuous_features(df, categorical)

    latest_window = Window.partitionBy(ID_COL).orderBy(F.col(DATE_COL).desc())
    earliest_window = Window.partitionBy(ID_COL).orderBy(F.col(DATE_COL).asc())

    df = df.withColumn("_rn_desc", F.row_number().over(latest_window)).withColumn(
        "_rn_asc",
        F.row_number().over(earliest_window),
    )

    aggregate_exprs = []
    if TARGET_COL in df.columns:
        aggregate_exprs.append(F.max(TARGET_COL).alias(TARGET_COL))

    if continuous:
        diff_columns = [
            (F.col(c) - F.lag(F.col(c)).over(earliest_window)).alias(f"{c}_diff")
            for c in continuous
        ]
        df = df.select("*", *diff_columns)

        for c in continuous:
            aggregate_exprs.extend(
                [
                    F.mean(c).alias(f"{c}_mean"),
                    F.stddev(c).alias(f"{c}_std"),
                    F.min(c).alias(f"{c}_min"),
                    F.max(c).alias(f"{c}_max"),
                    F.expr(f"percentile_approx(`{c}`, 0.5)").alias(f"{c}_median"),
                    F.max(F.when(F.col("_rn_desc") == 1, F.col(c))).alias(f"{c}_last"),
                    F.max(F.when(F.col("_rn_asc") == 1, F.col(c))).alias(f"{c}_first"),
                ]
            )

        for n in (3, 6):
            suffix = f"{n}m"
            for c in continuous:
                recent_value = F.when(F.col("_rn_desc") <= n, F.col(c))
                aggregate_exprs.extend(
                    [
                        F.mean(recent_value).alias(f"{c}_mean_{suffix}"),
                        F.stddev(recent_value).alias(f"{c}_std_{suffix}"),
                        F.min(recent_value).alias(f"{c}_min_{suffix}"),
                        F.max(recent_value).alias(f"{c}_max_{suffix}"),
                        F.max(F.when(F.col("_rn_desc") == 1, F.col(c))).alias(
                            f"{c}_last_{suffix}"
                        ),
                    ]
                )

        for c in continuous:
            aggregate_exprs.extend(
                [
                    F.mean(f"{c}_diff").alias(f"{c}_diff_mean"),
                    F.stddev(f"{c}_diff").alias(f"{c}_diff_std"),
                    F.min(f"{c}_diff").alias(f"{c}_diff_min"),
                    F.max(f"{c}_diff").alias(f"{c}_diff_max"),
                    (F.max(F.when(F.col("_rn_desc") == 1, F.col(c))) - F.mean(c)).alias(
                        f"{c}_last_minus_mean"
                    ),
                ]
            )

    if categorical:
        for c in categorical:
            aggregate_exprs.extend(
                [
                    F.count(c).alias(f"{c}_count"),
                    F.approx_count_distinct(c).alias(f"{c}_nunique"),
                    F.max(F.when(F.col("_rn_desc") == 1, F.col(c))).alias(f"{c}_last"),
                ]
            )

    if not aggregate_exprs:
        return df.select(ID_COL).distinct()

    return df.groupBy(ID_COL).agg(*aggregate_exprs)
