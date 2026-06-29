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


def join_frames(frames: list[DataFrame]) -> DataFrame:
    result = frames[0]
    for frame in frames[1:]:
        result = result.join(frame, on=ID_COL, how="left")
    return result


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

    base = df.select(ID_COL).distinct()
    frames = [base]

    if TARGET_COL in df.columns:
        target = df.groupBy(ID_COL).agg(F.max(TARGET_COL).alias(TARGET_COL))
        frames.append(target)

    if continuous:
        full_exprs = []
        for c in continuous:
            full_exprs.extend(
                [
                    F.mean(c).alias(f"{c}_mean"),
                    F.stddev(c).alias(f"{c}_std"),
                    F.min(c).alias(f"{c}_min"),
                    F.max(c).alias(f"{c}_max"),
                    F.expr(f"percentile_approx(`{c}`, 0.5)").alias(f"{c}_median"),
                ]
            )

        full_agg = df.groupBy(ID_COL).agg(*full_exprs)

        latest = (
            df.withColumn("_rn", F.row_number().over(latest_window))
            .where(F.col("_rn") == 1)
            .select(ID_COL, *[F.col(c).alias(f"{c}_last") for c in continuous])
        )

        earliest = (
            df.withColumn("_rn", F.row_number().over(earliest_window))
            .where(F.col("_rn") == 1)
            .select(ID_COL, *[F.col(c).alias(f"{c}_first") for c in continuous])
        )

        full_features = full_agg.join(latest, ID_COL, "left").join(
            earliest, ID_COL, "left"
        )

        lag_features = full_features.select(
            ID_COL,
            *[
                (F.col(f"{c}_last") - F.col(f"{c}_mean")).alias(f"{c}_last_minus_mean")
                for c in continuous
            ],
        )

        frames.extend([full_features, lag_features])

        for n in (3, 6):
            suffix = f"{n}m"

            recent = (
                df.withColumn("_rn", F.row_number().over(latest_window))
                .where(F.col("_rn") <= n)
                .drop("_rn")
            )

            recent_exprs = []
            for c in continuous:
                recent_exprs.extend(
                    [
                        F.mean(c).alias(f"{c}_mean_{suffix}"),
                        F.stddev(c).alias(f"{c}_std_{suffix}"),
                        F.min(c).alias(f"{c}_min_{suffix}"),
                        F.max(c).alias(f"{c}_max_{suffix}"),
                    ]
                )

            recent_agg = recent.groupBy(ID_COL).agg(*recent_exprs)

            recent_last = (
                recent.withColumn("_rn", F.row_number().over(latest_window))
                .where(F.col("_rn") == 1)
                .select(
                    ID_COL,
                    *[F.col(c).alias(f"{c}_last_{suffix}") for c in continuous],
                )
            )

            frames.append(recent_agg.join(recent_last, ID_COL, "left"))

        diff_df = df
        for c in continuous:
            diff_df = diff_df.withColumn(
                f"{c}_diff",
                F.col(c) - F.lag(F.col(c)).over(earliest_window),
            )

        diff_exprs = []
        for c in continuous:
            diff_exprs.extend(
                [
                    F.mean(f"{c}_diff").alias(f"{c}_diff_mean"),
                    F.stddev(f"{c}_diff").alias(f"{c}_diff_std"),
                    F.min(f"{c}_diff").alias(f"{c}_diff_min"),
                    F.max(f"{c}_diff").alias(f"{c}_diff_max"),
                ]
            )

        frames.append(diff_df.groupBy(ID_COL).agg(*diff_exprs))

    if categorical:
        cat_exprs = []
        for c in categorical:
            cat_exprs.extend(
                [
                    F.count(c).alias(f"{c}_count"),
                    F.approx_count_distinct(c).alias(f"{c}_nunique"),
                ]
            )

        cat_agg = df.groupBy(ID_COL).agg(*cat_exprs)

        cat_last = (
            df.withColumn("_rn", F.row_number().over(latest_window))
            .where(F.col("_rn") == 1)
            .select(ID_COL, *[F.col(c).alias(f"{c}_last") for c in categorical])
        )

        frames.append(cat_agg.join(cat_last, ID_COL, "left"))

    return join_frames(frames)
