from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from amex_default.config import CATEGORICAL_FEATURES, DATE_COL, ID_COL, TARGET_COL

_RECENT_AGG_STATS = ("mean", "std", "min", "max", "last")
_FULL_AGG_STATS = (*_RECENT_AGG_STATS, "median")
_DIFF_AGG_STATS = ("mean", "std", "min", "max")


def _strip_feature_suffix(feature_name: str) -> str | None:
    for suffix in ("_lag", "_first"):
        if feature_name.endswith(suffix):
            return feature_name[: -len(suffix)]

    for window_suffix in ("_3m", "_6m", "_diff"):
        if not feature_name.endswith(window_suffix):
            continue
        without_window = feature_name[: -len(window_suffix)]
        for stat in _DIFF_AGG_STATS if window_suffix == "_diff" else _RECENT_AGG_STATS:
            stat_suffix = f"_{stat}"
            if without_window.endswith(stat_suffix):
                return without_window[: -len(stat_suffix)]

    for stat in _FULL_AGG_STATS:
        stat_suffix = f"_{stat}"
        if feature_name.endswith(stat_suffix):
            return feature_name[: -len(stat_suffix)]

    return None


def infer_continuous_features(feature_names: Sequence[str]) -> list[str]:
    """Recover raw continuous feature names from trained engineered feature names."""
    features: list[str] = []
    seen = set()
    categorical = set(CATEGORICAL_FEATURES)

    for feature_name in feature_names:
        raw_name = _strip_feature_suffix(feature_name)
        if raw_name and raw_name not in categorical and raw_name not in seen:
            seen.add(raw_name)
            features.append(raw_name)

    return features


def _default_continuous_features(df: pd.DataFrame) -> list[str]:
    excluded = {ID_COL, TARGET_COL, DATE_COL, *CATEGORICAL_FEATURES}
    return [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]


def _prepare_statement_frame(
    statements: pd.DataFrame,
    continuous_features: Sequence[str] | None,
    categorical_features: Sequence[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    if statements.empty:
        raise ValueError("At least one statement row is required.")
    if ID_COL not in statements.columns:
        raise ValueError(f"Raw statements must include '{ID_COL}'.")

    df = statements.copy()
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df = df.sort_values([ID_COL, DATE_COL])

    continuous = list(continuous_features or _default_continuous_features(df))
    categorical = [col for col in categorical_features if col in CATEGORICAL_FEATURES]

    for col in continuous:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in categorical:
        if col not in df.columns:
            df[col] = np.nan

    return df, continuous, categorical


def build_customer_features(
    statements: pd.DataFrame,
    continuous_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
) -> pd.DataFrame:
    """Build the customer-level feature matrix used by model training.

    This mirrors notebooks/02_feature_engineering.ipynb: full-history numeric
    aggregations, categorical aggregations, lag, first, 3M, 6M, and diff features.
    """
    df, continuous, categorical = _prepare_statement_frame(
        statements,
        continuous_features=continuous_features,
        categorical_features=categorical_features,
    )

    frames: list[pd.DataFrame] = []

    if continuous:
        df_agg = df.groupby(ID_COL)[continuous].agg(_FULL_AGG_STATS)
        df_agg.columns = ["_".join(col) for col in df_agg.columns]
        df_agg = df_agg.reset_index()
        frames.append(df_agg)
    else:
        df_agg = df[[ID_COL]].drop_duplicates().reset_index(drop=True)
        frames.append(df_agg)

    if categorical:
        cat_agg = df.groupby(ID_COL)[categorical].agg(["count", "last", "nunique"])
        cat_agg.columns = ["_".join(col) for col in cat_agg.columns]
        frames.append(cat_agg.reset_index())

    if continuous:
        lag_features = {
            f"{col}_lag": df_agg[f"{col}_last"] - df_agg[f"{col}_mean"]
            for col in continuous
        }
        lag_df = pd.concat(
            [df_agg[[ID_COL]], pd.DataFrame(lag_features)],
            axis=1,
        )
        frames.append(lag_df)

        first_df = df.groupby(ID_COL)[continuous].first()
        first_df.columns = [f"{col}_first" for col in first_df.columns]
        frames.append(first_df.reset_index())

        df_3m = df.groupby(ID_COL).tail(3)
        agg_3m = df_3m.groupby(ID_COL)[continuous].agg(_RECENT_AGG_STATS)
        agg_3m.columns = ["_".join(col) + "_3m" for col in agg_3m.columns]
        frames.append(agg_3m.reset_index())

        df_6m = df.groupby(ID_COL).tail(6)
        agg_6m = df_6m.groupby(ID_COL)[continuous].agg(_RECENT_AGG_STATS)
        agg_6m.columns = ["_".join(col) + "_6m" for col in agg_6m.columns]
        frames.append(agg_6m.reset_index())

        sort_cols = [ID_COL, DATE_COL] if DATE_COL in df.columns else [ID_COL]
        df_sorted = df.sort_values(sort_cols)
        diff_df = df_sorted.groupby(ID_COL)[continuous].diff()
        diff_df = pd.concat(
            [
                diff_df.reset_index(drop=True),
                df_sorted[[ID_COL]].reset_index(drop=True),
            ],
            axis=1,
        )
        diff_agg = diff_df.groupby(ID_COL).agg(_DIFF_AGG_STATS)
        diff_agg.columns = ["_".join(col) + "_diff" for col in diff_agg.columns]
        frames.append(diff_agg.reset_index())

    feature_frame = frames[0]
    for frame in frames[1:]:
        feature_frame = feature_frame.merge(frame, on=ID_COL, how="left")

    return feature_frame
