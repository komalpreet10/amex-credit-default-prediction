from __future__ import annotations

from pathlib import Path

import pandas as pd

from amex_default.config import (
    ID_COL,
    TARGET_COL,
    TRAIN_FEATURES_PATH,
)


def load_train_features(path: str | Path = TRAIN_FEATURES_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in {ID_COL, TARGET_COL}]


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = get_feature_columns(df)
    return df[feature_cols], df[TARGET_COL]
