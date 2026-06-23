from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from amex_default.config import MODEL_DIR
from amex_default.features import build_customer_features, infer_continuous_features


def load_feature_list(path: str | Path | None = None) -> list[str]:
    path = Path(path or MODEL_DIR / "final" / "feature_list.json")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_categorical_feature_list(path: str | Path | None = None) -> list[str]:
    path = Path(path or MODEL_DIR / "final" / "categorical_feature_list.json")
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_final_model(path: str | Path | None = None):
    path = Path(path or MODEL_DIR / "final" / "final_model.txt")
    return lgb.Booster(model_file=str(path))


def _categorical_feature_map(
    model,
    feature_list: list[str],
    categorical_feature_list: list[str],
) -> dict[str, list[str]]:
    categories = getattr(model, "pandas_categorical", None) or []
    cat_features = [
        feature for feature in categorical_feature_list if feature in feature_list
    ]
    return {
        feature: list(values)
        for feature, values in zip(cat_features, categories, strict=False)
    }


def align_features(
    features: dict[str, float | str],
    feature_list: list[str],
    model=None,
    categorical_feature_list: list[str] | None = None,
) -> pd.DataFrame:
    categorical_map = (
        _categorical_feature_map(
            model,
            feature_list,
            categorical_feature_list or [],
        )
        if model
        else {}
    )
    row = {}
    for feature in feature_list:
        if feature in categorical_map:
            values = categorical_map[feature]
            row[feature] = features.get(feature, values[0] if values else "")
        else:
            row[feature] = features.get(feature, 0.0)

    frame = pd.DataFrame([row], columns=feature_list)
    for feature, values in categorical_map.items():
        frame[feature] = pd.Categorical(frame[feature], categories=values)
    return frame


def predict_default_probability(model, features: dict[str, float | str]) -> float:
    feature_list = load_feature_list()
    categorical_feature_list = load_categorical_feature_list()
    X = align_features(
        features,
        feature_list,
        model=model,
        categorical_feature_list=categorical_feature_list,
    )
    prediction = model.predict(X)
    return float(prediction[0])


def predict_default_probability_from_frame(model, features: pd.DataFrame) -> float:
    feature_list = load_feature_list()
    categorical_feature_list = load_categorical_feature_list()
    feature_dict = features.iloc[0].to_dict()
    X = align_features(
        feature_dict,
        feature_list,
        model=model,
        categorical_feature_list=categorical_feature_list,
    )
    prediction = model.predict(X)
    return float(prediction[0])


def predict_default_probability_from_statements(
    model,
    statements: list[dict[str, object]] | pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    feature_list = load_feature_list()
    statement_frame = (
        statements.copy()
        if isinstance(statements, pd.DataFrame)
        else pd.DataFrame(statements)
    )
    engineered_features = build_customer_features(
        statement_frame,
        continuous_features=infer_continuous_features(feature_list),
    )
    if len(engineered_features) != 1:
        raise ValueError(
            "Raw prediction expects statement rows for exactly one customer_ID."
        )
    probability = predict_default_probability_from_frame(model, engineered_features)
    return probability, engineered_features


def assign_risk_category(probability: float) -> str:
    if probability < 0.25:
        return "low"
    if probability < 0.60:
        return "medium"
    return "high"
