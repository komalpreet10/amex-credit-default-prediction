from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import lightgbm as lgb
import pandas as pd
from fastapi import FastAPI, HTTPException
from google.cloud import storage


MODEL_DIR = Path(os.getenv("AIP_MODEL_DIR", "/tmp/vertex_model"))
MODEL_URI = os.getenv("AIP_STORAGE_URI", "")
PREDICT_ROUTE = os.getenv("AIP_PREDICT_ROUTE", "/predict")
HEALTH_ROUTE = os.getenv("AIP_HEALTH_ROUTE", "/health")

app = FastAPI(title="AMEX LightGBM Vertex Serving")

_model: lgb.Booster | None = None
_feature_list: list[str] | None = None


def download_gcs_artifacts(uri: str, destination: Path) -> None:
    if not uri.startswith("gs://"):
        return
    bucket_name, prefix = uri.removeprefix("gs://").split("/", 1)
    prefix = prefix.rstrip("/")
    destination.mkdir(parents=True, exist_ok=True)
    bucket = storage.Client().bucket(bucket_name)
    for filename in ["model.txt", "selected_feature_list.json", "feature_list.json"]:
        blob = bucket.blob(f"{prefix}/{filename}")
        if not blob.exists():
            continue
        local_path = destination / filename
        blob.download_to_filename(local_path)


def ensure_model_artifacts() -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_URI.startswith("gs://") and not (MODEL_DIR / "model.txt").exists():
        download_gcs_artifacts(MODEL_URI, MODEL_DIR)
    return MODEL_DIR


def load_model() -> tuple[lgb.Booster, list[str]]:
    global _model, _feature_list
    if _model is not None and _feature_list is not None:
        return _model, _feature_list

    model_dir = ensure_model_artifacts()
    model_path = model_dir / "model.txt"
    feature_path = model_dir / "selected_feature_list.json"
    if not feature_path.exists():
        feature_path = model_dir / "feature_list.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_path}")
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature list artifact: {feature_path}")

    _model = lgb.Booster(model_file=str(model_path))
    _feature_list = json.loads(feature_path.read_text(encoding="utf-8"))
    return _model, _feature_list


def align_instances(instances: list[dict[str, Any]], feature_list: list[str]) -> pd.DataFrame:
    rows = []
    for instance in instances:
        rows.append({feature: instance.get(feature, 0.0) for feature in feature_list})
    return pd.DataFrame(rows, columns=feature_list)


def risk_category(probability: float) -> str:
    if probability < 0.25:
        return "low"
    if probability < 0.60:
        return "medium"
    return "high"


@app.on_event("startup")
def startup() -> None:
    load_model()


@app.get(HEALTH_ROUTE)
def health() -> dict[str, str]:
    load_model()
    return {"status": "ok"}


@app.post(PREDICT_ROUTE)
def predict(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    try:
        instances = payload.get("instances")
        if not isinstance(instances, list) or not instances:
            raise ValueError("Request body must include a non-empty 'instances' list.")
        model, feature_list = load_model()
        X = align_instances(instances, feature_list)
        probabilities = model.predict(X)
        return {
            "predictions": [
                {
                    "default_probability": float(probability),
                    "risk_category": risk_category(float(probability)),
                }
                for probability in probabilities
            ]
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
