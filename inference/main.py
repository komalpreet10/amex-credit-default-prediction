from __future__ import annotations

import json
import logging
import os
from typing import Any

import functions_framework
import redis
from google.cloud import aiplatform, bigquery, storage

from gcp.config import (
    CUSTOMER_FEATURES_TABLE,
    PROJECT_ID,
    REGION,
    SELECTED_FEATURES_URI,
)

LOGGER = logging.getLogger(__name__)

_bq_client: bigquery.Client | None = None
_endpoint: aiplatform.Endpoint | None = None
_redis_client: redis.Redis | None = None
_selected_features: list[str] | None = None


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} environment variable is required.")
    return value


def load_selected_features() -> list[str]:
    global _selected_features
    if _selected_features is not None:
        return _selected_features

    uri = os.environ.get("SELECTED_FEATURES_URI", SELECTED_FEATURES_URI)
    if not uri.startswith("gs://"):
        raise ValueError("SELECTED_FEATURES_URI must be a GCS URI.")
    bucket_name, blob_name = uri.removeprefix("gs://").split("/", 1)
    payload = (
        storage.Client()
        .bucket(bucket_name)
        .blob(blob_name)
        .download_as_text(encoding="utf-8")
    )
    features = json.loads(payload)
    if not isinstance(features, list) or not features:
        raise ValueError("Selected feature list must be a non-empty JSON list.")
    _selected_features = [str(feature) for feature in features]
    return _selected_features


def endpoint_resource_name(project: str, location: str, endpoint_id: str) -> str:
    if endpoint_id.startswith("projects/"):
        return endpoint_id
    return f"projects/{project}/locations/{location}/endpoints/{endpoint_id}"


def init_clients() -> None:
    global _bq_client, _endpoint, _redis_client
    if (
        _bq_client is not None
        and _endpoint is not None
        and _redis_client is not None
    ):
        return

    project = os.environ.get("PROJECT_ID", os.environ.get("GCP_PROJECT_ID", PROJECT_ID))
    endpoint_id = required_env("VERTEX_ENDPOINT_ID")
    location = os.environ.get("LOCATION", os.environ.get("REGION", REGION))

    _bq_client = bigquery.Client(project=project)
    _redis_client = redis.Redis(
        host=os.environ["REDIS_HOST"],
        port=int(os.environ.get("REDIS_PORT", "6379")),
        db=int(os.environ.get("REDIS_DB", "0")),
        ssl=os.environ.get("REDIS_SSL", "false").lower() in {"1", "true", "yes"},
        decode_responses=True,
    )
    aiplatform.init(project=project, location=location)
    _endpoint = aiplatform.Endpoint(
        endpoint_name=endpoint_resource_name(project, location, endpoint_id)
    )
    load_selected_features()


def quote_identifier(name: str) -> str:
    return f"`{name.replace('`', '')}`"


def selected_feature_vector(row: dict[str, Any], features: list[str]) -> dict[str, Any]:
    return {feature: row.get(feature, 0.0) for feature in features}


def redis_key_prefix() -> str:
    return os.environ.get("REDIS_KEY_PREFIX", "amex")


def feature_vector_key(customer_id: str) -> str:
    return f"{redis_key_prefix()}:features:{customer_id}"


def predict_risk(feature_vector: dict[str, Any]) -> tuple[float, str | None]:
    if _endpoint is None:
        raise RuntimeError("Vertex AI Endpoint client is not initialized.")
    response = _endpoint.predict(instances=[feature_vector])
    prediction = response.predictions[0]
    if isinstance(prediction, dict):
        risk_score = prediction.get("default_probability", prediction.get("risk_score"))
    else:
        risk_score = prediction
    return float(risk_score), getattr(response, "deployed_model_id", None)


def lookup_realtime_feature_cache(customer_id: str) -> dict[str, Any] | None:
    if _redis_client is None:
        raise RuntimeError("Redis client is not initialized.")
    payload = _redis_client.get(feature_vector_key(customer_id))
    if not payload:
        return None
    cache_entry = json.loads(payload)
    features = cache_entry.get("features")
    if isinstance(features, dict):
        return features
    return None


def lookup_bigquery_features(
    customer_id: str, features: list[str]
) -> dict[str, Any] | None:
    if _bq_client is None:
        raise RuntimeError("BigQuery client is not initialized.")
    table = os.environ.get("BQ_TABLE", CUSTOMER_FEATURES_TABLE)
    query = (
        "SELECT "
        + ", ".join(quote_identifier(feature) for feature in features)
        + f" FROM `{table}` WHERE customer_ID = @customer_id LIMIT 1"
    )
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("customer_id", "STRING", customer_id)
        ]
    )
    rows = list(_bq_client.query(query, job_config=job_config).result())
    if not rows:
        return None
    return dict(rows[0].items())


def lookup_features(
    customer_id: str, features: list[str]
) -> tuple[dict[str, Any] | None, str]:
    try:
        row = lookup_realtime_feature_cache(customer_id)
        if row is not None:
            return row, "REDIS_REALTIME_FEATURE_CACHE"
    except Exception:
        LOGGER.exception(
            "Realtime feature cache lookup failed for customer_ID=%s; falling back",
            customer_id,
        )

    row = lookup_bigquery_features(customer_id, features)
    if row is not None:
        return row, "BIGQUERY_FEATURE_FALLBACK"
    return None, "INSUFFICIENT_DATA"


@functions_framework.http
def score(request):
    if request.method != "POST":
        return ({"error": "Only POST is supported."}, 405)

    body = request.get_json(silent=True) or {}
    customer_id = body.get("customer_ID")
    if not customer_id:
        return ({"error": "customer_ID is required."}, 400)
    customer_id = str(customer_id)

    init_clients()
    features = load_selected_features()

    row, tier_used = lookup_features(customer_id, features)
    if row is None:
        return {
            "customer_ID": customer_id,
            "risk_score": None,
            "tier_used": tier_used,
        }

    feature_vector = selected_feature_vector(row, features)
    risk_score, model_version = predict_risk(feature_vector)
    return {
        "customer_ID": customer_id,
        "risk_score": risk_score,
        "tier_used": tier_used,
        "model_version": model_version,
    }
