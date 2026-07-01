from __future__ import annotations

import json
import logging
import os
from typing import Any

import functions_framework
import redis
from google.cloud import aiplatform, bigquery, storage

from amex_default.redis_config import redis_ssl_ca_certs
from gcp.config import (
    FEATURE_TABLE,
    PROJECT_ID,
    REDIS_FEATURE_TTL_SECONDS,
    REDIS_PORT,
    REDIS_SSL_ENABLED,
    REGION,
    SELECTED_FEATURES_URI,
)

LOGGER = logging.getLogger(__name__)

_redis_client: redis.Redis | None = None
_bq_client: bigquery.Client | None = None
_endpoint: aiplatform.Endpoint | None = None
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
    global _redis_client, _bq_client, _endpoint
    if _redis_client is not None and _bq_client is not None and _endpoint is not None:
        return

    project = os.environ.get("PROJECT_ID", os.environ.get("GCP_PROJECT_ID", PROJECT_ID))
    redis_host = required_env("REDIS_HOST")
    endpoint_id = required_env("VERTEX_ENDPOINT_ID")
    location = os.environ.get("LOCATION", os.environ.get("REGION", REGION))
    redis_port = int(os.environ.get("REDIS_PORT", str(REDIS_PORT)))

    _redis_client = redis.Redis(
        host=redis_host,
        port=redis_port,
        decode_responses=True,
        socket_timeout=10,
        ssl=REDIS_SSL_ENABLED,
        ssl_ca_certs=redis_ssl_ca_certs(),
    )
    _redis_client.ping()
    _bq_client = bigquery.Client(project=project)
    aiplatform.init(project=project, location=location)
    _endpoint = aiplatform.Endpoint(
        endpoint_name=endpoint_resource_name(project, location, endpoint_id)
    )
    load_selected_features()


def quote_identifier(name: str) -> str:
    return f"`{name.replace('`', '')}`"


def selected_feature_vector(row: dict[str, Any], features: list[str]) -> dict[str, Any]:
    return {feature: row.get(feature, 0.0) for feature in features}


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


def write_through_redis(customer_id: str, feature_vector: dict[str, Any]) -> None:
    if _redis_client is None:
        raise RuntimeError("Redis client is not initialized.")
    try:
        _redis_client.setex(
            f"features:{customer_id}",
            REDIS_FEATURE_TTL_SECONDS,
            json.dumps(feature_vector, separators=(",", ":")),
        )
    except Exception:
        LOGGER.warning(
            "Redis write-through failed for customer_ID=%s", customer_id, exc_info=True
        )


def lookup_bigquery(customer_id: str, features: list[str]) -> dict[str, Any] | None:
    if _bq_client is None:
        raise RuntimeError("BigQuery client is not initialized.")
    table = os.environ.get("BQ_TABLE", FEATURE_TABLE)
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
    if _redis_client is None:
        raise RuntimeError("Redis client is not initialized.")

    cached = _redis_client.get(f"features:{customer_id}")
    if cached:
        feature_vector = selected_feature_vector(json.loads(cached), features)
        risk_score, model_version = predict_risk(feature_vector)
        return {
            "customer_ID": customer_id,
            "risk_score": risk_score,
            "tier_used": "REDIS_CACHE",
            "model_version": model_version,
        }

    row = lookup_bigquery(customer_id, features)
    if row is None:
        return {
            "customer_ID": customer_id,
            "risk_score": None,
            "tier_used": "INSUFFICIENT_DATA",
        }

    feature_vector = selected_feature_vector(row, features)
    write_through_redis(customer_id, feature_vector)
    risk_score, model_version = predict_risk(feature_vector)
    return {
        "customer_ID": customer_id,
        "risk_score": risk_score,
        "tier_used": "BIGQUERY_LOOKUP",
        "model_version": model_version,
    }
