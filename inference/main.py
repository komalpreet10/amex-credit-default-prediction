from __future__ import annotations

import json
import logging
import os
from typing import Any

import functions_framework
from google.cloud import aiplatform, bigquery, storage
from google.cloud.aiplatform_v1 import FeatureOnlineStoreServiceClient
from google.cloud.aiplatform_v1.types import (
    feature_online_store_service as feature_store_service,
)
from google.protobuf.json_format import MessageToDict

from gcp.config import (
    CUSTOMER_FEATURES_TABLE,
    FEATURE_STORE_NAME,
    FEATURE_VIEW_NAME,
    PROJECT_ID,
    REGION,
    SELECTED_FEATURES_URI,
)

LOGGER = logging.getLogger(__name__)

_bq_client: bigquery.Client | None = None
_endpoint: aiplatform.Endpoint | None = None
_feature_store_client: FeatureOnlineStoreServiceClient | None = None
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
    global _bq_client, _endpoint, _feature_store_client
    if (
        _bq_client is not None
        and _endpoint is not None
        and _feature_store_client is not None
    ):
        return

    project = os.environ.get("PROJECT_ID", os.environ.get("GCP_PROJECT_ID", PROJECT_ID))
    endpoint_id = required_env("VERTEX_ENDPOINT_ID")
    location = os.environ.get("LOCATION", os.environ.get("REGION", REGION))
    feature_store_location = os.environ.get("FEATURE_STORE_LOCATION", location)

    _bq_client = bigquery.Client(project=project)
    _feature_store_client = FeatureOnlineStoreServiceClient(
        client_options={
            "api_endpoint": f"{feature_store_location}-aiplatform.googleapis.com"
        }
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


def feature_view_resource_name() -> str:
    if _feature_store_client is None:
        raise RuntimeError("Vertex AI Feature Store client is not initialized.")
    project = os.environ.get(
        "FEATURE_STORE_PROJECT", os.environ.get("PROJECT_ID", PROJECT_ID)
    )
    location = os.environ.get(
        "FEATURE_STORE_LOCATION",
        os.environ.get("LOCATION", os.environ.get("REGION", REGION)),
    )
    feature_store = os.environ.get("FEATURE_STORE_NAME", FEATURE_STORE_NAME)
    feature_view = os.environ.get("FEATURE_VIEW_NAME", FEATURE_VIEW_NAME)
    if feature_view.startswith("projects/"):
        return feature_view
    return _feature_store_client.feature_view_path(
        project=project,
        location=location,
        feature_online_store=feature_store,
        feature_view=feature_view,
    )


def lookup_vertex_feature_store(customer_id: str) -> dict[str, Any] | None:
    if _feature_store_client is None:
        raise RuntimeError("Vertex AI Feature Store client is not initialized.")
    request = feature_store_service.FetchFeatureValuesRequest(
        feature_view=feature_view_resource_name(),
        data_key=feature_store_service.FeatureViewDataKey(key=customer_id),
        data_format=feature_store_service.FeatureViewDataFormat.PROTO_STRUCT,
    )
    response = _feature_store_client.fetch_feature_values(request=request)
    if response.proto_struct:
        return MessageToDict(response.proto_struct)
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
        row = lookup_vertex_feature_store(customer_id)
        if row is not None:
            return row, "VERTEX_AI_FEATURE_STORE"
        LOGGER.info("No Feature Store row found for customer_ID=%s", customer_id)
    except Exception:
        LOGGER.exception(
            "Feature Store lookup failed for customer_ID=%s; falling back to BigQuery",
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
