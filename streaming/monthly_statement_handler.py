from __future__ import annotations

import base64
import json
import logging
import os
from datetime import UTC, datetime
from typing import Any

import functions_framework
import pandas as pd
import redis
from google.cloud import bigquery
from google.cloud import storage

from amex_default.config import CATEGORICAL_FEATURES, DATE_COL, ID_COL
from amex_default.features import build_customer_features, infer_continuous_features
from gcp.config import (
    CHANGED_CUSTOMERS_TABLE,
    PROJECT_ID,
    SELECTED_FEATURES_URI,
    STATEMENT_HISTORY_TABLE,
)

LOGGER = logging.getLogger(__name__)

_bq_client: bigquery.Client | None = None
_redis_client: redis.Redis | None = None
_selected_features: list[str] | None = None


def bq_client(project: str) -> bigquery.Client:
    global _bq_client
    if _bq_client is None:
        _bq_client = bigquery.Client(project=project)
    return _bq_client


def redis_client() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=os.environ["REDIS_HOST"],
            port=int(os.environ.get("REDIS_PORT", "6379")),
            db=int(os.environ.get("REDIS_DB", "0")),
            ssl=os.environ.get("REDIS_SSL", "false").lower()
            in {"1", "true", "yes"},
            decode_responses=True,
        )
    return _redis_client


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def load_selected_features() -> list[str]:
    global _selected_features
    if _selected_features is not None:
        return _selected_features

    uri = env("SELECTED_FEATURES_URI", SELECTED_FEATURES_URI)
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


def parse_pubsub_payload(cloud_event) -> dict[str, Any]:
    message = cloud_event.data.get("message", {})
    encoded = message.get("data")
    if not encoded:
        raise ValueError("Pub/Sub message is missing data.")
    payload = json.loads(base64.b64decode(encoded).decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Monthly statement payload must be a JSON object.")
    if not payload.get(ID_COL):
        raise ValueError(f"Monthly statement payload is missing {ID_COL}.")
    return payload


def statement_cycle(payload: dict[str, Any]) -> str:
    explicit_cycle = payload.get("statement_cycle")
    if explicit_cycle:
        return str(explicit_cycle)
    statement_date = payload.get(DATE_COL)
    if isinstance(statement_date, str) and len(statement_date) >= 7:
        return statement_date[:7]
    return datetime.now(UTC).strftime("%Y-%m")


def insert_statement(
    client: bigquery.Client,
    table: str,
    payload: dict[str, Any],
) -> None:
    errors = client.insert_rows_json(table, [payload])
    if errors:
        raise RuntimeError(f"Failed to insert monthly statement into {table}: {errors}")


def record_changed_customer(
    client: bigquery.Client,
    table: str,
    customer_id: str,
    cycle: str,
) -> None:
    row = {
        ID_COL: customer_id,
        "statement_cycle": cycle,
        "source_table": env("STATEMENT_HISTORY_TABLE", STATEMENT_HISTORY_TABLE),
        "updated_at": datetime.now(UTC).isoformat(),
    }
    errors = client.insert_rows_json(table, [row])
    if errors:
        raise RuntimeError(f"Failed to insert changed customer into {table}: {errors}")


def redis_key_prefix() -> str:
    return env("REDIS_KEY_PREFIX", "amex")


def history_limit() -> int:
    return int(os.environ.get("CUSTOMER_HISTORY_LIMIT", "13"))


def statement_history_key(customer_id: str) -> str:
    return f"{redis_key_prefix()}:statements:{customer_id}"


def feature_vector_key(customer_id: str) -> str:
    return f"{redis_key_prefix()}:features:{customer_id}"


def update_customer_feature_cache(
    payload: dict[str, Any],
    cycle: str,
) -> dict[str, Any]:
    selected_features = load_selected_features()
    customer_id = str(payload[ID_COL])
    redis_conn = redis_client()
    history_key = statement_history_key(customer_id)
    features_key = feature_vector_key(customer_id)

    statement = dict(payload)
    statement["statement_cycle"] = cycle
    pipeline = redis_conn.pipeline()
    pipeline.lpush(history_key, json.dumps(statement, default=str))
    pipeline.ltrim(history_key, 0, history_limit() - 1)
    pipeline.execute()

    statements = [
        json.loads(item)
        for item in redis_conn.lrange(history_key, 0, history_limit() - 1)
    ]

    frame = pd.DataFrame(list(reversed(statements)))
    continuous_features = infer_continuous_features(selected_features)
    categorical_features = [
        feature
        for feature in CATEGORICAL_FEATURES
        if feature in frame.columns
    ]
    engineered = build_customer_features(
        frame,
        continuous_features=continuous_features,
        categorical_features=categorical_features,
    )
    if engineered.empty:
        raise ValueError(f"No engineered features produced for {customer_id}.")

    row = engineered.iloc[0].to_dict()
    feature_vector = {
        feature: normalize_feature_value(row.get(feature, 0.0))
        for feature in selected_features
    }
    cache_payload = {
        ID_COL: customer_id,
        "statement_cycle": cycle,
        "features": feature_vector,
        "selected_feature_count": len(selected_features),
        "updated_at": datetime.now(UTC).isoformat(),
    }

    redis_conn.set(features_key, json.dumps(cache_payload, default=str))
    return feature_vector


def normalize_feature_value(value: Any) -> Any:
    if pd.isna(value):
        return 0.0
    if hasattr(value, "item"):
        return value.item()
    return value


@functions_framework.cloud_event
def ingest_monthly_statement(cloud_event):
    logging.basicConfig(level=logging.INFO)
    project = env("PROJECT_ID", env("GCP_PROJECT_ID", PROJECT_ID))
    payload = parse_pubsub_payload(cloud_event)
    customer_id = str(payload[ID_COL])
    cycle = statement_cycle(payload)
    client = bq_client(project)

    insert_statement(
        client=client,
        table=env("STATEMENT_HISTORY_TABLE", STATEMENT_HISTORY_TABLE),
        payload=payload,
    )
    record_changed_customer(
        client=client,
        table=env("CHANGED_CUSTOMERS_TABLE", CHANGED_CUSTOMERS_TABLE),
        customer_id=customer_id,
        cycle=cycle,
    )
    feature_vector = update_customer_feature_cache(
        payload=payload,
        cycle=cycle,
    )
    LOGGER.info(
        "Updated Redis realtime feature cache for customer_ID=%s statement_cycle=%s features=%d",
        customer_id,
        cycle,
        len(feature_vector),
    )
