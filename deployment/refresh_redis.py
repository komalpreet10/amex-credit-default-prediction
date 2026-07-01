from __future__ import annotations

import argparse
import json
import logging
import os
from decimal import Decimal
from typing import Any

import redis
from google.cloud import bigquery, storage

from amex_default.config import ID_COL
from amex_default.redis_config import redis_ssl_ca_certs
from gcp.config import (
    FEATURE_TABLE,
    REDIS_FEATURE_TTL_SECONDS,
    REDIS_PORT,
    REDIS_SSL_ENABLED,
    SELECTED_FEATURES_URI,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bq-table", default=FEATURE_TABLE)
    parser.add_argument("--features-uri", default=SELECTED_FEATURES_URI)
    parser.add_argument("--redis-port", type=int, default=REDIS_PORT)
    parser.add_argument("--batch-size", type=int, default=1000)
    return parser.parse_args()


def load_selected_features(uri: str) -> list[str]:
    if not uri.startswith("gs://"):
        raise ValueError("--features-uri must be a GCS URI.")
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
    return [str(feature) for feature in features]


def quote_identifier(name: str) -> str:
    return f"`{name.replace('`', '')}`"


def json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def flush_pipeline(pipe: redis.client.Pipeline, count: int) -> None:
    if count:
        pipe.execute()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    redis_host = os.environ.get("REDIS_HOST")
    if not redis_host:
        raise RuntimeError("REDIS_HOST environment variable is required.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    selected_features = load_selected_features(args.features_uri)
    columns = [ID_COL, *selected_features]
    query = (
        "SELECT "
        + ", ".join(quote_identifier(column) for column in columns)
        + f" FROM `{args.bq_table}`"
    )

    bq_client = bigquery.Client()
    redis_client = redis.Redis(
        host=redis_host,
        port=args.redis_port,
        decode_responses=True,
        socket_timeout=10,
        ssl=REDIS_SSL_ENABLED,
        ssl_ca_certs=redis_ssl_ca_certs(),
    )
    redis_client.ping()

    LOGGER.info(
        "Reading %d selected features from %s", len(selected_features), args.bq_table
    )
    rows = bq_client.query(query).result(page_size=args.batch_size)

    pipe = redis_client.pipeline(transaction=False)
    pending = 0
    total = 0
    for row in rows:
        record = dict(row.items())
        customer_id = record.pop(ID_COL)
        key = f"features:{customer_id}"
        payload = json.dumps(record, default=json_default, separators=(",", ":"))
        pipe.setex(key, REDIS_FEATURE_TTL_SECONDS, payload)
        pending += 1
        total += 1

        if pending >= args.batch_size:
            flush_pipeline(pipe, pending)
            pending = 0

        if total % 10_000 == 0:
            LOGGER.info("Refreshed Redis features for %d customers", total)

    flush_pipeline(pipe, pending)
    LOGGER.info("Completed Redis refresh for %d customers", total)


if __name__ == "__main__":
    main()
