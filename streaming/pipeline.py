from __future__ import annotations

import argparse
import json
import logging
import math
import time
from typing import Any

import apache_beam as beam
import pandas as pd
import redis
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions
from google.cloud import bigquery, pubsub_v1, storage

from amex_default.config import DATE_COL, ID_COL
from amex_default.features import build_customer_features, infer_continuous_features
from amex_default.redis_config import redis_ssl_ca_certs
from gcp.config import (
    PROJECT_ID,
    REDIS_FEATURE_TTL_SECONDS,
    REDIS_PORT,
    REDIS_SSL_ENABLED,
    REDIS_NETWORK,
    REGION,
    SELECTED_FEATURES_URI,
    STATEMENT_DLQ_TOPIC,
    STATEMENT_HISTORY_TABLE,
    STATEMENT_SUBSCRIPTION_PATH,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=PROJECT_ID)
    parser.add_argument("--region", default=REGION)
    parser.add_argument("--redis-host", required=True)
    parser.add_argument("--selected-features-uri", default=SELECTED_FEATURES_URI)
    parser.add_argument("--statement-history-table", default=STATEMENT_HISTORY_TABLE)
    parser.add_argument("--subscription", default=STATEMENT_SUBSCRIPTION_PATH)
    parser.add_argument("--network", default=REDIS_NETWORK)
    parser.add_argument("--subnetwork", default=None)
    return parser.parse_known_args()


def read_gcs_json(uri: str) -> Any:
    bucket_name, blob_name = uri.removeprefix("gs://").split("/", 1)
    payload = (
        storage.Client()
        .bucket(bucket_name)
        .blob(blob_name)
        .download_as_text(encoding="utf-8")
    )
    return json.loads(payload)


def quote_identifier(name: str) -> str:
    return f"`{name.replace('`', '')}`"


def to_json_value(value: Any) -> Any:
    if pd.isna(value):
        return 0.0
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return 0.0
    return value


class RebuildRedisFeatures(beam.DoFn):
    def __init__(
        self,
        redis_host: str,
        selected_features_uri: str,
        statement_history_table: str,
        project: str,
    ):
        self.redis_host = redis_host
        self.selected_features_uri = selected_features_uri
        self.statement_history_table = statement_history_table
        self.project = project
        self.redis_client = None
        self.bq_client = None
        self.selected_features = None
        self.continuous_features = None
        self.publisher = None
        self.dlq_topic_path = None

    def setup(self) -> None:
        self.redis_client = redis.Redis(
            host=self.redis_host,
            port=REDIS_PORT,
            decode_responses=True,
            socket_timeout=10,
            ssl=REDIS_SSL_ENABLED,
            ssl_ca_certs=redis_ssl_ca_certs(),
        )
        self.redis_client.ping()
        self.bq_client = bigquery.Client(project=self.project)
        self.selected_features = read_gcs_json(self.selected_features_uri)
        if not isinstance(self.selected_features, list) or not self.selected_features:
            raise ValueError("Selected feature list must be a non-empty JSON list.")
        self.selected_features = [str(feature) for feature in self.selected_features]
        self.continuous_features = infer_continuous_features(self.selected_features)
        self.publisher = pubsub_v1.PublisherClient()
        self.dlq_topic_path = self.publisher.topic_path(
            self.project, STATEMENT_DLQ_TOPIC
        )

    def publish_dlq(
        self, raw_message: bytes, customer_id: str | None, error: Exception
    ) -> None:
        if self.publisher is None or self.dlq_topic_path is None:
            raise RuntimeError("DLQ publisher is not initialized.")
        self.publisher.publish(
            self.dlq_topic_path,
            raw_message,
            customer_ID=customer_id or "",
            error=str(error),
        )

    def read_statement_history(
        self,
        customer_id: str,
        raw_statement_fields: dict[str, Any],
    ) -> pd.DataFrame:
        if self.bq_client is None:
            raise RuntimeError("BigQuery client is not initialized.")

        query = (
            f"SELECT * FROM `{self.statement_history_table}` "
            f"WHERE {quote_identifier(ID_COL)} = @customer_id "
            f"ORDER BY {quote_identifier(DATE_COL)}"
        )
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("customer_id", "STRING", customer_id)
            ]
        )
        rows = [
            dict(row.items())
            for row in self.bq_client.query(query, job_config=job_config).result()
        ]

        event_row = {**raw_statement_fields, ID_COL: customer_id}
        if DATE_COL in event_row and not self.history_contains_event(rows, event_row):
            rows.append(event_row)

        if not rows:
            raise ValueError(f"No statement history found for customer_ID={customer_id}.")
        return pd.DataFrame(rows)

    def history_contains_event(
        self,
        rows: list[dict[str, Any]],
        event_row: dict[str, Any],
    ) -> bool:
        event_date = event_row.get(DATE_COL)
        if event_date is None:
            return False
        return any(str(row.get(DATE_COL)) == str(event_date) for row in rows)

    def rebuild_feature_vector(
        self,
        customer_id: str,
        raw_statement_fields: dict[str, Any],
    ) -> dict[str, Any]:
        if self.selected_features is None or self.continuous_features is None:
            raise RuntimeError("Feature metadata is not initialized.")

        statements = self.read_statement_history(customer_id, raw_statement_fields)
        engineered = build_customer_features(
            statements,
            continuous_features=self.continuous_features,
        )
        if len(engineered) != 1:
            raise ValueError(
                f"Expected one engineered row for customer_ID={customer_id}, "
                f"got {len(engineered)}."
            )

        row = engineered.iloc[0].to_dict()
        return {
            feature: to_json_value(row.get(feature, 0.0))
            for feature in self.selected_features
        }

    def process(self, message: bytes):
        customer_id = None
        for attempt in range(1, 4):
            try:
                if self.redis_client is None:
                    raise RuntimeError("Redis client is not initialized.")
                payload = json.loads(message.decode("utf-8"))
                customer_id = payload.get("customer_ID")
                raw_statement_fields = payload.get("raw_statement_fields")
                if not customer_id:
                    raise ValueError("Missing customer_ID in Pub/Sub message.")
                if not isinstance(raw_statement_fields, dict):
                    raise ValueError("raw_statement_fields must be a dictionary.")

                key = f"features:{customer_id}"
                feature_vector = self.rebuild_feature_vector(
                    customer_id,
                    raw_statement_fields,
                )
                self.redis_client.setex(
                    key,
                    REDIS_FEATURE_TTL_SECONDS,
                    json.dumps(feature_vector, separators=(",", ":")),
                )
                yield {
                    "customer_ID": customer_id,
                    "updated_feature_count": len(feature_vector),
                }
                return
            except Exception as exc:
                LOGGER.exception(
                    "Streaming update failed for customer_ID=%s on attempt=%d",
                    customer_id,
                    attempt,
                )
                if attempt == 3:
                    self.publish_dlq(message, customer_id, exc)
                    return
                time.sleep(2**attempt)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args, beam_args = parse_args()
    options = PipelineOptions(
        beam_args,
        project=args.project,
        region=args.region,
        runner="DataflowRunner",
        streaming=True,
        save_main_session=True,
        experiments=["enable_streaming_engine"],
        network=args.network,
        subnetwork=args.subnetwork,
    )
    options.view_as(StandardOptions).streaming = True

    with beam.Pipeline(options=options) as pipeline:
        (
            pipeline
            | "Read PubSub" >> beam.io.ReadFromPubSub(subscription=args.subscription)
            | "Rebuild Redis Features"
            >> beam.ParDo(
                RebuildRedisFeatures(
                    redis_host=args.redis_host,
                    selected_features_uri=args.selected_features_uri,
                    statement_history_table=args.statement_history_table,
                    project=args.project,
                )
            )
        )


if __name__ == "__main__":
    main()
