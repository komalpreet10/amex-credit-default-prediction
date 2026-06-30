from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Any

import apache_beam as beam
import redis
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions
from google.cloud import pubsub_v1, storage

TTL_SECONDS = 2_592_000
DEFAULT_SUBSCRIPTION = (
    "projects/amex-credit-risk-ml/subscriptions/statement-cycle-close-sub"
)
DEFAULT_FEATURE_CONFIG = "gs://amex-credit-risk-ml-data/config/streaming_features.json"
DLQ_TOPIC = "statement-cycle-close-dlq"

LOGGER = logging.getLogger(__name__)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="amex-credit-risk-ml")
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--redis-host", required=True)
    parser.add_argument("--streaming-features-uri", default=DEFAULT_FEATURE_CONFIG)
    parser.add_argument("--subscription", default=DEFAULT_SUBSCRIPTION)
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


def resolve_feature_updates(
    raw_statement_fields: dict[str, Any],
    feature_config: Any,
) -> dict[str, Any]:
    updates = {}
    if isinstance(feature_config, list):
        for feature in feature_config:
            if feature in raw_statement_fields:
                updates[feature] = raw_statement_fields[feature]
        return updates

    if isinstance(feature_config, dict):
        for feature, source in feature_config.items():
            if isinstance(source, str):
                raw_key = source
            elif isinstance(source, dict):
                raw_key = source.get("source", feature)
            else:
                raw_key = feature
            if raw_key in raw_statement_fields:
                updates[feature] = raw_statement_fields[raw_key]
    return updates


class UpdateRedisFeatures(beam.DoFn):
    def __init__(self, redis_host: str, streaming_features_uri: str, project: str):
        self.redis_host = redis_host
        self.streaming_features_uri = streaming_features_uri
        self.project = project
        self.redis_client = None
        self.feature_config = None
        self.publisher = None
        self.dlq_topic_path = None

    def setup(self) -> None:
        self.redis_client = redis.Redis(
            host=self.redis_host,
            port=6379,
            decode_responses=True,
            socket_timeout=10,
        )
        self.redis_client.ping()
        self.feature_config = read_gcs_json(self.streaming_features_uri)
        self.publisher = pubsub_v1.PublisherClient()
        self.dlq_topic_path = self.publisher.topic_path(self.project, DLQ_TOPIC)

    def publish_dlq(self, raw_message: bytes, customer_id: str | None, error: Exception) -> None:
        if self.publisher is None or self.dlq_topic_path is None:
            raise RuntimeError("DLQ publisher is not initialized.")
        self.publisher.publish(
            self.dlq_topic_path,
            raw_message,
            customer_ID=customer_id or "",
            error=str(error),
        )

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
                existing = self.redis_client.get(key)
                feature_vector = json.loads(existing) if existing else {}
                feature_vector.update(
                    resolve_feature_updates(raw_statement_fields, self.feature_config)
                )
                self.redis_client.setex(
                    key,
                    TTL_SECONDS,
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
                time.sleep(2 ** attempt)


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
    )
    options.view_as(StandardOptions).streaming = True

    with beam.Pipeline(options=options) as pipeline:
        (
            pipeline
            | "Read PubSub"
            >> beam.io.ReadFromPubSub(subscription=args.subscription)
            | "Update Redis Features"
            >> beam.ParDo(
                UpdateRedisFeatures(
                    redis_host=args.redis_host,
                    streaming_features_uri=args.streaming_features_uri,
                    project=args.project,
                )
            )
        )


if __name__ == "__main__":
    main()
