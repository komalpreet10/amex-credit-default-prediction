from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from typing import Any

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions
from google.cloud import pubsub_v1

from amex_default.config import ID_COL
from gcp.config import (
    CHANGED_CUSTOMERS_TABLE,
    PROJECT_ID,
    REGION,
    STATEMENT_DLQ_TOPIC,
    STATEMENT_SUBSCRIPTION_PATH,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=PROJECT_ID)
    parser.add_argument("--region", default=REGION)
    parser.add_argument("--subscription", default=STATEMENT_SUBSCRIPTION_PATH)
    parser.add_argument("--changed-customers-table", default=CHANGED_CUSTOMERS_TABLE)
    return parser.parse_known_args()


class ExtractChangedCustomer(beam.DoFn):
    def __init__(self, project: str):
        self.project = project
        self.publisher = None
        self.dlq_topic_path = None

    def setup(self) -> None:
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

    def process(self, message: bytes):
        customer_id = None
        try:
            payload = json.loads(message.decode("utf-8"))
            customer_id = payload.get(ID_COL)
            if not customer_id:
                raise ValueError(f"Missing {ID_COL} in Pub/Sub message.")

            now = datetime.now(UTC).isoformat()
            yield {
                ID_COL: str(customer_id),
                "statement_cycle": payload.get("statement_cycle", ""),
                "source_table": payload.get("source_table", ""),
                "updated_at": now,
            }
        except Exception as exc:
            LOGGER.exception("Changed-customer extraction failed")
            self.publish_dlq(message, customer_id, exc)


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

    schema = {
        "fields": [
            {"name": ID_COL, "type": "STRING", "mode": "REQUIRED"},
            {"name": "statement_cycle", "type": "STRING", "mode": "NULLABLE"},
            {"name": "source_table", "type": "STRING", "mode": "NULLABLE"},
            {"name": "updated_at", "type": "TIMESTAMP", "mode": "REQUIRED"},
        ]
    }

    with beam.Pipeline(options=options) as pipeline:
        (
            pipeline
            | "Read PubSub" >> beam.io.ReadFromPubSub(subscription=args.subscription)
            | "Extract Changed Customer"
            >> beam.ParDo(ExtractChangedCustomer(project=args.project))
            | "Write Changed Customers"
            >> beam.io.WriteToBigQuery(
                args.changed_customers_table,
                schema=schema,
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
            )
        )


if __name__ == "__main__":
    main()
