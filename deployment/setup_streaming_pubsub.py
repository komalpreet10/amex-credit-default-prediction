from __future__ import annotations

import argparse
import logging

from google.api_core.exceptions import NotFound
from google.cloud import pubsub_v1

from gcp.config import PROJECT_ID, STATEMENT_DLQ_TOPIC, STATEMENT_TOPIC

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=PROJECT_ID)
    parser.add_argument("--statement-topic", default=STATEMENT_TOPIC)
    parser.add_argument("--dlq-topic", default=STATEMENT_DLQ_TOPIC)
    return parser.parse_args()


def ensure_topic(
    publisher: pubsub_v1.PublisherClient,
    project: str,
    topic: str,
) -> str:
    topic_path = publisher.topic_path(project, topic)
    try:
        publisher.get_topic(request={"topic": topic_path})
        LOGGER.info("Using existing topic: %s", topic_path)
    except NotFound:
        publisher.create_topic(request={"name": topic_path})
        LOGGER.info("Created topic: %s", topic_path)
    return topic_path


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    publisher = pubsub_v1.PublisherClient()
    ensure_topic(publisher, args.project, args.statement_topic)
    ensure_topic(publisher, args.project, args.dlq_topic)


if __name__ == "__main__":
    main()
