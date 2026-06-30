from __future__ import annotations

import argparse
import logging
import subprocess

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="amex-credit-risk-ml")
    return parser.parse_args()


def run(command: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=check, text=True, capture_output=True)


def ensure_topic(project: str, topic: str) -> None:
    describe = run(
        ["gcloud", "pubsub", "topics", "describe", topic, "--project", project],
        check=False,
    )
    if describe.returncode == 0:
        LOGGER.info("Topic already exists: %s", topic)
        return
    run(["gcloud", "pubsub", "topics", "create", topic, "--project", project])
    LOGGER.info("Created topic: %s", topic)


def ensure_subscription(project: str, subscription: str, topic: str) -> None:
    describe = run(
        [
            "gcloud",
            "pubsub",
            "subscriptions",
            "describe",
            subscription,
            "--project",
            project,
        ],
        check=False,
    )
    if describe.returncode == 0:
        LOGGER.info("Subscription already exists: %s", subscription)
        return
    run(
        [
            "gcloud",
            "pubsub",
            "subscriptions",
            "create",
            subscription,
            "--topic",
            topic,
            "--ack-deadline",
            "60",
            "--project",
            project,
        ]
    )
    LOGGER.info("Created subscription: %s", subscription)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    ensure_topic(args.project, "statement-cycle-close")
    ensure_topic(args.project, "statement-cycle-close-dlq")
    ensure_subscription(args.project, "statement-cycle-close-sub", "statement-cycle-close")
    ensure_topic(args.project, "amex-pipeline-trigger")


if __name__ == "__main__":
    main()
