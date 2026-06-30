from __future__ import annotations

import argparse
import logging
import subprocess

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="amex-credit-risk-ml")
    parser.add_argument("--location", default="us-central1")
    return parser.parse_args()


def run(command: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=check, text=True, capture_output=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    job_name = "amex-monthly-training"
    describe = run(
        [
            "gcloud",
            "scheduler",
            "jobs",
            "describe",
            job_name,
            "--location",
            args.location,
            "--project",
            args.project,
        ],
        check=False,
    )
    if describe.returncode == 0:
        LOGGER.info("Cloud Scheduler job already exists: %s", job_name)
        return

    run(
        [
            "gcloud",
            "scheduler",
            "jobs",
            "create",
            "pubsub",
            job_name,
            "--schedule",
            "0 2 1 * *",
            "--topic",
            "amex-pipeline-trigger",
            "--message-body",
            '{"trigger":"monthly_training"}',
            "--location",
            args.location,
            "--project",
            args.project,
        ]
    )
    LOGGER.info("Created Cloud Scheduler job: %s", job_name)


if __name__ == "__main__":
    main()
