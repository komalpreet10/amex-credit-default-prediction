from __future__ import annotations

import argparse
import logging
import subprocess

from gcp.config import (
    MONTHLY_TRAINING_JOB,
    MONTHLY_TRAINING_MESSAGE,
    MONTHLY_TRAINING_SCHEDULE,
    PIPELINE_TRIGGER_TOPIC,
    PROJECT_ID,
    REGION,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=PROJECT_ID)
    parser.add_argument("--location", default=REGION)
    return parser.parse_args()


def run(command: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=check, text=True, capture_output=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    job_name = MONTHLY_TRAINING_JOB
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
            MONTHLY_TRAINING_SCHEDULE,
            "--topic",
            PIPELINE_TRIGGER_TOPIC,
            "--message-body",
            MONTHLY_TRAINING_MESSAGE,
            "--location",
            args.location,
            "--project",
            args.project,
        ]
    )
    LOGGER.info("Created Cloud Scheduler job: %s", job_name)


if __name__ == "__main__":
    main()
