from __future__ import annotations

import argparse
import logging
import subprocess

from gcp.config import (
    PROJECT_ID,
    REDIS_NETWORK,
    REGION,
    VPC_CONNECTOR_NAME,
    VPC_CONNECTOR_RANGE,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=PROJECT_ID)
    parser.add_argument("--region", default=REGION)
    parser.add_argument("--network", default=REDIS_NETWORK)
    parser.add_argument("--name", default=VPC_CONNECTOR_NAME)
    parser.add_argument("--range", default=VPC_CONNECTOR_RANGE)
    return parser.parse_args()


def run(command: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=check, text=True, capture_output=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    run(
        [
            "gcloud",
            "services",
            "enable",
            "vpcaccess.googleapis.com",
            "--project",
            args.project,
        ]
    )
    describe = run(
        [
            "gcloud",
            "compute",
            "networks",
            "vpc-access",
            "connectors",
            "describe",
            args.name,
            "--project",
            args.project,
            "--region",
            args.region,
        ],
        check=False,
    )
    if describe.returncode == 0:
        LOGGER.info("VPC connector already exists: %s", args.name)
        return

    run(
        [
            "gcloud",
            "compute",
            "networks",
            "vpc-access",
            "connectors",
            "create",
            args.name,
            "--project",
            args.project,
            "--region",
            args.region,
            "--network",
            args.network,
            "--range",
            args.range,
        ]
    )
    LOGGER.info("Created VPC connector: %s", args.name)


if __name__ == "__main__":
    main()
