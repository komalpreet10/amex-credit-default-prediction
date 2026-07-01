from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path

from gcp.config import PROJECT_ID, REDIS_SSL_CA_CERT_SECRET

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=PROJECT_ID)
    parser.add_argument("--secret", default=REDIS_SSL_CA_CERT_SECRET)
    parser.add_argument("--cert-file", required=True)
    return parser.parse_args()


def run(command: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=check, text=True, capture_output=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    cert_file = Path(args.cert_file)
    if not cert_file.is_file():
        raise FileNotFoundError(f"Redis CA cert file does not exist: {cert_file}")

    run(
        [
            "gcloud",
            "services",
            "enable",
            "secretmanager.googleapis.com",
            "--project",
            args.project,
        ]
    )
    describe = run(
        [
            "gcloud",
            "secrets",
            "describe",
            args.secret,
            "--project",
            args.project,
        ],
        check=False,
    )
    if describe.returncode != 0:
        run(
            [
                "gcloud",
                "secrets",
                "create",
                args.secret,
                "--project",
                args.project,
                "--replication-policy",
                "automatic",
            ]
        )
        LOGGER.info("Created Secret Manager secret: %s", args.secret)

    run(
        [
            "gcloud",
            "secrets",
            "versions",
            "add",
            args.secret,
            "--project",
            args.project,
            "--data-file",
            str(cert_file),
        ]
    )
    LOGGER.info("Added Redis CA cert version to secret: %s", args.secret)


if __name__ == "__main__":
    main()
