from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="amex-credit-risk-ml")
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--output-dir", default="gs://amex-credit-risk-ml-data/config/")
    parser.add_argument("--skip-monitoring", action="store_true")
    return parser.parse_args()


def run_step(name: str, command: list[str]) -> None:
    LOGGER.info("Starting deployment step: %s", name)
    subprocess.run(command, check=True)
    LOGGER.info("Completed deployment step: %s", name)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    python = sys.executable
    serving_image = os.environ.get("SERVING_IMAGE_URI")
    if not serving_image:
        raise RuntimeError("SERVING_IMAGE_URI environment variable is required.")

    steps = [
        (
            "deploy_model",
            [
                python,
                "deployment/deploy_model.py",
                "--project",
                args.project,
                "--location",
                args.location,
                "--output-dir",
                args.output_dir,
                "--serving-image",
                serving_image,
            ],
        ),
        ("refresh_redis", [python, "deployment/refresh_redis.py"]),
        ("setup_pubsub", [python, "deployment/setup_pubsub.py", "--project", args.project]),
        (
            "setup_scheduler",
            [
                python,
                "deployment/setup_scheduler.py",
                "--project",
                args.project,
                "--location",
                args.location,
            ],
        ),
    ]
    if not args.skip_monitoring:
        steps.append(
            (
                "setup_monitoring",
                [
                    python,
                    "deployment/setup_monitoring.py",
                    "--project",
                    args.project,
                    "--location",
                    args.location,
                ],
            )
        )

    summary = {name: "PENDING" for name, _ in steps}
    for name, command in steps:
        try:
            run_step(name, command)
            summary[name] = "SUCCEEDED"
        except subprocess.CalledProcessError as exc:
            summary[name] = f"FAILED: {exc}"
            LOGGER.error("Deployment step failed: %s", name, exc_info=True)
            break

    print("Deployment summary:")
    for name, status in summary.items():
        print(f"- {name}: {status}")

    if any(status.startswith("FAILED") for status in summary.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
