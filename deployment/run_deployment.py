from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys

from gcp.config import DEPLOYMENT_CONFIG_DIR, PROJECT_ID, REGION

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=PROJECT_ID)
    parser.add_argument("--location", default=REGION)
    parser.add_argument("--output-dir", default=DEPLOYMENT_CONFIG_DIR)
    parser.add_argument("--redis-host", default=os.environ.get("REDIS_HOST"))
    parser.add_argument("--redis-ca-cert-file", default=None)
    parser.add_argument(
        "--vertex-endpoint-id", default=os.environ.get("VERTEX_ENDPOINT_ID")
    )
    parser.add_argument("--skip-monitoring", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--run-redis-refresh", action="store_true")
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
    if not args.skip_inference and not args.redis_host:
        raise RuntimeError(
            "--redis-host or REDIS_HOST is required for inference deploy."
        )
    if not args.skip_inference and not args.redis_ca_cert_file:
        raise RuntimeError("--redis-ca-cert-file is required for inference deploy.")
    if not args.skip_inference and not args.vertex_endpoint_id:
        raise RuntimeError(
            "--vertex-endpoint-id or VERTEX_ENDPOINT_ID is required for inference deploy."
        )

    steps = [
        (
            "setup_vpc_connector",
            [
                python,
                "deployment/setup_vpc_connector.py",
                "--project",
                args.project,
                "--region",
                args.location,
            ],
        ),
    ]
    if args.redis_ca_cert_file:
        steps.append(
            (
                "setup_redis_ca_secret",
                [
                    python,
                    "deployment/setup_redis_ca_secret.py",
                    "--project",
                    args.project,
                    "--cert-file",
                    args.redis_ca_cert_file,
                ],
            )
        )

    steps.extend(
        [
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
            (
                "setup_pubsub",
                [python, "deployment/setup_pubsub.py", "--project", args.project],
            ),
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
    )
    if args.run_redis_refresh:
        steps.insert(2, ("refresh_redis", [python, "deployment/refresh_redis.py"]))

    if not args.skip_inference:
        steps.append(
            (
                "deploy_inference_function",
                [
                    python,
                    "deployment/deploy_inference_function.py",
                    "--project",
                    args.project,
                    "--region",
                    args.location,
                    "--redis-host",
                    args.redis_host,
                    "--vertex-endpoint-id",
                    args.vertex_endpoint_id,
                ],
            )
        )

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
