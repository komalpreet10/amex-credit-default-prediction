from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from gcp.config import (
    FEATURE_TABLE,
    INFERENCE_FUNCTION_NAME,
    PROJECT_ID,
    REDIS_PORT,
    REDIS_SSL_CA_CERT_SECRET,
    REGION,
    SELECTED_FEATURES_URI,
    VPC_CONNECTOR_NAME,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=PROJECT_ID)
    parser.add_argument("--region", default=REGION)
    parser.add_argument("--function-name", default=INFERENCE_FUNCTION_NAME)
    parser.add_argument("--entry-point", default="score")
    parser.add_argument("--runtime", default="python311")
    parser.add_argument("--vpc-connector", default=VPC_CONNECTOR_NAME)
    parser.add_argument(
        "--redis-host", default=os.environ.get("REDIS_HOST"), required=False
    )
    parser.add_argument("--redis-port", type=int, default=REDIS_PORT)
    parser.add_argument(
        "--vertex-endpoint-id", default=os.environ.get("VERTEX_ENDPOINT_ID")
    )
    parser.add_argument("--selected-features-uri", default=SELECTED_FEATURES_URI)
    parser.add_argument("--bq-table", default=FEATURE_TABLE)
    parser.add_argument("--redis-ca-secret", default=REDIS_SSL_CA_CERT_SECRET)
    return parser.parse_args()


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def build_source_bundle(repo_root: Path, bundle_dir: Path) -> None:
    shutil.copy(repo_root / "inference" / "main.py", bundle_dir / "main.py")
    shutil.copy(
        repo_root / "inference" / "requirements.txt",
        bundle_dir / "requirements.txt",
    )
    shutil.copytree(repo_root / "src" / "amex_default", bundle_dir / "amex_default")
    shutil.copytree(repo_root / "gcp", bundle_dir / "gcp")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if not args.redis_host:
        raise RuntimeError("--redis-host or REDIS_HOST is required.")
    if not args.vertex_endpoint_id:
        raise RuntimeError("--vertex-endpoint-id or VERTEX_ENDPOINT_ID is required.")

    env_vars = {
        "PROJECT_ID": args.project,
        "REGION": args.region,
        "REDIS_HOST": args.redis_host,
        "REDIS_PORT": str(args.redis_port),
        "REDIS_SSL_ENABLED": "true",
        "VERTEX_ENDPOINT_ID": args.vertex_endpoint_id,
        "SELECTED_FEATURES_URI": args.selected_features_uri,
        "BQ_TABLE": args.bq_table,
    }
    set_env_vars = ",".join(f"{key}={value}" for key, value in env_vars.items())

    repo_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as temp_dir:
        bundle_dir = Path(temp_dir) / "inference_source"
        bundle_dir.mkdir()
        build_source_bundle(repo_root, bundle_dir)
        run(
            [
                "gcloud",
                "functions",
                "deploy",
                args.function_name,
                "--gen2",
                "--project",
                args.project,
                "--region",
                args.region,
                "--runtime",
                args.runtime,
                "--source",
                str(bundle_dir),
                "--entry-point",
                args.entry_point,
                "--trigger-http",
                "--vpc-connector",
                args.vpc_connector,
                "--egress-settings",
                "private-ranges-only",
                "--set-env-vars",
                set_env_vars,
                "--set-secrets",
                f"REDIS_SSL_CA_CERT_CONTENT={args.redis_ca_secret}:latest",
            ]
        )
    LOGGER.info("Deployed Cloud Function: %s", args.function_name)


if __name__ == "__main__":
    main()
