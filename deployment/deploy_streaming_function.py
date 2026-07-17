from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from gcp.config import (
    CHANGED_CUSTOMERS_TABLE,
    PROJECT_ID,
    REGION,
    SELECTED_FEATURES_URI,
    STATEMENT_HISTORY_TABLE,
    STATEMENT_INGEST_FUNCTION_NAME,
    STATEMENT_TOPIC,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=PROJECT_ID)
    parser.add_argument("--region", default=REGION)
    parser.add_argument("--function-name", default=STATEMENT_INGEST_FUNCTION_NAME)
    parser.add_argument("--entry-point", default="ingest_monthly_statement")
    parser.add_argument("--runtime", default="python311")
    parser.add_argument("--timeout", default="3600s")
    parser.add_argument("--trigger-topic", default=STATEMENT_TOPIC)
    parser.add_argument("--statement-history-table", default=STATEMENT_HISTORY_TABLE)
    parser.add_argument("--changed-customers-table", default=CHANGED_CUSTOMERS_TABLE)
    parser.add_argument("--selected-features-uri", default=SELECTED_FEATURES_URI)
    parser.add_argument("--redis-host", required=True)
    parser.add_argument("--redis-port", default="6379")
    parser.add_argument("--redis-db", default="0")
    parser.add_argument("--redis-key-prefix", default="amex")
    parser.add_argument("--customer-history-limit", default="13")
    parser.add_argument("--vpc-connector", default=None)
    parser.add_argument("--egress-settings", default="private-ranges-only")
    return parser.parse_args()


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def build_source_bundle(repo_root: Path, bundle_dir: Path) -> None:
    shutil.copy(
        repo_root / "streaming" / "monthly_statement_handler.py",
        bundle_dir / "main.py",
    )
    shutil.copy(
        repo_root / "streaming" / "requirements.txt",
        bundle_dir / "requirements.txt",
    )
    shutil.copytree(repo_root / "src" / "amex_default", bundle_dir / "amex_default")
    shutil.copytree(repo_root / "gcp", bundle_dir / "gcp")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    env_vars = {
        "PROJECT_ID": args.project,
        "REGION": args.region,
        "STATEMENT_HISTORY_TABLE": args.statement_history_table,
        "CHANGED_CUSTOMERS_TABLE": args.changed_customers_table,
        "SELECTED_FEATURES_URI": args.selected_features_uri,
        "REDIS_HOST": args.redis_host,
        "REDIS_PORT": args.redis_port,
        "REDIS_DB": args.redis_db,
        "REDIS_KEY_PREFIX": args.redis_key_prefix,
        "CUSTOMER_HISTORY_LIMIT": args.customer_history_limit,
    }
    set_env_vars = ",".join(f"{key}={value}" for key, value in env_vars.items())

    repo_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as temp_dir:
        bundle_dir = Path(temp_dir) / "streaming_source"
        bundle_dir.mkdir()
        build_source_bundle(repo_root, bundle_dir)
        command = [
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
                "--trigger-topic",
                args.trigger_topic,
                "--timeout",
                args.timeout,
                "--set-env-vars",
                set_env_vars,
        ]
        if args.vpc_connector:
            command.extend(
                [
                    "--vpc-connector",
                    args.vpc_connector,
                    "--egress-settings",
                    args.egress_settings,
                ]
            )
        run(command)
    LOGGER.info("Deployed streaming Cloud Function: %s", args.function_name)


if __name__ == "__main__":
    main()
