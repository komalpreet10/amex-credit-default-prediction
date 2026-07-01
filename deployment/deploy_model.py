from __future__ import annotations

import argparse
import json
import logging

from google.cloud import aiplatform, storage

DEFAULT_ARTIFACT_URI = "gs://amex-credit-risk-ml-data/models/lightgbm/"
DEFAULT_CONFIG_URI = "gs://amex-credit-risk-ml-data/config/deployment_config.json"

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="amex-credit-risk-ml")
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--artifact-uri", default=DEFAULT_ARTIFACT_URI)
    parser.add_argument("--serving-image", required=True)
    parser.add_argument("--output-dir", default="gs://amex-credit-risk-ml-data/config/")
    return parser.parse_args()


def write_gcs_json(uri: str, payload: dict[str, str]) -> None:
    bucket_name, blob_name = uri.removeprefix("gs://").split("/", 1)
    storage.Client().bucket(bucket_name).blob(blob_name).upload_from_string(
        json.dumps(payload, indent=2),
        content_type="application/json",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    aiplatform.init(project=args.project, location=args.location)
    model = aiplatform.Model.upload(
        display_name="amex-lightgbm-credit-default",
        artifact_uri=args.artifact_uri,
        serving_container_image_uri=args.serving_image,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        serving_container_ports=[8080],
        sync=True,
    )
    endpoint = aiplatform.Endpoint.create(
        display_name="amex-credit-default-endpoint",
        sync=True,
    )
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name="amex-lightgbm",
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=1,
        traffic_percentage=100,
        sync=True,
    )
    config_uri = f"{args.output_dir.rstrip('/')}/deployment_config.json"
    payload = {
        "model_resource_name": model.resource_name,
        "endpoint_resource_name": endpoint.resource_name,
        "artifact_uri": args.artifact_uri,
        "serving_image": args.serving_image,
    }
    write_gcs_json(config_uri, payload)
    LOGGER.info("Deployment config written to %s", config_uri)
    LOGGER.info("Endpoint resource: %s", endpoint.resource_name)


if __name__ == "__main__":
    main()
