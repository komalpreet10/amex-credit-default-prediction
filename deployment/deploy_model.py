from __future__ import annotations

import argparse
import json
import logging

from google.cloud import aiplatform, storage

from gcp.config import (
    DEPLOYED_MODEL_DISPLAY_NAME,
    DEPLOYMENT_CONFIG_DIR,
    ENDPOINT_DISPLAY_NAME,
    ENDPOINT_MACHINE_TYPE,
    ENDPOINT_MAX_REPLICA_COUNT,
    ENDPOINT_MIN_REPLICA_COUNT,
    ENDPOINT_TRAFFIC_PERCENTAGE,
    MODEL_ARTIFACTS,
    MODEL_DISPLAY_NAME,
    PROJECT_ID,
    REGION,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=PROJECT_ID)
    parser.add_argument("--location", default=REGION)
    parser.add_argument("--artifact-uri", default=MODEL_ARTIFACTS)
    parser.add_argument("--serving-image", required=True)
    parser.add_argument("--output-dir", default=DEPLOYMENT_CONFIG_DIR)
    return parser.parse_args()


def write_gcs_json(uri: str, payload: dict[str, str]) -> None:
    bucket_name, blob_name = uri.removeprefix("gs://").split("/", 1)
    storage.Client().bucket(bucket_name).blob(blob_name).upload_from_string(
        json.dumps(payload, indent=2),
        content_type="application/json",
    )


def get_or_create_endpoint(endpoint_display_name: str) -> aiplatform.Endpoint:
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_display_name}"',
        order_by="create_time desc",
    )
    if endpoints:
        endpoint = endpoints[0]
        LOGGER.info("Using existing endpoint: %s", endpoint.resource_name)
        return endpoint

    LOGGER.info("Creating endpoint: %s", endpoint_display_name)
    return aiplatform.Endpoint.create(
        display_name=endpoint_display_name,
        sync=True,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    aiplatform.init(project=args.project, location=args.location)
    model = aiplatform.Model.upload(
        display_name=MODEL_DISPLAY_NAME,
        artifact_uri=args.artifact_uri,
        serving_container_image_uri=args.serving_image,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        serving_container_ports=[8080],
        sync=True,
    )
    endpoint = get_or_create_endpoint(ENDPOINT_DISPLAY_NAME)
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=DEPLOYED_MODEL_DISPLAY_NAME,
        machine_type=ENDPOINT_MACHINE_TYPE,
        min_replica_count=ENDPOINT_MIN_REPLICA_COUNT,
        max_replica_count=ENDPOINT_MAX_REPLICA_COUNT,
        traffic_percentage=ENDPOINT_TRAFFIC_PERCENTAGE,
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
