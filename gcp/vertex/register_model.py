from __future__ import annotations

import argparse
import logging

from google.cloud import aiplatform

from gcp.config import (
    MODEL_ARTIFACTS,
    MODEL_DISPLAY_NAME,
    PROJECT_ID,
    REGION,
    SERVING_IMAGE,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=PROJECT_ID)
    parser.add_argument("--location", default=REGION)
    parser.add_argument("--display-name", default=MODEL_DISPLAY_NAME)
    parser.add_argument("--artifact-uri", default=MODEL_ARTIFACTS)
    parser.add_argument("--serving-container-image-uri", default=SERVING_IMAGE)
    return parser.parse_args()


def register_model(args: argparse.Namespace) -> aiplatform.Model:
    if not args.serving_container_image_uri:
        raise ValueError("--serving-container-image-uri is required.")

    aiplatform.init(project=args.project, location=args.location)
    LOGGER.info("Registering model artifacts from %s", args.artifact_uri)
    model = aiplatform.Model.upload(
        display_name=args.display_name,
        artifact_uri=args.artifact_uri,
        serving_container_image_uri=args.serving_container_image_uri,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        serving_container_ports=[8080],
        labels={"project": "amex-credit-default", "model": "lightgbm"},
        sync=True,
    )
    LOGGER.info("Registered model: %s", model.resource_name)
    return model


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    register_model(parse_args())


if __name__ == "__main__":
    main()
