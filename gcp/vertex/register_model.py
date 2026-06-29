from __future__ import annotations

import argparse
import logging

from google.cloud import aiplatform

PROJECT_ID = "amex-credit-risk-ml"
LOCATION = "us-central1"
DISPLAY_NAME = "amex-lightgbm-credit-default"
ARTIFACT_URI = "gs://amex-credit-risk-ml-data/models/lightgbm/"

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=PROJECT_ID)
    parser.add_argument("--location", default=LOCATION)
    parser.add_argument("--display-name", default=DISPLAY_NAME)
    parser.add_argument("--artifact-uri", default=ARTIFACT_URI)
    parser.add_argument("--description", default="LightGBM AMEX credit default model")
    parser.add_argument("--serving-container-image-uri", default=None)
    return parser.parse_args()


def register_model(args: argparse.Namespace) -> aiplatform.Model:
    aiplatform.init(project=args.project, location=args.location)

    upload_kwargs = {
        "display_name": args.display_name,
        "artifact_uri": args.artifact_uri,
        "description": args.description,
        "labels": {
            "project": "amex-credit-default",
            "model": "lightgbm",
        },
    }
    if args.serving_container_image_uri:
        upload_kwargs["serving_container_image_uri"] = args.serving_container_image_uri

    LOGGER.info("Registering model artifacts from %s", args.artifact_uri)
    model = aiplatform.Model.upload(**upload_kwargs)
    model.wait()

    LOGGER.info("Registered model: %s", model.resource_name)
    return model


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    register_model(parse_args())


if __name__ == "__main__":
    main()
