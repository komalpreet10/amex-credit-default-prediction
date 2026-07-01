from __future__ import annotations

import argparse
import logging

from google.cloud import aiplatform

from gcp.config import ENDPOINT_DISPLAY_NAME, PROJECT_ID, REGION

LOGGER = logging.getLogger(__name__)
MACHINE_TYPE = "n1-standard-2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=PROJECT_ID)
    parser.add_argument("--location", default=REGION)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--endpoint-name", default=ENDPOINT_DISPLAY_NAME)
    parser.add_argument("--deployed-model-name", default="amex-lightgbm")
    parser.add_argument("--machine-type", default=MACHINE_TYPE)
    parser.add_argument("--min-replica-count", type=int, default=1)
    parser.add_argument("--max-replica-count", type=int, default=1)
    return parser.parse_args()


def get_or_create_endpoint(args: argparse.Namespace) -> aiplatform.Endpoint:
    if args.endpoint:
        LOGGER.info("Using existing endpoint %s", args.endpoint)
        return aiplatform.Endpoint(args.endpoint)

    LOGGER.info("Creating endpoint %s", args.endpoint_name)
    return aiplatform.Endpoint.create(display_name=args.endpoint_name, sync=True)


def deploy_model(args: argparse.Namespace) -> aiplatform.Endpoint:
    aiplatform.init(project=args.project, location=args.location)
    model = aiplatform.Model(args.model)
    endpoint = get_or_create_endpoint(args)

    LOGGER.info("Deploying model %s to endpoint %s", model.resource_name, endpoint)
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=args.deployed_model_name,
        machine_type=args.machine_type,
        min_replica_count=args.min_replica_count,
        max_replica_count=args.max_replica_count,
        traffic_percentage=100,
        sync=True,
    )
    LOGGER.info("Deployment complete: %s", endpoint.resource_name)
    return endpoint


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    deploy_model(parse_args())


if __name__ == "__main__":
    main()
