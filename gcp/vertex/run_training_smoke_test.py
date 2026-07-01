from __future__ import annotations

import argparse
import logging

from google.cloud import aiplatform

from gcp.config import (
    MODEL_ARTIFACTS,
    PROJECT_ID,
    REGION,
    TRAINING_IMAGE,
    TUNED_PARAMS_URI,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=PROJECT_ID)
    parser.add_argument("--region", default=REGION)
    parser.add_argument("--training-image", default=TRAINING_IMAGE, required=False)
    parser.add_argument("--params-uri", default=TUNED_PARAMS_URI)
    parser.add_argument("--output-dir", default=f"{MODEL_ARTIFACTS.rstrip('/')}/smoke/")
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--selector-rounds", type=int, default=10)
    parser.add_argument("--final-rounds", type=int, default=10)
    parser.add_argument("--min-selected-features", type=int, default=25)
    parser.add_argument("--max-selected-features", type=int, default=100)
    parser.add_argument("--machine-type", default="n2-standard-4")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if not args.training_image:
        raise RuntimeError(
            "--training-image or TRAINING_IMAGE_URI is required for Vertex smoke test."
        )

    aiplatform.init(
        project=args.project, location=args.region, staging_bucket=args.output_dir
    )
    job = aiplatform.CustomContainerTrainingJob(
        display_name="amex-lightgbm-training-smoke-test",
        container_uri=args.training_image,
        command=["python", "gcp/vertex/train.py"],
    )
    job.run(
        args=[
            "--params-uri",
            args.params_uri,
            "--output-dir",
            args.output_dir,
            "--max-rows",
            str(args.max_rows),
            "--balanced-smoke-sample",
            "--selector-num-boost-round",
            str(args.selector_rounds),
            "--final-num-boost-round",
            str(args.final_rounds),
            "--min-selected-features",
            str(args.min_selected_features),
            "--max-selected-features",
            str(args.max_selected_features),
            "--disable-shap",
        ],
        replica_count=1,
        machine_type=args.machine_type,
        sync=True,
    )
    LOGGER.info("Smoke training artifacts written to %s", args.output_dir)


if __name__ == "__main__":
    main()
