from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import timedelta

from google.cloud import aiplatform_v1, storage
from google.protobuf.duration_pb2 import Duration

from gcp.config import (
    DEPLOYMENT_CONFIG_URI,
    MONITORING_DISPLAY_NAME,
    MONITORING_DRIFT_THRESHOLDS,
    MONITORING_INTERVAL_HOURS,
    MONITORING_SAMPLE_RATE,
    PROJECT_ID,
    REGION,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=PROJECT_ID)
    parser.add_argument("--location", default=REGION)
    return parser.parse_args()


def read_gcs_json(uri: str) -> dict[str, str]:
    bucket_name, blob_name = uri.removeprefix("gs://").split("/", 1)
    payload = (
        storage.Client()
        .bucket(bucket_name)
        .blob(blob_name)
        .download_as_text(encoding="utf-8")
    )
    return json.loads(payload)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    alert_email = os.environ.get("ALERT_EMAIL")
    if not alert_email:
        raise RuntimeError("ALERT_EMAIL environment variable is required.")

    config = read_gcs_json(DEPLOYMENT_CONFIG_URI)
    endpoint = config["endpoint_resource_name"]
    parent = f"projects/{args.project}/locations/{args.location}"
    client = aiplatform_v1.JobServiceClient(
        client_options={"api_endpoint": f"{args.location}-aiplatform.googleapis.com"}
    )

    thresholds = {
        feature: aiplatform_v1.ThresholdConfig(value=value)
        for feature, value in MONITORING_DRIFT_THRESHOLDS.items()
    }
    job = aiplatform_v1.ModelDeploymentMonitoringJob(
        display_name=MONITORING_DISPLAY_NAME,
        endpoint=endpoint,
        logging_sampling_strategy=aiplatform_v1.SamplingStrategy(
            random_sample_config=aiplatform_v1.SamplingStrategy.RandomSampleConfig(
                sample_rate=MONITORING_SAMPLE_RATE
            )
        ),
        model_deployment_monitoring_schedule_config=(
            aiplatform_v1.ModelDeploymentMonitoringScheduleConfig(
                monitor_interval=Duration(
                    seconds=int(
                        timedelta(hours=MONITORING_INTERVAL_HOURS).total_seconds()
                    )
                )
            )
        ),
        model_monitoring_alert_config=aiplatform_v1.ModelMonitoringAlertConfig(
            email_alert_config=aiplatform_v1.ModelMonitoringAlertConfig.EmailAlertConfig(
                user_emails=[alert_email]
            )
        ),
        model_deployment_monitoring_objective_configs=[
            aiplatform_v1.ModelDeploymentMonitoringObjectiveConfig(
                objective_config=aiplatform_v1.ModelMonitoringObjectiveConfig(
                    prediction_drift_detection_config=(
                        aiplatform_v1.ModelMonitoringObjectiveConfig.PredictionDriftDetectionConfig(
                            drift_thresholds=thresholds
                        )
                    )
                )
            )
        ],
    )
    response = client.create_model_deployment_monitoring_job(
        parent=parent,
        model_deployment_monitoring_job=job,
    )
    LOGGER.info("Created Vertex AI monitoring job: %s", response.name)


if __name__ == "__main__":
    main()
