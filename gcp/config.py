from __future__ import annotations

import os

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "amex-credit-risk-ml")
REGION = os.getenv("GCP_REGION", "us-central1")
BQ_LOCATION = "US"
BUCKET = "amex-credit-risk-ml-data"

PIPELINE_ROOT = os.getenv(
    "VERTEX_PIPELINE_ROOT",
    f"gs://{BUCKET}/pipeline-root/",
)
TRAINING_IMAGE = os.getenv("TRAINING_IMAGE_URI", "")
SERVING_IMAGE = os.getenv("SERVING_IMAGE_URI", "")

EXPERIMENT = "amex-credit-default"
DATASET = "amex_ml"
MODEL_DISPLAY_NAME = "amex-lightgbm-credit-default"
ENDPOINT_DISPLAY_NAME = "amex-credit-default-endpoint"

REDIS_INSTANCE_ID = os.getenv("REDIS_INSTANCE_ID", "amex-feature-cache")
REDIS_TIER = os.getenv("REDIS_TIER", "basic")
REDIS_SIZE_GB = os.getenv("REDIS_SIZE_GB", "1")
REDIS_VERSION = os.getenv("REDIS_VERSION", "redis_7_0")
REDIS_NETWORK = os.getenv("REDIS_NETWORK", "default")

RAW_DATA = f"gs://{BUCKET}/raw/train_data.csv"
RAW_LABELS = f"gs://{BUCKET}/raw/train_labels.csv"
PREPROCESSED = f"gs://{BUCKET}/processed/v1/train_preprocessed/"
FEATURES = f"gs://{BUCKET}/processed/v1/train_features/"

FEATURE_TABLE = f"{PROJECT_ID}.{DATASET}.train_features"
DRIFT_TABLE = f"{PROJECT_ID}.{DATASET}.drift_metrics"

MODEL_ARTIFACTS = f"gs://{BUCKET}/models/lightgbm/"
TUNING_ARTIFACTS = f"gs://{BUCKET}/models/lightgbm/tuning/"
TUNED_PARAMS_URI = f"{TUNING_ARTIFACTS}lightgbm_optuna_best_params.json"
DRIFT_REPORT = f"gs://{BUCKET}/monitoring/train_vs_scoring_drift_report.csv"

PREPROCESS_SCRIPT = f"gs://{BUCKET}/code/gcp/spark/preprocess.py"
FEATURE_SCRIPT = f"gs://{BUCKET}/code/gcp/spark/build_features.py"
PY_FILES = [f"gs://{BUCKET}/code/gcp/spark/amex_default.zip"]

DATAPROC_RUNTIME_PROPERTIES = {
    "spark.executor.instances": "2",
    "spark.driver.cores": "2",
    "spark.executor.cores": "2",
    "spark.driver.memory": "8g",
    "spark.executor.memory": "8g",
    "spark.dataproc.driver.disk.size": "250g",
    "spark.dataproc.executor.disk.size": "250g",
}
