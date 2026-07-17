from __future__ import annotations

import os

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "amex-credit-risk-ml")
REGION = os.getenv("GCP_REGION", "us-central1")
BQ_LOCATION = "US"
BUCKET = "amex-credit-risk-ml-data"

# Vertex AI pipeline runtime.
PIPELINE_ROOT = os.getenv(
    "VERTEX_PIPELINE_ROOT",
    f"gs://{BUCKET}/pipeline-root/",
)
TRAINING_IMAGE = os.getenv("TRAINING_IMAGE_URI", "")
SERVING_IMAGE = os.getenv("SERVING_IMAGE_URI", "")

EXPERIMENT = "amex-credit-default"
DATASET = "amex_ml"
FEATURE_TABLE_ID = "train_features"
TRAIN_FEATURE_TABLE_ID = "train_features_train"
TEST_FEATURE_TABLE_ID = "train_features_test"
CUSTOMER_FEATURES_TABLE_ID = os.getenv(
    "CUSTOMER_FEATURES_TABLE_ID",
    "customer_features_current",
)
BATCH_PREDICTIONS_TABLE_ID = os.getenv(
    "BATCH_PREDICTIONS_TABLE_ID",
    "customer_default_predictions",
)
STATEMENT_HISTORY_TABLE_ID = os.getenv(
    "STATEMENT_HISTORY_TABLE_ID",
    "raw_monthly_statements_amex",
)
CHANGED_CUSTOMERS_TABLE_ID = os.getenv(
    "CHANGED_CUSTOMERS_TABLE_ID",
    "changed_customers_statement_cycle",
)
DRIFT_TABLE_ID = "drift_metrics"
MODEL_DISPLAY_NAME = "amex-lightgbm-credit-default"
ENDPOINT_DISPLAY_NAME = "amex-credit-default-endpoint"
DEPLOYED_MODEL_DISPLAY_NAME = "amex-lightgbm"
ENDPOINT_MACHINE_TYPE = "n1-standard-2"
ENDPOINT_MIN_REPLICA_COUNT = 1
ENDPOINT_MAX_REPLICA_COUNT = 1
ENDPOINT_TRAFFIC_PERCENTAGE = 100
BATCH_PREDICTION_JOB_DISPLAY_NAME = "amex-lightgbm-batch-inference"

# Vertex custom training defaults.
TUNING_JOB_DISPLAY_NAME = "amex-lightgbm-optuna-tuning"
TRAINING_JOB_DISPLAY_NAME = "amex-lightgbm-training"
TRAINING_REPLICA_COUNT = 1
TRAINING_MACHINE_TYPE = "n2-highmem-8"
TUNING_REPLICA_COUNT = 1
TUNING_MACHINE_TYPE = "n2-highmem-8"
TUNING_N_TRIALS = 15
TUNING_N_SPLITS = 5
TRAINING_SHAP_SAMPLE_SIZE = 3000
TRAINING_SHAP_MAX_DISPLAY = 30

# Optional online feature serving resources. The primary production path uses
# batch inference from BigQuery features to a BigQuery predictions table.
FEATURE_STORE_NAME = os.getenv(
    "FEATURE_STORE_NAME", "amex_credit_default_feature_store"
)
FEATURE_VIEW_NAME = os.getenv("FEATURE_VIEW_NAME", "customer_features_current")

# Raw and engineered feature data.
RAW_DATA = f"gs://{BUCKET}/raw/train_data.csv"
RAW_LABELS = f"gs://{BUCKET}/raw/train_labels.csv"
PREPROCESSED = f"gs://{BUCKET}/processed/v1/train_preprocessed/"
FEATURES = f"gs://{BUCKET}/processed/v1/train_features/"
FEATURES_PARQUET_URI = f"{FEATURES}*.parquet"

FEATURE_TABLE = f"{PROJECT_ID}.{DATASET}.{FEATURE_TABLE_ID}"
TRAIN_FEATURE_TABLE = f"{PROJECT_ID}.{DATASET}.{TRAIN_FEATURE_TABLE_ID}"
TEST_FEATURE_TABLE = f"{PROJECT_ID}.{DATASET}.{TEST_FEATURE_TABLE_ID}"
CUSTOMER_FEATURES_TABLE = f"{PROJECT_ID}.{DATASET}.{CUSTOMER_FEATURES_TABLE_ID}"
BATCH_PREDICTIONS_TABLE = f"{PROJECT_ID}.{DATASET}.{BATCH_PREDICTIONS_TABLE_ID}"
STATEMENT_HISTORY_TABLE = f"{PROJECT_ID}.{DATASET}.{STATEMENT_HISTORY_TABLE_ID}"
CHANGED_CUSTOMERS_TABLE = f"{PROJECT_ID}.{DATASET}.{CHANGED_CUSTOMERS_TABLE_ID}"
DRIFT_TABLE = f"{PROJECT_ID}.{DATASET}.{DRIFT_TABLE_ID}"

# Model, tuning, monitoring, and deployment artifacts.
MODEL_ARTIFACTS = f"gs://{BUCKET}/models/lightgbm/"
TUNING_ARTIFACTS = f"gs://{BUCKET}/models/lightgbm/tuning/"
TUNED_PARAMS_URI = f"{TUNING_ARTIFACTS}lightgbm_optuna_best_params.json"
SELECTED_FEATURES_URI = f"{MODEL_ARTIFACTS}selected_feature_list.json"
DRIFT_REPORT = f"gs://{BUCKET}/monitoring/train_vs_scoring_drift_report.csv"
DEPLOYMENT_CONFIG_DIR = f"gs://{BUCKET}/config/"
DEPLOYMENT_CONFIG_URI = f"{DEPLOYMENT_CONFIG_DIR}deployment_config.json"
INFERENCE_FUNCTION_NAME = "amex-credit-default-score"

# Monthly statement streaming ingest.
STATEMENT_TOPIC = os.getenv("STATEMENT_TOPIC", "amex-monthly-statements")
STATEMENT_DLQ_TOPIC = os.getenv("STATEMENT_DLQ_TOPIC", "amex-monthly-statements-dlq")
STATEMENT_INGEST_FUNCTION_NAME = os.getenv(
    "STATEMENT_INGEST_FUNCTION_NAME",
    "amex-monthly-statement-ingest",
)

# Dataproc Serverless Spark jobs for full feature-engineering reruns.
PREPROCESS_SCRIPT = f"gs://{BUCKET}/code/gcp/spark/preprocess.py"
FEATURE_SCRIPT = f"gs://{BUCKET}/code/gcp/spark/build_features.py"
FEATURE_REFRESH_SCRIPT = f"gs://{BUCKET}/code/gcp/spark/refresh_selected_features.py"
PY_FILES = [f"gs://{BUCKET}/code/gcp/spark/amex_default.zip"]

DATAPROC_RUNTIME_PROPERTIES = {
    "spark.executor.instances": "2",
    "spark.driver.cores": "4",
    "spark.executor.cores": "4",
    "spark.driver.memory": "9600m",
    "spark.executor.memory": "9600m",
    "spark.dynamicAllocation.executorAllocationRatio": "0.3",
    "spark.dataproc.scaling.version": "2",
    "spark.dataproc.driver.disk.size": "250g",
    "spark.dataproc.executor.disk.size": "250g",
    "spark.sql.shuffle.partitions": "32",
}

# Vertex AI Model Monitoring defaults.
MONITORING_DISPLAY_NAME = "amex-credit-default-monitoring"
MONITORING_SAMPLE_RATE = 1.0
MONITORING_INTERVAL_HOURS = 1
MONITORING_DRIFT_THRESHOLDS = {
    "P_2_mean": 0.3,
    "B_1_mean": 0.3,
    "D_39_last": 0.3,
    "S_3_mean": 0.3,
    "B_1_last": 0.3,
}
