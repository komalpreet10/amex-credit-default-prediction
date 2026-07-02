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
DRIFT_TABLE_ID = "drift_metrics"
MODEL_DISPLAY_NAME = "amex-lightgbm-credit-default"
ENDPOINT_DISPLAY_NAME = "amex-credit-default-endpoint"
DEPLOYED_MODEL_DISPLAY_NAME = "amex-lightgbm"
ENDPOINT_MACHINE_TYPE = "n1-standard-2"
ENDPOINT_MIN_REPLICA_COUNT = 1
ENDPOINT_MAX_REPLICA_COUNT = 1
ENDPOINT_TRAFFIC_PERCENTAGE = 100

# Vertex custom training defaults.
TUNING_JOB_DISPLAY_NAME = "amex-lightgbm-optuna-tuning"
TRAINING_JOB_DISPLAY_NAME = "amex-lightgbm-training"
TRAINING_REPLICA_COUNT = 1
TRAINING_MACHINE_TYPE = "n2-standard-4"
TUNING_REPLICA_COUNT = 1
TUNING_MACHINE_TYPE = "n2-standard-4"
TUNING_N_TRIALS = 15
TUNING_N_SPLITS = 5
TRAINING_SHAP_SAMPLE_SIZE = 3000
TRAINING_SHAP_MAX_DISPLAY = 30

# Online inference cache.
REDIS_INSTANCE_ID = os.getenv("REDIS_INSTANCE_ID", "amex-feature-cache")
REDIS_TIER = os.getenv("REDIS_TIER", "standard")
REDIS_SIZE_GB = os.getenv("REDIS_SIZE_GB", "5")
REDIS_VERSION = os.getenv("REDIS_VERSION", "redis_7_0")
REDIS_NETWORK = os.getenv("REDIS_NETWORK", "default")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6378"))
REDIS_TRANSIT_ENCRYPTION_MODE = os.getenv(
    "REDIS_TRANSIT_ENCRYPTION_MODE",
    "SERVER_AUTHENTICATION",
)
REDIS_SSL_ENABLED = os.getenv("REDIS_SSL_ENABLED", "true").lower() == "true"
REDIS_SSL_CA_CERTS = os.getenv("REDIS_SSL_CA_CERTS")
REDIS_SSL_CA_CERT_CONTENT = os.getenv("REDIS_SSL_CA_CERT_CONTENT")
REDIS_SSL_CA_CERT_SECRET = os.getenv("REDIS_SSL_CA_CERT_SECRET", "redis-ca-cert")
REDIS_FEATURE_TTL_SECONDS = (
    3_024_000  # 35 days (30-day cycle + 5-day buffer for Dataflow recompute lag).
)
REDIS_EVICTION_POLICY = os.getenv("REDIS_EVICTION_POLICY", "volatile-ttl")
VPC_CONNECTOR_NAME = os.getenv("VPC_CONNECTOR_NAME", "amex-vpc-connector")
VPC_CONNECTOR_RANGE = os.getenv("VPC_CONNECTOR_RANGE", "10.8.0.0/28")

# Raw and engineered feature data.
RAW_DATA = f"gs://{BUCKET}/raw/train_data.csv"
RAW_LABELS = f"gs://{BUCKET}/raw/train_labels.csv"
PREPROCESSED = f"gs://{BUCKET}/processed/v1/train_preprocessed/"
FEATURES = f"gs://{BUCKET}/processed/v1/train_features/"
FEATURES_PARQUET_URI = f"{FEATURES}*.parquet"

FEATURE_TABLE = f"{PROJECT_ID}.{DATASET}.{FEATURE_TABLE_ID}"
TRAIN_FEATURE_TABLE = f"{PROJECT_ID}.{DATASET}.{TRAIN_FEATURE_TABLE_ID}"
TEST_FEATURE_TABLE = f"{PROJECT_ID}.{DATASET}.{TEST_FEATURE_TABLE_ID}"
DRIFT_TABLE = f"{PROJECT_ID}.{DATASET}.{DRIFT_TABLE_ID}"

# Model, tuning, monitoring, and deployment artifacts.
MODEL_ARTIFACTS = f"gs://{BUCKET}/models/lightgbm/"
TUNING_ARTIFACTS = f"gs://{BUCKET}/models/lightgbm/tuning/"
TUNED_PARAMS_URI = f"{TUNING_ARTIFACTS}lightgbm_optuna_best_params.json"
SELECTED_FEATURES_URI = f"{MODEL_ARTIFACTS}selected_feature_list.json"
DRIFT_REPORT = f"gs://{BUCKET}/monitoring/train_vs_scoring_drift_report.csv"
DEPLOYMENT_CONFIG_DIR = f"gs://{BUCKET}/config/"
DEPLOYMENT_CONFIG_URI = f"{DEPLOYMENT_CONFIG_DIR}deployment_config.json"
STREAMING_FEATURES_URI = f"{DEPLOYMENT_CONFIG_DIR}streaming_features.json"
INFERENCE_FUNCTION_NAME = "amex-credit-default-score"

# Dataproc Serverless Spark jobs for full feature-engineering reruns.
PREPROCESS_SCRIPT = f"gs://{BUCKET}/code/gcp/spark/preprocess.py"
FEATURE_SCRIPT = f"gs://{BUCKET}/code/gcp/spark/build_features.py"
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

# Pub/Sub and scheduler resources for online/automated workflows.
STATEMENT_TOPIC = "statement-cycle-close"
STATEMENT_SUBSCRIPTION = "statement-cycle-close-sub"
STATEMENT_DLQ_TOPIC = "statement-cycle-close-dlq"
STATEMENT_DLQ_MAX_DELIVERY_ATTEMPTS = 5
STATEMENT_RETRY_MIN_BACKOFF_SECONDS = 10
STATEMENT_RETRY_MAX_BACKOFF_SECONDS = 600
STATEMENT_ACK_DEADLINE_SECONDS = 60
STATEMENT_DLQ_ALERT_DISPLAY_NAME = "statement-cycle-close-dlq-message-alert"
STATEMENT_DLQ_ALERT_THRESHOLD = 0
STATEMENT_DLQ_ALERT_ALIGNMENT_SECONDS = 60
PIPELINE_TRIGGER_TOPIC = "amex-pipeline-trigger"
STATEMENT_SUBSCRIPTION_PATH = (
    f"projects/{PROJECT_ID}/subscriptions/{STATEMENT_SUBSCRIPTION}"
)
MONTHLY_TRAINING_JOB = "amex-monthly-training"
MONTHLY_TRAINING_SCHEDULE = "0 2 1 * *"
MONTHLY_TRAINING_MESSAGE = '{"trigger":"monthly_training"}'

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
