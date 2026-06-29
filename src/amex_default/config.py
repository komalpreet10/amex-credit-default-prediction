from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "models"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

TRAIN_FEATURES_PATH = PROCESSED_DIR / "train_features.parquet"
MODEL_COMPARISON_PATH = PROCESSED_DIR / "model_comparison.csv"

ID_COL = "customer_ID"
TARGET_COL = "target"
DATE_COL = "S_2"

CATEGORICAL_FEATURES = [
    "D_63",
    "D_64",
    "B_30",
    "B_38",
    "D_114",
    "D_116",
    "D_117",
    "D_120",
    "D_126",
    "D_68",
]

RANDOM_STATE = 42
N_SPLITS = 5
DEFAULT_THRESHOLD = 0.5

ACTIVE_MODELS = ["lightgbm", "xgboost"]
