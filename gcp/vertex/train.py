from __future__ import annotations

import argparse
import json
import logging
import tempfile
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.cloud import bigquery, storage
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

from amex_default.interpret import save_best_fold_shap_plots

PROJECT_ID = "amex-credit-risk-ml"
LOCATION = "us-central1"
BQ_LOCATION = "US"
TABLE = "amex-credit-risk-ml.amex_ml.train_features"
OUTPUT_DIR = "gs://amex-credit-risk-ml-data/models/lightgbm/"
EXPERIMENT = "amex-credit-default"

ID_COL = "customer_ID"
TARGET_COL = "target"
THRESHOLD = 0.5
RANDOM_STATE = 42

LOGGER = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": -1,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbosity": -1,
    "seed": RANDOM_STATE,
    "feature_fraction_seed": RANDOM_STATE,
    "bagging_seed": RANDOM_STATE,
    "data_random_seed": RANDOM_STATE,
    "num_threads": -1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=PROJECT_ID)
    parser.add_argument("--location", default=LOCATION)
    parser.add_argument("--bq-location", default=BQ_LOCATION)
    parser.add_argument("--table", default=TABLE)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--params-uri", default=None)
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--num-boost-round", type=int, default=1000)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--final-num-boost-round", type=int, default=300)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--shap-sample-size", type=int, default=None)
    parser.add_argument("--shap-max-display", type=int, default=30)
    parser.add_argument("--disable-shap", action="store_true")
    parser.add_argument("--disable-experiment", action="store_true")
    return parser.parse_args()


def load_tuned_params(params_uri: str | None) -> dict[str, object]:
    if not params_uri:
        return {}
    if not params_uri.startswith("gs://"):
        raise ValueError("--params-uri must be a GCS URI.")

    bucket_name, blob_name = params_uri.removeprefix("gs://").split("/", 1)
    payload = (
        storage.Client()
        .bucket(bucket_name)
        .blob(blob_name)
        .download_as_text(encoding="utf-8")
    )
    data = json.loads(payload)
    params = data.get("best_params", data)
    LOGGER.info("Loaded tuned LightGBM params from %s", params_uri)
    return params


def read_training_data(args: argparse.Namespace) -> pd.DataFrame:
    client = bigquery.Client(project=args.project, location=args.bq_location)
    query = f"SELECT * FROM `{args.table}`"
    if args.max_rows:
        query += f" LIMIT {args.max_rows}"

    LOGGER.info("Reading training data from BigQuery table %s", args.table)
    return client.query(query).result().to_dataframe()


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    required = {ID_COL, TARGET_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    customer_ids = df[ID_COL]
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[ID_COL, TARGET_COL])

    for column in X.select_dtypes(include=["object", "string"]).columns:
        X[column] = X[column].astype("category")

    return customer_ids, X, y


def calculate_metrics(y_true, y_pred_proba) -> dict[str, float]:
    y_pred = (y_pred_proba >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
        "pr_auc": float(average_precision_score(y_true, y_pred_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def train_cross_validated_model(
    X: pd.DataFrame,
    y: pd.Series,
    args: argparse.Namespace,
    tuned_params: dict[str, object] | None = None,
) -> tuple[list[lgb.Booster], np.ndarray, list[dict[str, float]], pd.DataFrame]:
    model_params = {**DEFAULT_PARAMS, **(tuned_params or {})}
    cv = StratifiedKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    categorical_features = X.select_dtypes(include=["category"]).columns.tolist()
    oof_predictions = np.zeros(len(X))
    fold_metrics = []
    models = []
    importance_frames = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        LOGGER.info("Training LightGBM fold %d", fold)
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=categorical_features,
            free_raw_data=False,
        )
        valid_data = lgb.Dataset(
            X_valid,
            label=y_valid,
            categorical_feature=categorical_features,
            reference=train_data,
            free_raw_data=False,
        )

        model = lgb.train(
            model_params,
            train_data,
            num_boost_round=args.num_boost_round,
            valid_sets=[valid_data],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(args.early_stopping_rounds),
                lgb.log_evaluation(100),
            ],
        )

        predictions = model.predict(X_valid, num_iteration=model.best_iteration)
        oof_predictions[valid_idx] = predictions

        metrics = calculate_metrics(y_valid, predictions)
        metrics["fold"] = fold
        metrics["best_iteration"] = int(
            model.best_iteration or model.current_iteration()
        )
        fold_metrics.append(metrics)
        models.append(model)

        importance_frames.append(
            pd.DataFrame(
                {
                    "feature": model.feature_name(),
                    f"fold_{fold}": model.feature_importance(importance_type="gain"),
                }
            )
        )

    feature_importance = importance_frames[0]
    for fold_importance in importance_frames[1:]:
        feature_importance = feature_importance.merge(
            fold_importance,
            on="feature",
            how="outer",
        )
    fold_columns = [c for c in feature_importance.columns if c.startswith("fold_")]
    feature_importance = feature_importance.fillna(0)
    feature_importance["importance_mean"] = feature_importance[fold_columns].mean(
        axis=1
    )
    feature_importance = feature_importance.sort_values(
        "importance_mean",
        ascending=False,
    )

    return models, oof_predictions, fold_metrics, feature_importance


def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    args: argparse.Namespace,
    tuned_params: dict[str, object] | None = None,
) -> lgb.Booster:
    model_params = {**DEFAULT_PARAMS, **(tuned_params or {})}
    categorical_features = X.select_dtypes(include=["category"]).columns.tolist()
    train_data = lgb.Dataset(
        X,
        label=y,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )
    return lgb.train(
        model_params,
        train_data,
        num_boost_round=args.final_num_boost_round,
    )


def save_plots(y: pd.Series, predictions: np.ndarray, output_dir: Path) -> None:
    fpr, tpr, _ = roc_curve(y, predictions)
    precision, recall, _ = precision_recall_curve(y, predictions)
    y_pred = (predictions >= THRESHOLD).astype(int)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_score(y, predictions):.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        recall,
        precision,
        label=f"PR AUC = {average_precision_score(y, predictions):.4f}",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(y, y_pred, ax=ax, colorbar=False)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def save_shap_plots(
    model: lgb.Booster,
    X: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    if args.disable_shap:
        LOGGER.info("SHAP plot generation disabled")
        return
    if args.shap_sample_size is None:
        LOGGER.info("Generating SHAP plots from all %d rows", len(X))
        X_sample = X
    else:
        if args.shap_sample_size <= 0:
            LOGGER.info("Skipping SHAP because --shap-sample-size is <= 0")
            return

        sample_size = min(args.shap_sample_size, len(X))
        LOGGER.info("Generating SHAP plots from %d sampled rows", sample_size)
        X_sample = X.sample(n=sample_size, random_state=RANDOM_STATE)

    save_best_fold_shap_plots(
        model,
        X_sample,
        "lightgbm",
        output_dir,
        max_display=args.shap_max_display,
    )


def upload_directory(local_dir: Path, gcs_dir: str) -> None:
    if not gcs_dir.startswith("gs://"):
        raise ValueError("--output-dir must be a GCS URI.")

    bucket_name, prefix = gcs_dir.removeprefix("gs://").split("/", 1)
    prefix = prefix.rstrip("/")
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for path in local_dir.rglob("*"):
        if path.is_file():
            blob_name = f"{prefix}/{path.relative_to(local_dir)}"
            bucket.blob(blob_name).upload_from_filename(path)


def log_vertex_experiment(
    args: argparse.Namespace,
    metrics: dict[str, object],
) -> None:
    if args.disable_experiment:
        return

    try:
        from google.cloud import aiplatform

        aiplatform.init(
            project=args.project,
            location=args.location,
            experiment=args.experiment,
            staging_bucket=args.output_dir,
        )
        with aiplatform.start_run(run=args.run_name) as run:
            run.log_params(
                {
                    "model": "lightgbm",
                    "n_splits": args.n_splits,
                    "num_boost_round": args.num_boost_round,
                    "early_stopping_rounds": args.early_stopping_rounds,
                    "final_num_boost_round": args.final_num_boost_round,
                    "shap_sample_size": args.shap_sample_size,
                    "shap_max_display": args.shap_max_display,
                    "disable_shap": args.disable_shap,
                }
            )
            run.log_metrics(
                {
                    key: value
                    for key, value in metrics.items()
                    if isinstance(value, (int, float))
                }
            )
    except Exception:
        LOGGER.exception("Vertex AI Experiment logging failed")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    start = time.perf_counter()
    df = read_training_data(args)
    customer_ids, X, y = split_features_target(df)
    tuned_params = load_tuned_params(args.params_uri)

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir)

        models, oof_predictions, fold_metrics, feature_importance = (
            train_cross_validated_model(X, y, args, tuned_params=tuned_params)
        )
        final_model = train_final_model(X, y, args, tuned_params=tuned_params)

        metrics = {
            "model": "lightgbm",
            "n_rows": int(len(X)),
            "n_features": int(X.shape[1]),
            "threshold": THRESHOLD,
            "training_time_seconds": time.perf_counter() - start,
            **calculate_metrics(y, oof_predictions),
            "fold_metrics": fold_metrics,
            "tuned_params": tuned_params,
        }

        final_model.save_model(str(output_dir / "model.txt"))
        feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)
        pd.DataFrame(
            {
                ID_COL: customer_ids,
                TARGET_COL: y,
                "prediction": oof_predictions,
            }
        ).to_parquet(output_dir / "oof_predictions.parquet", index=False)

        (output_dir / "feature_list.json").write_text(
            json.dumps(X.columns.tolist(), indent=2),
            encoding="utf-8",
        )
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )
        save_plots(y, oof_predictions, output_dir)
        save_shap_plots(final_model, X, args, output_dir)

        LOGGER.info("Uploading model artifacts to %s", args.output_dir)
        upload_directory(output_dir, args.output_dir)
        log_vertex_experiment(args, metrics)


if __name__ == "__main__":
    main()
