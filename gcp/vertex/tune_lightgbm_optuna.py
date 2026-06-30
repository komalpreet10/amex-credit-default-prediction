from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from google.cloud import bigquery, storage
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from amex_default.config import ID_COL, RANDOM_STATE, TARGET_COL
from gcp.config import BQ_LOCATION, FEATURE_TABLE, PROJECT_ID, TUNING_ARTIFACTS

LOGGER = logging.getLogger(__name__)

BASE_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "verbosity": -1,
    "seed": RANDOM_STATE,
    "feature_fraction_seed": RANDOM_STATE,
    "bagging_seed": RANDOM_STATE,
    "data_random_seed": RANDOM_STATE,
    "num_threads": -1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-name", default="lightgbm-optuna")
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--num-boost-round", type=int, default=700)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--metric", choices=["roc_auc", "pr_auc"], default="roc_auc")
    parser.add_argument("--timeout", type=int, default=None)
    return parser.parse_args()


def read_training_data(args: argparse.Namespace) -> pd.DataFrame:
    client = bigquery.Client(project=PROJECT_ID, location=BQ_LOCATION)
    query = f"SELECT * FROM `{FEATURE_TABLE}`"

    LOGGER.info("Reading tuning data from BigQuery table %s", FEATURE_TABLE)
    return client.query(query).result().to_dataframe()


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    required = {ID_COL, TARGET_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[ID_COL, TARGET_COL])
    for column in X.select_dtypes(include=["object", "string"]).columns:
        X[column] = X[column].astype("category")
    return X, y


def suggest_params(trial: optuna.Trial) -> dict[str, object]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 24, 256),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 2000),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.55, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.55, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 100.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 100.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 10.0),
        "path_smooth": trial.suggest_float("path_smooth", 0.0, 1.0),
    }


def compute_fold_metrics(
    y_true: pd.Series,
    pred: np.ndarray,
    pred_label: np.ndarray,
) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, pred)),
        "pr_auc": float(average_precision_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred_label, zero_division=0)),
        "recall": float(recall_score(y_true, pred_label, zero_division=0)),
        "f1": float(f1_score(y_true, pred_label, zero_division=0)),
    }


def build_objective(
    X: pd.DataFrame,
    y: pd.Series,
    args: argparse.Namespace,
):
    categorical_features = X.select_dtypes(include=["category"]).columns.tolist()
    cv = StratifiedKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    def objective(trial: optuna.Trial) -> float:
        params = {**BASE_PARAMS, **suggest_params(trial)}
        fold_scores = []
        fold_metrics = []
        best_iterations = []
        oof_true = []
        oof_pred = []

        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
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
                params,
                train_data,
                num_boost_round=args.num_boost_round,
                valid_sets=[valid_data],
                valid_names=["valid"],
                callbacks=[
                    lgb.early_stopping(args.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
            pred = model.predict(X_valid, num_iteration=model.best_iteration)
            pred_label = (pred >= 0.5).astype(int)
            metrics = compute_fold_metrics(y_valid, pred, pred_label)
            score = metrics[args.metric]
            tn, fp, fn, tp = confusion_matrix(
                y_valid,
                pred_label,
                labels=[0, 1],
            ).ravel()
            best_iteration = int(model.best_iteration or model.current_iteration())
            fold_scores.append(score)
            best_iterations.append(best_iteration)
            oof_true.extend(y_valid.tolist())
            oof_pred.extend(pred_label.tolist())
            fold_metrics.append({
                "fold": fold,
                **metrics,
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
                "best_iteration": best_iteration,
            })
            trial.report(float(np.mean(fold_scores)), step=fold)

            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr("fold_scores", fold_scores)
        trial.set_user_attr("fold_metrics", fold_metrics)
        for metric_name in ["roc_auc", "pr_auc", "precision", "recall", "f1"]:
            trial.set_user_attr(
                f"mean_{metric_name}",
                float(np.mean([fold[metric_name] for fold in fold_metrics])),
            )
        trial.set_user_attr(
            "confusion_matrix",
            confusion_matrix(oof_true, oof_pred, labels=[0, 1]).tolist(),
        )
        trial.set_user_attr(
            "classification_report",
            classification_report(
                oof_true,
                oof_pred,
                labels=[0, 1],
                output_dict=True,
                zero_division=0,
            ),
        )
        trial.set_user_attr("best_iterations", best_iterations)
        trial.set_user_attr("mean_best_iteration", float(np.mean(best_iterations)))
        return float(np.mean(fold_scores))

    return objective


def upload_directory(local_dir: Path, gcs_dir: str) -> None:
    if not gcs_dir.startswith("gs://"):
        raise ValueError("Artifact output directory must be a GCS URI.")

    bucket_name, prefix = gcs_dir.removeprefix("gs://").split("/", 1)
    prefix = prefix.rstrip("/")
    bucket = storage.Client().bucket(bucket_name)
    for path in local_dir.rglob("*"):
        if path.is_file():
            blob_name = f"{prefix}/{path.relative_to(local_dir)}"
            bucket.blob(blob_name).upload_from_filename(path)


def save_outputs(
    study: optuna.Study,
    local_dir: Path,
    args: argparse.Namespace,
    elapsed_seconds: float,
) -> None:
    best_user_attrs = study.best_trial.user_attrs
    best = {
        "study_name": study.study_name,
        "metric": args.metric,
        "n_trials": len(study.trials),
        "best_trial_number": study.best_trial.number,
        "best_score": study.best_value,
        "best_params": study.best_params,
        "best_user_attrs": best_user_attrs,
        "elapsed_seconds": elapsed_seconds,
    }
    (local_dir / "lightgbm_optuna_best_params.json").write_text(
        json.dumps(best, indent=2),
        encoding="utf-8",
    )
    study.trials_dataframe().to_csv(
        local_dir / "lightgbm_optuna_trials.csv",
        index=False,
    )
    cv_metrics = {
        "evaluation_source": "optuna_stratified_cross_validation",
        "metric": args.metric,
        "best_score": study.best_value,
        "n_splits": args.n_splits,
        "n_trials": len(study.trials),
        "best_trial_number": study.best_trial.number,
        "fold_metrics": best_user_attrs.get("fold_metrics", []),
        "mean_roc_auc": best_user_attrs.get("mean_roc_auc"),
        "mean_pr_auc": best_user_attrs.get("mean_pr_auc"),
        "mean_precision": best_user_attrs.get("mean_precision"),
        "mean_recall": best_user_attrs.get("mean_recall"),
        "mean_f1": best_user_attrs.get("mean_f1"),
        "confusion_matrix": best_user_attrs.get("confusion_matrix", {}),
        "classification_report": best_user_attrs.get("classification_report", {}),
    }
    (local_dir / "cv_metrics.json").write_text(
        json.dumps(cv_metrics, indent=2),
        encoding="utf-8",
    )
    (local_dir / "cv_classification_report.json").write_text(
        json.dumps(best_user_attrs.get("classification_report", {}), indent=2),
        encoding="utf-8",
    )
    (local_dir / "cv_confusion_matrix.json").write_text(
        json.dumps(best_user_attrs.get("confusion_matrix", {}), indent=2),
        encoding="utf-8",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    df = read_training_data(args)
    X, y = split_features_target(df)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )

    start = time.perf_counter()
    study.optimize(
        build_objective(X, y, args),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )
    elapsed_seconds = time.perf_counter() - start

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_dir = Path(tmp_dir)
        save_outputs(study, local_dir, args, elapsed_seconds)
        LOGGER.info("Uploading Optuna artifacts to %s", TUNING_ARTIFACTS)
        upload_directory(local_dir, TUNING_ARTIFACTS)

    LOGGER.info("Best %s: %.6f", args.metric, study.best_value)
    LOGGER.info("Best params: %s", json.dumps(study.best_params, sort_keys=True))


if __name__ == "__main__":
    main()
