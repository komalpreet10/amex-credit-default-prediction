from __future__ import annotations

import argparse
import json
import logging
import tempfile
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from google.cloud import bigquery, storage
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ID = "amex-credit-risk-ml"
LOCATION = "us-central1"
BQ_LOCATION = "US"
TABLE = "amex-credit-risk-ml.amex_ml.train_features"
OUTPUT_DIR = "gs://amex-credit-risk-ml-data/models/lightgbm/tuning/"

ID_COL = "customer_ID"
TARGET_COL = "target"
RANDOM_STATE = 42

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
    parser.add_argument("--project", default=PROJECT_ID)
    parser.add_argument("--location", default=LOCATION)
    parser.add_argument("--bq-location", default=BQ_LOCATION)
    parser.add_argument("--table", default=TABLE)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--study-name", default="lightgbm-optuna")
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--max-rows", type=int, default=100000)
    parser.add_argument("--num-boost-round", type=int, default=700)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--metric", choices=["roc_auc", "pr_auc"], default="roc_auc")
    parser.add_argument("--timeout", type=int, default=None)
    return parser.parse_args()


def read_training_data(args: argparse.Namespace) -> pd.DataFrame:
    client = bigquery.Client(project=args.project, location=args.bq_location)
    query = f"SELECT * FROM `{args.table}`"
    if args.max_rows:
        query += f" LIMIT {args.max_rows}"

    LOGGER.info("Reading tuning data from BigQuery table %s", args.table)
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
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 24, 256),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 500),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.55, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.55, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 10.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 2.0),
    }


def score_predictions(metric: str, y_true: pd.Series, pred: np.ndarray) -> float:
    if metric == "pr_auc":
        return float(average_precision_score(y_true, pred))
    return float(roc_auc_score(y_true, pred))


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
        best_iterations = []

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
            fold_scores.append(score_predictions(args.metric, y_valid, pred))
            best_iterations.append(
                int(model.best_iteration or model.current_iteration())
            )
            trial.report(float(np.mean(fold_scores)), step=fold)

            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr("fold_scores", fold_scores)
        trial.set_user_attr("best_iterations", best_iterations)
        trial.set_user_attr("mean_best_iteration", float(np.mean(best_iterations)))
        return float(np.mean(fold_scores))

    return objective


def upload_directory(local_dir: Path, gcs_dir: str) -> None:
    if not gcs_dir.startswith("gs://"):
        raise ValueError("--output-dir must be a GCS URI.")

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
    best = {
        "study_name": study.study_name,
        "metric": args.metric,
        "n_trials": len(study.trials),
        "best_trial_number": study.best_trial.number,
        "best_score": study.best_value,
        "best_params": study.best_params,
        "best_user_attrs": study.best_trial.user_attrs,
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
        LOGGER.info("Uploading Optuna artifacts to %s", args.output_dir)
        upload_directory(local_dir, args.output_dir)

    LOGGER.info("Best %s: %.6f", args.metric, study.best_value)
    LOGGER.info("Best params: %s", json.dumps(study.best_params, sort_keys=True))


if __name__ == "__main__":
    main()
