from __future__ import annotations

import argparse
import json
import logging
import tempfile
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import optuna
import pandas as pd
from google.cloud import bigquery, storage
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from amex_default.config import ID_COL, RANDOM_STATE, TARGET_COL
from gcp.config import BQ_LOCATION, FEATURE_TABLE, PROJECT_ID, TUNING_ARTIFACTS

LOGGER = logging.getLogger(__name__)
matplotlib.use("Agg")

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
    parser.add_argument("--table", default=FEATURE_TABLE)
    parser.add_argument("--output-dir", default=TUNING_ARTIFACTS)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--balanced-smoke-sample", action="store_true")
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--num-boost-round", type=int, default=700)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--metric", choices=["roc_auc", "pr_auc"], default="roc_auc")
    parser.add_argument("--timeout", type=int, default=None)
    return parser.parse_args()


def read_training_data(args: argparse.Namespace) -> pd.DataFrame:
    client = bigquery.Client(project=PROJECT_ID, location=BQ_LOCATION)
    if args.max_rows is None:
        query = f"SELECT * FROM `{args.table}`"
        LOGGER.info("Reading tuning data from BigQuery table %s", args.table)
        return client.query(query).result().to_dataframe()

    if args.balanced_smoke_sample:
        positive_rows = max(1, args.max_rows // 2)
        negative_rows = args.max_rows - positive_rows
        query = f"""
        (
          SELECT * FROM `{args.table}`
          WHERE {TARGET_COL} = 1
          LIMIT {positive_rows}
        )
        UNION ALL
        (
          SELECT * FROM `{args.table}`
          WHERE {TARGET_COL} = 0
          LIMIT {negative_rows}
        )
        """
        LOGGER.info(
            "Reading balanced tuning smoke sample from %s: %d positive rows, %d negative rows",
            args.table,
            positive_rows,
            negative_rows,
        )
        return client.query(query).result().to_dataframe()

    query = f"SELECT * FROM `{args.table}` LIMIT {args.max_rows}"
    LOGGER.info(
        "Reading tuning smoke sample from %s: max_rows=%d",
        args.table,
        args.max_rows,
    )
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
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 1e-3, 10.0, log=True
        ),
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
            best_iteration = int(model.best_iteration or model.current_iteration())
            fold_scores.append(score)
            best_iterations.append(best_iteration)
            fold_metrics.append(
                {
                    "fold": fold,
                    **metrics,
                    "best_iteration": best_iteration,
                }
            )
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
        trial.set_user_attr("best_iterations", best_iterations)
        trial.set_user_attr("mean_best_iteration", float(np.mean(best_iterations)))
        return float(np.mean(fold_scores))

    return objective


def plot_roc_curve(y_true: list[int], y_score: list[float], path: Path) -> None:
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Cross-Validated ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_pr_curve(y_true: list[int], y_score: list[float], path: Path) -> None:
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auc = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, label=f"PR-AUC = {auc:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Cross-Validated Precision-Recall Curve")
    ax.legend(loc="upper right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluate_best_params(
    X: pd.DataFrame,
    y: pd.Series,
    best_params: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    categorical_features = X.select_dtypes(include=["category"]).columns.tolist()
    cv = StratifiedKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    params = {**BASE_PARAMS, **best_params}
    fold_metrics = []
    best_iterations = []
    oof_true: list[int] = []
    oof_score: list[float] = []

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
        best_iteration = int(model.best_iteration or model.current_iteration())
        fold_metrics.append(
            {
                "fold": fold,
                **compute_fold_metrics(y_valid, pred, pred_label),
                "best_iteration": best_iteration,
            }
        )
        best_iterations.append(best_iteration)
        oof_true.extend(y_valid.astype(int).tolist())
        oof_score.extend([float(value) for value in pred])

    return {
        "evaluation_source": "best_params_stratified_cross_validation",
        "fold_metrics": fold_metrics,
        "mean_roc_auc": float(np.mean([fold["roc_auc"] for fold in fold_metrics])),
        "mean_pr_auc": float(np.mean([fold["pr_auc"] for fold in fold_metrics])),
        "mean_precision": float(np.mean([fold["precision"] for fold in fold_metrics])),
        "mean_recall": float(np.mean([fold["recall"] for fold in fold_metrics])),
        "mean_f1": float(np.mean([fold["f1"] for fold in fold_metrics])),
        "best_iterations": best_iterations,
        "mean_best_iteration": float(np.mean(best_iterations)),
        "oof_roc_auc": float(roc_auc_score(oof_true, oof_score)),
        "oof_pr_auc": float(average_precision_score(oof_true, oof_score)),
        "oof_true": oof_true,
        "oof_score": oof_score,
    }


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
    cv_evaluation: dict[str, object],
) -> None:
    cv_summary = {
        key: value
        for key, value in cv_evaluation.items()
        if key not in {"oof_true", "oof_score"}
    }
    best_user_attrs = {**study.best_trial.user_attrs, **cv_summary}
    best = {
        "study_name": study.study_name,
        "metric": args.metric,
        "n_trials": len(study.trials),
        "best_trial_number": study.best_trial.number,
        "best_score": study.best_value,
        "best_params": study.best_params,
        "best_user_attrs": best_user_attrs,
        "max_rows": args.max_rows,
        "balanced_smoke_sample": args.balanced_smoke_sample,
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
        **cv_summary,
    }
    (local_dir / "cv_metrics.json").write_text(
        json.dumps(cv_metrics, indent=2),
        encoding="utf-8",
    )
    plot_roc_curve(
        cv_evaluation["oof_true"],
        cv_evaluation["oof_score"],
        local_dir / "plots" / "cv_roc_curve.png",
    )
    plot_pr_curve(
        cv_evaluation["oof_true"],
        cv_evaluation["oof_score"],
        local_dir / "plots" / "cv_pr_curve.png",
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
    cv_evaluation = evaluate_best_params(X, y, study.best_params, args)

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_dir = Path(tmp_dir)
        save_outputs(study, local_dir, args, elapsed_seconds, cv_evaluation)
        LOGGER.info("Uploading Optuna artifacts to %s", args.output_dir)
        upload_directory(local_dir, args.output_dir)

    LOGGER.info("Best %s: %.6f", args.metric, study.best_value)
    LOGGER.info("Best params: %s", json.dumps(study.best_params, sort_keys=True))


if __name__ == "__main__":
    main()
