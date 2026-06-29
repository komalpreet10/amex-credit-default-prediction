from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow

from amex_default.config import (
    ACTIVE_MODELS,
    MODEL_DIR,
    PLOTS_DIR,
    PREDICTIONS_DIR,
    REPORTS_DIR,
)

METRIC_KEYS = [
    "roc_auc",
    "pr_auc",
    "precision",
    "recall",
    "f1",
    "threshold",
    "true_negative",
    "false_positive",
    "false_negative",
    "true_positive",
    "training_time_seconds",
    "inference_time_seconds",
    "total_time_seconds",
]


def configure_mlflow(project_root: str | Path, experiment_name: str) -> None:
    project_root = Path(project_root)
    mlflow.set_tracking_uri(f"file://{project_root / 'mlruns'}")
    mlflow.set_experiment(experiment_name)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _log_artifact_if_exists(path: Path, artifact_path: str | None = None) -> None:
    if path.exists() and path.is_file():
        mlflow.log_artifact(str(path), artifact_path=artifact_path)


def _log_dir_if_exists(path: Path, artifact_path: str | None = None) -> None:
    if path.exists() and any(path.iterdir()):
        mlflow.log_artifacts(str(path), artifact_path=artifact_path)


def log_model_run(model_name: str) -> dict[str, Any] | None:
    metrics_path = REPORTS_DIR / f"{model_name}_metrics.json"
    if not metrics_path.exists():
        print(f"Skipping {model_name}: missing {metrics_path}")
        return None

    metrics = _load_json(metrics_path)
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model", model_name)
        mlflow.log_param("n_rows", metrics.get("n_rows"))
        mlflow.log_param("n_features", metrics.get("n_features"))

        for key in METRIC_KEYS:
            value = metrics.get(key)
            if value is not None:
                mlflow.log_metric(key, value)

        for fold_metric in metrics.get("fold_metrics", []):
            fold = int(fold_metric["fold"])
            for key in ["roc_auc", "pr_auc", "precision", "recall", "f1"]:
                if key in fold_metric:
                    mlflow.log_metric(f"fold_{fold}_{key}", fold_metric[key])

        _log_artifact_if_exists(metrics_path, artifact_path="reports")
        _log_artifact_if_exists(
            metrics_path.with_suffix(".csv"), artifact_path="reports"
        )
        _log_artifact_if_exists(
            REPORTS_DIR / f"{model_name}_feature_importance.csv",
            artifact_path="reports",
        )
        _log_artifact_if_exists(
            PREDICTIONS_DIR / f"{model_name}_oof.parquet",
            artifact_path="predictions",
        )

        for plot_name in [
            f"{model_name}_roc_curve.png",
            f"{model_name}_pr_curve.png",
            f"{model_name}_confusion_matrix.png",
            f"{model_name}_feature_importance.png",
            f"{model_name}_shap_summary.png",
            f"{model_name}_shap_bar.png",
        ]:
            _log_artifact_if_exists(PLOTS_DIR / plot_name, artifact_path="plots")

        _log_dir_if_exists(MODEL_DIR / model_name, artifact_path="models")

    return metrics


def log_comparison_artifacts() -> None:
    with mlflow.start_run(run_name="model_comparison"):
        _log_artifact_if_exists(REPORTS_DIR / "model_comparison.csv", "reports")
        for plot_name in [
            "model_comparison_auc.png",
            "model_comparison_roc_auc.png",
            "model_comparison_pr_auc.png",
            "model_comparison_f1.png",
            "model_comparison_training_time_seconds.png",
            "model_comparison_inference_time_seconds.png",
        ]:
            _log_artifact_if_exists(PLOTS_DIR / plot_name, artifact_path="plots")


def log_final_model_artifacts() -> None:
    with mlflow.start_run(run_name="final_lightgbm_model"):
        mlflow.log_param("model", "lightgbm")
        mlflow.log_param("purpose", "serving")
        _log_dir_if_exists(MODEL_DIR / "final", artifact_path="models")
        _log_artifact_if_exists(REPORTS_DIR / "final_model_summary.md", "reports")


def log_project_artifacts() -> list[dict[str, Any]]:
    logged_metrics = []
    for model_name in ACTIVE_MODELS:
        metrics = log_model_run(model_name)
        if metrics is not None:
            logged_metrics.append(metrics)

    if (REPORTS_DIR / "model_comparison.csv").exists():
        log_comparison_artifacts()

    if (MODEL_DIR / "final").exists():
        log_final_model_artifacts()

    return logged_metrics
