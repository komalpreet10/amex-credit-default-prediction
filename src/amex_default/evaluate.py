from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
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

from amex_default.config import DEFAULT_THRESHOLD, ID_COL, TARGET_COL


def calculate_metrics(
    y_true,
    y_pred_proba,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float]:
    y_pred = (y_pred_proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
        "pr_auc": float(average_precision_score(y_true, y_pred_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def confusion_matrix_values(
    y_true,
    y_pred_proba,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, int]:
    y_pred = (y_pred_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def save_metrics(metrics: dict[str, object], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def build_oof_frame(
    customer_ids,
    y_true,
    y_pred_proba,
    folds,
    model_name: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_ID": customer_ids,
            "target": y_true,
            "prediction": y_pred_proba,
            "fold": folds,
            "model": model_name,
        }
    )


def standardize_oof_frame(
    oof: pd.DataFrame,
    model_name: str,
    prediction_col: str | None = None,
    folds=None,
) -> pd.DataFrame:
    """Return the shared OOF schema used by training, comparison, and reports."""
    if prediction_col is None:
        if "prediction" in oof.columns:
            prediction_col = "prediction"
        elif f"{model_name}_oof" in oof.columns:
            prediction_col = f"{model_name}_oof"
        elif model_name == "lightgbm" and "lgbm_oof" in oof.columns:
            prediction_col = "lgbm_oof"
        elif model_name == "xgboost" and "xgb_oof" in oof.columns:
            prediction_col = "xgb_oof"
        else:
            raise ValueError(f"Could not find prediction column for {model_name}.")

    result = pd.DataFrame(
        {
            ID_COL: oof[ID_COL],
            TARGET_COL: oof[TARGET_COL],
            "prediction": oof[prediction_col],
            "model": model_name,
        }
    )
    if "fold" in oof.columns:
        result["fold"] = oof["fold"]
    elif folds is not None:
        result["fold"] = folds
    else:
        result["fold"] = -1
    return result[[ID_COL, TARGET_COL, "prediction", "fold", "model"]]


def save_oof_frame(oof: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(path, index=False)


def evaluate_oof_frame(
    oof: pd.DataFrame,
    model_name: str,
    threshold: float = DEFAULT_THRESHOLD,
    training_time_seconds: float | None = None,
    inference_time_seconds: float | None = None,
    n_features: int | None = None,
    fold_metrics: list[dict[str, float]] | None = None,
) -> dict[str, object]:
    metrics = {
        "model": model_name,
        "n_rows": int(len(oof)),
        "n_features": n_features,
        **calculate_metrics(oof[TARGET_COL], oof["prediction"], threshold=threshold),
        "threshold": threshold,
        **confusion_matrix_values(
            oof[TARGET_COL],
            oof["prediction"],
            threshold=threshold,
        ),
        "training_time_seconds": training_time_seconds,
        "inference_time_seconds": inference_time_seconds,
        "total_time_seconds": (
            training_time_seconds + inference_time_seconds
            if training_time_seconds is not None and inference_time_seconds is not None
            else None
        ),
    }
    if fold_metrics is not None:
        metrics["fold_metrics"] = fold_metrics
    return metrics


def save_metrics_report(metrics: dict[str, object], json_path: str | Path) -> None:
    json_path = Path(json_path)
    save_metrics(metrics, json_path)
    flat_metrics = {k: v for k, v in metrics.items() if k != "fold_metrics"}
    pd.DataFrame([flat_metrics]).to_csv(json_path.with_suffix(".csv"), index=False)


def plot_roc_curve(
    y_true,
    y_pred_proba,
    path: str | Path,
    title: str = "ROC Curve",
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_pr_curve(
    y_true,
    y_pred_proba,
    path: str | Path,
    title: str = "Precision-Recall Curve",
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    baseline = pd.Series(y_true).mean()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")
    ax.axhline(
        baseline,
        linestyle="--",
        color="gray",
        label=f"Baseline = {baseline:.4f}",
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(
    y_true,
    y_pred_proba,
    path: str | Path,
    threshold: float = DEFAULT_THRESHOLD,
    title: str | None = None,
) -> None:
    y_pred = (y_pred_proba >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(
        matrix,
        display_labels=["No Default", "Default"],
    ).plot(ax=ax, colorbar=False, values_format="d")
    ax.set_title(title or f"Confusion Matrix at threshold={threshold}")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_evaluation_plots(
    oof: pd.DataFrame,
    model_name: str,
    plots_dir: str | Path,
    threshold: float = DEFAULT_THRESHOLD,
) -> None:
    plots_dir = Path(plots_dir)
    label = model_name.replace("_", " ").title()
    plot_roc_curve(
        oof[TARGET_COL],
        oof["prediction"],
        plots_dir / f"{model_name}_roc_curve.png",
        title=f"{label} ROC Curve",
    )
    plot_pr_curve(
        oof[TARGET_COL],
        oof["prediction"],
        plots_dir / f"{model_name}_pr_curve.png",
        title=f"{label} Precision-Recall Curve",
    )
    plot_confusion_matrix(
        oof[TARGET_COL],
        oof["prediction"],
        plots_dir / f"{model_name}_confusion_matrix.png",
        threshold=threshold,
        title=f"{label} Confusion Matrix",
    )


def comparison_from_metrics(metrics_by_model: list[dict[str, object]]) -> pd.DataFrame:
    columns = [
        "model",
        "n_rows",
        "n_features",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "threshold",
        "training_time_seconds",
        "inference_time_seconds",
        "total_time_seconds",
    ]
    return pd.DataFrame(metrics_by_model).reindex(columns=columns)


def plot_metric_comparison(
    comparison: pd.DataFrame,
    metric: str,
    path: str | Path,
    title: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))

    ordered = comparison.sort_values(metric, ascending=False)

    ax.bar(
        ordered["model"],
        ordered[metric],
        color="#4C78A8",
    )

    ax.set_title(title or metric.replace("_", " ").title())
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.tick_params(axis="x", rotation=0)

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
