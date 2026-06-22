from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap


def plot_feature_importance(
    importance_df: pd.DataFrame,
    path: str | Path,
    top_n: int = 30,
    title: str = "Feature Importance",
) -> pd.DataFrame:
    score_col = (
        "importance_mean"
        if "importance_mean" in importance_df.columns
        else "importance"
    )
    importance = importance_df.sort_values(score_col, ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(importance["feature"][::-1], importance[score_col][::-1])
    ax.set_title(title)
    ax.set_xlabel(score_col.replace("_", " ").title())
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return importance


def _positive_class_values(shap_values):
    return shap_values[1] if isinstance(shap_values, list) else shap_values


def shap_summary_plot(
    model,
    X_sample: pd.DataFrame,
    path: str | Path,
    max_display: int = 20,
) -> None:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    values = _positive_class_values(shap_values)
    shap.summary_plot(values, X_sample, max_display=max_display, show=False)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def shap_bar_plot(
    model,
    X_sample: pd.DataFrame,
    path: str | Path,
    max_display: int = 20,
) -> None:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    if isinstance(shap_values.values, list):
        shap_values.values = shap_values.values[1]
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def save_best_fold_shap_plots(
    model,
    X_valid: pd.DataFrame,
    model_name: str,
    plots_dir: str | Path,
    max_display: int = 20,
) -> None:
    plots_dir = Path(plots_dir)
    shap_summary_plot(
        model,
        X_valid,
        plots_dir / f"{model_name}_shap_summary.png",
        max_display=max_display,
    )
    shap_bar_plot(
        model,
        X_valid,
        plots_dir / f"{model_name}_shap_bar.png",
        max_display=max_display,
    )
