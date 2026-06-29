from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
import xgboost as xgb


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


def _prepare_xgboost_shap_data(X_sample: pd.DataFrame) -> pd.DataFrame:
    X_sample = X_sample.copy()

    cat_cols = X_sample.select_dtypes(include="category").columns
    for col in cat_cols:
        X_sample[col] = X_sample[col].cat.codes.astype("int32")

    return X_sample


def _prepare_shap_data(model, X_sample: pd.DataFrame) -> pd.DataFrame:
    if isinstance(model, xgb.Booster):
        return _prepare_xgboost_shap_data(X_sample)

    return X_sample


def shap_summary_plot(
    model,
    X_sample: pd.DataFrame,
    path: str | Path,
    max_display: int = 20,
) -> None:
    X_shap = _prepare_shap_data(model, X_sample)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
    values = _positive_class_values(shap_values)

    shap.summary_plot(
        values,
        X_shap,
        max_display=max_display,
        show=False,
    )

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=160)
    plt.close()


def shap_bar_plot(
    model,
    X_sample: pd.DataFrame,
    path: str | Path,
    max_display: int = 20,
) -> None:
    X_shap = _prepare_shap_data(model, X_sample)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
    values = _positive_class_values(shap_values)

    shap.summary_plot(
        values,
        X_shap,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=160)
    plt.close()


def save_best_fold_shap_plots(
    model,
    X_valid: pd.DataFrame,
    model_name: str,
    plots_dir: str | Path,
    max_display: int = 20,
) -> None:
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

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
