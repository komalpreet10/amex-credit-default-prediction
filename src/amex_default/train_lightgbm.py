from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from amex_default.config import (
    MODEL_DIR,
    N_SPLITS,
    PLOTS_DIR,
    PREDICTIONS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
)
from amex_default.evaluate import (
    build_oof_frame,
    calculate_metrics,
    evaluate_oof_frame,
    save_evaluation_plots,
    save_metrics_report,
    save_oof_frame,
)
from amex_default.interpret import plot_feature_importance


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


def get_categorical_features(X: pd.DataFrame) -> list[str]:
    return X.select_dtypes(include=["category"]).columns.tolist()


def average_feature_importance(models: list[lgb.Booster]) -> pd.DataFrame:
    importance_frames = []
    for fold, model in enumerate(models, start=1):
        importance_frames.append(
            pd.DataFrame(
                {
                    "feature": model.feature_name(),
                    f"fold_{fold}": model.feature_importance(importance_type="gain"),
                }
            )
        )

    importance = importance_frames[0]
    for fold_importance in importance_frames[1:]:
        importance = importance.merge(fold_importance, on="feature", how="outer")

    importance = importance.fillna(0)
    fold_columns = [col for col in importance.columns if col.startswith("fold_")]
    importance["importance_mean"] = importance[fold_columns].mean(axis=1)
    importance["importance_std"] = importance[fold_columns].std(axis=1).fillna(0)
    return importance.sort_values("importance_mean", ascending=False)


def train_cv(
    X: pd.DataFrame,
    y: pd.Series,
    customer_ids=None,
    params: dict | None = None,
    n_splits: int = N_SPLITS,
) -> dict[str, object]:
    model_params = {**DEFAULT_PARAMS, **(params or {})}
    categorical_features = get_categorical_features(X)
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof = np.zeros(len(X))
    fold_ids = np.full(len(X), -1)
    models: list[lgb.Booster] = []
    fold_metrics: list[dict[str, float]] = []
    fold_validation_indices: list[np.ndarray] = []
    training_time_seconds = 0.0
    inference_time_seconds = 0.0

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        print(f"\nLightGBM fold {fold}")
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        dtrain = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=categorical_features,
            free_raw_data=False,
        )
        dvalid = lgb.Dataset(
            X_valid,
            label=y_valid,
            categorical_feature=categorical_features,
            reference=dtrain,
            free_raw_data=False,
        )

        start = time.perf_counter()
        model = lgb.train(
            model_params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(100),
            ],
        )
        training_time_seconds += time.perf_counter() - start

        start = time.perf_counter()
        pred = model.predict(X_valid, num_iteration=model.best_iteration)
        inference_time_seconds += time.perf_counter() - start

        oof[valid_idx] = pred
        fold_ids[valid_idx] = fold
        fold_validation_indices.append(valid_idx)
        fold_metric = {
            "fold": fold,
            **calculate_metrics(y_valid, pred),
            "best_iteration": int(model.best_iteration or model.current_iteration()),
        }
        fold_metrics.append(fold_metric)
        models.append(model)
        print(f"Fold {fold} ROC-AUC: {fold_metric['roc_auc']:.4f}")

    oof_frame = build_oof_frame(
        customer_ids=customer_ids if customer_ids is not None else X.index,
        y_true=y,
        y_pred_proba=oof,
        folds=fold_ids,
        model_name="lightgbm",
    )
    metrics = evaluate_oof_frame(
        oof_frame,
        "lightgbm",
        training_time_seconds=training_time_seconds,
        inference_time_seconds=inference_time_seconds,
        n_features=X.shape[1],
        fold_metrics=fold_metrics,
    )
    best_fold_index = int(
        max(range(len(fold_metrics)), key=lambda i: fold_metrics[i]["roc_auc"])
    )

    return {
        "models": models,
        "oof": oof_frame,
        "metrics": metrics,
        "fold_metrics": fold_metrics,
        "fold_validation_indices": fold_validation_indices,
        "best_fold_index": best_fold_index,
        "feature_importance": average_feature_importance(models),
    }


def save_cv_artifacts(
    result: dict[str, object],
    model_dir: str | Path = MODEL_DIR / "lightgbm",
    predictions_dir: str | Path = PREDICTIONS_DIR,
    reports_dir: str | Path = REPORTS_DIR,
    plots_dir: str | Path = PLOTS_DIR,
) -> None:
    model_dir = Path(model_dir)
    reports_dir = Path(reports_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    for fold, model in enumerate(result["models"], start=1):
        model.save_model(str(model_dir / f"lgbm_fold_{fold}.txt"))

    save_oof_frame(result["oof"], Path(predictions_dir) / "lightgbm_oof.parquet")
    save_metrics_report(result["metrics"], reports_dir / "lightgbm_metrics.json")

    importance = result["feature_importance"]
    importance.to_csv(reports_dir / "lightgbm_feature_importance.csv", index=False)
    plot_feature_importance(
        importance,
        Path(plots_dir) / "lightgbm_feature_importance.png",
        title="LightGBM Feature Importance",
    )
    save_evaluation_plots(result["oof"], "lightgbm", plots_dir)


def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict | None = None,
    num_boost_round: int = 300,
) -> lgb.Booster:
    model_params = {**DEFAULT_PARAMS, **(params or {})}
    categorical_features = get_categorical_features(X)
    dtrain = lgb.Dataset(
        X,
        label=y,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )

    model = lgb.train(
        model_params,
        dtrain,
        num_boost_round=num_boost_round,
    )
    model.categorical_feature_names = categorical_features
    return model


def save_final_model(
    model: lgb.Booster,
    feature_names: list[str],
    metrics: dict[str, object] | None = None,
    categorical_feature_names: list[str] | None = None,
    output_dir: str | Path = MODEL_DIR / "final",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model.save_model(str(output_dir / "final_model.txt"))

    with (output_dir / "feature_list.json").open("w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    categorical_feature_names = categorical_feature_names or getattr(
        model,
        "categorical_feature_names",
        [],
    )
    with (output_dir / "categorical_feature_list.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(categorical_feature_names, f, indent=2)

    summary = [
        "# Final Model Summary",
        "",
        "Final serving model: LightGBM trained on all engineered training rows.",
        f"Final num_boost_round: {model.current_iteration()}",
        f"Categorical features: {len(categorical_feature_names)}",
        "Reported performance should use 5-fold OOF CV metrics, not in-sample final-model metrics.",
    ]

    if metrics:
        summary.extend(
            [
                "",
                "Selected CV metrics:",
                f"- ROC-AUC: {metrics.get('roc_auc'):.6f}",
                f"- PR-AUC: {metrics.get('pr_auc'):.6f}",
                f"- F1: {metrics.get('f1'):.6f}",
                f"- Training time seconds: {metrics.get('training_time_seconds')}",
                f"- Inference time seconds: {metrics.get('inference_time_seconds')}",
            ]
        )

    (REPORTS_DIR / "final_model_summary.md").write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )
