from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
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
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "tree_method": "hist",
    "nthread": -1,
    "seed": RANDOM_STATE,
}


def average_feature_importance(
    models: list[xgb.Booster],
    importance_type: str = "gain",
) -> pd.DataFrame:
    importance_frames = []
    for fold, model in enumerate(models, start=1):
        scores = model.get_score(importance_type=importance_type)
        importance_frames.append(
            pd.DataFrame(
                {
                    "feature": list(scores.keys()),
                    f"fold_{fold}": list(scores.values()),
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
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof = np.zeros(len(X))
    fold_ids = np.full(len(X), -1)
    models: list[xgb.Booster] = []
    fold_metrics: list[dict[str, float]] = []
    fold_validation_indices: list[np.ndarray] = []
    training_time_seconds = 0.0
    inference_time_seconds = 0.0

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        print(f"\nXGBoost fold {fold}")
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dvalid = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)

        start = time.perf_counter()
        model = xgb.train(
            model_params,
            dtrain,
            num_boost_round=1000,
            evals=[(dvalid, "valid")],
            early_stopping_rounds=50,
            verbose_eval=100,
        )
        training_time_seconds += time.perf_counter() - start

        start = time.perf_counter()
        pred = model.predict(
            dvalid,
            iteration_range=(0, model.best_iteration + 1),
        )
        inference_time_seconds += time.perf_counter() - start

        oof[valid_idx] = pred
        fold_ids[valid_idx] = fold
        fold_validation_indices.append(valid_idx)
        fold_metric = {
            "fold": fold,
            **calculate_metrics(y_valid, pred),
            "best_iteration": int(model.best_iteration or 0),
        }
        fold_metrics.append(fold_metric)
        models.append(model)
        print(f"Fold {fold} ROC-AUC: {fold_metric['roc_auc']:.4f}")

    oof_frame = build_oof_frame(
        customer_ids=customer_ids if customer_ids is not None else X.index,
        y_true=y,
        y_pred_proba=oof,
        folds=fold_ids,
        model_name="xgboost",
    )
    metrics = evaluate_oof_frame(
        oof_frame,
        "xgboost",
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
    model_dir: str | Path = MODEL_DIR / "xgboost",
    predictions_dir: str | Path = PREDICTIONS_DIR,
    reports_dir: str | Path = REPORTS_DIR,
    plots_dir: str | Path = PLOTS_DIR,
) -> None:
    model_dir = Path(model_dir)
    reports_dir = Path(reports_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    for fold, model in enumerate(result["models"], start=1):
        model.save_model(str(model_dir / f"xgb_fold_{fold}.json"))

    save_oof_frame(result["oof"], Path(predictions_dir) / "xgboost_oof.parquet")
    save_metrics_report(result["metrics"], reports_dir / "xgboost_metrics.json")

    importance = result["feature_importance"]
    importance.to_csv(reports_dir / "xgboost_feature_importance.csv", index=False)
    plot_feature_importance(
        importance,
        Path(plots_dir) / "xgboost_feature_importance.png",
        title="XGBoost Feature Importance",
    )
    save_evaluation_plots(result["oof"], "xgboost", plots_dir)
