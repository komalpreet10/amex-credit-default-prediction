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
import pandas as pd
from google.cloud import bigquery, storage

from amex_default.config import ID_COL, RANDOM_STATE, TARGET_COL
from amex_default.interpret import save_best_fold_shap_plots
from gcp.config import (
    BQ_LOCATION,
    EXPERIMENT,
    FEATURE_TABLE,
    MODEL_ARTIFACTS,
    PROJECT_ID,
    REGION,
)

SELECTOR_NUM_BOOST_ROUND = 100  # selector model only needs feature ranking

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
    parser.add_argument("--params-uri", default=None)
    parser.add_argument("--final-num-boost-round", type=int, default=300)
    parser.add_argument("--feature-selection-threshold", type=float, default=0.95)
    parser.add_argument("--min-selected-features", type=int, default=300)
    parser.add_argument("--max-selected-features", type=int, default=1000)
    parser.add_argument("--shap-sample-size", type=int, default=None)
    parser.add_argument("--shap-max-display", type=int, default=30)
    parser.add_argument("--disable-shap", action="store_true")
    return parser.parse_args()


def load_tuning_result(
    params_uri: str | None,
) -> tuple[dict[str, object], dict[str, object]]:
    if not params_uri:
        return {}, {}
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
    return params, data


def read_training_data() -> pd.DataFrame:
    client = bigquery.Client(project=PROJECT_ID, location=BQ_LOCATION)
    query = f"SELECT * FROM `{FEATURE_TABLE}`"
    LOGGER.info("Reading training data from BigQuery table %s", FEATURE_TABLE)
    return client.query(query).result().to_dataframe()


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    required = {ID_COL, TARGET_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[ID_COL, TARGET_COL])
    for column in X.select_dtypes(include=["object", "string"]).columns:
        X[column] = X[column].astype("category")
    return X, y


def compute_scale_pos_weight(y: pd.Series) -> float:
    pos = int(y.sum())
    neg = len(y) - pos
    if pos == 0:
        raise ValueError("No positive examples in target.")
    spw = neg / pos
    LOGGER.info(
        "Class distribution — positives: %d (%.2f%%), negatives: %d, scale_pos_weight: %.4f",
        pos, 100 * pos / len(y), neg, spw,
    )
    return spw


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_params: dict[str, object],
    num_boost_round: int,
) -> lgb.Booster:
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
        num_boost_round=num_boost_round,
    )


def build_feature_importance(model: lgb.Booster) -> pd.DataFrame:
    importance = pd.DataFrame(
        {
            "feature": model.feature_name(),
            "importance": model.feature_importance(importance_type="gain"),
        }
    ).sort_values("importance", ascending=False)
    total_importance = float(importance["importance"].sum())
    if total_importance > 0:
        importance["importance_fraction"] = importance["importance"] / total_importance
        importance["cumulative_importance"] = importance["importance_fraction"].cumsum()
    else:
        importance["importance_fraction"] = 0.0
        importance["cumulative_importance"] = 0.0
    return importance


def select_features(
    feature_importance: pd.DataFrame,
    threshold: float,
    min_features: int,
    max_features: int,
) -> list[str]:
    if feature_importance.empty:
        raise ValueError("Cannot select features from an empty importance frame.")
    if not 0 < threshold <= 1:
        raise ValueError("--feature-selection-threshold must be in (0, 1].")
    if min_features <= 0:
        raise ValueError("--min-selected-features must be positive.")
    if max_features < min_features:
        raise ValueError("--max-selected-features must be >= --min-selected-features.")
    ranked = feature_importance.reset_index(drop=True)
    selected_count = int((ranked["cumulative_importance"] <= threshold).sum())
    if selected_count < len(ranked):
        selected_count += 1
    selected_count = max(selected_count, min_features)
    selected_count = min(selected_count, max_features, len(ranked))
    return ranked.head(selected_count)["feature"].tolist()


def build_metrics(
    X: pd.DataFrame,
    args: argparse.Namespace,
    tuned_params: dict[str, object],
    tuning_result: dict[str, object],
    scale_pos_weight: float,
    elapsed_seconds: float,
) -> dict[str, object]:
    best_user_attrs = tuning_result.get("best_user_attrs", {})
    confusion = best_user_attrs.get("confusion_matrix") or [[None, None], [None, None]]
    true_negative, false_positive = confusion[0]
    false_negative, true_positive = confusion[1]
    return {
        "model": "lightgbm",
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "full_feature_count": int(tuning_result.get("full_feature_count", X.shape[1])),
        "selected_feature_count": int(X.shape[1]),
        "scale_pos_weight": scale_pos_weight,
        "training_time_seconds": elapsed_seconds,
        "evaluation_source": "optuna_cross_validation",
        "tuning_metric": tuning_result.get("metric"),
        "tuning_cv_score": tuning_result.get("best_score"),
        "tuning_n_trials": tuning_result.get("n_trials"),
        "tuning_best_trial_number": tuning_result.get("best_trial_number"),
        "cv_roc_auc": best_user_attrs.get("mean_roc_auc"),
        "cv_pr_auc": best_user_attrs.get("mean_pr_auc"),
        "cv_precision": best_user_attrs.get("mean_precision"),
        "cv_recall": best_user_attrs.get("mean_recall"),
        "cv_f1": best_user_attrs.get("mean_f1"),
        "cv_true_negative": true_negative,
        "cv_false_positive": false_positive,
        "cv_false_negative": false_negative,
        "cv_true_positive": true_positive,
        "tuning_best_user_attrs": tuning_result.get("best_user_attrs", {}),
        "final_num_boost_round": args.final_num_boost_round,
        "feature_selection_threshold": args.feature_selection_threshold,
        "min_selected_features": args.min_selected_features,
        "max_selected_features": args.max_selected_features,
        "tuned_params": tuned_params,
    }


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
        raise ValueError("Artifact output directory must be a GCS URI.")
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
    try:
        from google.cloud import aiplatform
        aiplatform.init(
            project=PROJECT_ID,
            location=REGION,
            experiment=EXPERIMENT,
            staging_bucket=MODEL_ARTIFACTS,
        )
        with aiplatform.start_run() as run:
            run.log_params({
                "model": "lightgbm",
                "final_num_boost_round": args.final_num_boost_round,
                "feature_selection_threshold": args.feature_selection_threshold,
                "min_selected_features": args.min_selected_features,
                "max_selected_features": args.max_selected_features,
                "shap_sample_size": args.shap_sample_size,
                "shap_max_display": args.shap_max_display,
                "disable_shap": args.disable_shap,
                "evaluation_source": metrics.get("evaluation_source"),
                "scale_pos_weight": metrics.get("scale_pos_weight"),
            })
            run.log_metrics({
                key: value
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            })
    except Exception:
        LOGGER.exception("Vertex AI Experiment logging failed — continuing")


def log_mlflow_run(
    args: argparse.Namespace,
    metrics: dict[str, object],
    artifact_dir: Path,
) -> None:
    try:
        import mlflow

        tracking_dir = artifact_dir / "mlruns"
        mlflow.set_tracking_uri(f"file://{tracking_dir}")
        mlflow.set_experiment(EXPERIMENT)

        with mlflow.start_run(run_name="final-lightgbm-training"):
            mlflow.log_params({
                "model": "lightgbm",
                "feature_table": FEATURE_TABLE,
                "final_num_boost_round": args.final_num_boost_round,
                "feature_selection_threshold": args.feature_selection_threshold,
                "min_selected_features": args.min_selected_features,
                "max_selected_features": args.max_selected_features,
                "shap_sample_size": args.shap_sample_size,
                "shap_max_display": args.shap_max_display,
                "disable_shap": args.disable_shap,
                "evaluation_source": metrics.get("evaluation_source"),
            })
            mlflow.log_metrics({
                key: value
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            })
            for path in artifact_dir.rglob("*"):
                if path.is_file() and "mlruns" not in path.parts:
                    artifact_path = path.relative_to(artifact_dir).parent
                    mlflow.log_artifact(
                        str(path),
                        artifact_path=None
                        if str(artifact_path) == "."
                        else str(artifact_path),
                    )
        LOGGER.info("MLflow run logged under %s", tracking_dir)
    except Exception:
        LOGGER.exception("MLflow logging failed — continuing")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    start = time.perf_counter()
    df = read_training_data()
    X, y = split_features_target(df)
    scale_pos_weight = compute_scale_pos_weight(y)

    tuned_params, tuning_result = load_tuning_result(args.params_uri)

    # Inject scale_pos_weight — from actual class ratio, not tuned
    model_params = {
        **DEFAULT_PARAMS,
        "scale_pos_weight": scale_pos_weight,
        **(tuned_params or {}),
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir)

        # ── Step 1: Selector model (100 rounds — only needs feature ranking) ──
        LOGGER.info("Training selector model (%d rounds) for feature importance",
                    SELECTOR_NUM_BOOST_ROUND)
        selector_model = train_model(X, y, model_params, SELECTOR_NUM_BOOST_ROUND)
        feature_importance = build_feature_importance(selector_model)
        selected_features = select_features(
            feature_importance,
            threshold=args.feature_selection_threshold,
            min_features=args.min_selected_features,
            max_features=args.max_selected_features,
        )
        LOGGER.info(
            "Selected %d of %d features (threshold=%.3f)",
            len(selected_features), X.shape[1], args.feature_selection_threshold,
        )

        # ── Step 2: Final model (full rounds, selected features only) ─────────
        X_selected = X[selected_features]
        LOGGER.info("Training final model (%d rounds) on %d selected features",
                    args.final_num_boost_round, len(selected_features))
        final_model = train_model(X_selected, y, model_params, args.final_num_boost_round)
        feature_importance["selected"] = feature_importance["feature"].isin(selected_features)

        # ── Step 3: Build combined metrics ────────────────────────────────────
        metrics = build_metrics(
            X_selected, args,
            tuned_params=tuned_params,
            tuning_result={**tuning_result, "full_feature_count": int(X.shape[1])},
            scale_pos_weight=scale_pos_weight,
            elapsed_seconds=time.perf_counter() - start,
        )

        # ── Step 4: Save artifacts ────────────────────────────────────────────
        final_model.save_model(str(output_dir / "model.txt"))
        feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)

        for fname, content in (
            ("feature_list.json", selected_features),
            ("selected_feature_list.json", selected_features),
            ("full_feature_list.json", X.columns.tolist()),
        ):
            (output_dir / fname).write_text(
                json.dumps(content, indent=2), encoding="utf-8"
            )

        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )

        # ── Step 5: SHAP plots ────────────────────────────────────────────────
        save_shap_plots(final_model, X_selected, args, output_dir)

        # ── Step 6: Log to MLflow and Vertex AI Experiments ───────────────────
        log_mlflow_run(args, metrics, output_dir)
        log_vertex_experiment(args, metrics)

        # ── Step 7: Upload to GCS ─────────────────────────────────────────────
        LOGGER.info("Uploading model artifacts to %s", MODEL_ARTIFACTS)
        upload_directory(output_dir, MODEL_ARTIFACTS)

    LOGGER.info(
        "Done — tuning %s: %s | selected features: %d | time: %.1fs",
        metrics.get("tuning_metric"),
        metrics.get("tuning_cv_score"),
        metrics["selected_feature_count"],
        metrics["training_time_seconds"],
    )


if __name__ == "__main__":
    main()
