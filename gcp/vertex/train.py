from __future__ import annotations

import argparse
import json
import logging
import tempfile
import time
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from google.cloud import bigquery, storage

from amex_default.interpret import save_best_fold_shap_plots

PROJECT_ID = "amex-credit-risk-ml"
LOCATION = "us-central1"
BQ_LOCATION = "US"
TABLE = "amex-credit-risk-ml.amex_ml.train_features"
OUTPUT_DIR = "gs://amex-credit-risk-ml-data/models/lightgbm/"
EXPERIMENT = "amex-credit-default"

ID_COL = "customer_ID"
TARGET_COL = "target"
RANDOM_STATE = 42

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
    parser.add_argument("--project", default=PROJECT_ID)
    parser.add_argument("--location", default=LOCATION)
    parser.add_argument("--bq-location", default=BQ_LOCATION)
    parser.add_argument("--table", default=TABLE)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--params-uri", default=None)
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--final-num-boost-round", type=int, default=300)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--shap-sample-size", type=int, default=None)
    parser.add_argument("--shap-max-display", type=int, default=30)
    parser.add_argument("--disable-shap", action="store_true")
    parser.add_argument("--disable-experiment", action="store_true")
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


def read_training_data(args: argparse.Namespace) -> pd.DataFrame:
    client = bigquery.Client(project=args.project, location=args.bq_location)
    query = f"SELECT * FROM `{args.table}`"
    if args.max_rows:
        query += f" LIMIT {args.max_rows}"

    LOGGER.info("Reading training data from BigQuery table %s", args.table)
    return client.query(query).result().to_dataframe()


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    required = {ID_COL, TARGET_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    customer_ids = df[ID_COL]
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[ID_COL, TARGET_COL])

    for column in X.select_dtypes(include=["object", "string"]).columns:
        X[column] = X[column].astype("category")

    return customer_ids, X, y


def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    args: argparse.Namespace,
    tuned_params: dict[str, object] | None = None,
) -> lgb.Booster:
    model_params = {**DEFAULT_PARAMS, **(tuned_params or {})}
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
        num_boost_round=args.final_num_boost_round,
    )


def build_feature_importance(model: lgb.Booster) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": model.feature_name(),
            "importance": model.feature_importance(importance_type="gain"),
        }
    ).sort_values("importance", ascending=False)


def build_metrics(
    X: pd.DataFrame,
    args: argparse.Namespace,
    tuned_params: dict[str, object],
    tuning_result: dict[str, object],
    elapsed_seconds: float,
) -> dict[str, object]:
    return {
        "model": "lightgbm",
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "training_time_seconds": elapsed_seconds,
        "evaluation_source": "optuna_cross_validation",
        "tuning_metric": tuning_result.get("metric"),
        "tuning_cv_score": tuning_result.get("best_score"),
        "tuning_n_trials": tuning_result.get("n_trials"),
        "tuning_best_trial_number": tuning_result.get("best_trial_number"),
        "tuning_best_user_attrs": tuning_result.get("best_user_attrs", {}),
        "final_num_boost_round": args.final_num_boost_round,
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
        raise ValueError("--output-dir must be a GCS URI.")

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
    if args.disable_experiment:
        return

    try:
        from google.cloud import aiplatform

        aiplatform.init(
            project=args.project,
            location=args.location,
            experiment=args.experiment,
            staging_bucket=args.output_dir,
        )
        with aiplatform.start_run(run=args.run_name) as run:
            run.log_params(
                {
                    "model": "lightgbm",
                    "final_num_boost_round": args.final_num_boost_round,
                    "shap_sample_size": args.shap_sample_size,
                    "shap_max_display": args.shap_max_display,
                    "disable_shap": args.disable_shap,
                    "evaluation_source": metrics.get("evaluation_source"),
                }
            )
            run.log_metrics(
                {
                    key: value
                    for key, value in metrics.items()
                    if isinstance(value, (int, float))
                }
            )
    except Exception:
        LOGGER.exception("Vertex AI Experiment logging failed")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    start = time.perf_counter()
    df = read_training_data(args)
    _, X, y = split_features_target(df)
    tuned_params, tuning_result = load_tuning_result(args.params_uri)

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir)

        final_model = train_final_model(X, y, args, tuned_params=tuned_params)
        feature_importance = build_feature_importance(final_model)

        metrics = build_metrics(
            X,
            args,
            tuned_params=tuned_params,
            tuning_result=tuning_result,
            elapsed_seconds=time.perf_counter() - start,
        )

        final_model.save_model(str(output_dir / "model.txt"))
        feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)

        (output_dir / "feature_list.json").write_text(
            json.dumps(X.columns.tolist(), indent=2),
            encoding="utf-8",
        )
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )
        save_shap_plots(final_model, X, args, output_dir)

        LOGGER.info("Uploading model artifacts to %s", args.output_dir)
        upload_directory(output_dir, args.output_dir)
        log_vertex_experiment(args, metrics)


if __name__ == "__main__":
    main()
