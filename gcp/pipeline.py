from kfp import compiler, dsl

from gcp.config import (
    BQ_LOCATION,
    DEPLOYED_MODEL_DISPLAY_NAME,
    ENDPOINT_DISPLAY_NAME,
    ENDPOINT_MACHINE_TYPE,
    ENDPOINT_MAX_REPLICA_COUNT,
    ENDPOINT_MIN_REPLICA_COUNT,
    ENDPOINT_TRAFFIC_PERCENTAGE,
    FEATURE_TABLE,
    FEATURES,
    MODEL_ARTIFACTS,
    MODEL_DISPLAY_NAME,
    PIPELINE_ROOT,
    PREPROCESSED,
    PROJECT_ID,
    RAW_DATA,
    RAW_LABELS,
    REGION,
    SERVING_IMAGE,
    TEST_FEATURE_TABLE,
    TRAINING_JOB_DISPLAY_NAME,
    TRAINING_MACHINE_TYPE,
    TRAINING_REPLICA_COUNT,
    TRAINING_SHAP_MAX_DISPLAY,
    TRAINING_SHAP_SAMPLE_SIZE,
    TRAINING_IMAGE,
    TRAIN_FEATURE_TABLE,
    TUNING_JOB_DISPLAY_NAME,
    TUNING_MACHINE_TYPE,
    TUNED_PARAMS_URI,
    TUNING_N_SPLITS,
    TUNING_N_TRIALS,
    TUNING_REPLICA_COUNT,
    TUNING_ARTIFACTS,
)


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["google-cloud-dataproc"],
)
def submit_dataproc_pyspark_batch(
    project: str,
    region: str,
    batch_id: str,
    main_python_file_uri: str,
    py_file_uris: list[str],
    args: list[str],
    runtime_properties: dict[str, str],
    timeout_seconds: int = 7200,
) -> str:
    import uuid

    from google.cloud.dataproc_v1 import BatchControllerClient
    from google.cloud.dataproc_v1.types import Batch, PySparkBatch, RuntimeConfig

    client = BatchControllerClient(
        client_options={"api_endpoint": f"{region}-dataproc.googleapis.com:443"}
    )
    parent = f"projects/{project}/locations/{region}"
    batch = Batch(
        pyspark_batch=PySparkBatch(
            main_python_file_uri=main_python_file_uri,
            python_file_uris=py_file_uris,
            args=args,
        ),
        runtime_config=RuntimeConfig(properties=runtime_properties),
    )

    actual_batch_id = f"{batch_id}-{uuid.uuid4().hex[:8]}"
    operation = client.create_batch(
        request={"parent": parent, "batch": batch, "batch_id": actual_batch_id}
    )
    response = operation.result(timeout=timeout_seconds)
    if response.state.name != "SUCCEEDED":
        raise RuntimeError(
            f"Dataproc batch {actual_batch_id} ended as {response.state.name}: "
            f"{response.state_message}"
        )
    return response.name


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["google-cloud-bigquery", "pyarrow"],
)
def load_features_to_bigquery(
    project: str,
    location: str,
    source_uri: str,
    table: str,
) -> str:
    from google.cloud import bigquery
    from google.cloud.exceptions import NotFound

    client = bigquery.Client(project=project, location=location)
    dataset_id = table.split(".")[-2]
    dataset_ref = bigquery.DatasetReference(project, dataset_id)

    try:
        client.get_dataset(dataset_ref)
    except NotFound:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = location
        client.create_dataset(dataset)

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.PARQUET,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    client.load_table_from_uri(
        source_uri,
        table,
        job_config=job_config,
        location=location,
    ).result()
    return table


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["google-cloud-bigquery"],
)
def split_bigquery_feature_table(
    project: str,
    location: str,
    source_table: str,
    train_table: str,
    test_table: str,
    train_ratio: float,
) -> str:
    from google.cloud import bigquery

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    client = bigquery.Client(project=project, location=location)
    threshold = int(train_ratio * 100)
    split_expr = "MOD(ABS(FARM_FINGERPRINT(CAST(customer_ID AS STRING))), 100)"
    for destination, condition in (
        (train_table, f"{split_expr} < {threshold}"),
        (test_table, f"{split_expr} >= {threshold}"),
    ):
        query = f"""
        CREATE OR REPLACE TABLE `{destination}` AS
        SELECT *
        FROM `{source_table}`
        WHERE {condition}
        """
        client.query(query, location=location).result()

    return train_table


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["google-cloud-aiplatform"],
)
def run_vertex_tuning_job(
    project: str,
    region: str,
    training_image: str,
    table: str,
    output_dir: str,
    display_name: str,
    metric: str,
    n_trials: int,
    n_splits: int,
    num_boost_round: int,
    early_stopping_rounds: int,
    max_rows: int,
    balanced_smoke_sample: bool,
    replica_count: int,
    machine_type: str,
) -> str:
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=region, staging_bucket=output_dir)
    job = aiplatform.CustomContainerTrainingJob(
        display_name=display_name,
        container_uri=training_image,
        command=["python", "gcp/vertex/tune_lightgbm_optuna.py"],
    )
    job_args = [
        "--table",
        table,
        "--metric",
        metric,
        "--n-trials",
        str(n_trials),
        "--n-splits",
        str(n_splits),
        "--num-boost-round",
        str(num_boost_round),
        "--early-stopping-rounds",
        str(early_stopping_rounds),
        "--output-dir",
        output_dir,
    ]
    if max_rows > 0:
        job_args.extend(["--max-rows", str(max_rows)])
    if balanced_smoke_sample:
        job_args.append("--balanced-smoke-sample")
    job.run(
        args=job_args,
        replica_count=replica_count,
        machine_type=machine_type,
        sync=True,
    )
    return f"{output_dir.rstrip('/')}/lightgbm_optuna_best_params.json"


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["google-cloud-aiplatform"],
)
def run_vertex_training_job(
    project: str,
    region: str,
    training_image: str,
    table: str,
    eval_table: str,
    output_dir: str,
    params_uri: str,
    display_name: str,
    shap_sample_size: int,
    shap_max_display: int,
    max_rows: int,
    balanced_smoke_sample: bool,
    selector_num_boost_round: int,
    final_num_boost_round: int,
    min_selected_features: int,
    max_selected_features: int,
    disable_shap: bool,
    replica_count: int,
    machine_type: str,
) -> str:
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=region, staging_bucket=output_dir)
    job = aiplatform.CustomContainerTrainingJob(
        display_name=display_name,
        container_uri=training_image,
        command=["python", "gcp/vertex/train.py"],
    )
    job_args = [
        "--table",
        table,
        "--eval-table",
        eval_table,
        "--params-uri",
        params_uri,
        "--output-dir",
        output_dir,
        "--selector-num-boost-round",
        str(selector_num_boost_round),
        "--final-num-boost-round",
        str(final_num_boost_round),
        "--min-selected-features",
        str(min_selected_features),
        "--max-selected-features",
        str(max_selected_features),
        "--shap-sample-size",
        str(shap_sample_size),
        "--shap-max-display",
        str(shap_max_display),
    ]
    if max_rows > 0:
        job_args.extend(["--max-rows", str(max_rows)])
    if balanced_smoke_sample:
        job_args.append("--balanced-smoke-sample")
    if disable_shap:
        job_args.append("--disable-shap")
    model = job.run(
        args=job_args,
        replica_count=replica_count,
        machine_type=machine_type,
        sync=True,
    )
    return model.resource_name if model else output_dir


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["google-cloud-aiplatform"],
)
def upload_vertex_model(
    project: str,
    region: str,
    artifact_uri: str,
    serving_image: str,
    model_display_name: str,
) -> str:
    from google.cloud import aiplatform

    if not serving_image:
        raise ValueError(
            "serving_image is required. Build and push docker/Dockerfile.serve, "
            "then pass SERVING_IMAGE_URI or the pipeline serving_image parameter."
        )

    aiplatform.init(project=project, location=region)
    model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri=serving_image,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        serving_container_ports=[8080],
        labels={"project": "amex-credit-default", "model": "lightgbm"},
        sync=True,
    )
    return model.resource_name


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["google-cloud-aiplatform"],
)
def deploy_model_to_endpoint(
    project: str,
    region: str,
    model_resource_name: str,
    endpoint_display_name: str,
    deployed_model_display_name: str,
    machine_type: str,
    min_replica_count: int,
    max_replica_count: int,
    traffic_percentage: int,
) -> str:
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=region)
    model = aiplatform.Model(model_resource_name)
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_display_name}"',
        order_by="create_time desc",
    )
    if endpoints:
        endpoint = endpoints[0]
    else:
        endpoint = aiplatform.Endpoint.create(display_name=endpoint_display_name)
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=deployed_model_display_name,
        machine_type=machine_type,
        min_replica_count=min_replica_count,
        max_replica_count=max_replica_count,
        traffic_percentage=traffic_percentage,
        sync=True,
    )
    return endpoint.resource_name


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "google-cloud-bigquery",
        "google-cloud-storage",
        "numpy",
        "pandas",
        "pyarrow",
    ],
)
def compute_feature_drift(
    project: str,
    location: str,
    baseline_table: str,
    current_table: str,
    metrics_table: str,
    output_uri: str,
) -> str:
    import numpy as np
    import pandas as pd
    from google.cloud import bigquery, storage

    id_col = "customer_ID"
    target_col = "target"
    threshold = 0.2

    client = bigquery.Client(project=project, location=location)
    baseline = client.query(f"SELECT * FROM `{baseline_table}`").result().to_dataframe()
    current = client.query(f"SELECT * FROM `{current_table}`").result().to_dataframe()

    rows = []
    numeric_columns = [
        column
        for column in baseline.select_dtypes(include=[np.number]).columns
        if column not in {id_col, target_col} and column in current.columns
    ]
    for column in numeric_columns:
        base_values = baseline[column].dropna()
        current_values = current[column].dropna()
        if base_values.empty or current_values.empty:
            psi = np.nan
        else:
            edges = np.unique(np.quantile(base_values, np.linspace(0, 1, 11)))
            if len(edges) < 2:
                psi = 0.0
            else:
                base_counts, _ = np.histogram(base_values, bins=edges)
                current_counts, _ = np.histogram(current_values, bins=edges)
                base_pct = base_counts / max(base_counts.sum(), 1)
                current_pct = current_counts / max(current_counts.sum(), 1)
                base_pct = np.where(base_pct == 0, 0.0001, base_pct)
                current_pct = np.where(current_pct == 0, 0.0001, current_pct)
                psi = float(
                    np.sum((current_pct - base_pct) * np.log(current_pct / base_pct))
                )

        rows.append(
            {
                "feature_name": column,
                "psi": psi,
                "drifted": bool(psi > threshold) if not np.isnan(psi) else False,
                "threshold": threshold,
            }
        )

    report = pd.DataFrame(rows).sort_values("psi", ascending=False)
    bucket_name, blob_name = output_uri.removeprefix("gs://").split("/", 1)
    storage.Client().bucket(bucket_name).blob(blob_name).upload_from_string(
        report.to_csv(index=False),
        content_type="text/csv",
    )
    client.load_table_from_dataframe(
        report,
        metrics_table,
        job_config=bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND
        ),
    ).result()
    return output_uri


@dsl.pipeline(
    name="amex-credit-default-mlops",
    pipeline_root=PIPELINE_ROOT,
)
def amex_pipeline(
    project: str = PROJECT_ID,
    region: str = REGION,
    bq_location: str = BQ_LOCATION,
    raw_data: str = RAW_DATA,
    raw_labels: str = RAW_LABELS,
    preprocessed_output: str = PREPROCESSED,
    feature_output: str = FEATURES,
    feature_table: str = FEATURE_TABLE,
    train_feature_table: str = TRAIN_FEATURE_TABLE,
    test_feature_table: str = TEST_FEATURE_TABLE,
    train_ratio: float = 0.8,
    model_artifacts: str = MODEL_ARTIFACTS,
    training_image: str = TRAINING_IMAGE,
    serving_image: str = SERVING_IMAGE,
) -> None:
    # The default deployable pipeline starts from the existing BigQuery feature
    # table because the current project quota is below Dataproc Serverless'
    # minimum valid CPU request. Spark components remain available for projects
    # with enough quota to rerun feature engineering from raw statement data.
    split = split_bigquery_feature_table(
        project=project,
        location=bq_location,
        source_table=feature_table,
        train_table=train_feature_table,
        test_table=test_feature_table,
        train_ratio=train_ratio,
    )

    training = run_vertex_training_job(
        project=project,
        region=region,
        training_image=training_image,
        table=train_feature_table,
        eval_table=test_feature_table,
        output_dir=model_artifacts,
        params_uri=TUNED_PARAMS_URI,
        display_name=TRAINING_JOB_DISPLAY_NAME,
        shap_sample_size=TRAINING_SHAP_SAMPLE_SIZE,
        shap_max_display=TRAINING_SHAP_MAX_DISPLAY,
        max_rows=0,
        balanced_smoke_sample=False,
        selector_num_boost_round=100,
        final_num_boost_round=300,
        min_selected_features=300,
        max_selected_features=1000,
        disable_shap=False,
        replica_count=TRAINING_REPLICA_COUNT,
        machine_type=TRAINING_MACHINE_TYPE,
    )
    tuning = run_vertex_tuning_job(
        project=project,
        region=region,
        training_image=training_image,
        table=train_feature_table,
        output_dir=TUNING_ARTIFACTS,
        display_name=TUNING_JOB_DISPLAY_NAME,
        metric="pr_auc",
        n_trials=TUNING_N_TRIALS,
        n_splits=TUNING_N_SPLITS,
        num_boost_round=700,
        early_stopping_rounds=50,
        max_rows=0,
        balanced_smoke_sample=False,
        replica_count=TUNING_REPLICA_COUNT,
        machine_type=TUNING_MACHINE_TYPE,
    )
    tuning.after(split)
    training.after(tuning)

    model = upload_vertex_model(
        project=project,
        region=region,
        artifact_uri=model_artifacts,
        serving_image=serving_image,
        model_display_name=MODEL_DISPLAY_NAME,
    )
    model.after(training)

    endpoint = deploy_model_to_endpoint(
        project=project,
        region=region,
        model_resource_name=model.output,
        endpoint_display_name=ENDPOINT_DISPLAY_NAME,
        deployed_model_display_name=DEPLOYED_MODEL_DISPLAY_NAME,
        machine_type=ENDPOINT_MACHINE_TYPE,
        min_replica_count=ENDPOINT_MIN_REPLICA_COUNT,
        max_replica_count=ENDPOINT_MAX_REPLICA_COUNT,
        traffic_percentage=ENDPOINT_TRAFFIC_PERCENTAGE,
    )
    endpoint.after(model)

    # Drift should compare two separately built feature tables, such as
    # train-window features vs a later scoring/current feature table. Keep this
    # disabled until a current feature table exists.
    # drift = compute_feature_drift(
    #     project=project,
    #     location=bq_location,
    #     baseline_table=feature_table,
    #     current_table="PROJECT.DATASET.current_features",
    #     metrics_table=DRIFT_TABLE,
    #     output_uri=DRIFT_REPORT,
    # )
    # drift.after(training)


def main() -> None:
    compiler.Compiler().compile(
        pipeline_func=amex_pipeline,
        package_path="amex_pipeline.json",
    )


if __name__ == "__main__":
    main()
