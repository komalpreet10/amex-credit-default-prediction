from kfp import compiler, dsl

try:
    from gcp.config import (
        BQ_LOCATION,
        ENDPOINT_DISPLAY_NAME,
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
        TRAINING_IMAGE,
        TUNED_PARAMS_URI,
        TUNING_ARTIFACTS,
    )
except ModuleNotFoundError:
    from config import (
        BQ_LOCATION,
        ENDPOINT_DISPLAY_NAME,
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
        TRAINING_IMAGE,
        TUNED_PARAMS_URI,
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
    packages_to_install=["google-cloud-aiplatform"],
)
def run_vertex_tuning_job(
    project: str,
    region: str,
    training_image: str,
    table: str,
    output_dir: str,
) -> str:
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=region, staging_bucket=output_dir)
    job = aiplatform.CustomContainerTrainingJob(
        display_name="amex-lightgbm-optuna-tuning",
        container_uri=training_image,
        command=["python", "gcp/vertex/tune_lightgbm_optuna.py"],
    )
    job.run(
        args=[
            "--n-trials",
            "15",
            "--n-splits",
            "5",
        ],
        replica_count=1,
        machine_type="n2-standard-16",
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
    output_dir: str,
    params_uri: str,
) -> str:
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=region, staging_bucket=output_dir)
    job = aiplatform.CustomContainerTrainingJob(
        display_name="amex-lightgbm-training",
        container_uri=training_image,
        command=["python", "gcp/vertex/train.py"],
    )
    model = job.run(
        args=[
            "--params-uri",
            params_uri,
            "--shap-sample-size",
            "3000",
            "--shap-max-display",
            "30",
        ],
        replica_count=1,
        machine_type="n2-standard-16",
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
) -> str:
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=region)
    model = aiplatform.Model(model_resource_name)
    endpoint = aiplatform.Endpoint.create(display_name=endpoint_display_name)
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name="amex-lightgbm",
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=1,
        traffic_percentage=100,
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
    model_artifacts: str = MODEL_ARTIFACTS,
    training_image: str = TRAINING_IMAGE,
    serving_image: str = SERVING_IMAGE,
) -> None:
    # Feature engineering has already been completed and loaded to BigQuery.
    # Leave these steps here for full reruns, but keep them disabled when starting
    # from the existing feature table.
    # preprocess = submit_dataproc_pyspark_batch(
    #     project=project,
    #     region=region,
    #     batch_id="amex-preprocess",
    #     main_python_file_uri=PREPROCESS_SCRIPT,
    #     py_file_uris=PY_FILES,
    #     runtime_properties=DATAPROC_RUNTIME_PROPERTIES,
    #     args=[
    #         "--input",
    #         raw_data,
    #         "--output",
    #         preprocessed_output,
    #         "--overwrite",
    #     ],
    # )

    # build_features = submit_dataproc_pyspark_batch(
    #     project=project,
    #     region=region,
    #     batch_id="amex-build-features",
    #     main_python_file_uri=FEATURE_SCRIPT,
    #     py_file_uris=PY_FILES,
    #     runtime_properties=DATAPROC_RUNTIME_PROPERTIES,
    #     args=[
    #         "--input",
    #         preprocessed_output,
    #         "--labels",
    #         raw_labels,
    #         "--output",
    #         feature_output,
    #         "--overwrite",
    #     ],
    # )
    # build_features.after(preprocess)

    # load_bq = load_features_to_bigquery(
    #     project=project,
    #     location=bq_location,
    #     source_uri=f"{feature_output}*.parquet",
    #     table=feature_table,
    # )
    # load_bq.after(build_features)

    training = run_vertex_training_job(
        project=project,
        region=region,
        training_image=training_image,
        table=feature_table,
        output_dir=model_artifacts,
        params_uri=TUNED_PARAMS_URI,
    )
    tuning = run_vertex_tuning_job(
        project=project,
        region=region,
        training_image=training_image,
        table=feature_table,
        output_dir=TUNING_ARTIFACTS,
    )
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
    )
    endpoint.after(model)

    # Drift should compare two separately built feature tables, such as
    # train-window features vs a later scoring/current feature table. Keep this
    # disabled until a current feature table exists.
    # drift = compute_feature_drift(
    #     project=project,
    #     location=bq_location,
    #     baseline_table=feature_table,
    #     current_table=f"{project}.amex_ml.drift_current_features",
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
