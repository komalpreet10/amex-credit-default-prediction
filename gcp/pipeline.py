from kfp import compiler, dsl

from gcp.config import (
    BATCH_PREDICTION_JOB_DISPLAY_NAME,
    BATCH_PREDICTIONS_TABLE,
    BQ_LOCATION,
    CUSTOMER_FEATURES_TABLE,
    DATAPROC_RUNTIME_PROPERTIES,
    DRIFT_REPORT,
    DRIFT_TABLE,
    FEATURE_TABLE,
    FEATURE_SCRIPT,
    FEATURE_STORE_NAME,
    FEATURE_VIEW_NAME,
    FEATURES,
    MODEL_ARTIFACTS,
    PIPELINE_ROOT,
    PREPROCESS_SCRIPT,
    PREPROCESSED,
    PROJECT_ID,
    PY_FILES,
    RAW_DATA,
    RAW_LABELS,
    REGION,
    SELECTED_FEATURES_URI,
    TEST_FEATURE_TABLE,
    TRAINING_IMAGE,
    TRAINING_JOB_DISPLAY_NAME,
    TRAINING_MACHINE_TYPE,
    TRAINING_REPLICA_COUNT,
    TRAINING_SHAP_MAX_DISPLAY,
    TRAINING_SHAP_SAMPLE_SIZE,
    TRAIN_FEATURE_TABLE,
    TUNED_PARAMS_URI,
)

PIP_ROOT_USER_OPTION = "--root-user-action=ignore"


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["google-cloud-dataproc", PIP_ROOT_USER_OPTION],
    use_venv=True,
)
def run_pyspark_batch(
    project: str,
    region: str,
    batch_id: str,
    script_uri: str,
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
    batch_name = f"{batch_id}-{uuid.uuid4().hex[:8]}"
    operation = client.create_batch(
        request={
            "parent": f"projects/{project}/locations/{region}",
            "batch_id": batch_name,
            "batch": Batch(
                pyspark_batch=PySparkBatch(
                    main_python_file_uri=script_uri,
                    python_file_uris=py_file_uris,
                    args=args,
                ),
                runtime_config=RuntimeConfig(properties=runtime_properties),
            ),
        }
    )
    response = operation.result(timeout=timeout_seconds)
    if response.state.name != "SUCCEEDED":
        raise RuntimeError(
            f"Dataproc batch {batch_name} ended as {response.state.name}: "
            f"{response.state_message}"
        )
    return response.name


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["google-cloud-bigquery", PIP_ROOT_USER_OPTION],
    use_venv=True,
)
def load_parquet_to_bigquery(
    project: str,
    location: str,
    source_uri: str,
    table: str,
) -> str:
    from google.cloud import bigquery

    client = bigquery.Client(project=project, location=location)
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
    packages_to_install=["google-cloud-aiplatform", PIP_ROOT_USER_OPTION],
    use_venv=True,
)
def sync_vertex_feature_store(
    project: str,
    region: str,
    feature_store_name: str,
    feature_view_name: str,
    source_table: str,
    entity_id_column: str,
) -> str:
    from google.api_core import exceptions
    from google.cloud.aiplatform_v1 import (
        CreateFeatureOnlineStoreRequest,
        CreateFeatureViewRequest,
        FeatureOnlineStore,
        FeatureOnlineStoreAdminServiceClient,
        FeatureView,
        GetFeatureOnlineStoreRequest,
        GetFeatureViewRequest,
        SyncFeatureViewRequest,
    )

    client = FeatureOnlineStoreAdminServiceClient(
        client_options={"api_endpoint": f"{region}-aiplatform.googleapis.com"}
    )
    parent = client.common_location_path(project, region)
    store_resource = client.feature_online_store_path(
        project=project,
        location=region,
        feature_online_store=feature_store_name,
    )

    try:
        client.get_feature_online_store(
            request=GetFeatureOnlineStoreRequest(name=store_resource)
        )
    except exceptions.NotFound:
        client.create_feature_online_store(
            request=CreateFeatureOnlineStoreRequest(
                parent=parent,
                feature_online_store_id=feature_store_name,
                feature_online_store=FeatureOnlineStore(
                    optimized=FeatureOnlineStore.Optimized()
                ),
            )
        ).result()

    view_resource = client.feature_view_path(
        project=project,
        location=region,
        feature_online_store=feature_store_name,
        feature_view=feature_view_name,
    )
    try:
        client.get_feature_view(request=GetFeatureViewRequest(name=view_resource))
    except exceptions.NotFound:
        client.create_feature_view(
            request=CreateFeatureViewRequest(
                parent=store_resource,
                feature_view_id=feature_view_name,
                feature_view=FeatureView(
                    big_query_source=FeatureView.BigQuerySource(
                        uri=f"bq://{source_table}",
                        entity_id_columns=[entity_id_column],
                    ),
                ),
                run_sync_immediately=True,
            )
        ).result()

    response = client.sync_feature_view(
        request=SyncFeatureViewRequest(feature_view=view_resource)
    )
    return response.feature_view_sync


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["google-cloud-bigquery", PIP_ROOT_USER_OPTION],
    use_venv=True,
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
    packages_to_install=["google-cloud-aiplatform", PIP_ROOT_USER_OPTION],
    use_venv=True,
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
    packages_to_install=[
        "google-cloud-bigquery",
        "google-cloud-storage",
        "lightgbm",
        PIP_ROOT_USER_OPTION,
    ],
    use_venv=True,
)
def run_batch_inference(
    project: str,
    location: str,
    model_artifacts_uri: str,
    selected_features_uri: str,
    source_table: str,
    output_table: str,
    job_display_name: str,
) -> str:
    from datetime import UTC, datetime
    import json
    import tempfile
    from pathlib import Path

    import lightgbm as lgb
    from google.cloud import bigquery, storage

    def download_text(uri: str) -> str:
        bucket_name, blob_name = uri.removeprefix("gs://").split("/", 1)
        return storage.Client().bucket(bucket_name).blob(blob_name).download_as_text()

    def download_file(uri: str, destination: Path) -> None:
        bucket_name, blob_name = uri.removeprefix("gs://").split("/", 1)
        storage.Client().bucket(bucket_name).blob(blob_name).download_to_filename(
            destination
        )

    feature_list = json.loads(download_text(selected_features_uri))
    if not isinstance(feature_list, list) or not feature_list:
        raise ValueError("Selected feature list must be a non-empty JSON list.")

    client = bigquery.Client(project=project, location=location)
    quoted_features = ", ".join(f"`{feature}`" for feature in feature_list)
    scoring_rows = [
        dict(row.items())
        for row in client.query(
            f"SELECT `customer_ID`, {quoted_features} FROM `{source_table}`"
        ).result()
    ]
    if not scoring_rows:
        raise ValueError(f"No rows available for batch inference in {source_table}.")

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.txt"
        download_file(f"{model_artifacts_uri.rstrip('/')}/model.txt", model_path)
        model = lgb.Booster(model_file=str(model_path))

    feature_matrix = [
        [
            row.get(feature) if row.get(feature) is not None else 0.0
            for feature in feature_list
        ]
        for row in scoring_rows
    ]
    probabilities = model.predict(feature_matrix)
    scored_at = datetime.now(UTC).isoformat()
    predictions = []
    for row, probability in zip(scoring_rows, probabilities, strict=True):
        probability = float(probability)
        if probability < 0.25:
            risk_category = "low"
        elif probability < 0.60:
            risk_category = "medium"
        else:
            risk_category = "high"
        predictions.append(
            {
                "customer_ID": row["customer_ID"],
                "default_probability": probability,
                "risk_category": risk_category,
                "scoring_job": job_display_name,
                "scored_at": scored_at,
            }
        )

    job_config = bigquery.LoadJobConfig(
        schema=[
            bigquery.SchemaField("customer_ID", "STRING"),
            bigquery.SchemaField("default_probability", "FLOAT"),
            bigquery.SchemaField("risk_category", "STRING"),
            bigquery.SchemaField("scoring_job", "STRING"),
            bigquery.SchemaField("scored_at", "TIMESTAMP"),
        ],
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    client.load_table_from_json(
        predictions,
        output_table,
        job_config=job_config,
        location=location,
    ).result()
    return output_table


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["google-cloud-aiplatform", PIP_ROOT_USER_OPTION],
    use_venv=True,
)
def run_psi_drift_job(
    project: str,
    region: str,
    training_image: str,
    bq_location: str,
    baseline_table: str,
    current_table: str,
    metrics_table: str,
    output_uri: str,
    display_name: str,
    max_rows: int,
) -> str:
    from google.cloud import aiplatform

    staging_bucket = output_uri.rsplit("/", 1)[0] + "/"
    aiplatform.init(project=project, location=region, staging_bucket=staging_bucket)
    job = aiplatform.CustomContainerTrainingJob(
        display_name=display_name,
        container_uri=training_image,
        command=["python", "gcp/monitoring/drift_psi.py"],
    )
    job_args = [
        "--project",
        project,
        "--bq-location",
        bq_location,
        "--baseline-table",
        baseline_table,
        "--current-table",
        current_table,
        "--metrics-table",
        metrics_table,
        "--output-uri",
        output_uri,
    ]
    if max_rows > 0:
        job_args.extend(["--max-rows", str(max_rows)])

    job.run(
        args=job_args,
        replica_count=1,
        machine_type="n2-standard-4",
        sync=True,
    )
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
    batch_source_table: str = CUSTOMER_FEATURES_TABLE,
    batch_prediction_table: str = BATCH_PREDICTIONS_TABLE,
    feature_store_name: str = FEATURE_STORE_NAME,
    feature_view_name: str = FEATURE_VIEW_NAME,
    drift_metrics_table: str = DRIFT_TABLE,
    drift_report_uri: str = DRIFT_REPORT,
    train_ratio: float = 0.8,
    model_artifacts: str = MODEL_ARTIFACTS,
    training_image: str = TRAINING_IMAGE,
) -> None:
    preprocess = run_pyspark_batch(
        project=project,
        region=region,
        batch_id="amex-preprocess",
        script_uri=PREPROCESS_SCRIPT,
        py_file_uris=PY_FILES,
        runtime_properties=DATAPROC_RUNTIME_PROPERTIES,
        args=[
            "--input",
            raw_data,
            "--output",
            preprocessed_output,
            "--overwrite",
        ],
    )

    build_features = run_pyspark_batch(
        project=project,
        region=region,
        batch_id="amex-build-features",
        script_uri=FEATURE_SCRIPT,
        py_file_uris=PY_FILES,
        runtime_properties=DATAPROC_RUNTIME_PROPERTIES,
        args=[
            "--input",
            preprocessed_output,
            "--labels",
            raw_labels,
            "--output",
            feature_output,
            "--overwrite",
        ],
    )
    build_features.after(preprocess)

    load_features = load_parquet_to_bigquery(
        project=project,
        location=bq_location,
        source_uri=f"{feature_output}*.parquet",
        table=feature_table,
    )
    load_features.after(build_features)

    split = split_bigquery_feature_table(
        project=project,
        location=bq_location,
        source_table=feature_table,
        train_table=train_feature_table,
        test_table=test_feature_table,
        train_ratio=train_ratio,
    )
    split.after(load_features)

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
    training.after(split)

    feature_store_sync = sync_vertex_feature_store(
        project=project,
        region=region,
        feature_store_name=feature_store_name,
        feature_view_name=feature_view_name,
        source_table=batch_source_table,
        entity_id_column="customer_ID",
    )
    feature_store_sync.after(training)

    batch_predictions = run_batch_inference(
        project=project,
        location=bq_location,
        model_artifacts_uri=model_artifacts,
        selected_features_uri=SELECTED_FEATURES_URI,
        source_table=batch_source_table,
        output_table=batch_prediction_table,
        job_display_name=BATCH_PREDICTION_JOB_DISPLAY_NAME,
    )
    batch_predictions.after(feature_store_sync)

    drift = run_psi_drift_job(
        project=project,
        region=region,
        training_image=training_image,
        bq_location=bq_location,
        baseline_table=feature_table,
        current_table=batch_source_table,
        metrics_table=drift_metrics_table,
        output_uri=drift_report_uri,
        display_name="amex-feature-psi-drift",
        max_rows=0,
    )
    drift.after(batch_predictions)


def main() -> None:
    compiler.Compiler().compile(
        pipeline_func=amex_pipeline,
        package_path="amex_pipeline.json",
    )


if __name__ == "__main__":
    main()
