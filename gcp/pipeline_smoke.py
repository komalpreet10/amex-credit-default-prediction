from kfp import compiler, dsl

from gcp.config import (
    FEATURE_TABLE,
    MODEL_ARTIFACTS,
    PIPELINE_ROOT,
    PROJECT_ID,
    REGION,
    TRAINING_IMAGE,
)
from gcp.pipeline import PIP_ROOT_USER_OPTION, run_vertex_training_job

SMOKE_TUNING_ARTIFACTS = f"{MODEL_ARTIFACTS.rstrip('/')}/smoke/tuning/"
SMOKE_MODEL_ARTIFACTS = f"{MODEL_ARTIFACTS.rstrip('/')}/smoke/"


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["google-cloud-aiplatform", PIP_ROOT_USER_OPTION],
    use_venv=True,
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


@dsl.pipeline(
    name="amex-credit-default-smoke-test",
    pipeline_root=PIPELINE_ROOT,
)
def amex_smoke_pipeline(
    project: str = PROJECT_ID,
    region: str = REGION,
    feature_table: str = FEATURE_TABLE,
    training_image: str = TRAINING_IMAGE,
    max_rows: int = 2000,
) -> None:
    tuning = run_vertex_tuning_job(
        project=project,
        region=region,
        training_image=training_image,
        table=feature_table,
        output_dir=SMOKE_TUNING_ARTIFACTS,
        display_name="amex-lightgbm-optuna-smoke-test",
        metric="pr_auc",
        n_trials=1,
        n_splits=2,
        num_boost_round=25,
        early_stopping_rounds=5,
        max_rows=max_rows,
        balanced_smoke_sample=True,
        replica_count=1,
        machine_type="n2-standard-4",
    )

    training = run_vertex_training_job(
        project=project,
        region=region,
        training_image=training_image,
        table=feature_table,
        eval_table="",
        output_dir=SMOKE_MODEL_ARTIFACTS,
        params_uri=tuning.output,
        display_name="amex-lightgbm-training-smoke-test",
        shap_sample_size=1,
        shap_max_display=10,
        max_rows=max_rows,
        balanced_smoke_sample=True,
        selector_num_boost_round=10,
        final_num_boost_round=10,
        min_selected_features=25,
        max_selected_features=100,
        disable_shap=True,
        replica_count=1,
        machine_type="n2-standard-4",
    )
    training.after(tuning)


def main() -> None:
    compiler.Compiler().compile(
        pipeline_func=amex_smoke_pipeline,
        package_path="amex_pipeline_smoke.json",
    )


if __name__ == "__main__":
    main()
