from kfp import compiler, dsl

from gcp.config import (
    FEATURE_TABLE,
    MODEL_ARTIFACTS,
    PIPELINE_ROOT,
    PROJECT_ID,
    REGION,
    TRAINING_IMAGE,
)
from gcp.pipeline import run_vertex_training_job, run_vertex_tuning_job

SMOKE_TUNING_ARTIFACTS = f"{MODEL_ARTIFACTS.rstrip('/')}/smoke/tuning/"
SMOKE_MODEL_ARTIFACTS = f"{MODEL_ARTIFACTS.rstrip('/')}/smoke/"


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
