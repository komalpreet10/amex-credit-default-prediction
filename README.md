# American Express Credit Default Prediction

End-to-end machine learning project for predicting customer credit default risk from monthly American Express statement data. The project covers exploratory analysis, PySpark customer-level feature engineering, BigQuery feature storage, Vertex AI Pipelines orchestration, Optuna-tuned LightGBM training, feature selection, SHAP explainability, and PSI drift analysis.

## Highlights

- Built a binary classification pipeline for credit default prediction using the American Express Default Prediction dataset.
- Engineered `3,418` customer-level features from raw monthly statement records using aggregation, lag, recent-window, first-value, and difference features.
- Compared LightGBM and XGBoost with cross-validation on `229,456` customer-level rows.
- Implemented a GCP-native MLOps path using Dataproc Serverless, BigQuery, Vertex AI Pipelines, Vertex AI Custom Training, and GCS.
- Added GCP-native model artifact versioning, feature selection, explainability, and drift monitoring components for reproducible credit risk modeling.

## Results

| Model | Rows | Features | ROC-AUC | PR-AUC | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 229,456 | 3,418 | 0.9593 | 0.8938 | 0.8104 | 0.8059 | 0.8081 |
| XGBoost | 229,456 | 3,418 | 0.9597 | 0.8948 | 0.8124 | 0.8035 | 0.8079 |

The final model is LightGBM trained on a selected feature subset. The current Vertex workflow uses Optuna 5-fold stratified cross-validation for tuning and evaluation, applies LightGBM cumulative gain feature selection, then trains the final model on the selected features.

## LightGBM Evaluation

![LightGBM ROC curve](docs/images/lightgbm_roc_curve.png)

![LightGBM precision recall curve](docs/images/lightgbm_pr_curve.png)

## Explainability

![LightGBM feature importance](docs/images/lightgbm_feature_importance.png)

![LightGBM SHAP summary](docs/images/lightgbm_shap_summary.png)

## Project Structure

```text
amex-credit-default/
├── app/                         # FastAPI application
│   ├── main.py
│   ├── model_loader.py
│   └── schemas.py
├── deployment/                  # GCP deployment and infrastructure scripts
├── inference/                   # Cloud Function online scoring entrypoint
├── gcp/                         # GCP pipeline, Spark jobs, Vertex training, monitoring
│   ├── pipeline.py
│   ├── bigquery/
│   ├── monitoring/
│   ├── redis/
│   ├── serving/
│   ├── spark/
│   └── vertex/
├── docker/                      # Vertex training container
├── docs/images/                 # README plots
├── notebooks/                   # EDA, training, comparison, SHAP, MLflow, API demo
├── streaming/                   # Dataflow streaming feature refresh job
├── src/amex_default/            # Reusable ML pipeline code
│   ├── data.py
│   ├── evaluate.py
│   ├── features.py
│   ├── interpret.py
│   ├── predict.py
│   ├── tracking.py
│   ├── train_lightgbm.py
│   └── train_xgboost.py
├── artifacts/                   # Local generated models, plots, reports
├── data/                        # Local raw, processed, and prediction data
├── mlruns/                      # Local MLflow experiment store
├── amex_pipeline.json           # Compiled Vertex AI Pipeline spec
├── requirements.txt
└── README.md
```

`data/`, `artifacts/`, and `mlruns/` are local generated outputs and are ignored by Git. The GCP pipeline uses GCS, BigQuery, and Vertex AI for cloud execution.

## GCP Architecture

```text
                           TRAINING / MLOPS

Raw AMEX CSVs in GCS
        |
        v
Dataproc Serverless PySpark preprocessing + feature engineering
        |
        v
Feature Parquet in GCS
        |
        v
BigQuery train_features
        |
        v
Vertex AI Pipeline
        |
        +--> Optuna + 5-fold stratified CV
        |       |
        |       v
        |   GCS tuning artifacts
        |
        +--> Final LightGBM training + feature selection
                |
                v
        GCS model artifacts + metrics + SHAP
                |
                v
        Vertex AI Model Registry
                |
                v
        Vertex AI Endpoint


                         ONLINE INFERENCE

New statement cycle / scoring request
        |
        v
Cloud Function: score(customer_ID)
        |
        +--> Tier 1: Memorystore Redis
        |       key = features:{customer_ID}
        |
        +--> Tier 2: BigQuery train_features lookup
        |       write-through to Redis on hit
        |
        +--> Tier 3: insufficient data response
        |
        v
Vertex AI Endpoint
        |
        v
Default probability + risk tier


                     STREAMING FEATURE REFRESH

Statement cycle close event
        |
        v
Pub/Sub: statement-cycle-close
        |
        v
Dataflow streaming job
        |
        v
Incremental Redis feature update
        |
        v
Pub/Sub DLQ on repeated failure
```

Current GCP project and storage:

```text
Project: amex-credit-risk-ml
Region: us-central1
Bucket: gs://amex-credit-risk-ml-data/
Feature table: amex-credit-risk-ml.amex_ml.train_features
Model artifacts: gs://amex-credit-risk-ml-data/models/lightgbm/
```

## Vertex AI Pipeline

The current compiled pipeline starts from existing BigQuery feature tables because feature engineering has already been completed and loaded. In `gcp/pipeline.py`, the Dataproc preprocessing, Dataproc feature build, and BigQuery load components are kept for full reruns, but are commented out for the current run.

Required BigQuery input before running the current pipeline:

```text
amex-credit-risk-ml.amex_ml.train_features
```

Active pipeline components:

| Order | Component | Script | Input | Output |
| ---: | --- | --- | --- | --- |
| 1 | `run-vertex-tuning-job` | `gcp/vertex/tune_lightgbm_optuna.py` | `train_features` | Optuna best params and trial history in GCS |
| 2 | `run-vertex-training-job` | `gcp/vertex/train.py` | `train_features` + tuned params | `model.txt`, `selected_feature_list.json`, metrics, SHAP, feature importance in GCS |
| 3 | `upload-vertex-model` | inline KFP component | trained GCS artifacts + serving image | Model registered in Vertex AI Model Registry |
| 4 | `deploy-model-to-endpoint` | inline KFP component | registered Vertex AI model | Deployed Vertex AI Endpoint |

Component details:

1. `run-vertex-tuning-job`
   - Runs Optuna tuning for LightGBM with 5-fold stratified cross-validation.
   - Uses full `train_features`; the production pipeline does not pass a row limit.
   - Writes `lightgbm_optuna_best_params.json` and `lightgbm_optuna_trials.csv` to GCS.

2. `run-vertex-training-job`
   - Loads tuned params from GCS through `--params-uri`.
   - Selects features using 95% cumulative LightGBM gain importance with 300/1000 min/max bounds.
   - Trains one final LightGBM model on the selected features.
   - Saves model artifacts, selected feature list, Optuna CV metrics, feature importance, and SHAP outputs to GCS.

3. `upload-vertex-model`
   - Registers `gs://amex-credit-risk-ml-data/models/lightgbm/` as a Vertex AI model.
   - Uses the custom serving image built from `docker/Dockerfile.serve`.

4. `deploy-model-to-endpoint`
   - Creates a Vertex AI Endpoint named `amex-credit-default-endpoint`.
   - Deploys the LightGBM serving container for online prediction.

Optional drift monitoring is implemented but remains disabled until a current/scoring feature table exists.

Required serving image before endpoint deployment:

```bash
docker build -f docker/Dockerfile.serve \
  -t us-central1-docker.pkg.dev/amex-credit-risk-ml/<repo>/amex-lightgbm-serving:<tag> .

docker push us-central1-docker.pkg.dev/amex-credit-risk-ml/<repo>/amex-lightgbm-serving:<tag>
```

Pass that image as `SERVING_IMAGE_URI` or the pipeline `serving_image` parameter.

## Online Feature Cache

Memorystore for Redis is used as the low-latency feature cache for online inference. It stores customer-level feature vectors or recently updated aggregate values so the scoring service does not recompute the full PySpark feature pipeline per request.

Default Redis settings are centralized in `gcp/config.py`:

```text
REDIS_INSTANCE_ID=amex-feature-cache
REDIS_TIER=basic
REDIS_SIZE_GB=1
REDIS_VERSION=redis_7_0
REDIS_NETWORK=default
```

Provision the Redis instance:

```bash
bash gcp/redis/provision_memorystore.sh
```

Override defaults if needed:

```bash
REDIS_INSTANCE_ID=amex-feature-cache \
REDIS_SIZE_GB=1 \
REDIS_NETWORK=default \
bash gcp/redis/provision_memorystore.sh
```

The script prints the Redis host and port after provisioning. The online scoring service should use those values through environment variables when feature lookup is added.

Compiled pipeline spec:

```text
amex_pipeline.json
```

To restore a full end-to-end rerun from raw data, uncomment the Dataproc and BigQuery task blocks in `gcp/pipeline.py`, recompile `amex_pipeline.json`, and rerun the pipeline.

## GCP LightGBM Tuning

The Vertex AI pipeline runs Optuna tuning as a cloud step before final LightGBM training. The tuning job reads `amex-credit-risk-ml.amex_ml.train_features` and writes its best parameters to:

```text
gs://amex-credit-risk-ml-data/models/lightgbm/tuning/lightgbm_optuna_best_params.json
```

The tuning job also stores CV evaluation artifacts in the same tuning prefix:

```text
cv_metrics.json
cv_classification_report.json
lightgbm_optuna_trials.csv
```

Final Vertex training loads the tuned parameter file through `--params-uri`, records the Optuna CV score, CV ROC-AUC, PR-AUC, precision, recall, and F1 in `metrics.json`, selects a compact feature set, and trains one final model on the selected features. Final model artifacts are written to:

```text
gs://amex-credit-risk-ml-data/models/lightgbm/
```

This prefix contains `model.txt`, `metrics.json`, feature lists, `feature_importance.csv`, SHAP plots, and the uploaded MLflow run directory under `mlruns/`.

## Drift Monitoring

Population Stability Index monitoring is implemented in `gcp/monitoring/drift_psi.py`. It compares a baseline BigQuery feature table against a current BigQuery feature table, writes drift metrics to BigQuery, and stores a CSV drift report in GCS.

Expected drift outputs:

```text
BigQuery metrics table: amex-credit-risk-ml.amex_ml.drift_metrics
GCS report: gs://amex-credit-risk-ml-data/monitoring/train_vs_scoring_drift_report.csv
```

## Local FastAPI Demo

Start the API from the project root:

```bash
PYTHONPATH=src uvicorn app.main:app --reload
```

Open the interactive API docs:

```text
http://127.0.0.1:8000/docs
```

Prediction endpoint:

```text
POST /predict
```

The endpoint accepts recent monthly statement rows for one `customer_ID`, runs lightweight request-level feature aggregation only for that customer, aligns the generated features to the trained model schema, and returns a default probability. It does not rerun the full Dataproc feature pipeline at inference time.

Example request:

```json
{
  "statements": [
    {
      "customer_ID": "customer-1",
      "S_2": "2018-03-01",
      "P_2": 0.72,
      "D_39": 0.01,
      "B_1": 0.05,
      "B_2": 0.81,
      "D_63": "CR",
      "D_64": "O"
    }
  ]
}
```

Example response:

```json
{
  "default_probability": 0.0008122114740618201,
  "risk_category": "low",
  "customer_id": "customer-1",
  "n_statements": 13,
  "n_engineered_features": 3418
}
```

## MLflow Tracking

MLflow is used for local experiment tracking only. The GCP workflow uses Vertex AI jobs, GCS artifacts, BigQuery tables, and Cloud Logging for cloud execution and observability.

Start the local MLflow UI:

```bash
MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri ./mlruns --port 5001
```

Open:

```text
http://127.0.0.1:5001
```

Tracked runs include:

- `lightgbm`
- `xgboost`
- `model_comparison`
- `final_lightgbm_model`

Tracked metrics include ROC-AUC, PR-AUC, precision, recall, F1, training time, inference time, and cross-validation metrics.

On Vertex AI, the final training job logs the same scalar metrics to Vertex AI Experiments and stores a self-contained MLflow run under `gs://amex-credit-risk-ml-data/models/lightgbm/mlruns/`.

Tracked artifacts include metrics reports, feature importance files, SHAP plots, model comparison plots, and final model files.

## Tech Stack

- Python
- Google Cloud Platform
- Dataproc Serverless
- BigQuery
- Vertex AI Pipelines
- Vertex AI Custom Training
- Google Cloud Storage
- Pandas, NumPy
- PySpark
- Scikit-learn
- LightGBM
- XGBoost
- Optuna
- SHAP
- MLflow
- FastAPI
