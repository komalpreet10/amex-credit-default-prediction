# American Express Credit Default Prediction

End-to-end credit default prediction pipeline on GCP using monthly AmEx statement data — from distributed feature engineering through real-time inference.

## Highlights
- Orchestrated distributed feature engineering, hyperparameter tuning, model training, and real-time inference across monthly statement cycles.
- Engineered 3,400+ behavioral, temporal, and statistical aggregations (22+ per numeric feature) across delinquency, spend, payment, balance, and risk variables.
- Tuned LightGBM via Optuna with 5-fold stratified CV, achieving **0.959 ROC-AUC / 0.894 PR-AUC** on imbalanced data.
- Added SHAP explainability and PSI drift monitoring; tracked runs with MLflow.
- Built a three-tier real-time inference path: **Redis** (online cache) → **BigQuery** (fallback lookup) → **Vertex AI Endpoint** (serving), with streaming feature refresh via Pub/Sub + Dataflow on each statement cycle close.

## Results

| Model | Rows | Features | ROC-AUC | PR-AUC | F1 |
|---|---:|---:|---:|---:|---:|
| LightGBM | 229,456 | 3,418 | 0.9593 | 0.8938 | 0.8081 |
| XGBoost | 229,456 | 3,418 | 0.9597 | 0.8948 | 0.8079 |

## Architecture
![AmEx Credit Default Prediction — GCP Pipeline Architecture](docs/amex_pipeline_architecture.svg)

```text
Training:   GCS → Dataproc Serverless (feature eng.) → BigQuery → Vertex AI Pipeline
            (Optuna + 5-fold CV → LightGBM) → Model Registry → Endpoint

Inference:  Cloud Function → Redis cache → BigQuery fallback → Vertex AI Endpoint

Streaming:  Pub/Sub (statement cycle close) → Dataflow → Redis feature refresh
```

## Stack
Python · PySpark · LightGBM · XGBoost · Optuna · SHAP · MLflow · BigQuery · Dataproc Serverless · Vertex AI (Pipelines, Endpoints) · Cloud Functions · Pub/Sub · Dataflow · Memorystore Redis

## Run

```bash
python -m gcp.pipeline                          # compile Vertex AI pipeline
bash gcp/redis/provision_memorystore.sh          # provision Redis
REDIS_HOST=<host> python deployment/refresh_redis.py   # refresh cache

SERVING_IMAGE_URI=<image> REDIS_HOST=<host> ALERT_EMAIL=<email> \
python deployment/run_deployment.py              # deploy online stack
```
