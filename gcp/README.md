# GCP Pipeline

Project: `amex-credit-risk-ml`

Bucket:

```text
gs://amex-credit-risk-ml-data/
```

Current raw inputs:

```text
gs://amex-credit-risk-ml-data/raw/train_data.csv
gs://amex-credit-risk-ml-data/raw/train_labels.csv
```

## PySpark Jobs

The Spark path mirrors the notebooks:

```text
notebooks/01_eda_preprocessing.ipynb
  raw CSV + labels -> preprocessed statement-level Parquet

notebooks/02_feature_engineering.ipynb
  preprocessed Parquet + labels -> customer-level feature Parquet
```

The notebook used a 50% stratified customer sample to reduce local compute.
The GCP path removes that sampling step and processes the full raw training data
while keeping the other preprocessing steps the same.

Preprocessing output:

```text
gs://amex-credit-risk-ml-data/processed/v1/train_preprocessed/
```

Feature output:

```text
gs://amex-credit-risk-ml-data/processed/v1/train_features/
```

Package the local `amex_default` source package before submitting a batch:

```bash
./gcp/spark/package_src.sh
```

Run the full preprocessing batch:

```bash
gcloud dataproc batches submit pyspark gcp/spark/preprocess.py \
  --project=amex-credit-risk-ml \
  --region=us-central1 \
  --deps-bucket=gs://amex-credit-risk-ml-data \
  --py-files=gcp/spark/amex_default.zip \
  -- \
  --output=gs://amex-credit-risk-ml-data/processed/v1/train_preprocessed/ \
  --overwrite
```

Then run the full feature batch:

```bash
gcloud dataproc batches submit pyspark gcp/spark/build_features.py \
  --project=amex-credit-risk-ml \
  --region=us-central1 \
  --deps-bucket=gs://amex-credit-risk-ml-data \
  --py-files=gcp/spark/amex_default.zip \
  -- \
  --input=gs://amex-credit-risk-ml-data/processed/v1/train_preprocessed/ \
  --labels=gs://amex-credit-risk-ml-data/raw/train_labels.csv \
  --output=gs://amex-credit-risk-ml-data/processed/v1/train_features/ \
  --overwrite
```

## Affected-customer feature refresh

For statement-cycle inference, do not recompute all customers and do not send raw
statement columns through Pub/Sub. Append the new monthly statement rows to
BigQuery, record the affected `customer_ID`s, then run a targeted Spark refresh.

Input tables:

```text
raw statements:      amex-credit-risk-ml.amex_ml.raw_monthly_statements_amex
changed customers:  amex-credit-risk-ml.amex_ml.changed_customers_statement_cycle
serving features:   amex-credit-risk-ml.amex_ml.train_features
```

The refresh job loads `selected_feature_list.json`, infers the raw columns needed
for those selected engineered features, recomputes features only for affected
customers, merges the refreshed rows into BigQuery, and optionally updates Redis.

```bash
./gcp/spark/package_src.sh

gcloud dataproc batches submit pyspark gcp/spark/refresh_selected_features.py \
  --project=amex-credit-risk-ml \
  --region=us-central1 \
  --deps-bucket=gs://amex-credit-risk-ml-data \
  --py-files=gcp/spark/amex_default.zip \
  -- \
  --raw-table=amex-credit-risk-ml.amex_ml.raw_monthly_statements_amex \
  --changed-customers-table=amex-credit-risk-ml.amex_ml.changed_customers_statement_cycle \
  --feature-table=amex-credit-risk-ml.amex_ml.train_features \
  --staging-table=amex-credit-risk-ml.amex_ml.train_features_refresh_staging \
  --selected-features-uri=gs://amex-credit-risk-ml-data/models/lightgbm/selected_feature_list.json \
  --statement-cycle=2026-07 \
  --redis-host=<memorystore-host>
```

Use `--skip-redis` when you only want to update BigQuery.

The Dataproc runtime must have the Spark BigQuery connector available. The
driver also needs `google-cloud-bigquery` for the `MERGE`; workers need `redis`
only when Redis refresh is enabled.

## Vertex training artifacts

The Vertex training job saves model artifacts, selected feature lists, Optuna CV
metrics, final-model feature importance, and SHAP explainability plots to:

```text
gs://amex-credit-risk-ml-data/models/lightgbm/
```

SHAP is generated on the full training feature table by default. To reduce
runtime for a smoke test, pass a bounded sample size:

```text
--shap-sample-size=3000
--shap-max-display=30
```

Use `--disable-shap` for the fastest smoke test.

## Date-windowed drift

Until external current data is added, use date windows from the available
statement history for a first drift check. The local processed data spans:

```text
2017-03-01 through 2018-03-31
```

Use the earlier history as the baseline population and the final month as the
current population:

```text
baseline: 2017-03-01 through 2018-02-28
current:  2018-03-01 through 2018-03-31
```

Build baseline-window features:

```bash
gcloud dataproc batches submit pyspark gcp/spark/build_features.py \
  --project=amex-credit-risk-ml \
  --region=us-central1 \
  --deps-bucket=gs://amex-credit-risk-ml-data \
  --py-files=gcp/spark/amex_default.zip \
  -- \
  --input=gs://amex-credit-risk-ml-data/processed/v1/train_preprocessed/ \
  --labels=gs://amex-credit-risk-ml-data/raw/train_labels.csv \
  --output=gs://amex-credit-risk-ml-data/processed/drift/baseline_features/ \
  --start-date=2017-03-01 \
  --end-date=2018-02-28 \
  --overwrite
```

Build current-window features:

```bash
gcloud dataproc batches submit pyspark gcp/spark/build_features.py \
  --project=amex-credit-risk-ml \
  --region=us-central1 \
  --deps-bucket=gs://amex-credit-risk-ml-data \
  --py-files=gcp/spark/amex_default.zip \
  -- \
  --input=gs://amex-credit-risk-ml-data/processed/v1/train_preprocessed/ \
  --labels=gs://amex-credit-risk-ml-data/raw/train_labels.csv \
  --output=gs://amex-credit-risk-ml-data/processed/drift/current_features/ \
  --start-date=2018-03-01 \
  --end-date=2018-03-31 \
  --overwrite
```

Load both outputs into separate BigQuery tables:

```bash
python gcp/bigquery/load_features.py \
  --project=amex-credit-risk-ml \
  --location=US \
  --dataset=amex_ml \
  --table=drift_baseline_features \
  --source-uri='gs://amex-credit-risk-ml-data/processed/drift/baseline_features/*.parquet'

python gcp/bigquery/load_features.py \
  --project=amex-credit-risk-ml \
  --location=US \
  --dataset=amex_ml \
  --table=drift_current_features \
  --source-uri='gs://amex-credit-risk-ml-data/processed/drift/current_features/*.parquet'
```

Then run `gcp/monitoring/drift_psi.py` with:

```text
baseline table: amex-credit-risk-ml.amex_ml.drift_baseline_features
current table:  amex-credit-risk-ml.amex_ml.drift_current_features
```

```bash
python gcp/monitoring/drift_psi.py \
  --project=amex-credit-risk-ml \
  --bq-location=US \
  --baseline-table=amex-credit-risk-ml.amex_ml.drift_baseline_features \
  --current-table=amex-credit-risk-ml.amex_ml.drift_current_features \
  --metrics-table=amex-credit-risk-ml.amex_ml.drift_metrics \
  --output-uri=gs://amex-credit-risk-ml-data/monitoring/train_window_drift_report.csv \
  --baseline-start-date=2017-03-01 \
  --baseline-end-date=2018-02-28 \
  --current-start-date=2018-03-01 \
  --current-end-date=2018-03-31
```
