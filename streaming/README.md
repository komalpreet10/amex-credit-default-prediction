# Monthly statement streaming

This path handles new monthly statement events from Pub/Sub.

Flow:

```text
Pub/Sub monthly statement JSON
  -> Cloud Function streaming/monthly_statement_handler.py
  -> append statement to BigQuery raw_monthly_statements_amex
  -> append customer_ID to changed_customers_statement_cycle
  -> update Redis rolling statement history
  -> recompute only selected LightGBM features for that customer
  -> store the selected feature vector in Redis
```

The Cloud Function stores recent statements in a Redis list per customer:

```text
amex:statements:{customer_ID}
```

The list is trimmed to the latest 13 statements after each update. The computed
selected feature vector is stored at:

```text
amex:features:{customer_ID}
```

Online scoring reads Redis first, so it can use the fresh selected features
immediately. BigQuery remains the durable raw statement store.

For Memorystore Redis, create a Serverless VPC Access connector and deploy the
streaming and inference functions with that connector:

```bash
python deployment/setup_vpc_connector.py
python deployment/deploy_streaming_function.py \
  --redis-host=<memorystore-private-ip> \
  --vpc-connector=amex-vpc-connector
python deployment/deploy_inference_function.py \
  --redis-host=<memorystore-private-ip> \
  --vpc-connector=amex-vpc-connector
```

Publish messages to the statement topic as JSON objects with at least:

```json
{
  "customer_ID": "customer_123",
  "S_2": "2026-07-01",
  "statement_cycle": "2026-07"
}
```

Include the raw AMEX statement columns needed by the selected model features.
The refresh job loads `selected_feature_list.json`, derives the required raw
columns, recomputes features only for changed customers, and updates only those
selected feature columns in the serving feature table.
