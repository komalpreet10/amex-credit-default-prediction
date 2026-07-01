#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:-amex-credit-risk-ml}"
REGION="${GCP_REGION:-us-central1}"
INSTANCE_ID="${REDIS_INSTANCE_ID:-amex-feature-cache}"
TIER="${REDIS_TIER:-basic}"
SIZE_GB="${REDIS_SIZE_GB:-1}"
REDIS_VERSION="${REDIS_VERSION:-redis_7_0}"
NETWORK="${REDIS_NETWORK:-default}"

if gcloud redis instances describe "${INSTANCE_ID}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" >/dev/null 2>&1; then
  echo "Memorystore Redis instance already exists: ${INSTANCE_ID}"
else
  gcloud redis instances create "${INSTANCE_ID}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --tier="${TIER}" \
    --size="${SIZE_GB}" \
    --redis-version="${REDIS_VERSION}" \
    --network="${NETWORK}" \
    --connect-mode=PRIVATE_SERVICE_ACCESS
fi

gcloud redis instances describe "${INSTANCE_ID}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --format="table(name,host,port,tier,memorySizeGb,redisVersion,state)"
