#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

config_value() {
  PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/src" python -c \
    "from gcp import config; print(getattr(config, '${1}'))"
}

PROJECT_ID="$(config_value PROJECT_ID)"
REGION="$(config_value REGION)"
INSTANCE_ID="$(config_value REDIS_INSTANCE_ID)"
TIER="$(config_value REDIS_TIER)"
SIZE_GB="$(config_value REDIS_SIZE_GB)"
REDIS_VERSION="$(config_value REDIS_VERSION)"
NETWORK="$(config_value REDIS_NETWORK)"
EVICTION_POLICY="$(config_value REDIS_EVICTION_POLICY)"

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
    --redis-config="maxmemory-policy=${EVICTION_POLICY}" \
    --connect-mode=PRIVATE_SERVICE_ACCESS \
    --transit-encryption-mode=SERVER_AUTHENTICATION
fi

gcloud redis instances describe "${INSTANCE_ID}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --format="table(name,host,port,tier,memorySizeGb,redisVersion,state,redisConfigs.maxmemory-policy,transitEncryptionMode)"