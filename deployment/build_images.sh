#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:-amex-credit-risk-ml}"
REGION="${GCP_REGION:-us-central1}"
REPOSITORY="${ARTIFACT_REGISTRY_REPOSITORY:-amex-credit-default}"
TAG="${IMAGE_TAG:-latest}"

REGISTRY="${REGION}-docker.pkg.dev"
IMAGE_BASE="${REGISTRY}/${PROJECT_ID}/${REPOSITORY}"
TRAINING_IMAGE_URI="${IMAGE_BASE}/training:${TAG}"
SERVING_IMAGE_URI="${IMAGE_BASE}/serving:${TAG}"

echo "Configuring Docker auth for ${REGISTRY}"
gcloud auth configure-docker "${REGISTRY}" --quiet

echo "Building training image: ${TRAINING_IMAGE_URI}"
docker build \
  -f docker/Dockerfile.train \
  -t "${TRAINING_IMAGE_URI}" \
  .

echo "Building serving image: ${SERVING_IMAGE_URI}"
docker build \
  -f docker/Dockerfile.serve \
  -t "${SERVING_IMAGE_URI}" \
  .

echo "Pushing training image"
docker push "${TRAINING_IMAGE_URI}"

echo "Pushing serving image"
docker push "${SERVING_IMAGE_URI}"

cat <<EOF

Images pushed.

Use these environment variables before compiling/running the Vertex pipeline or deployment:

export TRAINING_IMAGE_URI="${TRAINING_IMAGE_URI}"
export SERVING_IMAGE_URI="${SERVING_IMAGE_URI}"
EOF
