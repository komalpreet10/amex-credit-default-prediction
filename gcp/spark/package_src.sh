#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "${ROOT_DIR}/src"
zip -r "${ROOT_DIR}/gcp/spark/amex_default.zip" amex_default \
  -x '*/__pycache__/*' \
  -x '*/.ipynb_checkpoints/*'
