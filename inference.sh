#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: bash inference.sh RAW_TEST_ROOT OUTPUT_CSV" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${IMU_STAGE2_PYTHON:-python}"

exec "$PYTHON_BIN" \
  "$SCRIPT_DIR/scripts/infer_imu_stage2.py" \
  --raw-test-root "$1" \
  --output-csv "$2" \
  --bundle-root "$SCRIPT_DIR/inference_bundle"
