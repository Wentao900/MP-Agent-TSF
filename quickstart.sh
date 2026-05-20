#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
START_DATE="${START_DATE:-2024-01-01}"
END_DATE="${END_DATE:-2024-12-31}"
SYMBOLS="${SYMBOLS:-000001.SZ,600036.SH,600519.SH}"
FREQ="${FREQ:-daily}"
AK_START_DATE="${AK_START_DATE:-20240101}"
AK_END_DATE="${AK_END_DATE:-20241231}"

echo "[quickstart] installing dependencies"
"${PYTHON_BIN}" -m pip install -U pip
"${PYTHON_BIN}" -m pip install -r requirements.txt

echo "[quickstart] running full data pipeline"
"${PYTHON_BIN}" -u scripts/run_full_data_pipeline.py \
  --start-date "${START_DATE}" \
  --end-date "${END_DATE}" \
  --price-source akshare \
  --symbols "${SYMBOLS}" \
  --freq "${FREQ}" \
  --ak-start-date "${AK_START_DATE}" \
  --ak-end-date "${AK_END_DATE}"

echo "[quickstart] done"
