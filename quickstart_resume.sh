#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
BATCH_SIZE="${BATCH_SIZE:-100}"
FREQ="${FREQ:-daily}"
AK_START_DATE="${AK_START_DATE:-20240101}"
AK_END_DATE="${AK_END_DATE:-20241231}"
MINUTE_PERIOD="${MINUTE_PERIOD:-1}"
ADJUST="${ADJUST:-qfq}"
RETRY_FAILED_MAX="${RETRY_FAILED_MAX:-0}"

echo "[quickstart-resume] installing dependencies"
"${PYTHON_BIN}" -m pip install -U pip
"${PYTHON_BIN}" -m pip install -r requirements.txt

echo "[quickstart-resume] resume full-market price crawling from checkpoint"
"${PYTHON_BIN}" -u scripts/build_price_full_market_akshare.py \
  --batch-size "${BATCH_SIZE}" \
  --freq "${FREQ}" \
  --start-date "${AK_START_DATE}" \
  --end-date "${AK_END_DATE}" \
  --minute-period "${MINUTE_PERIOD}" \
  --adjust "${ADJUST}" \
  --output-csv outputs/data/price.csv

if [[ "${RETRY_FAILED_MAX}" != "0" ]]; then
  echo "[quickstart-resume] retry failed announcement downloads, max=${RETRY_FAILED_MAX}"
  "${PYTHON_BIN}" -u scripts/crawl_cninfo_content_fetch.py \
    --input-csv raw/cninfo_announcements.csv \
    --download-dir raw/cninfo_files \
    --output-csv raw/cninfo_announcements_with_content.csv \
    --extract-html-text \
    --max-rows "${RETRY_FAILED_MAX}"
fi

echo "[quickstart-resume] rebuild text features"
"${PYTHON_BIN}" -u scripts/build_text_from_cn_sources.py \
  --input-csv raw/cninfo_announcements_with_content.csv \
  --output-csv outputs/data/text.csv

echo "[quickstart-resume] validate dataset contract"
"${PYTHON_BIN}" -u scripts/validate_dataset_contract.py \
  --price-csv outputs/data/price.csv \
  --text-csv outputs/data/text.csv

echo "[quickstart-resume] done"
