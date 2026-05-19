#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
START_DATE="${START_DATE:-2024-01-01}"
END_DATE="${END_DATE:-2024-12-31}"
AK_START_DATE="${AK_START_DATE:-20240101}"
AK_END_DATE="${AK_END_DATE:-20241231}"
BATCH_SIZE="${BATCH_SIZE:-100}"
FREQ="${FREQ:-daily}"
MINUTE_PERIOD="${MINUTE_PERIOD:-1}"
ADJUST="${ADJUST:-qfq}"

echo "[quickstart-full] installing dependencies"
"${PYTHON_BIN}" -m pip install -U pip
"${PYTHON_BIN}" -m pip install -r requirements.txt

echo "[quickstart-full] crawl cninfo announcements"
"${PYTHON_BIN}" -u scripts/crawl_cninfo_incremental.py \
  --start-date "${START_DATE}" \
  --end-date "${END_DATE}" \
  --output-csv raw/cninfo_announcements.csv

echo "[quickstart-full] fetch announcement attachments"
"${PYTHON_BIN}" -u scripts/crawl_cninfo_content_fetch.py \
  --input-csv raw/cninfo_announcements.csv \
  --download-dir raw/cninfo_files \
  --output-csv raw/cninfo_announcements_with_content.csv \
  --extract-html-text

echo "[quickstart-full] fetch full-market prices via akshare"
"${PYTHON_BIN}" -u scripts/build_price_full_market_akshare.py \
  --refresh-symbols \
  --batch-size "${BATCH_SIZE}" \
  --freq "${FREQ}" \
  --start-date "${AK_START_DATE}" \
  --end-date "${AK_END_DATE}" \
  --minute-period "${MINUTE_PERIOD}" \
  --adjust "${ADJUST}" \
  --output-csv outputs/data/price.csv

echo "[quickstart-full] build text features"
"${PYTHON_BIN}" -u scripts/build_text_from_cn_sources.py \
  --input-csv raw/cninfo_announcements_with_content.csv \
  --output-csv outputs/data/text.csv

echo "[quickstart-full] validate dataset contract"
"${PYTHON_BIN}" -u scripts/validate_dataset_contract.py \
  --price-csv outputs/data/price.csv \
  --text-csv outputs/data/text.csv

echo "[quickstart-full] done"
