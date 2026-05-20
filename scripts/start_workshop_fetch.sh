#!/usr/bin/env bash
# Workshop 数据集一键抓取：50 股 × 2022-2024，标题文本（不下载 PDF）
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs raw/.checkpoints outputs/data

LOG="logs/workshop_fetch_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"

python3 scripts/auto_fetch_all.py \
  --start-date 2022-01-01 \
  --end-date 2024-12-31 \
  --universe-file data/universe_paper_50.txt \
  --ak-start-date 20220101 \
  --ak-end-date 20241231 \
  --skip-content-fetch \
  2>&1 | tee "$LOG"

echo ""
echo "Done. Next:"
echo "  python3 scripts/report_dataset_stats.py"
echo "  python3 scripts/validate_dataset_contract.py"
