# IMTSA 数据规模与抓取方案（论文 pilot）

## 1. 目标层级

| 层级 | 股票 | 时间 | 频率 | 训练序列(约) | 用途 |
|------|------|------|------|--------------|------|
| A 跑通 | 1 | 1 年 | 日 | ~200 | 仅验证 pipeline |
| B pilot（**推荐**） | **50** | **3 年** | **日** | **~2 万** | 论文主实验 / RQ2–4 |
| C 加强 | 50+ | 5 年 | 分钟 | 百万级 | 需更大算力与存储 |
| D 全市场 | 5000+ | 多年 | 日/分 | - | 仅当改横截面训练架构 |

**当前误区**：全市场 1 月 3.2 万条 PDF，但 2 只股票只有 **8** 条公告 → 对训练几乎无贡献。

## 2. 推荐方案 B（50 × 3 年日线）

### 规模估算

```bash
python scripts/estimate_data_scale.py --years 3 --download-pdf
```

| 指标 | 估算 |
|------|------|
| 公告条数 | ~1.1 万 |
| PDF 磁盘 | ~**4 GiB** |
| 价量行数 | ~3.6 万 |
| 训练序列 (60% split) | ~**2 万** |
| 文本事件 | 与公告同量级（对齐后仍远小于 K 线数，属正常） |

对比：全市场 1 年 PDF ≈ **100+ GiB**，且多数股票不在训练池。

### 一键抓取

```bash
python scripts/auto_fetch_all.py \
  --start-date 2022-01-01 \
  --end-date 2024-12-31 \
  --universe-file data/universe_paper_50.txt \
  --ak-start-date 20220101 \
  --ak-end-date 20241231 \
  --skip-content-fetch \
  2>&1 | tee logs/auto_fetch_paper50.log
```

- 论文 pilot 建议先加 `--skip-content-fetch`（只用标题情感），PDF 可后补。
- 需要 PDF 时去掉该参数，磁盘预留 **~5 GiB**。

### 分步命令

```bash
# 1) 按股票池抓公告列表（按月窗口 × 50 只）
python scripts/crawl_cninfo_incremental.py \
  --start-date 2022-01-01 --end-date 2024-12-31 \
  --universe-file data/universe_paper_50.txt \
  --window month \
  --output-csv raw/cninfo_announcements_universe.csv

# 2) 下载附件（可断点续跑）
python scripts/auto_fetch_all.py \
  --start-date 2022-01-01 --end-date 2024-12-31 \
  --universe-file data/universe_paper_50.txt \
  --skip-crawl --skip-price --skip-validate

# 3) 已有全市场 1 月数据时，可先过滤再决定是否重抓
python scripts/filter_cninfo_by_symbols.py \
  --input-csv raw/cninfo_announcements.csv \
  --output-csv raw/cninfo_announcements_universe_jan.csv
```

## 3. 样本是否“够预测”？

- Reflector 参数量 ~**7.4 万**；金融噪声下 **5k–20k** 训练序列是小型深度模型常见 pilot 下限。
- 50×3 年日线 → **~2 万** 训练序列 → **可达 pilot**，但结论需加：多 seed、成本、分 regime（RQ4）。
- 文本仍稀疏：大量 K 线共享最近一次公告特征 → 论文中应报告 **text coverage**（有公告的交易日占比）。

## 4. 不建议继续的方向

- ❌ 继续扒「全市场 × 1 月」剩余 PDF（训练池外股票占绝大多数）
- ❌ 在 2 只股票 + 1 月数据上声称预测有效
- ❌ 在未扩展 `price.csv` 股票数前，下载全年全市场公告

## 5. 训练前检查

```bash
python scripts/validate_dataset_contract.py \
  --price-csv outputs/data/price.csv \
  --text-csv outputs/data/text.csv
```

确认 `price` 与 `text` 的 `symbols` 与 `universe_paper_50.txt` 一致，且时间覆盖 2022–2024。
