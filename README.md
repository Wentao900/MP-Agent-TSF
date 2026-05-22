# IMTSA Research Scaffold

可复现的多模态时序决策研究脚手架，覆盖 **Parquet 数据集 → 训练 → 回测（含 Holdout）→ 消融 → 统计检验**。

论文对齐能力（Phase 3）：

| 研究问题 | 能力 |
|----------|------|
| **RQ2** | 轨迹级 Memory + K-step Reflector（`sequence_len` 展开） |
| **RQ3** | 解释–性能联合指标（faithfulness / stability / tradeoff） |
| **RQ4** | `market_regime` 分状态报表 + **Holdout 标的 OOS** |

详细字段与防泄漏规则见 [`docs/EXPERIMENT_DATA_GUIDE.md`](docs/EXPERIMENT_DATA_GUIDE.md)。

---

## 1. 数据集形式（核心）

训练与回测**只读 Parquet**，不直接读 `price.csv` / `text.csv`。中间 CSV 仅用于 `build_datasets.py` 编译。

### 1.1 应该使用的文件

| 用途 | 路径 | 说明 |
|------|------|------|
| 主实验特征 | `data/processed/aligned_daily_multimodal.parquet` | 价量 + 宏观 + 文本 embedding + `split` + `market_regime` |
| 主实验标签 | `data/processed/labels_trading.parquet` | `action` / `r_net` 等，按 `(Date, ticker)` 与 aligned merge |
| RQ4 Holdout | `data/processed/holdout_aligned_daily.parquet` | Holdout **标的**不得出现在主 parquet |
| Holdout 标签 | `data/processed/holdout_labels_trading.parquet` | 可选，有则 merge |
| 切分元数据 | `data/metadata/splits.json` | 日期范围与 `main_tickers` / `holdout_tickers` |

以上路径均在 `configs/base.yaml` 的 `data.*` 中配置；`data.source` 固定为 `parquet`。

### 1.2 表结构摘要

- **主键**：`(Date, ticker)`，**日频**一行
- **划分**：列 `split` ∈ `train` / `val` / `test`（禁止 `random_split`）
- **数值输入**（20 维）：`ret_1d`, `ret_5d`, `momentum_*`, `volatility_*`, 宏观列 `FEDFUNDS`…`VIXCLS` 等 → 见 `src/imtsa/data/experiment_config.py`
- **文本输入**（32 维）：`text_emb_0` … `text_emb_31`（决策时刻单行，窗口内广播）
- **标签**：`action`（0=Hold, 1=Buy, 2=Sell）、`r_net`（成本后收益，回测主指标）
- **禁止作特征**：`future_ret_1d/5d/20d`（前视偏差）

### 1.3 默认时间切分（`build_datasets.py`）

| Split | 条件（默认） |
|-------|----------------|
| train | `Date < 2023-01-01` |
| val | `2023-01-01 ≤ Date < 2024-01-01` |
| test | `Date ≥ 2024-01-01` |

可通过 `--val-start` / `--test-start` 调整。Workshop 配置见 `configs/*_workshop.yaml`（`seq_len: 32`）。

### 1.4 序列与标准化

- 回看窗口 **`seq_len: 60`**（交易日），按 **ticker** 分组滑窗，不跨股票
- **StandardScaler** 仅在 `train` 上 fit，保存为 `outputs/<exp>/scaler.joblib`，val/test/holdout 复用

---

## 2. 环境与安装

**Python >= 3.10**

```bash
python -m pip install -U pip
pip install -r requirements.txt
pip install -e .
```

**GPU（CUDA 12.8，可选）**：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

未安装 GPU 版时，训练会自动回退 CPU（`torch.cuda.is_available()`）。

可选依赖：`pip install -e ".[data]"`（AkShare / 巨潮抓取脚本）、`pip install -e ".[dev]"`（pytest）。

---

## 3. 构建数据集（必做）

### 3.1 路径 A：本地合成数据（ smoke / 开发）

```bash
python scripts/prepare_data.py          # outputs/data/price.csv + text.csv
python scripts/build_datasets.py      # 编译为 data/processed/*.parquet
python scripts/validate_dataset_contract.py
```

多标的 Holdout 示例：

```bash
python scripts/build_datasets.py --holdout-fraction 0.2
```

### 3.2 路径 B：真实 A 股数据（Workshop / 生产）

1. 抓取中间 CSV（**不进入训练**）：见 [`docs/DATA_PLAN.md`](docs/DATA_PLAN.md)、[`docs/WORKSHOP_SUBMISSION.md`](docs/WORKSHOP_SUBMISSION.md)  
   - 一键：`./scripts/start_workshop_fetch.sh` 或 `scripts/auto_fetch_all.py`
2. 编译 Parquet：

```bash
python scripts/build_price_from_akshare.py \
  --symbols-file data/universe_paper_50.txt \
  --start-date 20220101 --end-date 20241231 \
  --output-csv outputs/data/price.csv

python scripts/build_text_from_cn_sources.py \
  --input-csv raw/cninfo_announcements_with_content.csv \
  --output-csv outputs/data/text.csv

python scripts/build_datasets.py \
  --price-csv outputs/data/price.csv \
  --text-csv outputs/data/text.csv \
  --holdout-tickers-file path/to/holdout_tickers.txt   # 可选
```

`data/processed/` 与 `outputs/` 已在 `.gitignore` 中，需本地生成。

---

## 4. 训练与回测

### 4.1 单实验

```bash
python scripts/train.py --config configs/train_baseline.yaml
python scripts/backtest.py --config configs/train_baseline.yaml
```

回测自动完成：

- **Test**：`outputs/<exp>/metrics.json`、`trades.csv`、regime 报表等
- **Holdout（RQ4）**：`outputs/<exp>/holdout/`（存在 holdout parquet 时）
- **汇总**：`outputs/<exp>/eval_summary.json`（test + holdout）

跳过 Holdout：`python scripts/backtest.py --config ... --skip-holdout`

### 4.2 消融矩阵（3 模型 × 3 seed）

| 配置 | 模型 | `sequence_len` | 轨迹展开 |
|------|------|----------------|----------|
| `train_baseline.yaml` | baseline | 1 | 否 |
| `train_memory.yaml` | +Memory | 2 | 是 |
| `train_reflector.yaml` | +Reflector + Explain | 4 | 是 |

```bash
python scripts/run_ablation.py
python scripts/run_stats.py
```

Workshop 时间 OOD 版本：`*_workshop.yaml`（`seq_len: 32`）。

---

## 5. 目录结构

```text
configs/                 # base.yaml + train_*.yaml
data/
  processed/             # *.parquet（gitignore，本地生成）
  metadata/splits.json
  universe_paper_50.txt  # A 股股票池示例
docs/
  EXPERIMENT_DATA_GUIDE.md
  DATA_PLAN.md
  WORKSHOP_SUBMISSION.md
scripts/
  build_datasets.py      # CSV → Parquet（唯一训练入口数据源）
  train.py / backtest.py
  validate_dataset_contract.py
  run_ablation.py / run_stats.py
src/imtsa/
  data/                  # loader, pipeline, experiment_config
  models/ train/ backtest/ explain/ stats/
tests/
outputs/                 # 实验产物（gitignore）
```

---

## 6. 实验产物

每个 `outputs/<experiment_name>/`：

| 文件 | 内容 |
|------|------|
| `config_snapshot.json` | 复现用配置 |
| `scaler.joblib` | train 上 fit 的标准化器 |
| `model.pt` | 权重 |
| `metrics.json` / `trades.csv` | Test 集回测 |
| `metrics_by_regime.csv` | bull / bear / sideways |
| `explain_step.csv` / `tradeoff_summary.csv` | RQ3 |
| `holdout/*` | RQ4 OOS（同结构） |
| `eval_summary.json` | test + holdout 指标摘要 |

汇总表：`outputs/ablation_summary*.csv`、`outputs/paper_table_*.csv`。

---

## 7. 配置要点（`configs/base.yaml`）

```yaml
data:
  source: parquet
  seq_len: 60
  split_mode: column      # 使用 parquet 内 split 列
  fit_scaler_on_train: true
  eval_holdout: true

model:
  reflector_k: 15         # 与指南 REFLECT_EVERY_K 对齐

backtest:
  use_label_rewards: true # 以 r_net 为主
```

`ablation` 开关：`use_memory` / `use_reflector` / `use_explain_loss`（见各 `train_*.yaml`）。

---

## 8. 测试

```bash
python -m pytest tests/test_experiment_loader.py \
  tests/test_reflector_k_buffer.py \
  tests/test_trainer_sequence_unroll.py \
  tests/test_explain_outputs_contract.py \
  tests/test_regime_metrics_contract.py \
  tests/test_stats_pipeline_phase3.py
```

端到端 smoke 需先完成训练并存在 `outputs/baseline/` 产物。

---

## 9. 常见问题

**Q: 报错找不到 parquet？**  
先运行 `python scripts/build_datasets.py`，再 `validate_dataset_contract.py`。

**Q: 还能用 CSV 直接训练吗？**  
不能。`load_main_panel()` 强制 `data.source: parquet`。

**Q: Holdout 没有生成？**  
单标的或 `--holdout-fraction 0` 时会跳过；至少 2 个 ticker 且 `holdout-fraction > 0`，或使用 `--holdout-tickers-file`。

**Q: baseline 为何也有 explain / regime 文件？**  
统一产出合同；差异体现在指标数值与 `ablation` 开关。

**Q: 抓取脚本与实验代码关系？**  
抓取只产出 `outputs/data/*.csv`；**必须**经 `build_datasets.py` 转为 Parquet 后再训练。

---

## 10. 延伸阅读

- 字段级契约与代码映射：[`docs/EXPERIMENT_DATA_GUIDE.md`](docs/EXPERIMENT_DATA_GUIDE.md)
- A 股 pilot 规模与抓取：[`docs/DATA_PLAN.md`](docs/DATA_PLAN.md)
- Workshop 投稿清单：[`docs/WORKSHOP_SUBMISSION.md`](docs/WORKSHOP_SUBMISSION.md)

---

## 11. 许可证

本仓库用于研究与论文实验验证。使用真实金融数据前请完成合规与风险评估。
