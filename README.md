# IMTSA Research Scaffold（Phase 3 Enhanced）

本仓库是一个用于论文实验的可复现研究脚手架，覆盖从数据准备到训练、回测、消融与统计检验的完整闭环：

`data -> train -> backtest -> ablation -> stats`

当前版本已完成 **Phase 3 论文对齐增强**，重点补齐 RQ2 / RQ3 / RQ4：

- **RQ2**：轨迹级 Memory + K-step Reflector
- **RQ3**：解释-性能联合产物（step级、summary级、tradeoff级）
- **RQ4**：bull/bear/sideways 分状态收益风险与解释稳定性报表

---

## 1. 环境与安装

### 1.1 Python 版本
- 建议 `Python >= 3.10`（项目声明兼容到 3.12）

### 1.2 安装依赖
```bash
python -m pip install -U pip
python -m pip install -e .
```

---

## 2. 目录结构（核心）

```text
configs/
  base.yaml
  train_baseline.yaml
  train_memory.yaml
  train_reflector.yaml

scripts/
  prepare_data.py
  train.py
  backtest.py
  run_ablation.py
  run_stats.py

src/imtsa/
  data/
  models/
  train/
  backtest/
  explain/
  stats/

tests/
outputs/
```

---

## 3. 快速开始（最小闭环）

### Step 1) 生成示例数据
```bash
python scripts/prepare_data.py
```
将生成：
- `outputs/data/price.csv`
- `outputs/data/text.csv`

### Step 2) 单实验训练 + 回测（以 baseline 为例）
```bash
python scripts/train.py --config configs/train_baseline.yaml
python scripts/backtest.py --config configs/train_baseline.yaml
```

### Step 3) 全量消融（多seed）
```bash
python scripts/run_ablation.py
```

### Step 4) 统计检验与论文表
```bash
python scripts/run_stats.py
```

---

## 4. 配置说明（Phase 3 新增重点）

位置：`configs/base.yaml`

### 4.1 训练序列与损失权重
- `train.sequence_len`
  - 训练时轨迹展开步数。
  - `=1` 时保持旧版单步路径行为，确保向后兼容与可比性。
- `train.lambda_reflect_corr`
  - Reflector 校正损失权重。
- `train.lambda_exp_faith`
  - 解释 faithful 约束损失权重。
- `train.lambda_exp_stability`
  - 解释稳定性损失权重。
- `train.lambda_exp`
  - 解释主损失权重（当前实现下与 faithful/stability 联动）。

### 4.2 解释扰动参数
- `train.exp_perturb_ratio`
  - 对高贡献/低贡献时间步做扰动时的比例。

### 4.3 reward 归一化
- `train.reward_norm_scale`
  - Memory 中 prev_reward 归一化尺度（tanh 前缩放）。

---

## 5. 三组实验开关（可比性）

通过以下配置文件控制：
- `configs/train_baseline.yaml`
- `configs/train_memory.yaml`
- `configs/train_reflector.yaml`

核心开关（`ablation`）：
- `use_memory`
- `use_reflector`
- `use_explain_loss`

这样可保证 baseline / memory / memory_reflector 三组可直接比较。

---

## 6. Phase 3 关键产物清单

以下文件会在每个实验目录下（如 `outputs/baseline/`、`outputs/memory_reflector/`）生成。

### 6.1 原有基础合同（保留）
- `metrics.json`
- `trades.csv`
- `config_snapshot.json`

### 6.2 RQ3 解释-性能产物（新增）
- `explain_step.csv`
  - 逐步解释信息（alpha、动作概率、faithfulness、stability 等）
- `explain_summary.json`
  - 聚合解释指标
- `tradeoff_summary.csv`
  - 性能指标 + 解释指标联合行（论文表友好）

### 6.3 RQ4 分状态产物（新增）
- `metrics_by_regime.csv`
  - bull/bear/sideways 的收益风险交易指标
- `explain_by_regime.csv`
  - 分状态解释稳定性与faithfulness
- `risk_return_explain_state_table.csv`
  - 可直接贴论文的分状态联合总表

### 6.4 跨实验汇总与统计（新增）
- `outputs/ablation_summary_rq2_rq3.csv`
- `outputs/ablation_regime_summary.csv`
- `outputs/stats_report_rq2_rq3_rq4.json`
- `outputs/paper_table_main.csv`
- `outputs/paper_table_regime.csv`

---

## 7. 统计检验说明

`run_stats.py` 当前支持：
- paired t-test
- Wilcoxon signed-rank
- Cohen's d
- 多重比较校正：
  - Holm
  - BH-FDR

说明：当样本过少或差值退化时，Wilcoxon 可能出现 runtime warning，这是统计函数常见现象，不代表流程失败；请结合输出文件完整性判断。

---

## 8. 回归与测试

### 8.1 运行新增 Phase 3 合同测试
```bash
python -m pytest \
  tests/test_reflector_k_buffer.py \
  tests/test_trainer_sequence_unroll.py \
  tests/test_explain_outputs_contract.py \
  tests/test_regime_metrics_contract.py \
  tests/test_stats_pipeline_phase3.py \
  tests/test_smoke_e2e.py
```

### 8.2 覆盖点
- K-step reflector 接口
- 训练 sequence unroll
- explain 输出合同
- regime 输出合同
- 统计 pipeline 合同
- e2e smoke 合同

---

## 9. 复现实验建议

1. 固定 `experiment.seed`，并通过 `--seed-offset` 运行多seed。
2. 保持 `sequence_len=1` 先做 backward compatibility sanity check。
3. 再切换到更大 `sequence_len` 进行 RQ2 轨迹增强实验。
4. 统一使用 `run_ablation.py` + `run_stats.py` 生成论文总表，避免手工聚合口径不一致。

---

## 10. 常见问题

### Q1: 为什么 baseline 也会生成 explain/regime 文件？
为保证 pipeline 合同统一与 smoke 流程稳定，当前为全实验统一产出这类文件；差异由模型开关和指标数值体现。

### Q2: 如何确认 bull/bear/sideways 覆盖完整？
查看 `risk_return_explain_state_table.csv` 的 `market_state` 列，预期包含三类。样本不足时对应统计值会是 NaN，但行仍保留。

### Q3: 如何快速检查是否破坏旧流程？
用 `sequence_len=1` 运行 baseline，确认 `metrics.json` / `trades.csv` 仍正常产出，并通过 `tests/test_smoke_e2e.py`。

---

## 11. 论文复现实验配方（推荐）

本节给出可直接复现 RQ2/RQ3/RQ4 的命令与参数建议。

### 11.1 实验矩阵（baseline / memory / memory_reflector）

| 实验组 | 配置文件 | use_memory | use_reflector | use_explain_loss | sequence_len（建议） |
|---|---|---:|---:|---:|---:|
| baseline | `configs/train_baseline.yaml` | false | false | false | 1 |
| memory | `configs/train_memory.yaml` | true | false | false | 1 或 2 |
| memory_reflector | `configs/train_reflector.yaml` | true | true | true | 2 或 4 |

说明：
- 若目标是“严格向后兼容对照”，优先 `sequence_len=1`。
- 若目标是突出 RQ2 轨迹增强效果，建议对 `memory_reflector` 提高 `sequence_len`（如 2/4）并保持其他组同口径对照。

### 11.2 单组复现模板（以 memory_reflector 为例）

```bash
python scripts/train.py --config configs/train_reflector.yaml --seed-offset 0
python scripts/backtest.py --config configs/train_reflector.yaml --seed-offset 0
```

多 seed：
```bash
python scripts/train.py --config configs/train_reflector.yaml --seed-offset 1
python scripts/backtest.py --config configs/train_reflector.yaml --seed-offset 1
python scripts/train.py --config configs/train_reflector.yaml --seed-offset 2
python scripts/backtest.py --config configs/train_reflector.yaml --seed-offset 2
```

### 11.3 一键复现三组 + 统计

```bash
python scripts/run_ablation.py
python scripts/run_stats.py
```

### 11.3.1 方案 C 抓取（增量）

先抓 raw 公告数据（按日窗口断点续爬）：
```bash
python scripts/crawl_cninfo_incremental.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --output-csv raw/cninfo_announcements.csv
```

（可选）下载公告附件并提取文本（HTML/TXT）：
```bash
python scripts/crawl_cninfo_content_fetch.py \
  --input-csv raw/cninfo_announcements.csv \
  --download-dir raw/cninfo_files \
  --output-csv raw/cninfo_announcements_with_content.csv \
  --extract-html-text
```

再构建模型输入数据集（股票数据可二选一）：
```bash
# 方案A：Tushare 原始csv -> 价格特征
python scripts/build_price_from_tushare.py --input-csv raw/price_tushare.csv --output-csv outputs/data/price.csv --freq daily

# 方案B：AkShare 直接抓取 -> 价格特征（免Tushare积分）
python scripts/build_price_from_akshare.py --symbols 000001.SZ,600036.SH --freq daily --start-date 20240101 --end-date 20241231 --output-csv outputs/data/price.csv

# 文本特征 + 合同校验
python scripts/build_text_from_cn_sources.py --input-csv raw/cninfo_announcements_with_content.csv --output-csv outputs/data/text.csv
python scripts/validate_dataset_contract.py --price-csv outputs/data/price.csv --text-csv outputs/data/text.csv
```

### 11.4 关键超参数建议（可写入 `configs/base.yaml`）

- `train.sequence_len`: `1`（兼容） / `2~4`（轨迹增强）
- `train.lambda_reflect_corr`: `0.05 ~ 0.2`
- `train.lambda_exp_faith`: `0.05 ~ 0.2`
- `train.lambda_exp_stability`: `0.05 ~ 0.2`
- `train.exp_perturb_ratio`: `0.1 ~ 0.2`
- `train.reward_norm_scale`: `50 ~ 200`

建议做小网格：先固定除一个参数外其他值，观察 `tradeoff_summary.csv` 与 `risk_return_explain_state_table.csv` 的联动。

### 11.5 论文 RQ 与产物字段映射

#### RQ2（Memory / Reflector 轨迹增强）
- 主要看：
  - `outputs/ablation_summary_rq2_rq3.csv` 中性能字段（`total_return`, `sharpe`, `max_drawdown`, `win_rate`）
  - `outputs/stats_report_rq2_rq3_rq4.json` 的 `rq2_performance`

#### RQ3（解释-性能联合）
- 主要看：
  - 每实验目录 `tradeoff_summary.csv`
  - 每实验目录 `explain_summary.json`
  - 汇总 `outputs/ablation_summary_rq2_rq3.csv`
  - 统计 `stats_report_rq2_rq3_rq4.json` 的 `rq3_explain_tradeoff`

#### RQ4（分状态泛化）
- 主要看：
  - 每实验目录 `metrics_by_regime.csv`
  - 每实验目录 `explain_by_regime.csv`
  - 每实验目录 `risk_return_explain_state_table.csv`
  - 汇总 `outputs/ablation_regime_summary.csv`
  - 统计 `stats_report_rq2_rq3_rq4.json` 的 `regime_tests`

### 11.6 投稿表格直接来源建议

- 主表（跨指标族显著性）：`outputs/paper_table_main.csv`
- 分状态表（bull/bear/sideways）：`outputs/paper_table_regime.csv`

---

## 12. 许可证与用途

本仓库用于研究与论文实验验证。请在使用真实金融数据或部署到生产场景前，完成额外风险评估与合规审查。
