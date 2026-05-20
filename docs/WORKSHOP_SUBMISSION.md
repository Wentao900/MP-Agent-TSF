# NeurIPS / ICML FinML Workshop 投稿清单（IMTSA）

面向 **ML workshop**（非主会）的可执行路线。主实验：**50 股 × 2022–2024 日线 + 时间 OOD（test=2024）**。

---

## 1. 贡献一句话（投稿摘要用）

> 在 **稀疏事件文本 + 密集价量** 的序列决策中，提出 **K-step Reflector** 利用近期动作–收益历史修正策略，并在 **faithfulness / stability** 约束下分析解释–性能权衡；在 A 股 50 股、**2024 时间外推**测试上验证。

避免写：「beat 市场」「全 A 股普适预测」。

---

## 2. 数据（1 周）

```bash
python scripts/auto_fetch_all.py \
  --start-date 2022-01-01 \
  --end-date 2024-12-31 \
  --universe-file data/universe_paper_50.txt \
  --ak-start-date 20220101 \
  --ak-end-date 20241231 \
  --skip-content-fetch \
  2>&1 | tee logs/workshop_fetch.log
```

检查：

```bash
python scripts/validate_dataset_contract.py
python scripts/report_dataset_stats.py
```

论文 **Table 1（Data）** 用 `outputs/dataset_stats_workshop.json` 填数。

---

## 3. 实验矩阵（2 周）

### 3.1 主消融（RQ2 + 完整模型）

使用 **时间 OOD 划分**（`configs/*_workshop.yaml`：train&lt;2023, val=2023, test≥2024）：

```bash
for cfg in configs/train_baseline_workshop.yaml configs/train_memory_workshop.yaml configs/train_reflector_workshop.yaml; do
  for s in 0 1 2; do
    python scripts/train.py --config "$cfg" --seed-offset "$s"
    python scripts/backtest.py --config "$cfg" --seed-offset "$s"
  done
done
```

### 3.2 文本负对照（ workshop 必做）

```bash
python scripts/make_text_variants.py
# 将 config 中 text_csv 改为 text_null.csv / text_shuffled.csv 各跑 reflector
```

| 实验 | text_csv | 证明什么 |
|------|----------|----------|
| Full | `text.csv` | 完整模型 |
| No-text | `text_null.csv` | 增益不来自价量泄露 |
| Shuffled | `text_shuffled.csv` | 增益依赖时间对齐的文本 |

### 3.3 简单 baseline（正文至少 1 个）

Workshop 不强制 FinRL，但需 **非深度对照**：

- **Buy & Hold**（每只股票持有）
- 或 **价格动量规则**（ret_5 符号）

可在 `backtest` 外加脚本；表中与 IMTSA **同 test 段（2024）** 对比。

---

## 4. 论文图表（workshop 4–6 页典型）

| 图/表 | 内容 | 来源 |
|-------|------|------|
| Table 1 | 数据规模、coverage、split | `dataset_stats_workshop.json` |
| Table 2 | 主结果 OOD test：Return, Sharpe, MDD（mean±std, 3 seeds） | 各 `metrics.json` 聚合 |
| Table 3 | 消融：baseline / +memory / +reflector | workshop configs |
| Table 4 | 文本对照：full / null / shuffled | 手动汇总 |
| Fig 1 | 方法框图 | Reflector + Memory |
| Fig 2 | tradeoff：Sharpe vs faithfulness | `tradeoff_summary.csv` |
| Fig 3（可选） | regime 柱状图 | `metrics_by_regime.csv` |

统计：

```bash
python scripts/run_stats.py   # 若已对齐 workshop 输出目录
```

---

## 5. 写作模板（Limitations 必写）

- 规则/词典文本特征，非 PLM  
- 中国市场、50 只流动性筛选，非全市场  
- 简化成交（fee+slippage），非订单簿  
- 标签为短期收益三分类，非组合优化  

---

## 6. 开源（审稿加分）

- [ ] GitHub 公开（匿名审稿可用匿名仓库）  
- [ ] `README` 复现命令 + `data/universe_paper_50.txt`  
- [ ] 固定 seed、固定 `val_start` / `test_start`  
- [ ] 不上传巨潮 PDF；提供 **公告列表 CSV 构建脚本**  

---

## 7. 时间线（6 周）

| 周 | 任务 |
|----|------|
| W1 | 50×3 年数据 + `report_dataset_stats` |
| W2 | workshop 三模型 × 3 seeds |
| W3 | text null/shuffle + Buy&Hold |
| W4 | 图表 + `run_stats` |
| W5 | 4–8 页论文 + 附录 |
| W6 | 内审、提交、放 arXiv+代码 |

---

## 8. 常见拒稿点（workshop 也要避）

1. 只有 2 只股票 / 1 个月数据  
2. 没有 **时间 OOD**（随机划分）  
3. 没有 **text shuffle** 对照  
4. 只报 in-sample Sharpe  
5. 主张「预测 A 股」而非「机制 + 消融」  

---

## 9. 目标 venue（2025–2026 周期参考）

- NeurIPS Workshop: FinRL / AI for Science 等（每年 CFP 不同，关注 [https://neurips.cc](https://neurips.cc) workshops）  
- ICML Workshop: AFML、ML4F  
- ACM ICAIF（若错过 NeurIPS workshop）  

页数通常 **4–8 页 + 参考文献**，以当年 CFP 为准。
