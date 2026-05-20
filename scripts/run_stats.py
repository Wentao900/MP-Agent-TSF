from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from imtsa.stats.tests import paired_tests
from imtsa.utils import dump_json


def holm_correction(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.zeros(m)
    for i, idx in enumerate(order):
        adj[idx] = min(1.0, (m - i) * pvals[idx])
    for i in range(m - 2, -1, -1):
        adj[order[i]] = max(adj[order[i]], adj[order[i + 1]])
    return adj.tolist()


def bh_fdr(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.zeros(m)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        idx = order[i]
        val = pvals[idx] * m / (i + 1)
        prev = min(prev, val)
        adj[idx] = min(1.0, prev)
    return adj.tolist()


def apply_corrections(report: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = list(report.keys())
    t_p = [report[k]["t_pvalue"] for k in keys]
    w_p = [report[k]["wilcoxon_pvalue"] for k in keys]

    t_holm = holm_correction(t_p)
    t_bh = bh_fdr(t_p)
    w_holm = holm_correction(w_p)
    w_bh = bh_fdr(w_p)

    for i, k in enumerate(keys):
        report[k]["t_p_holm"] = float(t_holm[i])
        report[k]["t_p_bh_fdr"] = float(t_bh[i])
        report[k]["wilcoxon_p_holm"] = float(w_holm[i])
        report[k]["wilcoxon_p_bh_fdr"] = float(w_bh[i])
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="outputs/ablation_summary_rq2_rq3.csv")
    parser.add_argument("--regime-summary", default="outputs/ablation_regime_summary.csv")
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--candidate", default="memory_reflector")
    args = parser.parse_args()

    df = pd.read_csv(args.summary)
    bdf = df[df["experiment"] == args.baseline].sort_values("seed")
    cdf = df[df["experiment"] == args.candidate].sort_values("seed")

    metric_families = {
        "rq2_performance": ["total_return", "sharpe", "max_drawdown", "win_rate"],
        "rq3_explain_tradeoff": ["faithfulness_mean", "stability_mean", "alpha_price_mean", "alpha_text_mean"],
    }

    tests = {}
    for family, keys in metric_families.items():
        family_report = {}
        for k in keys:
            a = bdf[k].to_numpy(dtype=float)
            d = cdf[k].to_numpy(dtype=float)
            family_report[k] = paired_tests(d, a)
        tests[family] = apply_corrections(family_report)

    regime = pd.read_csv(args.regime_summary)
    rb = regime[regime["experiment"] == args.baseline]
    rc = regime[regime["experiment"] == args.candidate]
    regime_keys = ["total_pnl", "mean_pnl", "win_rate", "faithfulness", "stability"]

    regime_report = {}
    for state in ["bull", "bear", "sideways"]:
        sb = rb[rb["market_state"] == state].sort_values("seed")
        sc = rc[rc["market_state"] == state].sort_values("seed")
        per_state = {}
        for k in regime_keys:
            a = sb[k].to_numpy(dtype=float)
            d = sc[k].to_numpy(dtype=float)
            valid = np.isfinite(a) & np.isfinite(d)
            if valid.sum() < 2:
                per_state[k] = {
                    "t_stat": np.nan,
                    "t_pvalue": np.nan,
                    "wilcoxon_stat": np.nan,
                    "wilcoxon_pvalue": np.nan,
                    "cohens_d": np.nan,
                }
            else:
                per_state[k] = paired_tests(d[valid], a[valid])
        regime_report[state] = apply_corrections(per_state)

    out = {
        "baseline": args.baseline,
        "candidate": args.candidate,
        "n_seeds": int(min(len(bdf), len(cdf))),
        "tests": tests,
        "regime_tests": regime_report,
    }
    dump_json("outputs/stats_report_rq2_rq3_rq4.json", out)

    paper_main = []
    for family, vals in tests.items():
        for metric, rep in vals.items():
            paper_main.append({"family": family, "metric": metric, **rep})
    pd.DataFrame(paper_main).to_csv("outputs/paper_table_main.csv", index=False)

    paper_regime = []
    for state, vals in regime_report.items():
        for metric, rep in vals.items():
            paper_regime.append({"market_state": state, "metric": metric, **rep})
    pd.DataFrame(paper_regime).to_csv("outputs/paper_table_regime.csv", index=False)


if __name__ == "__main__":
    main()
