from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd

CONFIGS = [
    ("configs/train_baseline.yaml", "baseline"),
    ("configs/train_memory.yaml", "memory"),
    ("configs/train_reflector.yaml", "memory_reflector"),
]

SEED_OFFSETS = [0, 1, 2]


def run_cmd(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="python")
    args = parser.parse_args()

    rows = []
    rq_rows = []
    regime_rows = []
    for cfg, name in CONFIGS:
        for seed_offset in SEED_OFFSETS:
            run_cmd([args.python, "scripts/train.py", "--config", cfg, "--seed-offset", str(seed_offset)])
            run_cmd([args.python, "scripts/backtest.py", "--config", cfg, "--seed-offset", str(seed_offset)])

            exp_name = name if seed_offset == 0 else f"{name}_seed{seed_offset}"
            exp_dir = Path("outputs") / exp_name

            metrics = pd.read_json(exp_dir / "metrics.json", typ="series")
            tradeoff = pd.read_csv(exp_dir / "tradeoff_summary.csv").iloc[0].to_dict()
            explain_summary = pd.read_json(exp_dir / "explain_summary.json", typ="series").to_dict()
            regime = pd.read_csv(exp_dir / "risk_return_explain_state_table.csv")

            base_row = {"experiment": name, "seed": seed_offset, "eval_split": "test", **metrics.to_dict()}
            holdout_metrics_path = exp_dir / "holdout" / "metrics.json"
            if holdout_metrics_path.exists():
                holdout_metrics = pd.read_json(holdout_metrics_path, typ="series")
                rows.append({
                    "experiment": name,
                    "seed": seed_offset,
                    "eval_split": "holdout",
                    **holdout_metrics.to_dict(),
                })
            rows.append(base_row)
            rq_rows.append({
                "experiment": name,
                "seed": seed_offset,
                **tradeoff,
                **{f"explain_{k}": v for k, v in explain_summary.items()},
            })
            regime["experiment"] = name
            regime["seed"] = seed_offset
            regime_rows.append(regime)

    df = pd.DataFrame(rows)
    df.to_csv("outputs/ablation_summary.csv", index=False)

    summary = (
        df.groupby("experiment")
        .agg({
            "total_return": ["mean", "std"],
            "sharpe": ["mean", "std"],
            "max_drawdown": ["mean", "std"],
            "win_rate": ["mean", "std"],
            "turnover": ["mean", "std"],
        })
        .reset_index()
    )
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.values]
    summary.to_csv("outputs/ablation_mean_std.csv", index=False)

    pd.DataFrame(rq_rows).to_csv("outputs/ablation_summary_rq2_rq3.csv", index=False)

    regime_df = pd.concat(regime_rows, ignore_index=True)
    regime_df.to_csv("outputs/ablation_regime_summary.csv", index=False)


if __name__ == "__main__":
    main()
