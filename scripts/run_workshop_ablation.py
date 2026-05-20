from __future__ import annotations

"""Workshop ablation: 3 models x 3 seeds with date-based OOD split."""

import argparse
import subprocess

CONFIGS = [
    "configs/train_baseline_workshop.yaml",
    "configs/train_memory_workshop.yaml",
    "configs/train_reflector_workshop.yaml",
]
SEEDS = [0, 1, 2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="python")
    args = parser.parse_args()

    for cfg in CONFIGS:
        for seed in SEEDS:
            subprocess.run(
                [args.python, "scripts/train.py", "--config", cfg, "--seed-offset", str(seed)],
                check=True,
            )
            subprocess.run(
                [args.python, "scripts/backtest.py", "--config", cfg, "--seed-offset", str(seed)],
                check=True,
            )
    print("[done] workshop ablation finished; aggregate with scripts/run_stats.py or manual tables")


if __name__ == "__main__":
    main()
