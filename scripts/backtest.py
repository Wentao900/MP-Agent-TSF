from __future__ import annotations

import argparse
from pathlib import Path

from imtsa.backtest.engine import run_backtest
from imtsa.config import load_config
from imtsa.data.loader import build_supervised_targets, load_and_align_data, make_sequence_tensors
from imtsa.data.split import time_split
from imtsa.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_name = cfg["experiment"]["name"] if args.seed_offset == 0 else f"{cfg['experiment']['name']}_seed{args.seed_offset}"
    exp_dir = ensure_dir(Path(cfg["experiment"]["output_root"]) / exp_name)

    merged = load_and_align_data(cfg)
    labeled = build_supervised_targets(merged, cfg)
    _, _, test_df = time_split(labeled, cfg)
    test_bundle = make_sequence_tensors(test_df, cfg)

    run_backtest(test_bundle, test_df.reset_index(drop=True), cfg, exp_dir / "model.pt", exp_dir)


if __name__ == "__main__":
    main()
