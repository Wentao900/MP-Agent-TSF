from __future__ import annotations

import argparse
from pathlib import Path

from imtsa.config import load_config
from imtsa.data.loader import build_supervised_targets, load_and_align_data, make_sequence_tensors
from imtsa.data.split import time_split
from imtsa.train.trainer import train_model
from imtsa.utils import dump_json, ensure_dir, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["experiment"]["seed"]) + int(args.seed_offset)
    set_seed(seed)

    exp_name = cfg["experiment"]["name"] if args.seed_offset == 0 else f"{cfg['experiment']['name']}_seed{args.seed_offset}"
    exp_dir = ensure_dir(Path(cfg["experiment"]["output_root"]) / exp_name)
    dump_json(exp_dir / "config_snapshot.json", cfg)

    merged = load_and_align_data(cfg)
    labeled = build_supervised_targets(merged, cfg)
    train_df, val_df, test_df = time_split(labeled, cfg)

    train_bundle = make_sequence_tensors(train_df, cfg)
    val_bundle = make_sequence_tensors(val_df, cfg)
    _ = make_sequence_tensors(test_df, cfg)

    train_model(train_bundle, val_bundle, cfg, exp_dir)


if __name__ == "__main__":
    main()
