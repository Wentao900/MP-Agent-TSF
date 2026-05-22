from __future__ import annotations

import argparse
from pathlib import Path

from imtsa.config import load_config
from imtsa.data.pipeline import build_eval_bundle, prepare_main_splits, save_scaler
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

    train_df, val_df, test_df, scaler = prepare_main_splits(cfg)
    save_scaler(scaler, exp_dir / "scaler.joblib")

    train_bundle = build_eval_bundle(train_df, cfg)
    val_bundle = build_eval_bundle(val_df, cfg)
    _ = build_eval_bundle(test_df, cfg)

    train_model(train_bundle, val_bundle, cfg, exp_dir)


if __name__ == "__main__":
    main()
