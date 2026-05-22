from __future__ import annotations

import argparse
from pathlib import Path

from imtsa.backtest.engine import run_backtest
from imtsa.config import load_config
from imtsa.data.loader import fit_numeric_scaler, load_main_panel, split_panel
from imtsa.data.pipeline import apply_scaler, build_eval_bundle, load_scaler, prepare_holdout_frame
from imtsa.utils import dump_json, ensure_dir


def _get_test_frame(cfg: dict, exp_dir: Path):
    panel = load_main_panel(cfg)
    train_df, val_df, test_df = split_panel(panel, cfg)
    scaler = load_scaler(exp_dir / "scaler.joblib")
    if scaler is None:
        train_df, val_df, test_df, scaler = fit_numeric_scaler(train_df, val_df, test_df, cfg)
    else:
        test_df = apply_scaler(test_df, scaler, cfg)
    return test_df, scaler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--skip-holdout", action="store_true", help="skip RQ4 holdout OOS eval even if parquet exists")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.skip_holdout:
        cfg.setdefault("data", {})["eval_holdout"] = False

    exp_name = cfg["experiment"]["name"] if args.seed_offset == 0 else f"{cfg['experiment']['name']}_seed{args.seed_offset}"
    exp_dir = ensure_dir(Path(cfg["experiment"]["output_root"]) / exp_name)
    model_path = exp_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}; run scripts/train.py first")

    test_df, scaler = _get_test_frame(cfg, exp_dir)
    test_bundle = build_eval_bundle(test_df, cfg)
    test_metrics = run_backtest(
        test_bundle,
        test_df.reset_index(drop=True),
        cfg,
        model_path,
        exp_dir,
        output_subdir=None,
    )
    summary = {"test": test_metrics}

    holdout_df = prepare_holdout_frame(cfg, scaler)
    if holdout_df is not None and len(holdout_df) > 0:
        holdout_bundle = build_eval_bundle(holdout_df, cfg)
        holdout_metrics = run_backtest(
            holdout_bundle,
            holdout_df.reset_index(drop=True),
            cfg,
            model_path,
            exp_dir,
            output_subdir="holdout",
        )
        summary["holdout"] = holdout_metrics
        print(f"[holdout] total_return={holdout_metrics['total_return']:.4f} sharpe={holdout_metrics['sharpe']:.4f}")
    else:
        print("[holdout] skipped (no holdout parquet or eval_holdout=false)")

    dump_json(exp_dir / "eval_summary.json", summary)


if __name__ == "__main__":
    main()
