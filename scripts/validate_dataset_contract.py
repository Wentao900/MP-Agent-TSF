from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from imtsa.data import experiment_config as ec


def assert_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def assert_no_future_in_features() -> None:
    overlap = [c for c in ec.LEAKAGE_COLS if c in ec.NUMERIC_INPUT_COLS]
    if overlap:
        raise ValueError(f"numeric feature list must not include future labels: {overlap}")


def validate_parquet(aligned_path: Path, labels_path: Path) -> None:
    aligned = pd.read_parquet(aligned_path)
    labels = pd.read_parquet(labels_path)

    assert_columns(aligned, ["Date", "ticker", "split"] + ec.NUMERIC_INPUT_COLS, "aligned")
    assert_columns(labels, ["Date", "ticker", "action", "r_net"], "labels")
    assert_no_future_in_features()

    for col in ec.FUTURE_LABEL_COLS:
        if col in aligned.columns:
            assert col not in ec.NUMERIC_INPUT_COLS

    if not set(aligned["split"].unique()).issuperset({"train", "val", "test"}):
        raise ValueError("aligned.split must contain train/val/test")

    dup = aligned.duplicated(subset=["Date", "ticker"]).sum()
    if dup:
        raise ValueError(f"aligned has duplicate (Date, ticker) keys: {dup}")

    print("parquet dataset contract validation passed")
    print(f"aligned rows={len(aligned)}, tickers={aligned['ticker'].nunique()}")
    print(f"labels rows={len(labels)}, tickers={labels['ticker'].nunique()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate parquet dataset contract for IMTSA pipeline")
    parser.add_argument("--aligned-parquet", default="data/processed/aligned_daily_multimodal.parquet")
    parser.add_argument("--labels-parquet", default="data/processed/labels_trading.parquet")
    args = parser.parse_args()

    aligned_path = Path(args.aligned_parquet)
    labels_path = Path(args.labels_parquet)
    if not aligned_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"parquet not found under {aligned_path.parent}. Run: python scripts/build_datasets.py"
        )
    validate_parquet(aligned_path, labels_path)


if __name__ == "__main__":
    main()
