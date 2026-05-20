from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Build null/shuffled text.csv variants for workshop ablations")
    parser.add_argument("--input-csv", default="outputs/data/text.csv")
    parser.add_argument("--output-null", default="outputs/data/text_null.csv")
    parser.add_argument("--output-shuffled", default="outputs/data/text_shuffled.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    text = pd.read_csv(args.input_csv, parse_dates=["timestamp"])
    required = ["timestamp", "symbol", "sentiment", "relevance", "event_strength"]
    missing = [c for c in required if c not in text.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    null = text.copy()
    for c in ["sentiment", "relevance", "event_strength"]:
        null[c] = 0.0

    rng = np.random.default_rng(args.seed)
    shuffled = text.copy()
    for _, grp in shuffled.groupby("symbol"):
        idx = grp.index.to_numpy()
        perm = idx.copy()
        rng.shuffle(perm)
        for col in ["sentiment", "relevance", "event_strength"]:
            shuffled.loc[idx, col] = text.loc[perm, col].to_numpy()

    Path(args.output_null).parent.mkdir(parents=True, exist_ok=True)
    null.to_csv(args.output_null, index=False)
    shuffled.to_csv(args.output_shuffled, index=False)
    print(f"saved null text: {args.output_null} rows={len(null)}")
    print(f"saved shuffled text: {args.output_shuffled} rows={len(shuffled)}")


if __name__ == "__main__":
    main()
