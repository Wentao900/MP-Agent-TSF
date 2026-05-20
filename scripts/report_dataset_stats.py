from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Report dataset stats for workshop paper table")
    parser.add_argument("--price-csv", default="outputs/data/price.csv")
    parser.add_argument("--text-csv", default="outputs/data/text.csv")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--val-start", default="2023-01-01")
    parser.add_argument("--test-start", default="2024-01-01")
    parser.add_argument("--output-json", default="outputs/dataset_stats_workshop.json")
    args = parser.parse_args()

    price = pd.read_csv(args.price_csv, parse_dates=["timestamp"]).sort_values(["symbol", "timestamp"])
    text = pd.read_csv(args.text_csv, parse_dates=["timestamp"]).sort_values(["symbol", "timestamp"])

    merged = pd.merge_asof(
        price.sort_values("timestamp"),
        text.sort_values("timestamp"),
        on="timestamp",
        by="symbol",
        direction="backward",
        allow_exact_matches=True,
    )

    feat = ["open", "high", "low", "close", "volume", "ret_1", "ret_5", "vol_5", "sentiment", "relevance", "event_strength"]
    merged = merged.dropna(subset=feat)

    ts = merged["timestamp"]
    val_start = pd.Timestamp(args.val_start)
    test_start = pd.Timestamp(args.test_start)

    splits = {
        "train": merged[ts < val_start],
        "val": merged[(ts >= val_start) & (ts < test_start)],
        "test": merged[ts >= test_start],
    }

    # text coverage: days where text feature changed vs previous bar (per symbol)
    coverage_rows = []
    for sym, grp in merged.groupby("symbol"):
        g = grp.sort_values("timestamp").copy()
        changed = (
            g["sentiment"].diff().fillna(1).ne(0)
            | g["relevance"].diff().fillna(1).ne(0)
            | g["event_strength"].diff().fillna(1).ne(0)
        )
        coverage_rows.append({"symbol": sym, "text_update_ratio": float(changed.mean())})
    coverage_df = pd.DataFrame(coverage_rows)

    stats = {
        "n_symbols_price": int(price["symbol"].nunique()),
        "n_symbols_text": int(text["symbol"].nunique()),
        "n_price_rows": int(len(price)),
        "n_text_rows": int(len(text)),
        "n_merged_rows": int(len(merged)),
        "date_min": str(ts.min()),
        "date_max": str(ts.max()),
        "text_update_ratio_mean": float(coverage_df["text_update_ratio"].mean()) if len(coverage_df) else 0.0,
        "seq_len": args.seq_len,
        "n_sequence_samples": int(max(0, len(merged) - args.seq_len)),
        "splits": {},
    }
    for name, part in splits.items():
        stats["splits"][name] = {
            "rows": int(len(part)),
            "seq_samples": int(max(0, len(part) - args.seq_len)),
            "date_min": str(part["timestamp"].min()) if len(part) else None,
            "date_max": str(part["timestamp"].max()) if len(part) else None,
        }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
