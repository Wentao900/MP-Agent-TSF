from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_features(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    g = out.groupby("symbol", group_keys=False)
    out["ret_1"] = g["close"].pct_change(1).fillna(0.0)
    out["ret_5"] = g["close"].pct_change(5).fillna(0.0)

    vol_window = 5
    if freq == "minute":
        vol_window = 30
    out["vol_5"] = g["ret_1"].rolling(vol_window).std().reset_index(level=0, drop=True).fillna(0.0)
    return out


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "ts_code": "symbol",
        "trade_time": "timestamp",
        "trade_date": "timestamp",
        "vol": "volume",
    }
    out = df.rename(columns=rename_map)

    required = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"missing required columns in source csv: {missing}")

    out = out[required].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["symbol"] = out["symbol"].astype(str)

    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["timestamp", "symbol", "open", "high", "low", "close", "volume"])
    out = out.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pipeline-ready price.csv from Tushare-style export")
    parser.add_argument("--input-csv", required=True, help="Raw price csv path exported from Tushare or equivalent")
    parser.add_argument("--output-csv", default="outputs/data/price.csv")
    parser.add_argument("--freq", choices=["daily", "minute"], default="daily")
    args = parser.parse_args()

    src = Path(args.input_csv)
    if not src.exists():
        raise FileNotFoundError(f"input csv not found: {src}")

    raw = pd.read_csv(src)
    price = normalize_columns(raw)
    price = compute_features(price, args.freq)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    price.to_csv(out_path, index=False)

    print(f"saved price dataset: {out_path}")
    print(f"rows={len(price)}, symbols={price['symbol'].nunique()}")


if __name__ == "__main__":
    main()
