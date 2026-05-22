from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from imtsa.data import experiment_config as ec


def _engineer_price_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True).copy()
    g = out.groupby("symbol", group_keys=False)
    out["ret_1d"] = g["close"].pct_change(1)
    out["ret_5d"] = g["close"].pct_change(5)
    out["ret_20d"] = g["close"].pct_change(20)
    out["log_ret_1d"] = np.log1p(out["ret_1d"].fillna(0.0))
    out["momentum_20"] = g["close"].pct_change(20)
    out["momentum_60"] = g["close"].pct_change(60)
    out["volatility_20"] = g["ret_1d"].rolling(20).std().reset_index(level=0, drop=True)
    out["volatility_60"] = g["ret_1d"].rolling(60).std().reset_index(level=0, drop=True)
    ma20 = g["close"].rolling(20).mean().reset_index(level=0, drop=True)
    ma60 = g["close"].rolling(60).mean().reset_index(level=0, drop=True)
    out["ma_gap_20"] = out["close"] / ma20 - 1.0
    out["ma_gap_60"] = out["close"] / ma60 - 1.0
    out["volume_chg_20"] = g["volume"].pct_change(20)
    out["hl_range"] = (out["high"] - out["low"]) / out["close"]
    out["oc_gap"] = (out["open"] - out["close"].shift(1)) / out["close"].shift(1)
    for col in ec.MACRO_FEATURES:
        out[col] = np.nan
    feat = [c for c in ec.NUMERIC_INPUT_COLS if c in out.columns]
    out[feat] = out[feat].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _map_text_embeddings(text: pd.DataFrame) -> pd.DataFrame:
    out = text.copy()
    mapping = {
        "sentiment": "text_emb_0",
        "relevance": "text_emb_1",
        "event_strength": "text_emb_2",
    }
    for src, dst in mapping.items():
        if src in out.columns:
            out[dst] = out[src]
    for col in ec.TEXT_EMB_COLS:
        if col not in out.columns:
            out[col] = 0.0
    out["event_count"] = (out.get("event_strength", 0) > 0).astype(int)
    out["has_10k"] = 0
    out["has_10q"] = 0
    out["has_8k"] = 0
    out["days_since_event"] = 999
    return out


def _build_labels(df: pd.DataFrame, buy_th: float, sell_th: float, cost: float) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("ticker", group_keys=False)
    future = g["close"].shift(-1) / out["close"] - 1
    out["future_ret_1d"] = future
    out["future_ret_5d"] = g["close"].shift(-5) / out["close"] - 1
    out["future_ret_20d"] = g["close"].shift(-20) / out["close"] - 1
    out["action"] = np.select([future > buy_th, future < sell_th], [1, 2], default=0).astype(int)
    out["action_name"] = np.select(
        [out["action"] == 1, out["action"] == 2],
        ["Buy", "Sell"],
        default="Hold",
    )
    out["position"] = np.where(out["action"] == 1, 1.0, 0.0)
    out["delta_position"] = out.groupby("ticker")["position"].diff().fillna(out["position"])
    out["r_gross"] = out["position"] * out["ret_1d"].fillna(0.0)
    out["turnover"] = out["delta_position"].abs()
    out["r_net"] = out["r_gross"] - out["turnover"] * cost
    return out


def _assign_split(df: pd.DataFrame, val_start: str, test_start: str) -> pd.DataFrame:
    out = df.copy()
    val_ts = pd.Timestamp(val_start)
    test_ts = pd.Timestamp(test_start)
    out["split"] = np.select(
        [out["Date"] < val_ts, (out["Date"] >= val_ts) & (out["Date"] < test_ts)],
        ["train", "val"],
        default="test",
    )
    return out


def _spy_regime(df: pd.DataFrame, bench_tickers: set[str]) -> pd.DataFrame:
    out = df.copy()
    bench = out[out["ticker"].isin(bench_tickers)].sort_values("Date")
    if bench.empty:
        out["spy_roll_ret"] = 0.0
        out["market_regime"] = "sideways"
        return out

    spy = bench.groupby("Date")["ret_1d"].mean().sort_index()
    roll = spy.rolling(60, min_periods=20).sum()
    regime = pd.Series(
        np.select([roll > 0.05, roll < -0.05], ["bull", "bear"], default="sideways"),
        index=roll.index,
        name="market_regime",
    )
    regime_map = regime.to_dict()
    roll_map = roll.to_dict()
    out["market_regime"] = out["Date"].map(regime_map).fillna("sideways")
    out["spy_roll_ret"] = out["Date"].map(roll_map).fillna(0.0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build guide-compatible parquet datasets from CSV")
    parser.add_argument("--price-csv", default="outputs/data/price.csv")
    parser.add_argument("--text-csv", default="outputs/data/text.csv")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--metadata-dir", default="data/metadata")
    parser.add_argument("--val-start", default="2023-01-01")
    parser.add_argument("--test-start", default="2024-01-01")
    parser.add_argument("--buy-threshold", type=float, default=0.005)
    parser.add_argument("--sell-threshold", type=float, default=-0.005)
    parser.add_argument("--cost-rate", type=float, default=0.0005)
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.2,
        help="fraction of tickers reserved for holdout OOS (0 to disable); never appear in main parquet",
    )
    parser.add_argument("--holdout-tickers-file", default="", help="optional one ticker per line; overrides fraction")
    args = parser.parse_args()

    price = pd.read_csv(args.price_csv, parse_dates=["timestamp"])
    text = pd.read_csv(args.text_csv, parse_dates=["timestamp"])
    text = _map_text_embeddings(text)

    merged = pd.merge_asof(
        price.sort_values("timestamp"),
        text.sort_values("timestamp"),
        on="timestamp",
        by="symbol",
        direction="backward",
        allow_exact_matches=True,
    )
    merged = _engineer_price_features(merged)
    merged = merged.rename(columns={"timestamp": "Date", "symbol": "ticker"})
    merged = _spy_regime(merged, {"SPY", "QQQ"})
    labels = _build_labels(merged, args.buy_threshold, args.sell_threshold, args.cost_rate)

    tickers = sorted(labels["ticker"].unique().tolist())
    holdout_tickers: set[str] = set()
    if args.holdout_tickers_file:
        holdout_tickers = {
            line.strip()
            for line in Path(args.holdout_tickers_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    elif args.holdout_fraction > 0 and len(tickers) >= 2:
        n_hold = max(1, int(round(len(tickers) * args.holdout_fraction)))
        holdout_tickers = set(tickers[-n_hold:])

    main_df = labels[~labels["ticker"].isin(holdout_tickers)].copy()
    holdout_df = labels[labels["ticker"].isin(holdout_tickers)].copy()

    main_df = _assign_split(main_df, args.val_start, args.test_start)
    if len(holdout_df):
        holdout_df = holdout_df.copy()
        holdout_df["split"] = "holdout"

    label_cols = ["Date", "ticker"] + [c for c in ec.LABEL_COLS if c in labels.columns]
    feat_cols = (
        ["Date", "ticker", "split", "open", "high", "low", "close", "volume"]
        + ec.NUMERIC_INPUT_COLS
        + ec.TEXT_EMB_COLS
        + ec.TEXT_META_COLS
        + ec.FUTURE_LABEL_COLS
        + ["market_regime", "spy_roll_ret"]
    )

    def _pack(frame: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in feat_cols if c in frame.columns]
        return frame[cols].copy()

    aligned_main = _pack(main_df)
    labels_main = main_df[label_cols].copy()

    out_dir = Path(args.output_dir)
    meta_dir = Path(args.metadata_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    aligned_path = out_dir / "aligned_daily_multimodal.parquet"
    labels_path = out_dir / "labels_trading.parquet"
    aligned_main.to_parquet(aligned_path, index=False)
    labels_main.to_parquet(labels_path, index=False)

    splits = {
        "train": {
            "start": str(aligned_main.loc[aligned_main["split"] == "train", "Date"].min())
            if (aligned_main["split"] == "train").any()
            else None,
            "end": str(aligned_main.loc[aligned_main["split"] == "train", "Date"].max())
            if (aligned_main["split"] == "train").any()
            else None,
        },
        "val": {
            "start": str(aligned_main.loc[aligned_main["split"] == "val", "Date"].min())
            if (aligned_main["split"] == "val").any()
            else None,
            "end": str(aligned_main.loc[aligned_main["split"] == "val", "Date"].max())
            if (aligned_main["split"] == "val").any()
            else None,
        },
        "test": {
            "start": str(aligned_main.loc[aligned_main["split"] == "test", "Date"].min())
            if (aligned_main["split"] == "test").any()
            else None,
            "end": str(aligned_main.loc[aligned_main["split"] == "test", "Date"].max())
            if (aligned_main["split"] == "test").any()
            else None,
        },
        "main_tickers": sorted(aligned_main["ticker"].unique().tolist()),
        "holdout_tickers": sorted(holdout_tickers),
    }
    (meta_dir / "splits.json").write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote {aligned_path} rows={len(aligned_main)} tickers={aligned_main['ticker'].nunique()}")
    print(f"wrote {labels_path} rows={len(labels_main)}")

    if len(holdout_df):
        holdout_aligned_path = out_dir / "holdout_aligned_daily.parquet"
        holdout_labels_path = out_dir / "holdout_labels_trading.parquet"
        _pack(holdout_df).to_parquet(holdout_aligned_path, index=False)
        holdout_df[label_cols].to_parquet(holdout_labels_path, index=False)
        print(f"wrote {holdout_aligned_path} rows={len(holdout_df)} tickers={holdout_df['ticker'].nunique()}")
        print(f"wrote {holdout_labels_path}")
    else:
        print("[holdout] skipped (single ticker or holdout-fraction=0)")

    print(f"wrote {meta_dir / 'splits.json'}")


if __name__ == "__main__":
    main()
