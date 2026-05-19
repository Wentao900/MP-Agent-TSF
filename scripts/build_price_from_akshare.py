from __future__ import annotations

import argparse
from pathlib import Path

import akshare as ak
import pandas as pd


def _symbol_to_ak(symbol: str) -> str:
    # 000001.SZ -> 000001
    return symbol.split(".")[0]


def _normalize_daily(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    rename = {
        "日期": "timestamp",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
    }
    out = df.rename(columns=rename).copy()
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"daily data missing columns for {symbol}: {missing}")

    out = out[required]
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["symbol"] = symbol
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])  # type: ignore[arg-type]


def _normalize_minute(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    rename = {
        "时间": "timestamp",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
    }
    out = df.rename(columns=rename).copy()
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"minute data missing columns for {symbol}: {missing}")

    out = out[required]
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["symbol"] = symbol
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])  # type: ignore[arg-type]


def _feature_engineer(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    out = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True).copy()
    g = out.groupby("symbol", group_keys=False)
    out["ret_1"] = g["close"].pct_change(1).fillna(0.0)
    out["ret_5"] = g["close"].pct_change(5).fillna(0.0)
    vol_window = 5 if freq == "daily" else 30
    out["vol_5"] = g["ret_1"].rolling(vol_window).std().reset_index(level=0, drop=True).fillna(0.0)
    return out.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def fetch_daily(symbol: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    ak_symbol = _symbol_to_ak(symbol)
    raw = ak.stock_zh_a_hist(symbol=ak_symbol, period="daily", start_date=start, end_date=end, adjust=adjust)
    return _normalize_daily(raw, symbol)


def fetch_minute(symbol: str, period: str, adjust: str) -> pd.DataFrame:
    ak_symbol = _symbol_to_ak(symbol)
    raw = ak.stock_zh_a_hist_min_em(symbol=ak_symbol, period=period, adjust=adjust)
    return _normalize_minute(raw, symbol)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build project price.csv from AkShare")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols, e.g. 000001.SZ,600036.SH")
    parser.add_argument("--freq", choices=["daily", "minute"], default="daily")
    parser.add_argument("--start-date", default="20240101", help="daily only, YYYYMMDD")
    parser.add_argument("--end-date", default="20241231", help="daily only, YYYYMMDD")
    parser.add_argument("--minute-period", choices=["1", "5", "15", "30", "60"], default="1")
    parser.add_argument("--adjust", default="qfq", help="daily: qfq/hfq/'' ; minute: qfq/hfq/")
    parser.add_argument("--output-csv", default="outputs/data/price.csv")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        raise ValueError("no symbols provided")

    frames: list[pd.DataFrame] = []
    for i, symbol in enumerate(symbols, start=1):
        print(f"[fetch] {i}/{len(symbols)} symbol={symbol} freq={args.freq}", flush=True)
        if args.freq == "daily":
            df = fetch_daily(symbol, args.start_date, args.end_date, args.adjust)
        else:
            df = fetch_minute(symbol, args.minute_period, args.adjust)
        print(f"[ok] symbol={symbol} rows={len(df)}", flush=True)
        frames.append(df)

    price = pd.concat(frames, ignore_index=True)
    price = _feature_engineer(price, args.freq)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    price.to_csv(out_path, index=False)

    print(f"[done] saved={out_path} rows={len(price)} symbols={price['symbol'].nunique()}", flush=True)


if __name__ == "__main__":
    main()
