from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PRICE_REQUIRED = ["timestamp", "symbol", "open", "high", "low", "close", "volume", "ret_1", "ret_5", "vol_5"]
TEXT_REQUIRED = ["timestamp", "symbol", "sentiment", "relevance", "event_strength"]


def assert_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def assert_monotonic(df: pd.DataFrame, name: str) -> None:
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError(f"{name}.timestamp is not monotonic increasing")


def assert_no_null(df: pd.DataFrame, cols: list[str], name: str) -> None:
    null_cols = [c for c in cols if df[c].isna().any()]
    if null_cols:
        raise ValueError(f"{name} has null values in columns: {null_cols}")


def assert_ranges(text: pd.DataFrame) -> None:
    if ((text["relevance"] < 0) | (text["relevance"] > 1)).any():
        raise ValueError("text.relevance must be in [0,1]")
    if ((text["event_strength"] < 0) | (text["event_strength"] > 1)).any():
        raise ValueError("text.event_strength must be in [0,1]")
    if ((text["sentiment"] < -1) | (text["sentiment"] > 1)).any():
        raise ValueError("text.sentiment should be in [-1,1]")


def leakage_sanity(price: pd.DataFrame, text: pd.DataFrame) -> None:
    p = price[["timestamp", "symbol"]].sort_values(["timestamp", "symbol"]).copy()
    t = text[["timestamp", "symbol"]].sort_values(["timestamp", "symbol"]).copy()

    merged = pd.merge_asof(
        p.sort_values("timestamp"),
        t.sort_values("timestamp"),
        on="timestamp",
        by="symbol",
        direction="backward",
        allow_exact_matches=True,
        suffixes=("_price", "_text"),
    )

    # merge_asof backward 本身防未来信息；这里只做覆盖率提醒
    coverage = merged["timestamp"].notna().mean()
    print(f"text backward-match coverage (sanity): {coverage:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dataset contract for IMTSA pipeline")
    parser.add_argument("--price-csv", default="outputs/data/price.csv")
    parser.add_argument("--text-csv", default="outputs/data/text.csv")
    args = parser.parse_args()

    price_path = Path(args.price_csv)
    text_path = Path(args.text_csv)
    if not price_path.exists():
        raise FileNotFoundError(f"price csv not found: {price_path}")
    if not text_path.exists():
        raise FileNotFoundError(f"text csv not found: {text_path}")

    price = pd.read_csv(price_path)
    text = pd.read_csv(text_path)

    price["timestamp"] = pd.to_datetime(price["timestamp"])
    text["timestamp"] = pd.to_datetime(text["timestamp"])

    price = price.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    text = text.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    assert_columns(price, PRICE_REQUIRED, "price")
    assert_columns(text, TEXT_REQUIRED, "text")

    assert_no_null(price, PRICE_REQUIRED, "price")
    assert_no_null(text, TEXT_REQUIRED, "text")

    assert_monotonic(price, "price")
    assert_monotonic(text, "text")

    assert_ranges(text)
    leakage_sanity(price, text)

    print("dataset contract validation passed")
    print(f"price rows={len(price)}, symbols={price['symbol'].nunique()}")
    print(f"text rows={len(text)}, symbols={text['symbol'].nunique()}")


if __name__ == "__main__":
    main()
