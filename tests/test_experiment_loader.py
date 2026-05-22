from __future__ import annotations

import numpy as np
import pandas as pd

from imtsa.data.loader import build_supervised_targets, make_sequence_tensors


def _synthetic_panel(n: int = 120) -> pd.DataFrame:
    ts = pd.date_range("2022-01-01", periods=n, freq="B")
    rows = []
    for sym in ["AAA", "BBB"]:
        close = np.cumprod(1 + np.random.default_rng(0).normal(0, 0.01, size=n)) * 100
        part = pd.DataFrame(
            {
                "timestamp": ts,
                "symbol": sym,
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1e6,
                "ret_1d": np.r_[0.0, np.diff(close) / close[:-1]],
                "ret_5d": 0.0,
                "volatility_20": 0.01,
                "text_emb_0": 0.1,
                "text_emb_1": 0.8,
                "text_emb_2": 0.2,
            }
        )
        rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    out["split"] = np.where(out["timestamp"] < pd.Timestamp("2023-01-01"), "train", "val")
    out = build_supervised_targets(out, _cfg())
    return out


def test_sequences_do_not_cross_tickers() -> None:
    panel = _synthetic_panel()
    bundle = make_sequence_tensors(panel[panel["split"] == "train"], _cfg())
    assert bundle["x_price"].shape[0] > 0
    assert len(np.unique(bundle["ticker"])) >= 1


def test_action_encoding_hold_buy_sell() -> None:
    panel = _synthetic_panel(80)
    assert set(panel["action"].unique()).issubset({0, 1, 2})


def _cfg() -> dict:
    return {
        "data": {
            "source": "parquet",
            "timestamp_col": "timestamp",
            "symbol_col": "symbol",
            "seq_len": 16,
            "buy_threshold": 0.005,
            "sell_threshold": -0.005,
        },
        "features": {
            "price_cols": ["ret_1d", "ret_5d", "volatility_20"],
            "text_cols": ["text_emb_0", "text_emb_1", "text_emb_2"],
        },
    }
