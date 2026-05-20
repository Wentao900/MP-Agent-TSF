from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class DatasetBundle:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def _assert_no_leakage(df: pd.DataFrame, timestamp_col: str) -> None:
    if not df[timestamp_col].is_monotonic_increasing:
        raise ValueError("timestamps must be sorted ascending")


def load_and_align_data(config: dict) -> pd.DataFrame:
    data_cfg = config["data"]
    feat_cfg = config["features"]
    ts_col = data_cfg["timestamp_col"]

    price = pd.read_csv(data_cfg["price_csv"], parse_dates=[ts_col]).sort_values(ts_col)
    text = pd.read_csv(data_cfg["text_csv"], parse_dates=[ts_col]).sort_values(ts_col)

    _assert_no_leakage(price, ts_col)
    _assert_no_leakage(text, ts_col)

    merged = pd.merge_asof(
        price,
        text,
        on=ts_col,
        by=data_cfg["symbol_col"],
        direction="backward",
        allow_exact_matches=True,
    )
    merged = merged.dropna(subset=feat_cfg["price_cols"] + feat_cfg["text_cols"]).reset_index(drop=True)
    if not merged[ts_col].is_monotonic_increasing:
        raise ValueError("merged data must be monotonic by timestamp")
    return merged


def build_supervised_targets(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    horizon = int(config["data"]["prediction_horizon"])
    out = df.copy()
    out["future_return"] = out.groupby(config["data"]["symbol_col"])["close"].shift(-horizon) / out["close"] - 1
    out["action_label"] = np.select(
        [out["future_return"] > 0.001, out["future_return"] < -0.001],
        [0, 1],
        default=2,
    )
    out = out.dropna(subset=["future_return"]).reset_index(drop=True)
    return out


def make_sequence_tensors(df: pd.DataFrame, config: dict) -> Dict[str, np.ndarray]:
    seq_len = int(config["data"]["seq_len"])
    pcols = config["features"]["price_cols"]
    tcols = config["features"]["text_cols"]

    price_values = df[pcols].to_numpy(dtype=np.float32)
    text_values = df[tcols].to_numpy(dtype=np.float32)
    labels = df["action_label"].to_numpy(dtype=np.int64)
    future_ret = df["future_return"].to_numpy(dtype=np.float32)

    xs_p, xs_t, ys, y_ret, idx = [], [], [], [], []
    for i in range(seq_len - 1, len(df)):
        xs_p.append(price_values[i - seq_len + 1 : i + 1])
        xs_t.append(text_values[i - seq_len + 1 : i + 1])
        ys.append(labels[i])
        y_ret.append(future_ret[i])
        idx.append(i)

    return {
        "x_price": np.stack(xs_p),
        "x_text": np.stack(xs_t),
        "y_action": np.asarray(ys),
        "y_return": np.asarray(y_ret),
        "row_index": np.asarray(idx),
    }
