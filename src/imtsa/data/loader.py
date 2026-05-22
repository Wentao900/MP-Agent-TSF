from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from imtsa.data import experiment_config as ec
from imtsa.data.split import column_split


@dataclass
class DatasetBundle:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    scaler: StandardScaler | None = None


def _assert_no_leakage(df: pd.DataFrame, timestamp_col: str) -> None:
    if not df[timestamp_col].is_monotonic_increasing:
        raise ValueError("timestamps must be sorted ascending")


def _repo_root(config: dict) -> Path:
    root = config.get("project_root")
    return Path(root) if root else ec.PROJECT_ROOT


def _require_parquet_source(config: dict) -> None:
    source = str(config["data"].get("source", "parquet")).lower()
    if source != "parquet":
        raise ValueError(
            f"data.source must be 'parquet' (got {source!r}). "
            "Run: python scripts/build_datasets.py"
        )


def _normalize_panel_columns(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    data_cfg = config["data"]
    out = df.copy()
    rename = {}
    if "Date" in out.columns and data_cfg["timestamp_col"] not in out.columns:
        rename["Date"] = data_cfg["timestamp_col"]
    if "ticker" in out.columns and data_cfg["symbol_col"] not in out.columns:
        rename["ticker"] = data_cfg["symbol_col"]
    if "Close" in out.columns and "close" not in out.columns:
        rename["Close"] = "close"
    out = out.rename(columns=rename)
    out[data_cfg["timestamp_col"]] = pd.to_datetime(out[data_cfg["timestamp_col"]])
    return out.sort_values([data_cfg["symbol_col"], data_cfg["timestamp_col"]]).reset_index(drop=True)


def _feature_columns(config: dict, df: pd.DataFrame | None = None) -> tuple[list[str], list[str]]:
    feat_cfg = config["features"]
    price_default = ec.NUMERIC_INPUT_COLS
    text_default = ec.TEXT_EMB_COLS
    price_cols = list(feat_cfg.get("price_cols", price_default))
    text_cols = list(feat_cfg.get("text_cols", text_default))
    if df is not None:
        price_in_df = [c for c in price_cols if c in df.columns]
        text_in_df = [c for c in text_cols if c in df.columns]
        if not price_in_df:
            price_in_df = [c for c in price_default if c in df.columns]
        if not text_in_df:
            text_in_df = [c for c in text_default if c in df.columns]
        price_cols = price_in_df
        text_cols = text_in_df
    if not price_cols:
        raise ValueError("no numeric feature columns found in parquet panel")
    if not text_cols:
        raise ValueError("no text embedding columns found in parquet panel")
    return price_cols, text_cols


def _assert_no_future_leakage(df: pd.DataFrame, price_cols: list[str]) -> None:
    leaked = [c for c in ec.LEAKAGE_COLS if c in df.columns and c in price_cols]
    if leaked:
        raise ValueError(f"future-return columns must not be model inputs: {leaked}")


def _load_parquet_panel(config: dict) -> pd.DataFrame:
    data_cfg = config["data"]
    root = _repo_root(config)
    aligned_path = root / data_cfg.get("aligned_parquet", ec.DEFAULT_PATHS["aligned_parquet"])
    labels_path = root / data_cfg.get("labels_parquet", ec.DEFAULT_PATHS["labels_parquet"])
    if not aligned_path.exists():
        raise FileNotFoundError(
            f"aligned parquet not found: {aligned_path}. Run: python scripts/build_datasets.py"
        )
    if not labels_path.exists():
        raise FileNotFoundError(
            f"labels parquet not found: {labels_path}. Run: python scripts/build_datasets.py"
        )

    aligned = pd.read_parquet(aligned_path)
    labels = pd.read_parquet(labels_path)
    label_cols = ["Date", "ticker"] + [c for c in ec.LABEL_COLS if c in labels.columns]
    if "Date" not in labels.columns:
        label_cols = [data_cfg["timestamp_col"], data_cfg["symbol_col"]] + [
            c for c in ec.LABEL_COLS if c in labels.columns
        ]
    merge_keys = [c for c in label_cols[:2] if c in aligned.columns and c in labels.columns]
    if not merge_keys:
        merge_keys = [data_cfg["timestamp_col"], data_cfg["symbol_col"]]
    return aligned.merge(labels[[c for c in label_cols if c in labels.columns]], on=merge_keys, how="left")


def load_main_panel(config: dict) -> pd.DataFrame:
    _require_parquet_source(config)
    data_cfg = config["data"]

    df = _load_parquet_panel(config)
    df = _normalize_panel_columns(df, config)
    price_cols, text_cols = _feature_columns(config, df)
    _assert_no_future_leakage(df, price_cols)

    ts_col = data_cfg["timestamp_col"]
    sym_col = data_cfg["symbol_col"]
    required = [ts_col, sym_col, "close", "split", "action"] + price_cols + text_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"panel missing required columns: {missing}")

    return df.dropna(subset=price_cols + text_cols).reset_index(drop=True)


def load_holdout_panel(config: dict) -> pd.DataFrame:
    _require_parquet_source(config)
    data_cfg = config["data"]
    root = _repo_root(config)
    aligned_path = root / data_cfg.get("holdout_aligned_parquet", ec.DEFAULT_PATHS["holdout_aligned_parquet"])
    labels_path = root / data_cfg.get("holdout_labels_parquet", ec.DEFAULT_PATHS["holdout_labels_parquet"])
    if not aligned_path.exists():
        raise FileNotFoundError(f"holdout aligned parquet not found: {aligned_path}")

    aligned = pd.read_parquet(aligned_path)
    if labels_path.exists():
        labels = pd.read_parquet(labels_path)
        merge_keys = ["Date", "ticker"]
        if "Date" not in labels.columns:
            merge_keys = [data_cfg["timestamp_col"], data_cfg["symbol_col"]]
        label_cols = [c for c in merge_keys + ec.LABEL_COLS if c in labels.columns]
        aligned = aligned.merge(labels[label_cols], on=merge_keys, how="left")

    df = _normalize_panel_columns(aligned, config)
    price_cols, text_cols = _feature_columns(config, df)
    _assert_no_future_leakage(df, price_cols)
    ts_col = data_cfg["timestamp_col"]
    sym_col = data_cfg["symbol_col"]
    required = [ts_col, sym_col, "close", "action"] + price_cols + text_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"holdout panel missing required columns: {missing}")
    return df


def assign_split_column(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """仅 build_datasets 写盘前使用；训练管线要求 parquet 已含 split 列。"""
    data_cfg = config["data"]
    tcol = data_cfg["timestamp_col"]
    out = df.copy()
    val_start = pd.Timestamp(data_cfg.get("val_start_date", "2023-01-01"))
    test_start = pd.Timestamp(data_cfg.get("test_start_date", "2024-01-01"))
    out["split"] = np.select(
        [out[tcol] < val_start, (out[tcol] >= val_start) & (out[tcol] < test_start)],
        ["train", "val"],
        default="test",
    )
    return out


def build_supervised_targets(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """供 build_datasets / 单元测试生成标签，训练入口不调用。"""
    data_cfg = config["data"]
    sym_col = data_cfg["symbol_col"]
    close_col = "close" if "close" in df.columns else "Close"
    buy_th = float(data_cfg.get("buy_threshold", 0.005))
    sell_th = float(data_cfg.get("sell_threshold", -0.005))

    out = df.copy()
    g = out.groupby(sym_col, group_keys=False)
    future = g[close_col].shift(-1) / out[close_col] - 1
    out["future_ret_1d"] = future
    out["action"] = np.select(
        [future > buy_th, future < sell_th],
        [1, 2],
        default=0,
    ).astype(int)
    out["action_name"] = np.select(
        [out["action"] == 1, out["action"] == 2],
        ["Buy", "Sell"],
        default="Hold",
    )
    out["r_net"] = future.fillna(0.0)
    return out.dropna(subset=["future_ret_1d"]).reset_index(drop=True)


def split_panel(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "split" not in df.columns:
        raise ValueError("parquet panel must include precomputed 'split' column")
    return column_split(df, config)


def fit_numeric_scaler(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    price_cols, _ = _feature_columns(config, train_df)
    if not bool(config["data"].get("fit_scaler_on_train", True)):
        return train_df, val_df, test_df, None

    scaler = StandardScaler()
    scaler.fit(train_df[price_cols].to_numpy(dtype=np.float64))

    def _transform(part: pd.DataFrame) -> pd.DataFrame:
        out = part.copy()
        out[price_cols] = scaler.transform(out[price_cols].to_numpy(dtype=np.float64))
        return out

    return _transform(train_df), _transform(val_df), _transform(test_df), scaler


def _text_tensor_at_row(g: pd.DataFrame, i: int, text_cols: list[str], seq_len: int) -> np.ndarray:
    row = g.iloc[i][text_cols].to_numpy(dtype=np.float32)
    if len(text_cols) == 1:
        row = row.reshape(1)
    return np.tile(row, (seq_len, 1))


def make_sequence_tensors(df: pd.DataFrame, config: dict) -> Dict[str, np.ndarray]:
    data_cfg = config["data"]
    seq_len = int(data_cfg["seq_len"])
    sym_col = data_cfg["symbol_col"]
    price_cols, text_cols = _feature_columns(config, df)

    label_col = "action"
    if "future_ret_1d" in df.columns:
        ret_col = "future_ret_1d"
    elif "r_net" in df.columns:
        ret_col = "r_net"
    else:
        raise ValueError("panel must include future_ret_1d or r_net from labels_trading.parquet")

    xs_p, xs_t, ys, y_ret, idx, tickers = [], [], [], [], [], []
    for ticker, g in df.groupby(sym_col, sort=False):
        g = g.reset_index(drop=True)
        if len(g) <= seq_len:
            continue
        price_values = g[price_cols].to_numpy(dtype=np.float32)
        labels = g[label_col].to_numpy(dtype=np.int64)
        future_ret = g[ret_col].to_numpy(dtype=np.float32)

        for i in range(seq_len - 1, len(g) - 1):
            xs_p.append(price_values[i - seq_len + 1 : i + 1])
            xs_t.append(_text_tensor_at_row(g, i, text_cols, seq_len))
            ys.append(labels[i])
            y_ret.append(future_ret[i])
            idx.append(int(g.index[i]))
            tickers.append(str(ticker))

    if not xs_p:
        raise ValueError("no sequence samples produced; check split coverage and seq_len")

    return {
        "x_price": np.stack(xs_p),
        "x_text": np.stack(xs_t),
        "y_action": np.asarray(ys),
        "y_return": np.asarray(y_ret),
        "row_index": np.asarray(idx),
        "ticker": np.asarray(tickers),
    }


def iter_sequences(df: pd.DataFrame, split: str, config: dict) -> Iterable[tuple]:
    data_cfg = config["data"]
    seq_len = int(data_cfg["seq_len"])
    price_cols, text_cols = _feature_columns(config, df)
    sub = df[df["split"] == split]

    for ticker, g in sub.groupby(data_cfg["symbol_col"]):
        g = g.reset_index(drop=True)
        for i in range(seq_len, len(g) - 1):
            window = g.iloc[i - seq_len : i]
            row = g.iloc[i]
            x_num = window[price_cols].values.astype(np.float32)
            x_txt = row[text_cols].values.astype(np.float32)
            y_action = int(row["action"])
            y_ret = float(row.get("future_ret_1d", row.get("r_net", 0.0)))
            yield ticker, row[data_cfg["timestamp_col"]], x_num, x_txt, y_action, y_ret
