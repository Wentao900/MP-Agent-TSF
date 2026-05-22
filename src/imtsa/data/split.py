from __future__ import annotations

import numpy as np
import pandas as pd


def column_split(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_cfg = config["data"]
    split_col = data_cfg.get("split_col", "split")
    if split_col not in df.columns:
        raise ValueError(f"split column '{split_col}' missing; use split_mode=date or build datasets first")

    train = df[df[split_col] == "train"].reset_index(drop=True)
    val = df[df[split_col] == "val"].reset_index(drop=True)
    test = df[df[split_col] == "test"].reset_index(drop=True)
    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError("each split must be non-empty")
    return train, val, test


def time_split(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_cfg = config["data"]
    tcol = data_cfg["timestamp_col"]
    split_mode = str(data_cfg.get("split_mode", "ratio"))

    if split_mode == "column":
        return column_split(df, config)

    if split_mode == "date":
        val_start = pd.Timestamp(data_cfg["val_start_date"])
        test_start = pd.Timestamp(data_cfg["test_start_date"])
        train = df[df[tcol] < val_start].reset_index(drop=True)
        val = df[(df[tcol] >= val_start) & (df[tcol] < test_start)].reset_index(drop=True)
        test = df[df[tcol] >= test_start].reset_index(drop=True)
    else:
        ratios = (
            float(data_cfg["train_ratio"]),
            float(data_cfg["val_ratio"]),
            float(data_cfg["test_ratio"]),
        )
        if not np.isclose(sum(ratios), 1.0):
            raise ValueError("train/val/test ratios must sum to 1")

        n = len(df)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        train = df.iloc[:n_train].reset_index(drop=True)
        val = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
        test = df.iloc[n_train + n_val :].reset_index(drop=True)

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError("each split must be non-empty")

    if not (train[tcol].max() < val[tcol].min()):
        raise ValueError("train/val time split order violated")
    if not (val[tcol].max() < test[tcol].min()):
        raise ValueError("val/test time split order violated")
    return train, val, test


def label_market_states(df: pd.DataFrame, config: dict) -> pd.Series:
    regime_col = config["data"].get("regime_col", "market_regime")
    if regime_col in df.columns:
        states = df[regime_col].astype(str)
        states = states.replace({"nan": "sideways"}).fillna("sideways")
        return pd.Series(states.values, index=df.index, name="market_state")

    window = int(config["data"].get("market_state_window", 30))
    close_col = "close" if "close" in df.columns else "Close"
    rolling_ret = df[close_col].pct_change(window).fillna(0.0)
    states = np.where(rolling_ret > 0.01, "bull", np.where(rolling_ret < -0.01, "bear", "sideways"))
    return pd.Series(states, index=df.index, name="market_state")
