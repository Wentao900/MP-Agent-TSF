from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from imtsa.data import experiment_config as ec
from imtsa.data.loader import (
    _feature_columns,
    fit_numeric_scaler,
    load_holdout_panel,
    load_main_panel,
    make_sequence_tensors,
    split_panel,
)


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler | None, config: dict) -> pd.DataFrame:
    if scaler is None:
        return df
    price_cols, _ = _feature_columns(config, df)
    out = df.copy()
    out[price_cols] = scaler.transform(out[price_cols].to_numpy(dtype=np.float64))
    return out


def prepare_main_splits(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler | None]:
    panel = load_main_panel(config)
    train_df, val_df, test_df = split_panel(panel, config)
    return fit_numeric_scaler(train_df, val_df, test_df, config)


def prepare_holdout_frame(config: dict, scaler: StandardScaler | None) -> pd.DataFrame | None:
    data_cfg = config["data"]
    if not bool(data_cfg.get("eval_holdout", True)):
        return None
    root = Path(config["project_root"]) if config.get("project_root") else ec.PROJECT_ROOT
    holdout_path = root / data_cfg.get("holdout_aligned_parquet", ec.DEFAULT_PATHS["holdout_aligned_parquet"])
    if not holdout_path.exists():
        return None
    holdout = load_holdout_panel(config)
    price_cols, text_cols = _feature_columns(config, holdout)
    holdout = holdout.dropna(subset=price_cols + text_cols).reset_index(drop=True)
    return apply_scaler(holdout, scaler, config)


def save_scaler(scaler: StandardScaler | None, path: Path) -> None:
    if scaler is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path: Path) -> StandardScaler | None:
    if not path.exists():
        return None
    return joblib.load(path)


def build_eval_bundle(df: pd.DataFrame, config: dict) -> dict:
    return make_sequence_tensors(df, config)
