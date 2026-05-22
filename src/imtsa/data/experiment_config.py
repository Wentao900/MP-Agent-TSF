"""Constants aligned with docs/EXPERIMENT_DATA_GUIDE.md."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

LOOKBACK_DAYS = 60
REFLECT_EVERY_K = 15
AUX_HORIZONS = [1, 5, 20]
NUM_ACTIONS = 3
MAIN_TICKERS = 32
HOLDOUT_TICKERS = 15
TRAIN_END = "2022-12-31"
VAL_END = "2023-12-31"

PRICE_FEATURES = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "log_ret_1d",
    "momentum_20",
    "momentum_60",
    "volatility_20",
    "volatility_60",
    "ma_gap_20",
    "ma_gap_60",
    "volume_chg_20",
    "hl_range",
    "oc_gap",
]
MACRO_FEATURES = [
    "FEDFUNDS",
    "CPIAUCSL",
    "UNRATE",
    "DGS10",
    "DGS2",
    "T10Y2Y",
    "VIXCLS",
]
NUMERIC_INPUT_COLS = PRICE_FEATURES + MACRO_FEATURES

TEXT_EMB_COLS = [f"text_emb_{i}" for i in range(32)]
TEXT_META_COLS = ["event_count", "has_10k", "has_10q", "has_8k", "days_since_event"]

LABEL_COLS = [
    "action",
    "action_name",
    "position",
    "delta_position",
    "r_gross",
    "r_net",
    "turnover",
]

FUTURE_LABEL_COLS = ["future_ret_1d", "future_ret_5d", "future_ret_20d"]
LEAKAGE_COLS = FUTURE_LABEL_COLS

DEFAULT_PATHS = {
    "aligned_parquet": "data/processed/aligned_daily_multimodal.parquet",
    "labels_parquet": "data/processed/labels_trading.parquet",
    "holdout_aligned_parquet": "data/processed/holdout_aligned_daily.parquet",
    "holdout_labels_parquet": "data/processed/holdout_labels_trading.parquet",
    "splits_json": "data/metadata/splits.json",
}
