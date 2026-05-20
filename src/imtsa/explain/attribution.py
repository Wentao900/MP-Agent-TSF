from __future__ import annotations

import numpy as np
import pandas as pd


REGIMES = ["bull", "bear", "sideways"]


def modality_contribution(alpha: np.ndarray) -> dict:
    price_w = alpha.mean(axis=0)
    text_w = 1.0 - price_w
    return {
        "price_contrib_mean": float(price_w.mean()),
        "text_contrib_mean": float(text_w.mean()),
    }


def temporal_contribution(x_price: np.ndarray) -> np.ndarray:
    abs_signal = np.abs(x_price).mean(axis=2)
    weights = abs_signal / (abs_signal.sum(axis=1, keepdims=True) + 1e-8)
    return weights


def explain_step_frame(
    steps: np.ndarray,
    market_states: np.ndarray,
    alpha: np.ndarray,
    probs: np.ndarray,
    temporal_weights: np.ndarray,
    faithfulness: np.ndarray,
    stability: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame({
        "step": steps.astype(int),
        "market_state": market_states,
        "alpha_price": alpha[:, 0],
        "alpha_text": 1.0 - alpha[:, 0],
        "prob_buy": probs[:, 0],
        "prob_sell": probs[:, 1],
        "prob_hold": probs[:, 2],
        "temporal_peak": temporal_weights.max(axis=1),
        "faithfulness": faithfulness,
        "stability": stability,
    })


def summarize_explain(explain_step: pd.DataFrame) -> dict:
    return {
        "n_steps": int(len(explain_step)),
        "faithfulness_mean": float(explain_step["faithfulness"].mean()),
        "stability_mean": float(explain_step["stability"].mean()),
        "alpha_price_mean": float(explain_step["alpha_price"].mean()),
        "alpha_text_mean": float(explain_step["alpha_text"].mean()),
        "temporal_peak_mean": float(explain_step["temporal_peak"].mean()),
    }


def explain_by_regime(explain_step: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime in REGIMES:
        sdf = explain_step[explain_step["market_state"] == regime]
        if len(sdf) == 0:
            rows.append({
                "market_state": regime,
                "coverage": 0,
                "faithfulness": np.nan,
                "stability": np.nan,
                "price_contrib_mean": np.nan,
                "text_contrib_mean": np.nan,
            })
        else:
            rows.append({
                "market_state": regime,
                "coverage": int(len(sdf)),
                "faithfulness": float(sdf["faithfulness"].mean()),
                "stability": float(sdf["stability"].mean()),
                "price_contrib_mean": float(sdf["alpha_price"].mean()),
                "text_contrib_mean": float(sdf["alpha_text"].mean()),
            })
    return pd.DataFrame(rows)
