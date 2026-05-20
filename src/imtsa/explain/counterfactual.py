from __future__ import annotations

import numpy as np


def perturb_high_low(contrib: np.ndarray, x: np.ndarray, ratio: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    topk = max(1, int(contrib.shape[1] * ratio))
    sort_idx = np.argsort(contrib, axis=1)
    low_idx = sort_idx[:, :topk]
    high_idx = sort_idx[:, -topk:]

    x_high = x.copy()
    x_low = x.copy()
    for i in range(len(x)):
        x_high[i, high_idx[i], :] = 0.0
        x_low[i, low_idx[i], :] = 0.0
    return x_high, x_low


def faithfulness_score(delta_high: np.ndarray, delta_low: np.ndarray) -> float:
    return float(np.mean(delta_high - delta_low))


def stability_score(weights_a: np.ndarray, weights_b: np.ndarray) -> float:
    return float(np.mean(np.abs(weights_a - weights_b)))


def batch_faithfulness_stability(
    probs: np.ndarray,
    probs_high: np.ndarray,
    probs_low: np.ndarray,
    temporal_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p_main = probs.max(axis=1)
    p_high = probs_high.max(axis=1)
    p_low = probs_low.max(axis=1)
    faith = np.maximum(0.0, p_main - p_high) - np.maximum(0.0, p_main - p_low)

    if len(temporal_weights) <= 1:
        stab = np.zeros(len(temporal_weights), dtype=float)
    else:
        d = np.abs(temporal_weights[1:] - temporal_weights[:-1]).mean(axis=1)
        stab = np.concatenate([[d[0]], d])
    return faith, stab
