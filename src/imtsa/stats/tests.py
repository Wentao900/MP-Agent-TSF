from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


def paired_tests(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    t_stat, t_p = ttest_rel(a, b)
    w_stat, w_p = wilcoxon(a, b, zero_method="wilcox")
    diff = a - b
    effect = float(diff.mean() / (diff.std(ddof=1) + 1e-8))
    return {
        "t_stat": float(t_stat),
        "t_pvalue": float(t_p),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_pvalue": float(w_p),
        "cohens_d": effect,
    }
