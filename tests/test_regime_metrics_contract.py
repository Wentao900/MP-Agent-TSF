from __future__ import annotations

import pandas as pd
from pathlib import Path


def test_regime_metrics_contract() -> None:
    exp_dir = Path("outputs/memory_reflector")
    required = [
        exp_dir / "metrics_by_regime.csv",
        exp_dir / "explain_by_regime.csv",
        exp_dir / "risk_return_explain_state_table.csv",
    ]
    for p in required:
        assert p.exists(), f"missing artifact: {p}"

    df = pd.read_csv(exp_dir / "risk_return_explain_state_table.csv")
    assert {"bull", "bear", "sideways"}.issubset(set(df["market_state"].tolist()))
