from __future__ import annotations

from pathlib import Path


def test_expected_outputs_exist_after_run() -> None:
    # This smoke test validates artifact contract; run scripts first.
    required = [
        Path("outputs/baseline/metrics.json"),
        Path("outputs/baseline/trades.csv"),
        Path("outputs/baseline/config_snapshot.json"),
        Path("outputs/baseline/explain_step.csv"),
        Path("outputs/baseline/explain_summary.json"),
        Path("outputs/baseline/tradeoff_summary.csv"),
        Path("outputs/baseline/metrics_by_regime.csv"),
        Path("outputs/baseline/explain_by_regime.csv"),
        Path("outputs/baseline/risk_return_explain_state_table.csv"),
    ]
    for p in required:
        assert p.exists(), f"missing artifact: {p}"
