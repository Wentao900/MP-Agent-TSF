from __future__ import annotations

from pathlib import Path


def test_phase3_stats_outputs_contract() -> None:
    required = [
        Path("outputs/ablation_summary_rq2_rq3.csv"),
        Path("outputs/ablation_regime_summary.csv"),
        Path("outputs/stats_report_rq2_rq3_rq4.json"),
        Path("outputs/paper_table_main.csv"),
        Path("outputs/paper_table_regime.csv"),
    ]
    for p in required:
        assert p.exists(), f"missing artifact: {p}"
