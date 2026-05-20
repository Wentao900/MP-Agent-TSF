from __future__ import annotations

from pathlib import Path


def test_explain_outputs_contract() -> None:
    exp_dir = Path("outputs/memory_reflector")
    required = [
        exp_dir / "explain_step.csv",
        exp_dir / "explain_summary.json",
        exp_dir / "tradeoff_summary.csv",
    ]
    for p in required:
        assert p.exists(), f"missing artifact: {p}"
