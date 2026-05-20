from __future__ import annotations

from imtsa.backtest.costs import calc_trade_cost


def test_cost_formula() -> None:
    c = calc_trade_cost(delta_position=0.5, price=100.0, fee_rate=0.001, slippage_rate=0.002)
    assert abs(c - 0.15) < 1e-8
