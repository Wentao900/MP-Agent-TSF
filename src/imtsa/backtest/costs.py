from __future__ import annotations


def calc_trade_cost(delta_position: float, price: float, fee_rate: float, slippage_rate: float) -> float:
    notional = abs(delta_position) * price
    return notional * (fee_rate + slippage_rate)
