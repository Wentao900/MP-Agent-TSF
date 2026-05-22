from __future__ import annotations

import numpy as np
import pandas as pd


REGIMES = ["bull", "bear", "sideways"]


def compute_metrics(equity_curve: pd.Series, trades: pd.DataFrame) -> dict:
    returns = equity_curve.pct_change().fillna(0.0)
    ann_factor = np.sqrt(252)
    sharpe = 0.0 if returns.std() == 0 else (returns.mean() / returns.std()) * ann_factor
    cummax = equity_curve.cummax()
    drawdown = (equity_curve / cummax - 1.0).min()
    win_rate = float((trades["pnl"] > 0).mean()) if len(trades) else 0.0
    turnover = float(trades["abs_delta_pos"].sum()) if len(trades) else 0.0
    total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)
    return {
        "total_return": total_return,
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown),
        "win_rate": win_rate,
        "turnover": turnover,
        "num_trades": int(len(trades)),
    }


def compute_metrics_by_regime(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime in REGIMES:
        sdf = trades[trades["market_state"] == regime]
        if len(sdf) == 0:
            rows.append({
                "market_state": regime,
                "coverage": 0,
                "total_pnl": np.nan,
                "mean_pnl": np.nan,
                "vol_pnl": np.nan,
                "win_rate": np.nan,
                "turnover": np.nan,
                "num_trades": 0,
            })
            continue

        rows.append({
            "market_state": regime,
            "coverage": int(len(sdf)),
            "total_pnl": float(sdf["pnl"].sum()),
            "mean_pnl": float(sdf["pnl"].mean()),
            "vol_pnl": float(sdf["pnl"].std(ddof=0)),
            "win_rate": float((sdf["pnl"] > 0).mean()),
            "turnover": float(sdf["abs_delta_pos"].sum()),
            "num_trades": int(len(sdf)),
        })
    return pd.DataFrame(rows)


def build_risk_return_explain_state_table(metrics_by_regime: pd.DataFrame, explain_by_regime: pd.DataFrame) -> pd.DataFrame:
    out = metrics_by_regime.merge(explain_by_regime, on="market_state", how="outer")
    cols = [
        "market_state",
        "coverage",
        "num_trades",
        "total_pnl",
        "mean_pnl",
        "vol_pnl",
        "win_rate",
        "turnover",
        "faithfulness",
        "stability",
        "price_contrib_mean",
        "text_contrib_mean",
    ]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols].sort_values("market_state").reset_index(drop=True)
