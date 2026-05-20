from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from imtsa.backtest.costs import calc_trade_cost
from imtsa.backtest.metrics import build_risk_return_explain_state_table, compute_metrics, compute_metrics_by_regime
from imtsa.data.split import label_market_states
from imtsa.explain.attribution import explain_by_regime, explain_step_frame, summarize_explain, temporal_contribution
from imtsa.explain.counterfactual import batch_faithfulness_stability, perturb_high_low
from imtsa.models.policy_baseline import IMTSAPolicy
from imtsa.utils import dump_json


def run_backtest(test_bundle: dict, test_df: pd.DataFrame, config: dict, model_path: Path, out_dir: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IMTSAPolicy(test_bundle["x_price"].shape[-1], test_bundle["x_text"].shape[-1], config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    initial_cash = float(config["backtest"]["initial_cash"])
    fee_rate = float(config["backtest"]["fee_rate"])
    slippage_rate = float(config["backtest"]["slippage_rate"])

    cash = initial_cash
    position = 0.0
    equity = []
    trade_rows = []

    x_price = torch.from_numpy(test_bundle["x_price"]).to(device)
    x_text = torch.from_numpy(test_bundle["x_text"]).to(device)

    prev_action = torch.zeros(1, model.num_actions, device=device)
    prev_reward = torch.zeros(1, device=device)
    memory = model.init_memory(1, device)
    k_hist = int(model.reflector.k) if getattr(model, "use_reflector", False) else 1
    action_hist = torch.full((1, k_hist, model.num_actions), 1.0 / model.num_actions, device=device)
    reward_hist = torch.zeros(1, k_hist, device=device)

    explain_alpha = []
    explain_probs = []
    step_states = []

    for i in range(len(x_price)):
        state_prev_action = prev_action.clone()
        state_prev_reward = prev_reward.clone()
        state_memory = memory.clone()
        state_action_hist = action_hist.clone()
        state_reward_hist = reward_hist.clone()

        with torch.no_grad():
            out = model(x_price[i : i + 1], x_text[i : i + 1], prev_action, prev_reward, memory, action_hist, reward_hist)
            probs = torch.softmax(out["logits"], dim=-1)
            action = int(torch.argmax(probs, dim=-1).item())
            memory = out["next_memory"]

        explain_alpha.append(out["alpha"].detach().cpu().numpy()[0])
        explain_probs.append(probs.detach().cpu().numpy()[0])
        step_states.append({
            "prev_action": state_prev_action,
            "prev_reward": state_prev_reward,
            "memory": state_memory,
            "action_hist": state_action_hist,
            "reward_hist": state_reward_hist,
        })

        target_pos = 1.0 if action == 0 else (-1.0 if action == 1 else 0.0)
        delta_pos = target_pos - position

        row_idx = int(test_bundle["row_index"][i])
        close = float(test_df.iloc[row_idx]["close"])
        nxt = float(test_df.iloc[min(row_idx + 1, len(test_df) - 1)]["close"])

        cost = calc_trade_cost(delta_pos, close, fee_rate, slippage_rate)
        pnl = position * (nxt - close) - cost

        cash += pnl
        position = target_pos
        eq = cash + position * close
        equity.append(eq)

        prev_action = probs.detach()
        prev_reward = torch.tensor([pnl], device=device)
        action_hist = torch.cat([action_hist[:, 1:, :], probs.detach().unsqueeze(1)], dim=1)
        reward_hist = torch.cat([reward_hist[:, 1:], prev_reward.unsqueeze(1)], dim=1)

        trade_rows.append({
            "step": i,
            "action": action,
            "position": position,
            "abs_delta_pos": abs(delta_pos),
            "price": close,
            "pnl": pnl,
            "cost": cost,
            "equity": eq,
            "row_index": row_idx,
        })

    trades = pd.DataFrame(trade_rows)
    equity_curve = pd.Series(equity)
    metrics = compute_metrics(equity_curve, trades)

    x_np = test_bundle["x_price"].astype(np.float32)
    alpha_np = np.asarray(explain_alpha)
    probs_np = np.asarray(explain_probs)

    temporal = temporal_contribution(x_np)
    x_high, x_low = perturb_high_low(temporal, x_np, ratio=float(config.get("train", {}).get("exp_perturb_ratio", 0.15)))

    probs_high = []
    probs_low = []
    with torch.no_grad():
        for i in range(len(x_price)):
            s = step_states[i]
            hp = model(
                torch.from_numpy(x_high[i : i + 1]).to(device),
                x_text[i : i + 1],
                s["prev_action"],
                s["prev_reward"],
                s["memory"],
                s["action_hist"],
                s["reward_hist"],
            )
            lp = model(
                torch.from_numpy(x_low[i : i + 1]).to(device),
                x_text[i : i + 1],
                s["prev_action"],
                s["prev_reward"],
                s["memory"],
                s["action_hist"],
                s["reward_hist"],
            )
            probs_high.append(torch.softmax(hp["logits"], dim=-1).cpu().numpy()[0])
            probs_low.append(torch.softmax(lp["logits"], dim=-1).cpu().numpy()[0])

    probs_high = np.asarray(probs_high)
    probs_low = np.asarray(probs_low)
    faith_step, stab_step = batch_faithfulness_stability(probs_np, probs_high, probs_low, temporal)

    full_states = label_market_states(test_df, config)
    market_states = full_states.iloc[test_bundle["row_index"]].to_numpy()

    explain_step = explain_step_frame(
        steps=np.arange(len(trades)),
        market_states=market_states,
        alpha=alpha_np,
        probs=probs_np,
        temporal_weights=temporal,
        faithfulness=faith_step,
        stability=stab_step,
    )
    explain_summary = summarize_explain(explain_step)

    trades = trades.merge(
        pd.DataFrame({
            "step": np.arange(len(trades)),
            "market_state": market_states,
        }),
        on="step",
        how="left",
    )
    metrics_by_regime = compute_metrics_by_regime(trades)
    explain_regime = explain_by_regime(explain_step)
    state_table = build_risk_return_explain_state_table(metrics_by_regime, explain_regime)

    tradeoff = pd.DataFrame([
        {
            **metrics,
            "faithfulness_mean": explain_summary["faithfulness_mean"],
            "stability_mean": explain_summary["stability_mean"],
            "alpha_price_mean": explain_summary["alpha_price_mean"],
            "alpha_text_mean": explain_summary["alpha_text_mean"],
        }
    ])

    trades.to_csv(out_dir / "trades.csv", index=False)
    dump_json(out_dir / "metrics.json", metrics)

    explain_step.to_csv(out_dir / "explain_step.csv", index=False)
    dump_json(out_dir / "explain_summary.json", explain_summary)
    tradeoff.to_csv(out_dir / "tradeoff_summary.csv", index=False)

    metrics_by_regime.to_csv(out_dir / "metrics_by_regime.csv", index=False)
    explain_regime.to_csv(out_dir / "explain_by_regime.csv", index=False)
    state_table.to_csv(out_dir / "risk_return_explain_state_table.csv", index=False)

    return metrics
