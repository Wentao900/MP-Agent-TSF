from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from imtsa.models.policy_baseline import IMTSAPolicy
from imtsa.train.losses import (
    action_loss,
    faithfulness_loss,
    prediction_loss,
    reflector_correction_loss,
    smoothness_loss,
    stability_loss,
)
from imtsa.utils import dump_json, ensure_dir


@dataclass
class TrainArtifacts:
    model_path: Path
    metrics_path: Path


def _to_loader(bundle: dict[str, np.ndarray], batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(bundle["x_price"]),
        torch.from_numpy(bundle["x_text"]),
        torch.from_numpy(bundle["y_action"]),
        torch.from_numpy(bundle["y_return"]),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _onehot_action(action_idx: torch.Tensor, num_actions: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(action_idx.long(), num_classes=num_actions).float()


def _perturb_by_importance(x: torch.Tensor, contrib: torch.Tensor, ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    bs, seq_len, dim = x.shape
    topk = max(1, int(seq_len * ratio))
    hi = torch.topk(contrib, k=topk, dim=1).indices
    lo = torch.topk(-contrib, k=topk, dim=1).indices

    x_high = x.clone()
    x_low = x.clone()
    for i in range(bs):
        x_high[i, hi[i], :] = 0.0
        x_low[i, lo[i], :] = 0.0
    return x_high, x_low


def _run_seq(
    model: IMTSAPolicy,
    x_price: torch.Tensor,
    x_text: torch.Tensor,
    y_action: torch.Tensor,
    y_return: torch.Tensor,
    tcfg: dict[str, Any],
    use_explain_loss: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, float], dict[str, torch.Tensor]]:
    device = x_price.device
    bs = x_price.shape[0]
    seq_unroll = int(tcfg.get("sequence_len", 1))

    prev_action = torch.zeros(bs, model.num_actions, device=device)
    prev_reward = torch.zeros(bs, device=device)
    prev_memory = model.init_memory(bs, device)

    k_hist = int(model.reflector.k) if getattr(model, "use_reflector", False) else 1
    act_hist = torch.full((bs, k_hist, model.num_actions), 1.0 / model.num_actions, device=device)
    rew_hist = torch.zeros(bs, k_hist, device=device)

    out = None
    probs = None
    alpha_prev = None
    reflect_losses = []
    faith_losses = []
    stab_losses = []

    for _ in range(seq_unroll):
        out = model(x_price, x_text, prev_action, prev_reward, prev_memory, act_hist, rew_hist)
        probs = torch.softmax(out["logits"], dim=-1)

        reward_proxy = out["pred_return"].detach()
        prev_action = probs.detach()
        prev_reward = reward_proxy
        prev_memory = out["next_memory"]

        act_hist = torch.cat([act_hist[:, 1:, :], probs.detach().unsqueeze(1)], dim=1)
        rew_hist = torch.cat([rew_hist[:, 1:], reward_proxy.unsqueeze(1)], dim=1)

        if getattr(model, "use_reflector", False):
            reflect_losses.append(reflector_correction_loss(out["delta_logits"], y_action))

        if use_explain_loss:
            contrib = x_price.detach().abs().mean(dim=2)
            perturb_ratio = float(tcfg.get("exp_perturb_ratio", 0.15))
            x_high, x_low = _perturb_by_importance(x_price, contrib, perturb_ratio)

            out_high = model(x_high, x_text, prev_action, prev_reward, prev_memory, act_hist, rew_hist)
            out_low = model(x_low, x_text, prev_action, prev_reward, prev_memory, act_hist, rew_hist)
            p_main = probs.max(dim=-1).values
            p_high = torch.softmax(out_high["logits"], dim=-1).max(dim=-1).values
            p_low = torch.softmax(out_low["logits"], dim=-1).max(dim=-1).values
            delta_high = torch.relu(p_main - p_high)
            delta_low = torch.relu(p_main - p_low)
            faith_losses.append(faithfulness_loss(delta_high, delta_low))

            if alpha_prev is not None:
                stab_losses.append(stability_loss(out["alpha"], alpha_prev))
            alpha_prev = out["alpha"].detach()

    assert out is not None and probs is not None

    l_act = action_loss(out["logits"], y_action)
    l_pred = prediction_loss(out["pred_return"], y_return)
    prev_probs = torch.full_like(probs, 1.0 / probs.shape[-1])
    l_smooth = smoothness_loss(probs, prev_probs)

    if use_explain_loss:
        l_faith = torch.stack(faith_losses).mean() if faith_losses else torch.tensor(0.0, device=device)
        l_stab = torch.stack(stab_losses).mean() if stab_losses else torch.tensor(0.0, device=device)
        l_exp = 0.5 * (l_faith + l_stab)
    else:
        l_exp = torch.tensor(0.0, device=device)
        l_faith = torch.tensor(0.0, device=device)
        l_stab = torch.tensor(0.0, device=device)

    l_reflect = torch.stack(reflect_losses).mean() if reflect_losses else torch.tensor(0.0, device=device)

    losses = {
        "l_act": l_act,
        "l_pred": l_pred,
        "l_smooth": l_smooth,
        "l_exp": l_exp,
        "l_reflect": l_reflect,
        "l_faith": l_faith,
        "l_stab": l_stab,
    }
    scalars = {k: float(v.detach().item()) for k, v in losses.items()}
    state = {
        "prev_action": prev_action,
        "prev_reward": prev_reward,
        "prev_memory": prev_memory,
        "act_hist": act_hist,
        "rew_hist": rew_hist,
    }
    return losses, scalars, state


def train_model(train_bundle: dict[str, np.ndarray], val_bundle: dict[str, np.ndarray], config: dict, out_dir: Path) -> TrainArtifacts:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    price_dim = train_bundle["x_price"].shape[-1]
    text_dim = train_bundle["x_text"].shape[-1]

    model = IMTSAPolicy(price_dim, text_dim, config).to(device)
    tcfg = config["train"]
    optimizer = optim.AdamW(model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))

    train_loader = _to_loader(train_bundle, int(tcfg["batch_size"]), True)
    val_loader = _to_loader(val_bundle, int(tcfg["batch_size"]), False)

    best_val = float("inf")
    history: list[dict[str, float]] = []

    use_explain_loss = bool(config.get("ablation", {}).get("use_explain_loss", False))

    for epoch in range(int(tcfg["epochs"])):
        model.train()
        train_losses = []
        for x_price, x_text, y_action, y_return in train_loader:
            x_price, x_text = x_price.to(device), x_text.to(device)
            y_action, y_return = y_action.to(device), y_return.to(device)

            losses, scalars, _ = _run_seq(model, x_price, x_text, y_action, y_return, tcfg, use_explain_loss)
            loss = (
                float(tcfg["lambda_act"]) * losses["l_act"]
                + float(tcfg["lambda_pred"]) * losses["l_pred"]
                + float(tcfg["lambda_smooth"]) * losses["l_smooth"]
                + float(tcfg["lambda_exp"]) * losses["l_exp"]
                + float(tcfg.get("lambda_reflect_corr", 0.0)) * losses["l_reflect"]
                + float(tcfg.get("lambda_exp_faith", 0.0)) * losses["l_faith"]
                + float(tcfg.get("lambda_exp_stability", 0.0)) * losses["l_stab"]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_price, x_text, y_action, y_return in val_loader:
                x_price, x_text = x_price.to(device), x_text.to(device)
                y_action, y_return = y_action.to(device), y_return.to(device)

                losses, _, _ = _run_seq(model, x_price, x_text, y_action, y_return, tcfg, use_explain_loss=False)
                vloss = losses["l_act"] + losses["l_pred"]
                val_losses.append(float(vloss.item()))

        epoch_row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": float(np.mean(val_losses)),
            "sequence_len": float(tcfg.get("sequence_len", 1)),
        }
        history.append(epoch_row)
        if epoch_row["val_loss"] < best_val:
            best_val = epoch_row["val_loss"]
            ensure_dir(out_dir)
            torch.save(model.state_dict(), out_dir / "model.pt")

    metrics = {"best_val_loss": best_val, "history": history}
    metrics_path = out_dir / "train_metrics.json"
    dump_json(metrics_path, metrics)
    return TrainArtifacts(model_path=out_dir / "model.pt", metrics_path=metrics_path)
