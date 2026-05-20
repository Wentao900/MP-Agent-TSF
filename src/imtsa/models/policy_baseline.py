from __future__ import annotations

import torch
from torch import nn

from .encoders import PriceEncoder, TextEncoder
from .fusion import GatedFusion
from .memory import MemoryModule
from .reflector import Reflector


class IMTSAPolicy(nn.Module):
    def __init__(self, price_dim: int, text_dim: int, config: dict):
        super().__init__()
        mcfg = config["model"]
        self.hidden_dim = int(mcfg["hidden_dim"])
        self.num_actions = int(mcfg["num_actions"])

        self.price_encoder = PriceEncoder(price_dim, self.hidden_dim)
        self.text_encoder = TextEncoder(text_dim, self.hidden_dim)
        self.fusion = GatedFusion(self.hidden_dim)

        self.use_memory = bool(config["ablation"].get("use_memory", False))
        self.use_reflector = bool(config["ablation"].get("use_reflector", False))

        if self.use_memory:
            reward_norm_scale = float(config.get("train", {}).get("reward_norm_scale", 100.0))
            self.memory = MemoryModule(self.hidden_dim, int(mcfg["memory_dim"]), self.num_actions, reward_norm_scale)

        self.policy_head = nn.Linear(self.hidden_dim, self.num_actions)
        self.pred_head = nn.Linear(self.hidden_dim, 1)

        if self.use_reflector:
            self.reflector = Reflector(self.hidden_dim, self.num_actions, int(mcfg["reflector_k"]))

    def init_memory(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if not self.use_memory:
            return torch.zeros(batch_size, self.hidden_dim, device=device)
        return torch.zeros(batch_size, self.memory.cell.hidden_size, device=device)

    def forward(
        self,
        x_price: torch.Tensor,
        x_text: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        prev_memory: torch.Tensor,
        recent_action_probs: torch.Tensor | None = None,
        recent_rewards: torch.Tensor | None = None,
        state_history: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        h_price = self.price_encoder(x_price)
        h_text = self.text_encoder(x_text)
        z, alpha = self.fusion(h_price, h_text)

        state = z
        next_memory = prev_memory
        if self.use_memory:
            next_memory, mem_proj = self.memory(z, prev_action, prev_reward, prev_memory)
            state = state + mem_proj

        logits = self.policy_head(state)
        delta_logits = torch.zeros_like(logits)
        if self.use_reflector and recent_action_probs is not None and recent_rewards is not None:
            if recent_action_probs.dim() == 2:
                recent_action_probs = recent_action_probs.unsqueeze(1)
            if recent_rewards.dim() == 1:
                recent_rewards = recent_rewards.unsqueeze(1)
            delta_logits = self.reflector(state, recent_action_probs, recent_rewards)
            logits = logits + delta_logits

        return {
            "logits": logits,
            "delta_logits": delta_logits,
            "pred_return": self.pred_head(state).squeeze(-1),
            "alpha": alpha,
            "state": state,
            "next_memory": next_memory,
        }
