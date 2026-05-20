from __future__ import annotations

import torch
from torch import nn


class MemoryModule(nn.Module):
    def __init__(self, hidden_dim: int, memory_dim: int, num_actions: int, reward_norm_scale: float = 100.0):
        super().__init__()
        self.num_actions = num_actions
        self.reward_norm_scale = float(reward_norm_scale)
        self.cell = nn.GRUCell(hidden_dim + num_actions + 1, memory_dim)
        self.proj = nn.Linear(memory_dim, hidden_dim)

    def _action_to_one_hot(self, prev_action: torch.Tensor) -> torch.Tensor:
        if prev_action.dim() == 2 and prev_action.shape[-1] == self.num_actions:
            return prev_action
        act_idx = prev_action.squeeze(-1).long().clamp(min=0, max=self.num_actions - 1)
        return torch.nn.functional.one_hot(act_idx, num_classes=self.num_actions).float()

    def forward(
        self,
        z_t: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        prev_memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_oh = self._action_to_one_hot(prev_action)
        reward_norm = torch.tanh(prev_reward / max(self.reward_norm_scale, 1e-6)).unsqueeze(-1)
        x = torch.cat([z_t, action_oh, reward_norm], dim=-1)
        m = self.cell(x, prev_memory)
        return m, self.proj(m)
