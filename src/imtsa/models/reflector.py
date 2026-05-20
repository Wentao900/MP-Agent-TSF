from __future__ import annotations

import torch
from torch import nn


class Reflector(nn.Module):
    def __init__(self, hidden_dim: int, num_actions: int, k: int, use_gate: bool = True):
        super().__init__()
        self.k = k
        self.num_actions = num_actions
        self.hist_encoder = nn.GRU(input_size=num_actions + 1, hidden_size=hidden_dim, batch_first=True)
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, num_actions),
            nn.Sigmoid(),
        ) if use_gate else None

    def forward(
        self,
        features: torch.Tensor,
        action_prob_hist: torch.Tensor,
        reward_hist: torch.Tensor,
    ) -> torch.Tensor:
        hist = torch.cat([action_prob_hist, reward_hist.unsqueeze(-1)], dim=-1)
        _, h = self.hist_encoder(hist)
        h_last = h[-1]
        fused = torch.cat([features, h_last], dim=-1)
        delta_logits = self.delta_head(fused)
        if self.gate is not None:
            delta_logits = delta_logits * self.gate(fused)
        return delta_logits
