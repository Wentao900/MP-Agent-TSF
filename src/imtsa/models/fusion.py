from __future__ import annotations

import torch
from torch import nn


class GatedFusion(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, h_price: torch.Tensor, h_text: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = self.gate(torch.cat([h_price, h_text], dim=-1))
        z = alpha * h_price + (1.0 - alpha) * h_text
        return z, alpha
