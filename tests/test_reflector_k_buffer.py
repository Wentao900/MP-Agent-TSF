from __future__ import annotations

import torch

from imtsa.models.reflector import Reflector


def test_reflector_accepts_k_step_history() -> None:
    bs, hidden_dim, num_actions, k = 4, 8, 3, 5
    refl = Reflector(hidden_dim, num_actions, k)
    features = torch.randn(bs, hidden_dim)
    action_hist = torch.softmax(torch.randn(bs, k, num_actions), dim=-1)
    reward_hist = torch.randn(bs, k)

    out = refl(features, action_hist, reward_hist)
    assert out.shape == (bs, num_actions)
