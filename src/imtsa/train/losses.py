from __future__ import annotations

import torch
import torch.nn.functional as F


def action_loss(logits: torch.Tensor, y_action: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, y_action)


def prediction_loss(pred_return: torch.Tensor, y_return: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_return, y_return)


def smoothness_loss(curr_probs: torch.Tensor, prev_probs: torch.Tensor) -> torch.Tensor:
    return F.kl_div(curr_probs.log(), prev_probs, reduction="batchmean")


def explainability_loss(alpha: torch.Tensor, high_drop: torch.Tensor, low_drop: torch.Tensor, margin: float = 0.05) -> torch.Tensor:
    _ = alpha
    return torch.relu(margin - (high_drop - low_drop)).mean()


def reflector_correction_loss(delta_logits: torch.Tensor, target_action: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(delta_logits, target_action)


def faithfulness_loss(high_drop: torch.Tensor, low_drop: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    return torch.relu(margin - (high_drop - low_drop)).mean()


def stability_loss(alpha_curr: torch.Tensor, alpha_prev: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(alpha_curr - alpha_prev))
