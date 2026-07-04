"""Losses for bag-level malignancy supervision.

Focal loss down-weights easy (usually benign) bags so the 17 malignant nodules
are not drowned out by the 175 benign ones. It composes with minority
oversampling: oversampling balances *which* bags are seen, focal loss balances
*how much each contributes* once seen.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Binary focal loss (Lin et al., 2017) on raw logits.

    Args:
        logits: ``(B,)`` or ``(B, 1)`` raw scores.
        targets: same shape, float in {0, 1}.
        alpha: class-balancing weight for the positive class in ``[0, 1]``; set
            ``alpha < 0`` to disable alpha-weighting (then ``gamma=0`` is exactly
            BCE-with-logits).
        gamma: focusing parameter; ``0`` recovers (weighted) cross-entropy.
        reduction: ``"mean"`` | ``"sum"`` | ``"none"``.
    """
    logits = logits.reshape(-1)
    targets = targets.reshape(-1).to(logits.dtype)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    loss = ce * (1.0 - p_t).pow(gamma)
    if alpha >= 0.0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"unknown reduction {reduction!r}")


class FocalLoss(nn.Module):
    """Module wrapper around :func:`focal_loss_with_logits`."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return focal_loss_with_logits(
            logits, targets, self.alpha, self.gamma, self.reduction
        )
