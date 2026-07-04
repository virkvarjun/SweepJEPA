"""Focal loss: BCE reduction at gamma=0, easy-example down-weighting."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.track1.losses import FocalLoss, focal_loss_with_logits


def test_reduces_to_bce_at_gamma0_alpha_disabled():
    logits = torch.randn(32)
    targets = (torch.rand(32) > 0.5).float()
    focal = focal_loss_with_logits(logits, targets, alpha=-1.0, gamma=0.0)
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    assert torch.allclose(focal, bce, atol=1e-6)


def test_focal_downweights_easy_examples():
    # An easy correct example (confident + right) should contribute far less
    # under gamma=2 than under gamma=0.
    logits = torch.tensor([5.0])  # very confident positive
    targets = torch.tensor([1.0])
    easy_g0 = focal_loss_with_logits(logits, targets, alpha=-1.0, gamma=0.0)
    easy_g2 = focal_loss_with_logits(logits, targets, alpha=-1.0, gamma=2.0)
    assert easy_g2 < 0.1 * easy_g0


def test_nonnegative_and_shape_agnostic():
    logits = torch.randn(8, 1)
    targets = (torch.rand(8, 1) > 0.5).float()
    loss = focal_loss_with_logits(logits, targets)
    assert loss.ndim == 0 and loss.item() >= 0.0


def test_reduction_none_shape():
    logits = torch.randn(10)
    targets = torch.zeros(10)
    out = focal_loss_with_logits(logits, targets, reduction="none")
    assert out.shape == (10,)


def test_module_matches_functional():
    logits = torch.randn(16)
    targets = (torch.rand(16) > 0.5).float()
    mod = FocalLoss(alpha=0.25, gamma=2.0)
    assert torch.allclose(
        mod(logits, targets), focal_loss_with_logits(logits, targets, 0.25, 2.0)
    )
