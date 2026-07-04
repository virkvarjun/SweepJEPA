"""Mask-aware aggregators: padding invariance, variant shapes, factory."""

from __future__ import annotations

import pytest
import torch

from src.track1.mil import (
    GatedAttentionMIL,
    MeanPoolMIL,
    SetTransformerMIL,
    build_aggregator,
)


def _pad(bag: torch.Tensor, n_pad: int):
    """Append ``n_pad`` junk frames and a mask marking them invalid."""
    b, n, d = bag.shape
    junk = 1e3 * torch.randn(b, n_pad, d)
    padded = torch.cat([bag, junk], dim=1)
    mask = torch.zeros(b, n + n_pad, dtype=torch.bool)
    mask[:, :n] = True
    return padded, mask


@pytest.mark.parametrize("cls", [GatedAttentionMIL, MeanPoolMIL, SetTransformerMIL])
def test_padding_is_invariant(cls):
    torch.manual_seed(0)
    mod = cls(input_dim=16).eval()
    bag = torch.randn(2, 5, 16)
    with torch.no_grad():
        base, _ = mod(bag)
        padded, mask = _pad(bag, 4)
        masked, alpha = mod(padded, mask)
    # Padding frames (with huge values) must not change the prediction.
    assert torch.allclose(base, masked, atol=1e-4)
    # No attention mass lands on padded frames.
    assert alpha[:, 5:].abs().max().item() < 1e-5


@pytest.mark.parametrize("cls", [GatedAttentionMIL, MeanPoolMIL, SetTransformerMIL])
def test_shapes_and_attention_normalised(cls):
    mod = cls(input_dim=8).eval()
    bag = torch.randn(3, 6, 8)
    with torch.no_grad():
        logits, alpha = mod(bag)
    assert logits.shape == (3, 1)
    assert alpha.shape == (3, 6)
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(3), atol=1e-4)


def test_factory_builds_all():
    for name in ["gated_attention", "average", "mean", "set_transformer"]:
        mod = build_aggregator(name, input_dim=8, hidden_dim=16, dropout=0.1)
        logits, alpha = mod(torch.randn(2, 4, 8))
        assert logits.shape == (2, 1)


def test_factory_rejects_unknown():
    with pytest.raises(ValueError):
        build_aggregator("nope", input_dim=8)
