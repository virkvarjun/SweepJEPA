"""Gated attention-MIL: shapes, attention normalisation, learnability."""

from __future__ import annotations

import torch

from src.track1.mil import GatedAttentionMIL


def test_output_shapes():
    mil = GatedAttentionMIL(input_dim=32)
    bag = torch.randn(4, 7, 32)
    logits, alpha = mil(bag)
    assert logits.shape == (4, 1)
    assert alpha.shape == (4, 7)


def test_attention_is_a_distribution():
    mil = GatedAttentionMIL(input_dim=16)
    _, alpha = mil(torch.randn(3, 5, 16))
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(3), atol=1e-5)
    assert (alpha >= 0).all()


def test_gradients_flow():
    mil = GatedAttentionMIL(input_dim=16)
    logits, _ = mil(torch.randn(2, 4, 16))
    logits.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in mil.parameters())


def test_learns_toy_bag_signal():
    """Loss decreases on a separable bag task (bag-level supervision only)."""
    torch.manual_seed(0)
    dim = 8

    def make_batch(bs=16, n=6):
        labels = torch.randint(0, 2, (bs,)).float()
        bags = torch.randn(bs, n, dim)
        # Positive bags: one instance carries a +signal in channel 0.
        for b in range(bs):
            if labels[b] == 1:
                bags[b, torch.randint(0, n, (1,))] += 3.0 * torch.eye(dim)[0]
        return bags, labels

    mil = GatedAttentionMIL(input_dim=dim, hidden_dim=16)
    opt = torch.optim.Adam(mil.parameters(), lr=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    bags, labels = make_batch()
    first = None
    for step in range(60):
        opt.zero_grad()
        logits, _ = mil(bags)
        loss = loss_fn(logits.squeeze(-1), labels)
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
    assert loss.item() < first
