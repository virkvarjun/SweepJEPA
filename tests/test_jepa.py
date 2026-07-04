"""Pose-conditioned V-JEPA: masking, scalar latent loss, EMA/frozen, ablations.

Pins the feature-space L1 + EMA contract and the ablation toggles (pose-free,
frozen teacher, latent-vs-input) that the Track 2 experiments depend on.
"""

from __future__ import annotations

import torch

from src.track2.jepa import PoseConditionedJEPA, random_tubelet_mask


def _batch(b=4, t=8, d=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randn(b, t, d, generator=g)
    poses = 0.1 * torch.randn(b, t, 6, generator=g)
    mask = random_tubelet_mask(b, t, mask_ratio=0.5, generator=g)
    return tokens, mask, poses


def test_mask_has_visible_and_target():
    mask = random_tubelet_mask(16, 10, mask_ratio=0.75, generator=torch.Generator().manual_seed(0))
    assert mask.shape == (16, 10)
    for row in mask:
        assert row.any() and (~row).any()  # at least one target and one visible


def test_loss_is_nonnegative_scalar():
    jepa = PoseConditionedJEPA(embed_dim=32, depth=2, predictor_depth=1, num_heads=4)
    tokens, mask, poses = _batch(d=32)
    loss = jepa(tokens, mask, poses)
    assert loss.ndim == 0 and loss.item() >= 0.0


def test_target_encoder_frozen():
    jepa = PoseConditionedJEPA(embed_dim=16, num_heads=4)
    assert all(not p.requires_grad for p in jepa.target_encoder.parameters())


def test_ema_updates_target_but_frozen_does_not():
    tokens, mask, poses = _batch(d=16)

    ema = PoseConditionedJEPA(embed_dim=16, depth=2, predictor_depth=1, num_heads=4,
                              teacher_type="ema", ema_tau=0.5)
    with torch.no_grad():
        for p in ema.context_encoder.parameters():
            p.add_(0.5)
    before = torch.cat([p.flatten() for p in ema.target_encoder.parameters()]).clone()
    ema.update_ema()
    after = torch.cat([p.flatten() for p in ema.target_encoder.parameters()])
    assert not torch.allclose(before, after)  # EMA moved the teacher

    frozen = PoseConditionedJEPA(embed_dim=16, depth=2, predictor_depth=1, num_heads=4,
                                 teacher_type="frozen")
    with torch.no_grad():
        for p in frozen.context_encoder.parameters():
            p.add_(0.5)
    b0 = torch.cat([p.flatten() for p in frozen.target_encoder.parameters()]).clone()
    frozen.update_ema()
    b1 = torch.cat([p.flatten() for p in frozen.target_encoder.parameters()])
    assert torch.allclose(b0, b1)  # frozen teacher never moves


def _train_loss_curve(jepa, steps=60, seed=0):
    tokens, mask, poses = _batch(seed=seed, d=jepa.embed_dim)
    opt = torch.optim.Adam([p for p in jepa.parameters() if p.requires_grad], lr=1e-3)
    first = last = None
    for s in range(steps):
        opt.zero_grad()
        loss = jepa(tokens, mask, poses)
        loss.backward()
        opt.step()
        jepa.update_ema()
        if s == 0:
            first = loss.item()
        last = loss.item()
    return first, last


def test_loss_decreases_pose_conditioned():
    jepa = PoseConditionedJEPA(embed_dim=32, depth=2, predictor_depth=2, num_heads=4)
    first, last = _train_loss_curve(jepa)
    assert last < first


def test_pose_free_ablation_runs_and_learns():
    jepa = PoseConditionedJEPA(embed_dim=32, depth=2, predictor_depth=2, num_heads=4,
                               pose_conditioned=False)
    first, last = _train_loss_curve(jepa)
    assert last < first


def test_input_space_reconstruction_baseline():
    jepa = PoseConditionedJEPA(embed_dim=32, depth=2, predictor_depth=2, num_heads=4,
                               target_space="input")
    first, last = _train_loss_curve(jepa)
    assert last < first


def test_pose_conditioning_changes_output():
    jepa = PoseConditionedJEPA(embed_dim=32, depth=2, predictor_depth=1, num_heads=4).eval()
    tokens, mask, _ = _batch(d=32)
    poses_a = torch.zeros(*mask.shape, 6)
    poses_b = torch.ones(*mask.shape, 6)
    with torch.no_grad():
        la = jepa(tokens, mask, poses_a)
        lb = jepa(tokens, mask, poses_b)
    # Different actions -> different predictions -> different loss.
    assert abs(la.item() - lb.item()) > 1e-6
