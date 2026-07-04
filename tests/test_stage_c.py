"""Stage C/D: tubelet building, JEPA pretraining, geometry-aware Track 1 head."""

from __future__ import annotations

import numpy as np

from src.shared.synthetic import SyntheticCineDataset
from src.track1.encoder import USEncoder
from src.track1.pipeline import DecisionConfig, TrainConfig, cross_validate
from src.track2.jepa import PoseConditionedJEPA
from src.track2.stage_c import (
    build_clip_tubelets,
    jepa_embedding_bags,
    pretrain_jepa,
    stack_tubelets,
)


def _clips(n=12, n_frames=8, dim=1280, seed=0):
    ds = SyntheticCineDataset(
        n_nodules=n, n_malignant=max(2, n // 3), n_frames=n_frames, image_size=32, seed=seed
    )
    enc = USEncoder(name="imagenet_cnn", adaptation="frozen")
    return build_clip_tubelets(ds, enc, temporal_size=2, device="cpu"), enc.embed_dim


def test_build_clip_tubelets_shapes():
    clips, dim = _clips(n=6, n_frames=8)
    assert len(clips) == 6
    # 8 frames / temporal_size 2 -> 4 tubelets.
    assert clips[0].tokens.shape == (4, dim)
    assert clips[0].pose_deltas.shape == (4, 6)


def test_stack_requires_equal_lengths():
    clips, _ = _clips(n=4, n_frames=8)
    tokens, poses = stack_tubelets(clips)
    assert tokens.shape[0] == 4
    assert poses.shape[-1] == 6


def test_pretrain_jepa_loss_decreases():
    clips, dim = _clips(n=12, n_frames=8)
    tokens, poses = stack_tubelets(clips)
    jepa = PoseConditionedJEPA(embed_dim=dim, depth=2, predictor_depth=1, num_heads=4)
    history = pretrain_jepa(tokens, poses, jepa, epochs=15, batch_size=4, lr=1e-3, seed=0)
    assert history[-1] < history[0]


def test_geometry_aware_bags_feed_track1_head():
    clips, dim = _clips(n=20, n_frames=8, seed=1)
    tokens, poses = stack_tubelets(clips)
    jepa = PoseConditionedJEPA(embed_dim=dim, depth=2, predictor_depth=1, num_heads=4)
    pretrain_jepa(tokens, poses, jepa, epochs=5, batch_size=4, seed=1)

    bags = jepa_embedding_bags(clips, jepa)
    assert len(bags) == len(clips)
    assert bags[0].emb.shape == (4, dim)  # (tubelets, dim)
    # The geometry-aware bags run through the unchanged Track 1 CV pipeline.
    res = cross_validate(
        bags,
        TrainConfig(aggregator="gated_attention", epochs=5, lr=1e-2, hidden_dim=64, seed=1),
        DecisionConfig(type="rcps", sensitivity_floor=0.8, delta=0.1, calib_fraction=0.4),
        n_splits=5, seed=1,
    )
    assert not np.isnan(res.oof_scores).any()
    assert 0.0 <= res.auc.auc <= 1.0


def test_shuffle_tubelets_option_runs():
    clips, dim = _clips(n=8, n_frames=8)
    tokens, poses = stack_tubelets(clips)
    jepa = PoseConditionedJEPA(embed_dim=dim, depth=2, predictor_depth=1, num_heads=4)
    h = pretrain_jepa(tokens, poses, jepa, epochs=4, batch_size=4, shuffle_tubelets=True, seed=0)
    assert len(h) == 4
