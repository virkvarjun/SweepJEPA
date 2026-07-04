"""Track 1 CV pipeline: embedding caching + patient-level CV + conformal head."""

from __future__ import annotations

import numpy as np
import torch

from src.shared.synthetic import SyntheticCineDataset
from src.track1.encoder import USEncoder
from src.track1.pipeline import (
    DecisionConfig,
    EmbeddingBag,
    TrainConfig,
    cross_validate,
    precompute_embeddings,
    score_bags,
    train_aggregator,
)


def _synthetic_embedding_bags(n_patients=30, dim=16, seed=0):
    """Patient-grouped embedding bags with a generalizable bag-level signal.

    Malignant bags carry a consistent offset in channel 0 across the bag, so the
    signal *generalizes* to held-out patients (frame-level attention learnability
    is covered in-sample by ``test_mil``). This keeps the CV test about the
    pipeline, not about whether one strong frame memorizes.
    """
    torch.manual_seed(seed)
    ds = SyntheticCineDataset(
        n_nodules=n_patients + 12, n_malignant=15, n_frames=1, image_size=8, seed=seed
    )
    bags = []
    g = torch.Generator().manual_seed(seed)
    for s in ds.samples:
        n = 6
        emb = 0.5 * torch.randn(n, dim, generator=g)
        if s["label"] == 1:
            emb[:, 0] += 2.5  # consistent bag-level cue
        bags.append(
            EmbeddingBag(emb=emb, label=s["label"], patient_id=s["patient_id"], nodule_id=s["nodule_id"])
        )
    return bags


def test_cross_validate_covers_all_and_separates():
    bags = _synthetic_embedding_bags()
    train_cfg = TrainConfig(aggregator="gated_attention", epochs=25, lr=1e-2, seed=0)
    dec_cfg = DecisionConfig(type="rcps", sensitivity_floor=0.8, delta=0.1, calib_fraction=0.4)
    res = cross_validate(bags, train_cfg, dec_cfg, n_splits=5, seed=0)

    # Every nodule got an out-of-fold score and action (full val coverage).
    assert not np.isnan(res.oof_scores).any()
    assert all(a is not None for a in res.oof_actions)
    assert len(res.attention) == len(bags)
    # The signal is learnable, so OOF AUC should clear chance comfortably.
    assert res.auc.auc > 0.75
    assert res.auc.ci_low <= res.auc.auc <= res.auc.ci_high
    assert len(res.fold_summaries) == 5


def test_cross_validate_youden_decision():
    bags = _synthetic_embedding_bags(seed=1)
    train_cfg = TrainConfig(epochs=15, lr=1e-2, seed=1)
    dec_cfg = DecisionConfig(type="youden", sensitivity_weight=2.0)
    res = cross_validate(bags, train_cfg, dec_cfg, n_splits=5, seed=1)
    assert not np.isnan(res.oof_scores).any()
    # Youden emits only biopsy / no_biopsy (never defer).
    assert set(res.oof_actions) <= {"biopsy", "no_biopsy"}


def test_train_and_score_roundtrip():
    bags = _synthetic_embedding_bags(seed=2)
    model = train_aggregator(bags, input_dim=16, cfg=TrainConfig(epochs=20, lr=1e-2, seed=2))
    scores, attn = score_bags(model, bags)
    assert scores.shape == (len(bags),)
    assert len(attn) == len(bags)
    labels = np.array([b.label for b in bags])
    # Trained model ranks positives above negatives on the training set.
    assert scores[labels == 1].mean() > scores[labels == 0].mean()


def test_precompute_embeddings_with_real_encoder():
    ds = SyntheticCineDataset(n_nodules=4, n_malignant=2, n_frames=3, image_size=32, seed=0)
    enc = USEncoder(name="imagenet_cnn", adaptation="frozen")
    bags = precompute_embeddings(ds, enc, device="cpu", frame_batch=8)
    assert len(bags) == 4
    assert bags[0].emb.shape == (3, enc.embed_dim)
    assert bags[0].patient_id == ds.samples[0]["patient_id"]
