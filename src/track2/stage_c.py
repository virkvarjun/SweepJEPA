"""Stage C plumbing: cine clips -> tubelet tokens -> JEPA pretraining -> Stage D.

Turns a cine dataset into pose-annotated tubelet token sequences (using the same
US foundation encoder as Track 1, then R_psi or GT poses), pretrains the
pose-conditioned JEPA on them, and re-encodes clips into geometry-aware
embeddings that plug straight into the Track 1 MIL + conformal head (Stage D).

Matched backbone/compute (#6): the only structural difference from Track 1 is the
tubelet grouping + pose-conditioned predictor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.track1.pipeline import EmbeddingBag
from src.track2.jepa import PoseConditionedJEPA, random_tubelet_mask
from src.track2.pose_estimator import PoseEstimator
from src.track2.sweep_reconstruction import (
    compose_relative_transforms,
    compose_sweep_poses,
    partition_temporal_tubelets,
)


@dataclass
class ClipTubelets:
    """Tubelet tokens + per-tubelet relative pose for one clip."""

    tokens: torch.Tensor  # (T, D)
    pose_deltas: torch.Tensor  # (T, 6) relative pose action per tubelet
    label: int
    patient_id: str
    clip_id: str


def _synthetic_clip_poses(n_frames: int, seed: int, scale: float = 0.03) -> torch.Tensor:
    """Smooth synthetic absolute poses for a clip (dev stand-in for R_psi)."""
    g = torch.Generator().manual_seed(seed)
    deltas = scale * torch.randn(1, n_frames - 1, 6, generator=g)
    return compose_sweep_poses(deltas)[0]


def build_clip_tubelets(
    dataset,
    encoder: torch.nn.Module,
    temporal_size: int = 2,
    device: str = "cpu",
    pose_estimator: PoseEstimator | None = None,
    seed: int = 0,
) -> list[ClipTubelets]:
    """Encode clips to tubelet tokens with per-tubelet relative poses.

    ``pose_estimator`` (R_psi) supplies poses when given; otherwise smooth
    synthetic poses stand in (dev). Labels/patient ids are carried through so the
    same clips can be re-used for supervised Stage D.
    """
    encoder = encoder.to(device).eval()
    out: list[ClipTubelets] = []
    with torch.no_grad():
        for i in range(len(dataset)):
            bag = dataset[i]
            frames = bag.frames.to(device)
            feats = encoder(frames.unsqueeze(0)).squeeze(0)  # (N, D)
            if pose_estimator is not None:
                deltas = pose_estimator.estimate_sweep(frames).cpu()
                poses = compose_relative_transforms(deltas)
            else:
                poses = _synthetic_clip_poses(frames.shape[0], seed=seed * 100003 + i)
            tokens, _, pose_deltas = partition_temporal_tubelets(
                feats.cpu(), poses, temporal_size=temporal_size
            )
            out.append(
                ClipTubelets(
                    tokens=tokens,
                    pose_deltas=pose_deltas,
                    label=int(bag.label),
                    patient_id=bag.patient_id,
                    clip_id=bag.nodule_id,
                )
            )
    return out


def stack_tubelets(clips: list[ClipTubelets]):
    """Stack equal-length clips into ``(Nc, T, D)`` tokens and ``(Nc, T, 6)`` poses."""
    lengths = {c.tokens.shape[0] for c in clips}
    if len(lengths) != 1:
        raise ValueError(
            f"clips have varying tubelet counts {sorted(lengths)}; crop/pad first "
            "(variable-length JEPA batching is tracked in docs/STATUS.md)"
        )
    tokens = torch.stack([c.tokens for c in clips], dim=0)
    poses = torch.stack([c.pose_deltas for c in clips], dim=0)
    return tokens, poses


def pretrain_jepa(
    tokens: torch.Tensor,
    poses: torch.Tensor,
    jepa: PoseConditionedJEPA,
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-3,
    mask_ratio: float = 0.75,
    seed: int = 0,
    shuffle_tubelets: bool = False,
    device: str = "cpu",
) -> list[float]:
    """Pretrain the JEPA on stacked tubelet tokens; return the per-epoch loss.

    ``shuffle_tubelets`` permutes the temporal order (tokens + poses) each batch,
    destroying sweep order — the "true order vs shuffled" ablation.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    jepa = jepa.to(device)
    tokens = tokens.to(device)
    poses = poses.to(device)
    n, t, _ = tokens.shape
    opt = torch.optim.Adam([p for p in jepa.parameters() if p.requires_grad], lr=lr)
    gen = torch.Generator().manual_seed(seed)

    history: list[float] = []
    jepa.train()
    for _ in range(epochs):
        order = torch.randperm(n, generator=gen)
        total, count = 0.0, 0
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            tok = tokens[idx]
            pos = poses[idx]
            if shuffle_tubelets:
                perm = torch.randperm(t, generator=gen)
                tok = tok[:, perm]
                pos = pos[:, perm]
            mask = random_tubelet_mask(tok.shape[0], t, mask_ratio, generator=gen).to(device)
            opt.zero_grad()
            loss = jepa(tok, mask, pos)
            loss.backward()
            opt.step()
            jepa.update_ema()
            total += loss.item() * tok.shape[0]
            count += tok.shape[0]
        history.append(total / max(count, 1))
    return history


@torch.no_grad()
def jepa_embedding_bags(
    clips: list[ClipTubelets],
    jepa: PoseConditionedJEPA,
    device: str = "cpu",
) -> list[EmbeddingBag]:
    """Re-encode clips with the pretrained context encoder -> geometry-aware bags.

    The resulting ``EmbeddingBag`` (one embedding per tubelet) is the Track 2
    representation consumed unchanged by the Track 1 head (Stage D).
    """
    jepa = jepa.to(device).eval()
    bags: list[EmbeddingBag] = []
    for c in clips:
        tok = c.tokens.unsqueeze(0).to(device)  # (1, T, D)
        t = tok.shape[1]
        no_mask = torch.zeros(1, t, dtype=torch.bool, device=device)
        ctx = jepa.encode_context(tok, no_mask).squeeze(0).cpu()  # (T, D)
        bags.append(
            EmbeddingBag(emb=ctx, label=c.label, patient_id=c.patient_id, nodule_id=c.clip_id)
        )
    return bags
