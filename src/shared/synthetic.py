"""Deterministic synthetic ultrasound data for development and tests.

Real datasets are gated (Stanford AIMI, ThyroidXL) or large (TUS-REC2024). Until
access clears and downloads land under ``data/``, every training and evaluation
path is exercised against these generators. They are:

* **deterministic** — fully seeded; the same ``seed`` reproduces byte-identical
  tensors, so experiments are reproducible from a config + seed;
* **structured** — cine bags carry patient grouping and the ~17/192 malignant
  imbalance of Stanford AIMI, so patient-level splitting and minority
  oversampling have something real to bite on;
* **learnable** — malignant bags embed a faint label-correlated blob so a MIL
  head's loss actually decreases, letting loss-sanity tests be meaningful
  without leaking the label to individual frames deterministically.

Nothing here imports a real dataset; swap these for the real loaders once the
manifests exist under ``data/``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from src.shared.data import NoduleBag

__all__ = [
    "SyntheticCineDataset",
    "SyntheticStaticDataset",
    "SyntheticTrackedSweepDataset",
    "TrackedSweep",
    "make_cine_bag",
]

# Stanford AIMI Thyroid Cine-clip headline numbers, used to shape the synthetic
# distribution so imbalance-handling code is tested against a realistic ratio.
AIMI_N_NODULES = 192
AIMI_N_MALIGNANT = 17


def _generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def make_cine_bag(
    *,
    label: int,
    patient_id: str,
    nodule_id: str,
    n_frames: int = 16,
    image_size: int = 224,
    seed: int = 0,
    signal_strength: float = 0.35,
) -> NoduleBag:
    """Build a single synthetic cine-clip bag.

    A malignant bag (``label == 1``) has a faint bright blob added to a random
    subset of its frames — a *bag-level* signal that is present in only some
    frames, mirroring the fact that not every frame in a real sweep shows the
    diagnostic view. The label is never a deterministic per-frame feature.
    """
    g = _generator(seed)
    frames = 0.15 * torch.randn(n_frames, 3, image_size, image_size, generator=g)
    frames = frames + 0.5  # centre around mid-grey like a normalised B-mode frame

    if label == 1:
        # Diagnostic view appears in a random subset of frames only.
        n_signal = max(1, int(torch.randint(1, n_frames + 1, (1,), generator=g).item()))
        signal_frames = torch.randperm(n_frames, generator=g)[:n_signal]
        cy = int(torch.randint(image_size // 4, 3 * image_size // 4, (1,), generator=g).item())
        cx = int(torch.randint(image_size // 4, 3 * image_size // 4, (1,), generator=g).item())
        radius = image_size // 8
        ys = torch.arange(image_size).view(-1, 1)
        xs = torch.arange(image_size).view(1, -1)
        blob = torch.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2.0 * radius ** 2))
        for fi in signal_frames.tolist():
            frames[fi] = frames[fi] + signal_strength * blob

    frames = frames.clamp(0.0, 1.0)
    return NoduleBag(
        frames=frames,
        label=int(label),
        patient_id=patient_id,
        nodule_id=nodule_id,
    )


class SyntheticCineDataset(Dataset):
    """Synthetic Stanford-AIMI-shaped cine dataset with patient grouping.

    Patients own one or two nodules; malignancy is assigned to match the target
    prevalence. ``samples`` mirrors the schema of the real loader (a list of
    dicts with ``patient_id`` / ``nodule_id`` / ``label``) so patient-level
    splitting works identically on synthetic and real data.
    """

    def __init__(
        self,
        n_nodules: int = AIMI_N_NODULES,
        n_malignant: int = AIMI_N_MALIGNANT,
        n_frames: int = 16,
        image_size: int = 224,
        max_nodules_per_patient: int = 2,
        seed: int = 0,
    ) -> None:
        if n_malignant > n_nodules:
            raise ValueError("n_malignant cannot exceed n_nodules")
        self.n_frames = n_frames
        self.image_size = image_size
        self.seed = seed

        g = _generator(seed)
        labels = torch.zeros(n_nodules, dtype=torch.long)
        malignant_idx = torch.randperm(n_nodules, generator=g)[:n_malignant]
        labels[malignant_idx] = 1

        # Assign nodules to patients in contiguous groups of 1-2.
        self.samples: list[dict] = []
        patient_counter = 0
        i = 0
        while i < n_nodules:
            group = int(torch.randint(1, max_nodules_per_patient + 1, (1,), generator=g).item())
            group = min(group, n_nodules - i)
            pid = f"synthetic_patient_{patient_counter:04d}"
            for j in range(group):
                nid = f"nodule_{i + j:04d}"
                self.samples.append(
                    {
                        "patient_id": pid,
                        "nodule_id": nid,
                        "label": int(labels[i + j].item()),
                        # per-sample seed keeps __getitem__ deterministic & unique
                        "seed": seed * 100003 + (i + j),
                    }
                )
            patient_counter += 1
            i += group

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> NoduleBag:
        s = self.samples[idx]
        return make_cine_bag(
            label=s["label"],
            patient_id=s["patient_id"],
            nodule_id=s["nodule_id"],
            n_frames=self.n_frames,
            image_size=self.image_size,
            seed=s["seed"],
        )

    @property
    def labels(self) -> list[int]:
        return [s["label"] for s in self.samples]

    @property
    def patient_ids(self) -> list[str]:
        return [s["patient_id"] for s in self.samples]


class SyntheticStaticDataset(Dataset):
    """Synthetic ThyroidXL-shaped static B-mode images (one frame per sample).

    Validates the per-frame backbone + decision head — there is no cine
    aggregation here, matching ThyroidXL's role as an external per-image test.
    """

    def __init__(
        self,
        n_images: int = 512,
        n_malignant: int | None = None,
        image_size: int = 224,
        seed: int = 0,
    ) -> None:
        self.image_size = image_size
        self.seed = seed
        g = _generator(seed)
        if n_malignant is None:
            n_malignant = n_images // 3
        labels = torch.zeros(n_images, dtype=torch.long)
        labels[torch.randperm(n_images, generator=g)[:n_malignant]] = 1
        self.samples: list[dict] = [
            {
                "patient_id": f"xl_patient_{k:05d}",
                "image_id": f"xl_image_{k:05d}",
                "label": int(labels[k].item()),
                "seed": seed * 100003 + k,
            }
            for k in range(n_images)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        bag = make_cine_bag(
            label=s["label"],
            patient_id=s["patient_id"],
            nodule_id=s["image_id"],
            n_frames=1,
            image_size=self.image_size,
            seed=s["seed"],
        )
        return bag.frames[0], s["label"]

    @property
    def labels(self) -> list[int]:
        return [s["label"] for s in self.samples]


@dataclass
class TrackedSweep:
    """A freehand sweep with per-frame ground-truth probe poses.

    Mirrors TUS-REC2024: optically-tracked frames whose relative transforms
    train the pose estimator R_psi.
    """

    frames: torch.Tensor  # (N, 3, H, W)
    poses: torch.Tensor  # (N, 4, 4) absolute SE(3) probe pose per frame
    sweep_id: str


class SyntheticTrackedSweepDataset(Dataset):
    """Synthetic optically-tracked sweeps (TUS-REC2024-shaped).

    Each sweep is a smooth SE(3) trajectory (small, temporally-correlated
    inter-frame motion) with frames that shift coherently with translation, so
    the pose estimator has recoverable image-to-motion structure.
    """

    def __init__(
        self,
        n_sweeps: int = 32,
        n_frames: int = 24,
        image_size: int = 128,
        seed: int = 0,
    ) -> None:
        self.n_sweeps = n_sweeps
        self.n_frames = n_frames
        self.image_size = image_size
        self.seed = seed

    def __len__(self) -> int:
        return self.n_sweeps

    def __getitem__(self, idx: int) -> TrackedSweep:
        g = _generator(self.seed * 100003 + idx)
        n, size = self.n_frames, self.image_size

        # Smooth per-step motion: small translations + small rotations that
        # integrate into a coherent trajectory.
        step_trans = 0.02 * torch.randn(n - 1, 3, generator=g).cumsum(0) / (n ** 0.5)
        step_rot = 0.01 * torch.randn(n - 1, 3, generator=g)

        from src.track2.sweep_reconstruction import compose_sweep_poses

        deltas = torch.cat([step_trans, step_rot], dim=-1).unsqueeze(0)  # (1, N-1, 6)
        poses = compose_sweep_poses(deltas)[0]  # (N, 4, 4)

        # Frames: a textured field translated by the in-plane pose component so
        # image content moves consistently with the tracked motion.
        base = 0.1 * torch.randn(3, size, size, generator=g) + 0.5
        frames = torch.empty(n, 3, size, size)
        for i in range(n):
            shift = poses[i, :2, 3]
            sx = int(torch.round(shift[0] * size).item())
            sy = int(torch.round(shift[1] * size).item())
            frames[i] = torch.roll(base, shifts=(sy, sx), dims=(-2, -1))
        frames = frames.clamp(0.0, 1.0)

        return TrackedSweep(frames=frames, poses=poses, sweep_id=f"sweep_{idx:04d}")
