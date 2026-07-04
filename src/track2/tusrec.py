"""TUS-REC2024 loader: optically-tracked freehand sweeps for training R_psi.

TUS-REC2024 (open, Zenodo) provides freehand forearm sweeps with per-frame
optical-tracker poses. It is the only openly downloadable set, so R_psi is
pretrained here and transferred to thyroid cine.

Expected on-disk layout (fill in once downloaded — see docs/STATUS.md):
    data/tusrec2024/<scan_id>.npz  with arrays
        frames: (N, H, W) or (N, 3, H, W)  uint8/float
        poses:  (N, 4, 4)                   absolute SE(3) tracker poses
Until then, :func:`load_tusrec_or_synthetic` returns the synthetic tracked-sweep
dataset so the whole training path is exercisable offline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.shared.synthetic import SyntheticTrackedSweepDataset, TrackedSweep
from src.track2.sweep_reconstruction import relative_pose


class TUSRECSweepDataset(Dataset):
    """Sweeps read from ``.npz`` scan files (one sweep per file)."""

    def __init__(self, root: Path, image_size: int = 128) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.files = sorted(self.root.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"no .npz scans under {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> TrackedSweep:
        from torchvision.transforms.functional import resize

        data = np.load(self.files[idx])
        frames = torch.as_tensor(np.asarray(data["frames"]), dtype=torch.float32)
        if frames.ndim == 3:  # (N, H, W) -> (N, 3, H, W)
            frames = frames.unsqueeze(1).repeat(1, 3, 1, 1)
        if frames.max() > 1.5:
            frames = frames / 255.0
        frames = resize(frames, [self.image_size, self.image_size], antialias=True)
        poses = torch.as_tensor(np.asarray(data["poses"]), dtype=torch.float32)
        return TrackedSweep(frames=frames, poses=poses, sweep_id=self.files[idx].stem)


class FramePairDataset(Dataset):
    """Flatten sweeps into consecutive frame pairs with their relative transform.

    Yields ``(frame_i, frame_j, T_rel)`` where ``T_rel = T_i^-1 T_j`` — the
    supervised target for the pairwise pose estimator.
    """

    def __init__(self, sweeps: Dataset, stride: int = 1) -> None:
        self.sweeps = sweeps
        self.index: list[tuple[int, int]] = []
        for s in range(len(sweeps)):
            n = sweeps[s].frames.shape[0]
            for i in range(0, n - stride, stride):
                self.index.append((s, i))
        self.stride = stride

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        s, i = self.index[idx]
        sweep = self.sweeps[s]
        j = i + self.stride
        t_rel = relative_pose(sweep.poses[i], sweep.poses[j])
        return sweep.frames[i], sweep.frames[j], t_rel


def load_tusrec_or_synthetic(
    root: str | Path | None = "data/tusrec2024",
    image_size: int = 128,
    synthetic_kwargs: dict | None = None,
) -> Dataset:
    """Return the real TUS-REC dataset if present, else the synthetic fallback."""
    if root is not None:
        p = Path(root)
        if p.exists() and any(p.glob("*.npz")):
            return TUSRECSweepDataset(p, image_size=image_size)
    return SyntheticTrackedSweepDataset(image_size=image_size, **(synthetic_kwargs or {}))
