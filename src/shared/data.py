"""Dataset interfaces for Stanford AIMI, ThyroidXL, and unlabeled US corpora.

The cine dataset is *bag-structured*: one ordered clip of frames carries a
single nodule-level label (non-negotiable #2 — the label is never broadcast to
individual frames). Bags have variable frame counts, so batching pads to the
longest bag in the batch and returns a mask; downstream aggregation must respect
the mask so padding frames never contribute to attention or pooling.
"""

from __future__ import annotations

import csv
import glob
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset, WeightedRandomSampler

# ImageNet statistics — the default normalisation for timm ViT/CNN backbones and
# for most released US foundation-model weights.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class NoduleBag:
    """A cine-clip bag: ordered frames with a single nodule-level label."""

    frames: torch.Tensor  # (N, 3, H, W)
    label: int
    patient_id: str
    nodule_id: str


@dataclass
class BagBatch:
    """A padded batch of cine bags with a validity mask.

    ``frames`` is padded to the longest bag; ``mask[b, i]`` is True where frame
    ``i`` of bag ``b`` is real (not padding). ``lengths`` gives each bag's true
    frame count.
    """

    frames: torch.Tensor  # (B, N_max, 3, H, W)
    labels: torch.Tensor  # (B,)
    mask: torch.Tensor  # (B, N_max) bool
    lengths: torch.Tensor  # (B,)
    patient_ids: list[str]
    nodule_ids: list[str]

    def to(self, device) -> BagBatch:
        return BagBatch(
            frames=self.frames.to(device),
            labels=self.labels.to(device),
            mask=self.mask.to(device),
            lengths=self.lengths.to(device),
            patient_ids=self.patient_ids,
            nodule_ids=self.nodule_ids,
        )


def _load_frame(
    path: Path,
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    """Load one frame -> normalised (3, image_size, image_size) float tensor."""
    from torchvision.io import ImageReadMode, read_image
    from torchvision.transforms.functional import normalize, resize

    img = read_image(str(path), mode=ImageReadMode.RGB).float() / 255.0
    img = resize(img, [image_size, image_size], antialias=True)
    return normalize(img, list(mean), list(std))


class ThyroidCineDataset(Dataset):
    """Patient-level cine nodule bags from Stanford AIMI Thyroid Cine-clip.

    Build from a manifest via :meth:`from_manifest`. Each manifest row is one
    nodule: ``nodule_id, patient_id, label, ti_rads`` plus a way to enumerate its
    frames (``frame_dir`` directory or ``frame_glob`` pattern). Frames are loaded
    lazily in :meth:`__getitem__`.
    """

    def __init__(
        self,
        root: Path,
        split: str = "train",
        image_size: int = 224,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.transform = transform
        self.samples: list[dict] = []

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_manifest(
        cls,
        manifest_path: Path,
        root: Path,
        split: str = "train",
        image_size: int = 224,
        **kwargs,
    ) -> ThyroidCineDataset:
        """Parse a CSV manifest into a dataset.

        Required columns: ``nodule_id``, ``patient_id``, ``label``. Optional:
        ``ti_rads`` and one of ``frame_dir`` (a directory of frames) or
        ``frame_glob`` (a glob pattern, relative to ``root``).
        """
        ds = cls(root=root, split=split, image_size=image_size, **kwargs)
        manifest_path = Path(manifest_path)
        with manifest_path.open(newline="") as f:
            reader = csv.DictReader(f)
            required = {"nodule_id", "patient_id", "label"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"manifest missing columns: {sorted(missing)}")
            for row in reader:
                ds.samples.append(
                    {
                        "nodule_id": row["nodule_id"],
                        "patient_id": row["patient_id"],
                        "label": int(row["label"]),
                        "ti_rads": row.get("ti_rads") or None,
                        "frame_dir": row.get("frame_dir") or None,
                        "frame_glob": row.get("frame_glob") or None,
                    }
                )
        ds._resolve_frame_paths()
        return ds

    def _resolve_frame_paths(self) -> None:
        for s in self.samples:
            if s.get("frame_paths"):
                continue
            if s.get("frame_dir"):
                d = self.root / s["frame_dir"]
                paths = sorted(
                    p for p in d.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                )
            elif s.get("frame_glob"):
                paths = sorted(Path(p) for p in glob.glob(str(self.root / s["frame_glob"])))
            else:
                raise ValueError(
                    f"nodule {s['nodule_id']} has neither frame_dir nor frame_glob"
                )
            if not paths:
                raise ValueError(f"no frames found for nodule {s['nodule_id']}")
            s["frame_paths"] = [str(p) for p in paths]

    # ------------------------------------------------------------------ #
    # Dataset protocol
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> NoduleBag:
        s = self.samples[idx]
        frames = torch.stack(
            [
                _load_frame(Path(p), self.image_size, self.mean, self.std)
                for p in s["frame_paths"]
            ]
        )
        if self.transform is not None:
            frames = self.transform(frames)
        return NoduleBag(
            frames=frames,
            label=int(s["label"]),
            patient_id=s["patient_id"],
            nodule_id=s["nodule_id"],
        )

    def patient_ids(self) -> Iterator[str]:
        seen: set = set()
        for sample in self.samples:
            pid = sample["patient_id"]
            if pid not in seen:
                seen.add(pid)
                yield pid

    @property
    def labels(self) -> list[int]:
        return [int(s["label"]) for s in self.samples]

    @property
    def sample_patient_ids(self) -> list[str]:
        return [s["patient_id"] for s in self.samples]


def collate_bags(bags: Sequence[NoduleBag]) -> BagBatch:
    """Collate variable-length cine bags into a padded batch with a mask.

    Frames are padded with zeros to the longest bag; ``mask`` marks the real
    frames so masked aggregation ignores the padding.
    """
    lengths = torch.tensor([b.frames.shape[0] for b in bags], dtype=torch.long)
    n_max = int(lengths.max().item())
    b0 = bags[0].frames
    frames = b0.new_zeros((len(bags), n_max, *b0.shape[1:]))
    mask = torch.zeros((len(bags), n_max), dtype=torch.bool)
    for i, bag in enumerate(bags):
        n = bag.frames.shape[0]
        frames[i, :n] = bag.frames
        mask[i, :n] = True
    return BagBatch(
        frames=frames,
        labels=torch.tensor([b.label for b in bags], dtype=torch.long),
        mask=mask,
        lengths=lengths,
        patient_ids=[b.patient_id for b in bags],
        nodule_ids=[b.nodule_id for b in bags],
    )


def make_minority_sampler(
    labels: Sequence[int], indices: Sequence[int] | None = None
) -> WeightedRandomSampler:
    """Weighted sampler that oversamples the minority (malignant) class.

    Each sample is weighted by the inverse frequency of its class, so with 17
    positives in 192 nodules the positives are drawn ~11x more often, balancing
    the effective batch without duplicating data on disk.

    Args:
        labels: per-sample labels for the *whole* dataset.
        indices: optional subset (e.g. a training fold's indices). Weights and
            the sampler length are computed over this subset only.
    """
    labels = [int(x) for x in labels]
    if indices is None:
        indices = list(range(len(labels)))
    counts: dict[int, int] = {}
    for i in indices:
        counts[labels[i]] = counts.get(labels[i], 0) + 1
    weights = torch.tensor(
        [1.0 / counts[labels[i]] for i in indices], dtype=torch.double
    )
    return WeightedRandomSampler(weights, num_samples=len(indices), replacement=True)
