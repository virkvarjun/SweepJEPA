"""Dataset interfaces for Stanford AIMI, ThyroidXL, and unlabeled US corpora."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset


@dataclass
class NoduleBag:
    """A cine-clip bag: ordered frames with a single nodule-level label."""

    frames: torch.Tensor  # (N, 3, H, W)
    label: int
    patient_id: str
    nodule_id: str


class ThyroidCineDataset(Dataset):
    """Patient-level cine nodule bags from Stanford AIMI Thyroid."""

    def __init__(self, root: Path, split: str = "train") -> None:
        self.root = Path(root)
        self.split = split
        self.samples: list[dict] = []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> NoduleBag:
        raise NotImplementedError("Wire to Stanford AIMI Thyroid Cine-clip manifest.")

    def patient_ids(self) -> Iterator[str]:
        seen: set[str] = set()
        for sample in self.samples:
            pid = sample["patient_id"]
            if pid not in seen:
                seen.add(pid)
                yield pid
