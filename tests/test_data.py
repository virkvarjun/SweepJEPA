"""Cine loader: padded collation, mask correctness, minority oversampling,
manifest round-trip."""

from __future__ import annotations

import csv

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.io import write_png

from src.shared.data import (
    ThyroidCineDataset,
    collate_bags,
    make_minority_sampler,
)
from src.shared.synthetic import SyntheticCineDataset, make_cine_bag


def test_collate_pads_and_masks():
    bags = [
        make_cine_bag(label=0, patient_id="p0", nodule_id="n0", n_frames=3, image_size=16, seed=0),
        make_cine_bag(label=1, patient_id="p1", nodule_id="n1", n_frames=5, image_size=16, seed=1),
    ]
    batch = collate_bags(bags)
    assert batch.frames.shape == (2, 5, 3, 16, 16)
    assert batch.mask.tolist() == [
        [True, True, True, False, False],
        [True, True, True, True, True],
    ]
    assert batch.lengths.tolist() == [3, 5]
    assert batch.labels.tolist() == [0, 1]
    # Padding frames are exactly zero.
    assert torch.equal(batch.frames[0, 3:], torch.zeros_like(batch.frames[0, 3:]))
    # Real frames preserved.
    assert torch.equal(batch.frames[1, :5], bags[1].frames)


def test_minority_sampler_upweights_positives():
    ds = SyntheticCineDataset(n_nodules=192, n_malignant=17, n_frames=1, image_size=8, seed=0)
    sub = Subset(ds, list(range(len(ds))))
    sampler = make_minority_sampler(ds.labels)
    loader = DataLoader(sub, batch_size=1, sampler=sampler, collate_fn=collate_bags)
    drawn = [int(b.labels[0]) for b in loader]
    frac_pos = sum(drawn) / len(drawn)
    # Native prevalence ~0.089; oversampling should push it near balanced.
    assert frac_pos > 0.3


def test_minority_sampler_on_subset_indices():
    labels = [0, 0, 0, 0, 1]
    sampler = make_minority_sampler(labels, indices=[0, 1, 4])
    assert len(list(sampler)) == 3  # one sample per subset element


def test_manifest_roundtrip(tmp_path):
    root = tmp_path
    # Two nodules for one patient, one for another; write real PNG frames.
    layout = [
        ("n0", "pA", 0, 3),
        ("n1", "pA", 1, 2),
        ("n2", "pB", 0, 4),
    ]
    rows = []
    for nid, pid, label, n_frames in layout:
        d = root / f"frames_{nid}"
        d.mkdir()
        for k in range(n_frames):
            img = (torch.rand(3, 16, 16) * 255).to(torch.uint8)
            write_png(img, str(d / f"{k:03d}.png"))
        rows.append(
            {
                "nodule_id": nid,
                "patient_id": pid,
                "label": label,
                "ti_rads": "TR3",
                "frame_dir": f"frames_{nid}",
            }
        )
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["nodule_id", "patient_id", "label", "ti_rads", "frame_dir"]
        )
        w.writeheader()
        w.writerows(rows)

    ds = ThyroidCineDataset.from_manifest(manifest, root=root, image_size=16)
    assert len(ds) == 3
    assert ds.labels == [0, 1, 0]
    assert list(ds.patient_ids()) == ["pA", "pB"]
    bag = ds[1]
    assert bag.frames.shape == (2, 3, 16, 16)  # nodule n1 had 2 frames
    assert bag.patient_id == "pA"
