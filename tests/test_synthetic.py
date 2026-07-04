"""Synthetic data: determinism, shapes, imbalance, patient grouping."""

from __future__ import annotations

import torch

from src.shared.data import NoduleBag
from src.shared.synthetic import (
    SyntheticCineDataset,
    SyntheticStaticDataset,
    SyntheticTrackedSweepDataset,
    make_cine_bag,
)


def test_make_cine_bag_shapes_and_range():
    bag = make_cine_bag(label=1, patient_id="p", nodule_id="n", n_frames=8, image_size=64, seed=1)
    assert isinstance(bag, NoduleBag)
    assert bag.frames.shape == (8, 3, 64, 64)
    assert bag.frames.min() >= 0.0 and bag.frames.max() <= 1.0
    assert bag.label == 1


def test_synthetic_is_deterministic():
    a = make_cine_bag(label=1, patient_id="p", nodule_id="n", n_frames=4, image_size=32, seed=7)
    b = make_cine_bag(label=1, patient_id="p", nodule_id="n", n_frames=4, image_size=32, seed=7)
    assert torch.equal(a.frames, b.frames)


def test_different_seeds_differ():
    a = make_cine_bag(label=0, patient_id="p", nodule_id="n", n_frames=4, image_size=32, seed=1)
    b = make_cine_bag(label=0, patient_id="p", nodule_id="n", n_frames=4, image_size=32, seed=2)
    assert not torch.equal(a.frames, b.frames)


def test_dataset_matches_target_imbalance():
    ds = SyntheticCineDataset(n_nodules=192, n_malignant=17, n_frames=2, image_size=16, seed=0)
    assert len(ds) == 192
    assert sum(ds.labels) == 17


def test_dataset_patient_grouping_consistent():
    ds = SyntheticCineDataset(n_nodules=40, n_malignant=5, n_frames=2, image_size=16, seed=3)
    # Every sample's bag reports the same patient_id as its manifest entry.
    for i in range(len(ds)):
        assert ds[i].patient_id == ds.samples[i]["patient_id"]
    # Some patients own more than one nodule.
    assert len(set(ds.patient_ids)) < len(ds)


def test_dataset_getitem_deterministic():
    ds = SyntheticCineDataset(n_nodules=8, n_malignant=2, n_frames=3, image_size=16, seed=5)
    assert torch.equal(ds[0].frames, ds[0].frames)


def test_static_dataset():
    ds = SyntheticStaticDataset(n_images=32, n_malignant=10, image_size=16, seed=0)
    img, label = ds[0]
    assert img.shape == (3, 16, 16)
    assert label in (0, 1)
    assert sum(ds.labels) == 10


def test_tracked_sweep_shapes_and_poses():
    ds = SyntheticTrackedSweepDataset(n_sweeps=3, n_frames=6, image_size=32, seed=0)
    sweep = ds[0]
    assert sweep.frames.shape == (6, 3, 32, 32)
    assert sweep.poses.shape == (6, 4, 4)
    # First pose is identity; all are valid SE(3) (bottom row [0,0,0,1]).
    assert torch.allclose(sweep.poses[0], torch.eye(4), atol=1e-5)
    assert torch.allclose(sweep.poses[:, 3, :], torch.tensor([0.0, 0.0, 0.0, 1.0]), atol=1e-5)
