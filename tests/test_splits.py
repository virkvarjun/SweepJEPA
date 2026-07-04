"""Patient-level split correctness: the non-negotiable leakage guarantee."""

from __future__ import annotations

import pytest

from src.shared.splits import (
    assert_no_patient_leakage,
    carve_calibration_split,
    patient_level_folds,
)
from src.shared.synthetic import SyntheticCineDataset


def _dataset():
    return SyntheticCineDataset(n_nodules=192, n_malignant=17, n_frames=1, image_size=8, seed=0)


def test_no_patient_appears_in_train_and_val():
    ds = _dataset()
    folds = patient_level_folds(ds.patient_ids, ds.labels, n_splits=5, seed=0)
    assert len(folds) == 5
    for fold in folds:
        # Direct assertion + independent recomputation from indices.
        assert_no_patient_leakage(fold.train_patients, fold.val_patients)
        train_pat = {ds.patient_ids[i] for i in fold.train_idx}
        val_pat = {ds.patient_ids[i] for i in fold.val_idx}
        assert train_pat.isdisjoint(val_pat)


def test_every_patient_in_exactly_one_val_fold():
    ds = _dataset()
    folds = patient_level_folds(ds.patient_ids, ds.labels, n_splits=5, seed=0)
    val_counts = {}
    for fold in folds:
        for p in fold.val_patients:
            val_counts[p] = val_counts.get(p, 0) + 1
    assert set(val_counts) == set(ds.patient_ids)
    assert all(c == 1 for c in val_counts.values())


def test_positives_spread_across_folds():
    ds = _dataset()
    folds = patient_level_folds(ds.patient_ids, ds.labels, n_splits=5, seed=0)
    # With stratified assignment, most folds see at least one positive in val.
    labels = ds.labels
    pos_per_fold = [sum(labels[i] for i in f.val_idx) for f in folds]
    assert sum(pos_per_fold) == 17
    assert sum(p > 0 for p in pos_per_fold) >= 4


def test_folds_are_deterministic():
    ds = _dataset()
    f1 = patient_level_folds(ds.patient_ids, ds.labels, n_splits=5, seed=42)
    f2 = patient_level_folds(ds.patient_ids, ds.labels, n_splits=5, seed=42)
    assert [f.val_patients for f in f1] == [f.val_patients for f in f2]


def test_calibration_split_is_patient_disjoint():
    ds = _dataset()
    folds = patient_level_folds(ds.patient_ids, ds.labels, n_splits=5, seed=0)
    fit_idx, calib_idx = carve_calibration_split(
        folds[0].train_idx, ds.patient_ids, ds.labels, calib_fraction=0.3, seed=0
    )
    fit_pat = {ds.patient_ids[i] for i in fit_idx}
    calib_pat = {ds.patient_ids[i] for i in calib_idx}
    assert fit_pat.isdisjoint(calib_pat)
    assert set(fit_idx) | set(calib_idx) == set(folds[0].train_idx)


def test_leakage_assertion_fires():
    with pytest.raises(AssertionError):
        assert_no_patient_leakage(["a", "b"], ["b", "c"])


def test_invalid_n_splits():
    ds = SyntheticCineDataset(n_nodules=4, n_malignant=1, n_frames=1, image_size=8, seed=0)
    with pytest.raises(ValueError):
        patient_level_folds(ds.patient_ids, ds.labels, n_splits=100, seed=0)
