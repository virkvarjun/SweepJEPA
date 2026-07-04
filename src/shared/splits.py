"""Patient-level cross-validation splits with hard leakage guarantees.

Non-negotiable design decision #1: **no patient may appear in both train and
val**. Nodule-level splitting would leak texture/anatomy across the boundary
because one patient can contribute multiple nodules. Every training and
evaluation path must obtain its folds here, and every fold is checked for
patient leakage before it is returned.

Splits are stratified on *patient-level* positivity (a patient is positive if
any of their nodules is malignant) so the 17-positive minority is spread across
folds instead of collapsing into one. With so few positives, a fold can still
end up with zero calibration positives; callers that need a guaranteed positive
count (the conformal layer) should surface that honestly rather than pretend.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "Fold",
    "assert_no_patient_leakage",
    "patient_level_folds",
    "carve_calibration_split",
]


@dataclass(frozen=True)
class Fold:
    """One CV fold, as integer indices into the sample sequence."""

    index: int
    train_idx: list[int]
    val_idx: list[int]
    train_patients: list[str]
    val_patients: list[str]


def assert_no_patient_leakage(
    train_patients: Sequence[str], val_patients: Sequence[str]
) -> None:
    """Raise if any patient appears on both sides of a split."""
    overlap = set(train_patients) & set(val_patients)
    if overlap:
        raise AssertionError(
            f"Patient leakage across split: {sorted(overlap)[:5]}"
            f"{' ...' if len(overlap) > 5 else ''} ({len(overlap)} patients)"
        )


def _patient_positive(labels_by_patient: dict[str, list[int]]) -> dict[str, int]:
    return {pid: int(any(lbls)) for pid, lbls in labels_by_patient.items()}


def patient_level_folds(
    patient_ids: Sequence[str],
    labels: Sequence[int],
    n_splits: int = 5,
    seed: int = 0,
) -> list[Fold]:
    """Build ``n_splits`` stratified, patient-disjoint CV folds.

    Args:
        patient_ids: per-sample patient id (length == n_samples).
        labels: per-sample binary label (length == n_samples).
        n_splits: number of folds.
        seed: shuffling seed for reproducibility.

    Returns:
        A list of :class:`Fold`, each verified patient-disjoint. Every patient
        appears in exactly one validation fold.
    """
    if len(patient_ids) != len(labels):
        raise ValueError("patient_ids and labels must have equal length")
    if len(patient_ids) == 0:
        raise ValueError("cannot split an empty dataset")

    patient_ids = list(patient_ids)
    labels = [int(x) for x in labels]

    # Group sample indices by patient and derive a patient-level label.
    per_patient_idx: dict[str, list[int]] = {}
    per_patient_lbl: dict[str, list[int]] = {}
    order: list[str] = []
    for i, pid in enumerate(patient_ids):
        if pid not in per_patient_idx:
            per_patient_idx[pid] = []
            per_patient_lbl[pid] = []
            order.append(pid)
        per_patient_idx[pid].append(i)
        per_patient_lbl[pid].append(labels[i])

    unique_patients = order
    if n_splits < 2 or n_splits > len(unique_patients):
        raise ValueError(
            f"n_splits={n_splits} invalid for {len(unique_patients)} patients"
        )

    patient_pos = _patient_positive(per_patient_lbl)

    # Deterministic shuffle, then stratified round-robin assignment of patients
    # to folds within each positivity class. This keeps positives balanced
    # across folds without depending on sklearn's StratifiedGroupKFold (which is
    # unstable for very small positive counts).
    rng = np.random.default_rng(seed)
    fold_of_patient: dict[str, int] = {}
    for cls in (1, 0):
        cls_patients = [p for p in unique_patients if patient_pos[p] == cls]
        rng.shuffle(cls_patients)
        # Rotate the starting fold per class so tiny positive pools don't all
        # pile into fold 0.
        offset = 0 if cls == 1 else (len(cls_patients) % n_splits)
        for k, p in enumerate(cls_patients):
            fold_of_patient[p] = (k + offset) % n_splits

    folds: list[Fold] = []
    for f in range(n_splits):
        val_patients = [p for p in unique_patients if fold_of_patient[p] == f]
        train_patients = [p for p in unique_patients if fold_of_patient[p] != f]
        assert_no_patient_leakage(train_patients, val_patients)

        val_idx = [i for p in val_patients for i in per_patient_idx[p]]
        train_idx = [i for p in train_patients for i in per_patient_idx[p]]
        folds.append(
            Fold(
                index=f,
                train_idx=sorted(train_idx),
                val_idx=sorted(val_idx),
                train_patients=sorted(train_patients),
                val_patients=sorted(val_patients),
            )
        )

    _assert_full_val_coverage(folds, unique_patients)
    return folds


def _assert_full_val_coverage(folds: Sequence[Fold], unique_patients: Sequence[str]) -> None:
    """Every patient must be in exactly one validation fold."""
    seen: dict[str, int] = {}
    for fold in folds:
        for p in fold.val_patients:
            seen[p] = seen.get(p, 0) + 1
    for p in unique_patients:
        count = seen.get(p, 0)
        if count != 1:
            raise AssertionError(
                f"patient {p!r} appears in {count} validation folds (expected 1)"
            )


def carve_calibration_split(
    train_idx: Sequence[int],
    patient_ids: Sequence[str],
    labels: Sequence[int],
    calib_fraction: float = 0.3,
    seed: int = 0,
) -> tuple[list[int], list[int]]:
    """Split a training fold into (fit, calibration) sets, patient-disjoint.

    The calibration set feeds the conformal decision layer, so it must not share
    patients with the fit set. Positives are stratified across the two so the
    calibration split retains as many of the scarce positives as the fraction
    allows.

    Returns:
        ``(fit_idx, calib_idx)`` — sorted index lists, patient-disjoint.
    """
    if not 0.0 < calib_fraction < 1.0:
        raise ValueError("calib_fraction must be in (0, 1)")

    train_idx = list(train_idx)
    per_patient_idx: dict[str, list[int]] = {}
    per_patient_lbl: dict[str, list[int]] = {}
    order: list[str] = []
    for i in train_idx:
        pid = patient_ids[i]
        if pid not in per_patient_idx:
            per_patient_idx[pid] = []
            per_patient_lbl[pid] = []
            order.append(pid)
        per_patient_idx[pid].append(i)
        per_patient_lbl[pid].append(int(labels[i]))

    patient_pos = _patient_positive(per_patient_lbl)
    rng = np.random.default_rng(seed)

    calib_patients: set = set()
    for cls in (1, 0):
        cls_patients = [p for p in order if patient_pos[p] == cls]
        rng.shuffle(cls_patients)
        n_calib = int(round(calib_fraction * len(cls_patients)))
        calib_patients.update(cls_patients[:n_calib])

    calib_idx = [i for p in order if p in calib_patients for i in per_patient_idx[p]]
    fit_idx = [i for p in order if p not in calib_patients for i in per_patient_idx[p]]

    fit_patients = [p for p in order if p not in calib_patients]
    assert_no_patient_leakage(fit_patients, sorted(calib_patients))
    return sorted(fit_idx), sorted(calib_idx)
