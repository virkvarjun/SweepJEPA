"""AUC (DeLong CI) and clinical decision metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.track1.conformal import ClinicalAction
from src.track1.metrics import decision_metrics, delong_auc_ci, roc_auc


def test_perfect_separation_auc_one():
    scores = [0.1, 0.2, 0.3, 0.8, 0.9]
    labels = [0, 0, 0, 1, 1]
    r = delong_auc_ci(scores, labels)
    assert abs(r.auc - 1.0) < 1e-9


def test_auc_matches_sklearn():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=200)
    scores = rng.random(200) + 0.5 * labels  # mild signal
    ours = roc_auc(scores, labels)
    try:
        from sklearn.metrics import roc_auc_score

        assert abs(ours - roc_auc_score(labels, scores)) < 1e-9
    except ImportError:  # pragma: no cover
        assert 0.0 <= ours <= 1.0


def test_ci_brackets_point_estimate():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, size=120)
    scores = rng.random(120) + 0.4 * labels
    r = delong_auc_ci(scores, labels, alpha=0.05)
    assert r.ci_low <= r.auc <= r.ci_high
    assert 0.0 <= r.ci_low and r.ci_high <= 1.0
    assert r.se > 0.0


def test_auc_needs_both_classes():
    with pytest.raises(ValueError):
        delong_auc_ci([0.1, 0.2], [1, 1])


def test_decision_metrics_on_known_confusion():
    # 2 malignant, 3 benign.
    actions = [
        ClinicalAction.BIOPSY,     # malignant -> biopsied (caught)
        ClinicalAction.NO_BIOPSY,  # malignant -> missed
        ClinicalAction.NO_BIOPSY,  # benign -> avoided
        ClinicalAction.NO_BIOPSY,  # benign -> avoided
        ClinicalAction.BIOPSY,     # benign -> unnecessary biopsy
    ]
    labels = [1, 1, 0, 0, 0]
    m = decision_metrics(actions, labels)
    assert m.missed_cancers == 1
    assert m.miss_rate == 0.5
    assert m.sensitivity == 0.5
    assert m.biopsies_avoided == 2
    assert abs(m.specificity - 2 / 3) < 1e-9
    # 2 biopsies total, 1 is a true cancer -> NNB = 2.
    assert m.number_needed_to_biopsy == 2.0
    assert m.biopsies_avoided_per_missed_cancer == 2.0


def test_defer_counts_as_not_missed():
    actions = [ClinicalAction.DEFER, ClinicalAction.DEFER]
    labels = [1, 0]
    m = decision_metrics(actions, labels)
    assert m.missed_cancers == 0
    assert m.sensitivity == 1.0
    assert m.abstention_rate == 1.0
