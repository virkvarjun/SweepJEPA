"""RCPS conformal decision: exact-binomial floor, honest certification, routing."""

from __future__ import annotations

import math

import pytest
import torch

from src.track1.conformal import (
    ClinicalAction,
    RiskControlledDecision,
    WeightedYoudenDecision,
    clopper_pearson_upper,
)


def test_clopper_pearson_known_values():
    # Beta(1, n): CP_upper(0, n, delta) = 1 - delta**(1/n).
    for n in (10, 17, 45):
        expected = 1.0 - 0.1 ** (1.0 / n)
        assert abs(clopper_pearson_upper(0, n, 0.1) - expected) < 1e-6
    assert clopper_pearson_upper(5, 5, 0.1) == 1.0
    assert clopper_pearson_upper(0, 0, 0.1) == 1.0


def test_clopper_pearson_monotonic_in_k():
    vals = [clopper_pearson_upper(k, 20, 0.1) for k in range(21)]
    assert all(b <= a + 1e-12 for a, b in zip(vals[1:], vals[:-1])) or all(
        vals[i] <= vals[i + 1] + 1e-12 for i in range(len(vals) - 1)
    )


def test_binomial_beats_hoeffding_at_17_positives():
    # The motivating fact: exact binomial certifies ~13% miss at n=17, delta=0.1.
    cp = clopper_pearson_upper(0, 17, 0.1)
    assert 0.10 < cp < 0.15


def test_floor_not_certifiable_with_few_positives():
    dec = RiskControlledDecision(sensitivity_floor=0.95, delta=0.1)  # alpha = 0.05
    scores = torch.cat([torch.full((100,), 0.1), torch.full((17,), 0.9)])
    labels = torch.cat([torch.zeros(100), torch.ones(17)]).long()
    dec.calibrate(scores, labels)
    # 5% floor is NOT certifiable with 17 positives; report honestly.
    assert dec.certified_ is False
    assert dec.certified_floor_ is not None
    assert 0.85 < dec.certified_floor_ < 0.90


def test_floor_certifiable_when_relaxed():
    dec = RiskControlledDecision(sensitivity_floor=0.80, delta=0.1)  # alpha = 0.20
    scores = torch.cat([torch.full((100,), 0.1), torch.full((17,), 0.9)])
    labels = torch.cat([torch.zeros(100), torch.ones(17)]).long()
    dec.calibrate(scores, labels)
    assert dec.certified_ is True
    # Separable data: a no-biopsy region exists and its miss bound is under alpha.
    assert dec.threshold_no_biopsy > -math.inf
    assert dec.empirical_miss_bound_ <= 0.20 + 1e-9


def test_routing_order():
    dec = RiskControlledDecision(sensitivity_floor=0.80, delta=0.1, abstention_target=0.3)
    scores = torch.cat([torch.full((100,), 0.1), torch.full((20,), 0.9)])
    labels = torch.cat([torch.zeros(100), torch.ones(20)]).long()
    dec.calibrate(scores, labels)
    assert dec.predict(0.05) == ClinicalAction.NO_BIOPSY
    assert dec.predict(0.99) == ClinicalAction.BIOPSY
    assert dec.threshold_no_biopsy <= dec.threshold_biopsy


def test_no_biopsy_region_respects_certified_miss_bound():
    # Among calibration positives, the fraction landing in no-biopsy must not
    # exceed the reported upper bound.
    torch.manual_seed(0)
    dec = RiskControlledDecision(sensitivity_floor=0.85, delta=0.1)
    neg = torch.rand(200) * 0.4
    pos = 0.6 + torch.rand(30) * 0.4
    scores = torch.cat([neg, pos])
    labels = torch.cat([torch.zeros(200), torch.ones(30)]).long()
    dec.calibrate(scores, labels)
    missed = (pos <= dec.threshold_no_biopsy).float().mean().item()
    assert missed <= dec.empirical_miss_bound_ + 1e-9


def test_external_positive_pool_tightens_certification():
    internal = RiskControlledDecision(sensitivity_floor=0.90, delta=0.1)
    scores = torch.cat([torch.full((80,), 0.1), torch.full((10,), 0.9)])
    labels = torch.cat([torch.zeros(80), torch.ones(10)]).long()
    internal.calibrate(scores, labels)

    external_pos = torch.full((200,), 0.9)  # large external positive pool
    external = RiskControlledDecision(sensitivity_floor=0.90, delta=0.1)
    external.calibrate(scores, labels, positive_scores=external_pos)
    # More positives -> a higher certifiable floor.
    assert external.certified_floor_ > internal.certified_floor_


def test_calibrate_requires_positives():
    dec = RiskControlledDecision()
    with pytest.raises(ValueError):
        dec.calibrate(torch.tensor([0.1, 0.2]), torch.tensor([0, 0]))


def test_predict_before_calibrate_raises():
    with pytest.raises(RuntimeError):
        RiskControlledDecision().predict(0.5)


def test_invalid_params():
    with pytest.raises(ValueError):
        RiskControlledDecision(sensitivity_floor=1.5)
    with pytest.raises(ValueError):
        RiskControlledDecision(delta=0.0)


def test_summary_keys():
    dec = RiskControlledDecision(sensitivity_floor=0.8, delta=0.1)
    scores = torch.cat([torch.full((50,), 0.1), torch.full((20,), 0.9)])
    labels = torch.cat([torch.zeros(50), torch.ones(20)]).long()
    dec.calibrate(scores, labels)
    s = dec.summary()
    assert s["certified"] in (True, False)
    assert s["n_calibration_positives"] == 20
    assert "certifiable_sensitivity_floor" in s


def test_weighted_youden_baseline():
    dec = WeightedYoudenDecision(sensitivity_weight=2.0)
    scores = torch.cat([torch.full((50,), 0.2), torch.full((20,), 0.8)])
    labels = torch.cat([torch.zeros(50), torch.ones(20)]).long()
    dec.calibrate(scores, labels)
    assert dec.predict(0.9) == ClinicalAction.BIOPSY
    assert dec.predict(0.1) == ClinicalAction.NO_BIOPSY
