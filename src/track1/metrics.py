"""Evaluation metrics: malignancy AUC with DeLong CIs, and decision metrics.

The AUC confidence interval uses the fast DeLong estimator (Sun & Xu, 2014),
which gives a closed-form variance for the empirical AUC and matches pROC. With
17 positives the CI is wide, so reporting it honestly matters.

Decision metrics operate on the biopsy/no-biopsy/defer actions from the decision
head. A ``defer`` is never counted as a missed cancer — it escalates to further
workup — so sensitivity treats defer as "not missed".
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.track1.conformal import ClinicalAction


def _midrank(x: np.ndarray) -> np.ndarray:
    """Mid-ranks of ``x`` (ties share the average rank), 1-based."""
    order = np.argsort(x, kind="mergesort")
    ranked = x[order]
    n = len(x)
    out = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and ranked[j] == ranked[i]:
            j += 1
        out[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    result = np.empty(n, dtype=float)
    result[order] = out
    return result


@dataclass
class AUCResult:
    auc: float
    ci_low: float
    ci_high: float
    se: float
    n_pos: int
    n_neg: int


def delong_auc_ci(
    scores: Sequence[float], labels: Sequence[int], alpha: float = 0.05
) -> AUCResult:
    """Empirical AUC with a two-sided ``1 - alpha`` DeLong confidence interval."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        raise ValueError("AUC needs both positive and negative samples")

    tx = _midrank(pos)
    ty = _midrank(neg)
    tz = _midrank(np.concatenate([pos, neg]))
    auc = (tz[:m].sum() - m * (m + 1) / 2.0) / (m * n)

    # Structural components (placement values).
    v01 = (tz[:m] - tx) / n  # one per positive
    v10 = 1.0 - (tz[m:] - ty) / m  # one per negative
    s01 = np.var(v01, ddof=1) / m if m > 1 else 0.0
    s10 = np.var(v10, ddof=1) / n if n > 1 else 0.0
    se = math.sqrt(max(s01 + s10, 0.0))

    try:
        from scipy.stats import norm

        z = float(norm.ppf(1.0 - alpha / 2.0))
    except Exception:  # pragma: no cover
        z = 1.959963984540054 if abs(alpha - 0.05) < 1e-9 else _inv_norm(1 - alpha / 2)

    lo = max(0.0, auc - z * se)
    hi = min(1.0, auc + z * se)
    return AUCResult(auc=float(auc), ci_low=lo, ci_high=hi, se=se, n_pos=m, n_neg=n)


def _inv_norm(p: float) -> float:  # pragma: no cover - scipy fallback
    # Acklam's rational approximation to the normal quantile.
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


@dataclass
class DecisionMetrics:
    sensitivity: float  # 1 - P(no_biopsy | malignant); defer counts as caught
    specificity: float  # P(no_biopsy | benign)
    miss_rate: float  # P(no_biopsy | malignant)
    abstention_rate: float  # P(defer)
    biopsy_rate: float  # P(biopsy)
    number_needed_to_biopsy: float  # biopsies per cancer among biopsied (inf if 0)
    biopsies_avoided: int  # benign nodules routed to no_biopsy
    missed_cancers: int
    biopsies_avoided_per_missed_cancer: float


def decision_metrics(
    actions: Sequence[ClinicalAction], labels: Sequence[int]
) -> DecisionMetrics:
    """Compute clinical decision metrics from actions + ground-truth labels."""
    actions = list(actions)
    labels = [int(x) for x in labels]
    if len(actions) != len(labels):
        raise ValueError("actions and labels must align")

    n = len(labels)
    n_pos = sum(labels)
    n_neg = n - n_pos

    def is_(a, kind):
        return a == kind

    missed = sum(
        1 for a, y in zip(actions, labels) if y == 1 and is_(a, ClinicalAction.NO_BIOPSY)
    )
    pos_biopsied = sum(
        1 for a, y in zip(actions, labels) if y == 1 and is_(a, ClinicalAction.BIOPSY)
    )
    avoided = sum(
        1 for a, y in zip(actions, labels) if y == 0 and is_(a, ClinicalAction.NO_BIOPSY)
    )
    n_biopsy = sum(1 for a in actions if is_(a, ClinicalAction.BIOPSY))
    n_defer = sum(1 for a in actions if is_(a, ClinicalAction.DEFER))

    miss_rate = missed / n_pos if n_pos else 0.0
    nnb = (n_biopsy / pos_biopsied) if pos_biopsied else math.inf
    per_missed = avoided / missed if missed else (math.inf if avoided else 0.0)

    return DecisionMetrics(
        sensitivity=1.0 - miss_rate,
        specificity=(avoided / n_neg) if n_neg else 0.0,
        miss_rate=miss_rate,
        abstention_rate=n_defer / n if n else 0.0,
        biopsy_rate=n_biopsy / n if n else 0.0,
        number_needed_to_biopsy=nnb,
        biopsies_avoided=avoided,
        missed_cancers=missed,
        biopsies_avoided_per_missed_cancer=per_missed,
    )


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Plain empirical AUC (Mann-Whitney), for quick checks."""
    return delong_auc_ci(scores, labels).auc
