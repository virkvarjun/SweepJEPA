"""Risk-controlled biopsy decision with an exact-binomial sensitivity floor.

Non-negotiable design decision #3: the sensitivity guarantee uses the **exact
binomial (Clopper-Pearson) upper confidence bound** on the missed-cancer rate,
never Hoeffding. Hoeffding is far too loose at this sample size — with 17
positives it can only certify a ~26% miss rate, whereas the exact binomial
certifies ~13%. A 5% floor genuinely needs ~45 positives, so we:

* certify the floor honestly and expose :attr:`certified_` (was the *requested*
  floor met?) and :attr:`certified_floor_` (the best floor the positive pool can
  actually support), and
* allow the risk to be certified on a **larger external positive pool**
  (ThyroidXL) via ``positive_scores`` while the operating point is set on the
  internal data.

Decision structure (RCPS):

* ``no_biopsy``  when score <= tau_lo  — the risk-controlled region;
* ``biopsy``     when score >= tau_hi;
* ``defer``      in between            — abstain / escalate to further workup.

Only ``tau_lo`` carries the distribution-free guarantee: a defer or a biopsy is
never a miss, so the certified quantity is P(no_biopsy | malignant).
"""

from __future__ import annotations

import math
from enum import Enum

import torch


class ClinicalAction(str, Enum):
    BIOPSY = "biopsy"
    NO_BIOPSY = "no_biopsy"
    DEFER = "defer"


def clopper_pearson_upper(k: int, n: int, delta: float) -> float:
    """Exact-binomial (Clopper-Pearson) upper confidence bound on a rate.

    Returns the ``1 - delta`` upper bound on the true probability given ``k``
    successes in ``n`` independent trials. ``k == n`` -> 1.0; ``n == 0`` -> 1.0
    (nothing is known, so bound conservatively).

    Uses the Beta-quantile form: ``UCB = Beta_ppf(1 - delta; k + 1, n - k)``.
    """
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    if k < 0:
        raise ValueError("k must be non-negative")
    # Prefer scipy's exact Beta quantile; fall back to a bisection on the
    # regularized incomplete Beta if scipy is unavailable.
    try:
        from scipy.stats import beta

        return float(beta.ppf(1.0 - delta, k + 1, n - k))
    except Exception:  # pragma: no cover - exercised only without scipy
        return _beta_ppf_bisect(1.0 - delta, k + 1, n - k)


def _beta_ppf_bisect(q: float, a: float, b: float) -> float:  # pragma: no cover
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _betainc(a: float, b: float, x: float) -> float:  # pragma: no cover
    # Regularized incomplete beta via a continued fraction (Numerical Recipes).
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(1, 200):
        m = i // 2
        if i % 2 == 0:
            num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= d * c
        if abs(1.0 - d * c) < 1e-10:
            break
    return front * (f - 1.0)


class RiskControlledDecision:
    """RCPS biopsy/no-biopsy/defer head with a certified sensitivity floor.

    Args:
        sensitivity_floor: target sensitivity (``1 - max miss rate``).
        delta: confidence level; the bound holds with probability ``1 - delta``.
        abstention_target: fraction of the (score > tau_lo) mass routed to defer
            rather than biopsy — an operating-point knob, *not* risk-controlled.
    """

    def __init__(
        self,
        sensitivity_floor: float = 0.95,
        delta: float = 0.1,
        abstention_target: float = 0.2,
    ) -> None:
        if not 0.0 < sensitivity_floor <= 1.0:
            raise ValueError("sensitivity_floor must be in (0, 1]")
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        if not 0.0 <= abstention_target < 1.0:
            raise ValueError("abstention_target must be in [0, 1)")
        self.sensitivity_floor = sensitivity_floor
        self.delta = delta
        self.abstention_target = abstention_target

        self.threshold_no_biopsy: float | None = None  # tau_lo
        self.threshold_biopsy: float | None = None  # tau_hi
        # Honest reporting (populated by calibrate()):
        self.certified_: bool = False
        self.certified_floor_: float | None = None
        self.n_calibration_positives_: int = 0
        self.empirical_miss_bound_: float | None = None

    def calibrate(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        positive_scores: torch.Tensor | None = None,
    ) -> None:
        """Fit ``tau_lo`` (risk-controlled) and ``tau_hi`` (operating point).

        Args:
            scores: ``(M,)`` calibration malignancy scores.
            labels: ``(M,)`` binary labels aligned with ``scores``.
            positive_scores: optional external positive-only score pool used for
                the risk bound (e.g. ThyroidXL). When given, ``tau_lo`` and the
                certification are computed from it; the internal negatives still
                set ``tau_hi``.
        """
        scores = scores.reshape(-1).float()
        labels = labels.reshape(-1)
        pos = (
            positive_scores.reshape(-1).float()
            if positive_scores is not None
            else scores[labels == 1]
        )
        if pos.numel() == 0:
            raise ValueError("no positives available to certify a sensitivity floor")

        n_pos = int(pos.numel())
        self.n_calibration_positives_ = n_pos
        alpha = 1.0 - self.sensitivity_floor

        # Best floor achievable with zero observed misses (largest region below
        # the smallest positive score). This is what the pool can *ever* certify.
        self.certified_floor_ = 1.0 - clopper_pearson_upper(0, n_pos, self.delta)
        self.certified_ = clopper_pearson_upper(0, n_pos, self.delta) <= alpha + 1e-12

        # Largest tau_lo whose certified miss-rate bound stays under alpha. The
        # bound is monotone in the miss count k = #{positives <= tau_lo}, and k
        # is monotone in tau_lo, so we sweep candidate thresholds drawn from ALL
        # calibration scores (not just positives) and stop when the bound breaks.
        # Drawing from all scores lets tau_lo sit in the negative region, which
        # is what actually creates a no-biopsy zone under separable data.
        tau_lo = -math.inf
        best_bound = clopper_pearson_upper(0, n_pos, self.delta)
        if self.certified_:
            candidates, _ = torch.sort(torch.unique(scores))
            for cand in candidates:
                c = float(cand.item())
                k = int((pos <= c).sum().item())
                bound = clopper_pearson_upper(k, n_pos, self.delta)
                if bound <= alpha + 1e-12:
                    tau_lo = c
                    best_bound = bound
                else:
                    break
        self.threshold_no_biopsy = tau_lo
        self.empirical_miss_bound_ = best_bound

        # tau_hi: split the (score > tau_lo) region so a fraction defers. This is
        # an efficiency choice and is explicitly *not* risk-controlled.
        above = scores[scores > tau_lo]
        if above.numel() == 0:
            self.threshold_biopsy = tau_lo
        else:
            q = torch.quantile(above, self.abstention_target)
            self.threshold_biopsy = max(float(q.item()), tau_lo)

    def predict(self, score: float) -> ClinicalAction:
        if self.threshold_no_biopsy is None or self.threshold_biopsy is None:
            raise RuntimeError("call calibrate() before predict()")
        if score <= self.threshold_no_biopsy:
            return ClinicalAction.NO_BIOPSY
        if score >= self.threshold_biopsy:
            return ClinicalAction.BIOPSY
        return ClinicalAction.DEFER

    def predict_batch(self, scores: torch.Tensor) -> list[ClinicalAction]:
        return [self.predict(float(s)) for s in scores.reshape(-1)]

    def summary(self) -> dict:
        """Machine-readable, honest report of what was certified."""
        return {
            "requested_sensitivity_floor": self.sensitivity_floor,
            "delta": self.delta,
            "certified": self.certified_,
            "certifiable_sensitivity_floor": self.certified_floor_,
            "n_calibration_positives": self.n_calibration_positives_,
            "miss_rate_upper_bound": self.empirical_miss_bound_,
            "tau_lo": self.threshold_no_biopsy,
            "tau_hi": self.threshold_biopsy,
        }


class WeightedYoudenDecision:
    """Ablation baseline: a single threshold at the weighted-Youden optimum.

    Maximizes ``w * TPR - (1 - TPR_of_negatives)`` ... concretely
    ``w * sensitivity + specificity``. No distribution-free guarantee — this is
    the point of the RCPS-vs-Youden ablation. Emits only biopsy / no_biopsy.
    """

    def __init__(self, sensitivity_weight: float = 2.0) -> None:
        if sensitivity_weight <= 0:
            raise ValueError("sensitivity_weight must be positive")
        self.sensitivity_weight = sensitivity_weight
        self.threshold: float | None = None

    def calibrate(self, scores: torch.Tensor, labels: torch.Tensor) -> None:
        scores = scores.reshape(-1).float()
        labels = labels.reshape(-1)
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        if pos.numel() == 0 or neg.numel() == 0:
            raise ValueError("need both classes to fit a Youden threshold")
        candidates = torch.unique(scores)
        best_j, best_t = -math.inf, float(candidates[0].item())
        for t in candidates:
            tv = float(t.item())
            sens = float((pos >= tv).float().mean().item())
            spec = float((neg < tv).float().mean().item())
            j = self.sensitivity_weight * sens + spec
            if j > best_j:
                best_j, best_t = j, tv
        self.threshold = best_t

    def predict(self, score: float) -> ClinicalAction:
        if self.threshold is None:
            raise RuntimeError("call calibrate() before predict()")
        return ClinicalAction.BIOPSY if score >= self.threshold else ClinicalAction.NO_BIOPSY

    def predict_batch(self, scores: torch.Tensor) -> list[ClinicalAction]:
        return [self.predict(float(s)) for s in scores.reshape(-1)]


# Backward-compatible alias: the exported name stays `ConformalDecisionLayer`,
# now backed by the exact-binomial RCPS implementation.
ConformalDecisionLayer = RiskControlledDecision
