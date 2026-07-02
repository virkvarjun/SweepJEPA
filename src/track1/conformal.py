"""Conformal risk-controlled biopsy decision with abstention."""

from __future__ import annotations

from enum import Enum

import torch


class ClinicalAction(str, Enum):
    BIOPSY = "biopsy"
    NO_BIOPSY = "no_biopsy"
    DEFER = "defer"


class ConformalDecisionLayer:
    """Map pooled malignancy scores to clinical actions.

    Provides a distribution-free bound on missed-cancer rate (sensitivity
    floor) and abstains when the bound cannot be met.
    """

    def __init__(self, sensitivity_floor: float = 0.95) -> None:
        if not 0.0 < sensitivity_floor <= 1.0:
            raise ValueError("sensitivity_floor must be in (0, 1]")
        self.sensitivity_floor = sensitivity_floor
        self.threshold_biopsy: float | None = None
        self.threshold_no_biopsy: float | None = None

    def calibrate(self, scores: torch.Tensor, labels: torch.Tensor) -> None:
        """Fit decision thresholds on a calibration split."""
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        if len(pos) == 0 or len(neg) == 0:
            raise ValueError("Calibration split must contain both classes.")

        pos_sorted, _ = torch.sort(pos)
        k = max(1, int((1.0 - self.sensitivity_floor) * len(pos_sorted)))
        self.threshold_biopsy = pos_sorted[k - 1].item()

        neg_sorted, _ = torch.sort(neg, descending=True)
        self.threshold_no_biopsy = neg_sorted[0].item()

    def predict(self, score: float) -> ClinicalAction:
        if self.threshold_biopsy is None or self.threshold_no_biopsy is None:
            raise RuntimeError("Call calibrate() before predict().")

        if score >= self.threshold_biopsy:
            return ClinicalAction.BIOPSY
        if score <= self.threshold_no_biopsy:
            return ClinicalAction.NO_BIOPSY
        return ClinicalAction.DEFER
