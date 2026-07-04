"""Track 1: US foundation encoder, attention-MIL, conformal decision."""

from src.track1.conformal import (
    ClinicalAction,
    ConformalDecisionLayer,
    RiskControlledDecision,
    WeightedYoudenDecision,
    clopper_pearson_upper,
)
from src.track1.encoder import USEncoder
from src.track1.losses import FocalLoss, focal_loss_with_logits
from src.track1.metrics import (
    AUCResult,
    DecisionMetrics,
    decision_metrics,
    delong_auc_ci,
)
from src.track1.mil import (
    GatedAttentionMIL,
    MeanPoolMIL,
    SetTransformerMIL,
    build_aggregator,
)

__all__ = [
    "USEncoder",
    "GatedAttentionMIL",
    "MeanPoolMIL",
    "SetTransformerMIL",
    "build_aggregator",
    "ConformalDecisionLayer",
    "RiskControlledDecision",
    "WeightedYoudenDecision",
    "ClinicalAction",
    "clopper_pearson_upper",
    "FocalLoss",
    "focal_loss_with_logits",
    "AUCResult",
    "DecisionMetrics",
    "decision_metrics",
    "delong_auc_ci",
]
