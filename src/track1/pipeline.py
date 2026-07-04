"""Track 1 training pipeline: frozen-encoder embeddings + patient-level CV.

The recipe is the standard frozen-backbone MIL one: run the (frozen or
LoRA-adapted) US encoder once to cache per-frame embeddings, then cross-validate
the cheap aggregation head over patient-level folds. Caching makes CV fast and
keeps the geometry ablation honest — Track 2 will swap only the embeddings.

Each fold:
  1. patient-level split (no leakage — enforced upstream in ``splits``);
  2. carve a patient-disjoint calibration set from the training fold;
  3. train the aggregator on the fit set (focal loss + minority oversampling);
  4. fit the decision layer on calibration scores;
  5. score the held-out fold -> out-of-fold predictions + actions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.shared.splits import carve_calibration_split, patient_level_folds
from src.track1.conformal import (
    ClinicalAction,
    RiskControlledDecision,
    WeightedYoudenDecision,
)
from src.track1.losses import focal_loss_with_logits
from src.track1.metrics import AUCResult, decision_metrics, delong_auc_ci
from src.track1.mil import build_aggregator


@dataclass
class EmbeddingBag:
    """Cached per-frame embeddings for one nodule."""

    emb: torch.Tensor  # (N, D)
    label: int
    patient_id: str
    nodule_id: str


@dataclass
class TrainConfig:
    aggregator: str = "gated_attention"
    hidden_dim: int = 256
    dropout: float = 0.25
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    batch_size: int = 8
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    oversample: bool = True
    seed: int = 0


@dataclass
class DecisionConfig:
    type: str = "rcps"  # "rcps" | "youden"
    sensitivity_floor: float = 0.95
    delta: float = 0.1
    abstention_target: float = 0.2
    calib_fraction: float = 0.4
    sensitivity_weight: float = 2.0


@dataclass
class CVResults:
    oof_scores: np.ndarray
    oof_labels: np.ndarray
    oof_actions: list
    nodule_ids: list
    attention: list  # per-bag (N,) attention weights
    auc: AUCResult
    decision: object  # DecisionMetrics
    fold_summaries: list = field(default_factory=list)
    models: list = field(default_factory=list)  # trained per-fold aggregators
    input_dim: int = 0


# --------------------------------------------------------------------------- #
# Embedding extraction
# --------------------------------------------------------------------------- #
def precompute_embeddings(
    dataset: Dataset,
    encoder: torch.nn.Module,
    device: str = "cpu",
    frame_batch: int = 64,
) -> list[EmbeddingBag]:
    """Run the encoder once over every bag, caching ``(N, D)`` embeddings.

    Works with any dataset yielding ``NoduleBag`` (real or synthetic).
    """
    encoder = encoder.to(device).eval()
    out: list[EmbeddingBag] = []
    grad_ctx = torch.no_grad() if encoder.num_trainable() == 0 else torch.enable_grad()
    for i in range(len(dataset)):
        bag = dataset[i]
        frames = bag.frames.to(device)
        chunks = []
        with grad_ctx:
            for start in range(0, frames.shape[0], frame_batch):
                chunk = frames[start : start + frame_batch].unsqueeze(0)  # (1, n, 3, H, W)
                emb = encoder(chunk).squeeze(0)  # (n, D)
                chunks.append(emb.detach().cpu())
        out.append(
            EmbeddingBag(
                emb=torch.cat(chunks, dim=0),
                label=int(bag.label),
                patient_id=bag.patient_id,
                nodule_id=bag.nodule_id,
            )
        )
    return out


class _BagListDataset(Dataset):
    def __init__(self, bags: Sequence[EmbeddingBag]) -> None:
        self.bags = list(bags)

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, idx: int) -> EmbeddingBag:
        return self.bags[idx]


def collate_embedding_bags(bags: Sequence[EmbeddingBag]):
    """Pad ``(N, D)`` embedding bags to ``(B, N_max, D)`` with a validity mask."""
    lengths = [b.emb.shape[0] for b in bags]
    n_max = max(lengths)
    d = bags[0].emb.shape[1]
    emb = bags[0].emb.new_zeros((len(bags), n_max, d))
    mask = torch.zeros((len(bags), n_max), dtype=torch.bool)
    for i, b in enumerate(bags):
        emb[i, : lengths[i]] = b.emb
        mask[i, : lengths[i]] = True
    labels = torch.tensor([b.label for b in bags], dtype=torch.float32)
    return emb, labels, mask, [b.nodule_id for b in bags]


# --------------------------------------------------------------------------- #
# Training / scoring
# --------------------------------------------------------------------------- #
def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_aggregator(
    train_bags: Sequence[EmbeddingBag],
    input_dim: int,
    cfg: TrainConfig,
    device: str = "cpu",
) -> torch.nn.Module:
    """Train one aggregation head on cached embeddings."""
    _seed_everything(cfg.seed)
    model = build_aggregator(
        cfg.aggregator, input_dim=input_dim, hidden_dim=cfg.hidden_dim, dropout=cfg.dropout
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    ds = _BagListDataset(train_bags)
    gen = torch.Generator().manual_seed(cfg.seed)
    if cfg.oversample:
        labels = [b.label for b in train_bags]
        counts = {c: labels.count(c) for c in set(labels)}
        weights = torch.tensor([1.0 / counts[y] for y in labels], dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=len(labels), replacement=True, generator=gen
        )
        loader = DataLoader(
            ds, batch_size=cfg.batch_size, sampler=sampler, collate_fn=collate_embedding_bags
        )
    else:
        loader = DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=True, generator=gen,
            collate_fn=collate_embedding_bags,
        )

    model.train()
    for _ in range(cfg.epochs):
        for emb, labels_b, mask, _ in loader:
            emb, labels_b, mask = emb.to(device), labels_b.to(device), mask.to(device)
            opt.zero_grad()
            logits, _ = model(emb, mask)
            loss = focal_loss_with_logits(
                logits, labels_b, alpha=cfg.focal_alpha, gamma=cfg.focal_gamma
            )
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def score_bags(
    model: torch.nn.Module,
    bags: Sequence[EmbeddingBag],
    device: str = "cpu",
    batch_size: int = 16,
):
    """Return ``(scores, attention)`` for bags in input order."""
    model = model.to(device).eval()
    ds = _BagListDataset(bags)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_embedding_bags)
    scores: list[float] = []
    attention: list[np.ndarray] = []
    for emb, _, mask, _ in loader:
        emb, mask = emb.to(device), mask.to(device)
        logits, alpha = model(emb, mask)
        probs = torch.sigmoid(logits.reshape(-1)).cpu().numpy()
        scores.extend(probs.tolist())
        lengths = mask.sum(dim=1).cpu().numpy()
        for i, n in enumerate(lengths):
            attention.append(alpha[i, : int(n)].cpu().numpy())
    return np.asarray(scores), attention


def _build_decision(cfg: DecisionConfig):
    if cfg.type == "rcps":
        return RiskControlledDecision(
            sensitivity_floor=cfg.sensitivity_floor,
            delta=cfg.delta,
            abstention_target=cfg.abstention_target,
        )
    if cfg.type == "youden":
        return WeightedYoudenDecision(sensitivity_weight=cfg.sensitivity_weight)
    raise ValueError(f"unknown decision type {cfg.type!r}")


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
def cross_validate(
    bags: Sequence[EmbeddingBag],
    train_cfg: TrainConfig,
    decision_cfg: DecisionConfig,
    n_splits: int = 5,
    seed: int = 0,
    device: str = "cpu",
    external_positive_scores: torch.Tensor | None = None,
) -> CVResults:
    """Patient-level CV producing out-of-fold scores, actions, and metrics."""
    bags = list(bags)
    input_dim = bags[0].emb.shape[1]
    patient_ids = [b.patient_id for b in bags]
    labels = [b.label for b in bags]

    folds = patient_level_folds(patient_ids, labels, n_splits=n_splits, seed=seed)

    oof_scores = np.full(len(bags), np.nan)
    oof_actions: list = [None] * len(bags)
    attention: list = [None] * len(bags)
    summaries = []
    models = []

    for fold in folds:
        fit_idx, calib_idx = carve_calibration_split(
            fold.train_idx, patient_ids, labels,
            calib_fraction=decision_cfg.calib_fraction, seed=seed,
        )
        # Guard: a tiny fold may put all positives on one side. Fall back to the
        # whole training fold for calibration if the fit set lost a class.
        if len({labels[i] for i in fit_idx}) < 2 or not calib_idx:
            fit_idx, calib_idx = fold.train_idx, fold.train_idx

        fit_bags = [bags[i] for i in fit_idx]
        model = train_aggregator(fit_bags, input_dim, train_cfg, device=device)
        models.append(model)

        calib_bags = [bags[i] for i in calib_idx]
        calib_scores, _ = score_bags(model, calib_bags, device=device)
        calib_labels = torch.tensor([bags[i].label for i in calib_idx])

        decision = _build_decision(decision_cfg)
        if decision_cfg.type == "rcps":
            decision.calibrate(
                torch.tensor(calib_scores), calib_labels,
                positive_scores=external_positive_scores,
            )
            summaries.append({"fold": fold.index, **decision.summary()})
        else:
            decision.calibrate(torch.tensor(calib_scores), calib_labels)
            summaries.append({"fold": fold.index, "threshold": decision.threshold})

        val_bags = [bags[i] for i in fold.val_idx]
        val_scores, val_attn = score_bags(model, val_bags, device=device)
        val_acts = decision.predict_batch(torch.tensor(val_scores))
        for j, i in enumerate(fold.val_idx):
            oof_scores[i] = val_scores[j]
            oof_actions[i] = val_acts[j]
            attention[i] = val_attn[j]

    oof_labels = np.asarray(labels)
    auc = delong_auc_ci(oof_scores, oof_labels)
    dmetrics = decision_metrics(oof_actions, oof_labels.tolist())
    return CVResults(
        oof_scores=oof_scores,
        oof_labels=oof_labels,
        oof_actions=[a.value if isinstance(a, ClinicalAction) else a for a in oof_actions],
        nodule_ids=[b.nodule_id for b in bags],
        attention=attention,
        auc=auc,
        decision=dmetrics,
        fold_summaries=summaries,
        models=models,
        input_dim=input_dim,
    )


@torch.no_grad()
def ensemble_scores(
    models: Sequence[torch.nn.Module],
    bags: Sequence[EmbeddingBag],
    device: str = "cpu",
) -> np.ndarray:
    """Mean sigmoid score across an ensemble of per-fold aggregators.

    Used for the external ThyroidXL test, where each static image is scored as a
    one-frame bag by every fold model and averaged.
    """
    per_model = [score_bags(m, bags, device=device)[0] for m in models]
    return np.mean(np.stack(per_model, axis=0), axis=0)
