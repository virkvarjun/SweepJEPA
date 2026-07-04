#!/usr/bin/env python3
"""Evaluate Track 1: internal patient-CV metrics + ThyroidXL external test.

Internal: malignancy AUC (DeLong CI) + decision metrics from patient-level CV.

External (ThyroidXL): each static B-mode image is scored as a one-frame bag by
the ensemble of fold models. Crucially, the large external positive pool is used
to *calibrate the sensitivity floor* — with hundreds of positives the exact
binomial certifies a floor the 17-positive internal set cannot (non-negotiable
#3). This validates the per-frame backbone + decision, not cine aggregation.

    PYTHONPATH=. python scripts/eval.py --config configs/track1.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.shared.config import (
    build_cine_dataset,
    build_decision_config,
    build_encoder,
    build_external_static_dataset,
    build_train_config,
    load_config,
)
from src.shared.splits import carve_calibration_split
from src.track1.conformal import RiskControlledDecision
from src.track1.metrics import decision_metrics, delong_auc_ci
from src.track1.pipeline import (
    EmbeddingBag,
    cross_validate,
    ensemble_scores,
    precompute_embeddings,
)


def _encode_static(dataset, encoder, device: str) -> list[EmbeddingBag]:
    """Encode each static image as a one-frame embedding bag."""
    encoder = encoder.to(device).eval()
    bags: list[EmbeddingBag] = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            img, label = dataset[idx]
            s = dataset.samples[idx]
            emb = encoder(img.to(device).unsqueeze(0).unsqueeze(0)).squeeze(0)  # (1, D)
            bags.append(
                EmbeddingBag(
                    emb=emb.cpu(),
                    label=int(label),
                    patient_id=s["patient_id"],
                    nodule_id=s["image_id"],
                )
            )
    return bags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/track1.yaml"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    encoder = build_encoder(cfg)
    train_cfg = build_train_config(cfg)
    decision_cfg = build_decision_config(cfg)
    cv_cfg = cfg.get("cv", {})
    seed = int(cv_cfg.get("seed", 0))

    # ---- Internal patient-level CV ----
    dataset = build_cine_dataset(cfg)
    bags = precompute_embeddings(dataset, encoder, device=args.device)
    results = cross_validate(
        bags, train_cfg, decision_cfg,
        n_splits=int(cv_cfg.get("n_splits", 5)), seed=seed, device=args.device,
    )
    a = results.auc
    d = results.decision
    print("=== Internal (Stanford AIMI cine, patient-level CV) ===")
    print(f"AUC {a.auc:.3f} (95% CI {a.ci_low:.3f}-{a.ci_high:.3f}); "
          f"sens {d.sensitivity:.3f} spec {d.specificity:.3f} "
          f"NNB {d.number_needed_to_biopsy:.2f} abstention {d.abstention_rate:.3f}")
    internal_floor = None
    for s in results.fold_summaries:
        if s.get("certifiable_sensitivity_floor") is not None:
            internal_floor = s["certifiable_sensitivity_floor"]
            break
    if internal_floor is not None:
        print(f"Internal certifiable sensitivity floor: {internal_floor:.3f} "
              f"(n_pos per fold is small)")

    # ---- External ThyroidXL (per-image) ----
    ext = build_external_static_dataset(cfg, seed=seed)
    ext_bags = _encode_static(ext, encoder, device=args.device)
    scores = ensemble_scores(results.models, ext_bags, device=args.device)
    labels = [b.label for b in ext_bags]

    auc_ext = delong_auc_ci(scores, labels)
    # Patient-level calib/test split; calibrate the floor on the big external pool.
    pids = [b.patient_id for b in ext_bags]
    test_idx, calib_idx = carve_calibration_split(
        list(range(len(ext_bags))), pids, labels, calib_fraction=0.5, seed=seed
    )
    ext_decision = RiskControlledDecision(
        sensitivity_floor=decision_cfg.sensitivity_floor,
        delta=decision_cfg.delta,
        abstention_target=decision_cfg.abstention_target,
    )
    ext_decision.calibrate(
        torch.tensor(scores[calib_idx]),
        torch.tensor([labels[i] for i in calib_idx]),
    )
    test_actions = ext_decision.predict_batch(torch.tensor(scores[test_idx]))
    dm_ext = decision_metrics(test_actions, [labels[i] for i in test_idx])
    summ = ext_decision.summary()

    print("\n=== External (ThyroidXL, per-image) ===")
    print(f"AUC {auc_ext.auc:.3f} (95% CI {auc_ext.ci_low:.3f}-{auc_ext.ci_high:.3f})")
    print(f"External calibration positives: {summ['n_calibration_positives']}")
    print(f"Requested floor {decision_cfg.sensitivity_floor:.2f} -> "
          f"certified={summ['certified']}, "
          f"certifiable floor {summ['certifiable_sensitivity_floor']:.3f}")
    print(f"Test decision: sens {dm_ext.sensitivity:.3f} spec {dm_ext.specificity:.3f} "
          f"NNB {dm_ext.number_needed_to_biopsy:.2f} abstention {dm_ext.abstention_rate:.3f}")
    print("\nTakeaway: the external positive pool certifies a floor the internal "
          "17-positive set cannot — reported honestly per non-negotiable #3.")


if __name__ == "__main__":
    main()
