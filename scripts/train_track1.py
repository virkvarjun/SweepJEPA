#!/usr/bin/env python3
"""Train Track 1: US encoder -> attention-MIL -> conformal decision, patient-CV.

Frozen/LoRA encoder embeddings are cached once, then the aggregation head is
cross-validated over patient-level folds with focal loss + minority oversampling,
and a risk-controlled decision layer is fit per fold. Reports malignancy AUC with
DeLong CIs and clinical decision metrics, and saves out-of-fold predictions +
attention weights.

    PYTHONPATH=. python scripts/train_track1.py --config configs/track1.yaml
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.shared.config import (
    build_cine_dataset,
    build_decision_config,
    build_encoder,
    build_train_config,
    load_config,
)
from src.track1.pipeline import cross_validate, precompute_embeddings


def _maybe_wandb(cfg: dict):
    if not cfg.get("logging", {}).get("wandb", False):
        return None
    try:
        import wandb

        wandb.init(project=cfg["logging"].get("project", "sweepjepa-track1"), config=cfg)
        return wandb
    except Exception as e:  # noqa: BLE001
        print(f"[wandb] disabled ({e!r})")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/track1.yaml"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    wandb = _maybe_wandb(cfg)

    encoder = build_encoder(cfg)
    train_cfg = build_train_config(cfg)
    decision_cfg = build_decision_config(cfg)
    cv_cfg = cfg.get("cv", {})

    print(
        f"Encoder: {encoder.name} [{encoder.adaptation}] "
        f"dim={encoder.embed_dim} weights={encoder.load_info.source} "
        f"trainable={encoder.num_trainable():,}"
    )

    dataset = build_cine_dataset(cfg)
    print(f"Dataset: {len(dataset)} nodules; caching embeddings...")
    bags = precompute_embeddings(dataset, encoder, device=args.device)

    results = cross_validate(
        bags,
        train_cfg,
        decision_cfg,
        n_splits=int(cv_cfg.get("n_splits", 5)),
        seed=int(cv_cfg.get("seed", 0)),
        device=args.device,
    )

    auc = results.auc
    dm = results.decision
    print("\n=== Internal patient-level CV ===")
    print(f"Malignancy AUC: {auc.auc:.3f}  (95% DeLong CI {auc.ci_low:.3f}-{auc.ci_high:.3f})")
    print(
        f"Decision: sens={dm.sensitivity:.3f} spec={dm.specificity:.3f} "
        f"NNB={dm.number_needed_to_biopsy:.2f} abstention={dm.abstention_rate:.3f} "
        f"missed={dm.missed_cancers} avoided={dm.biopsies_avoided}"
    )
    if decision_cfg.type == "rcps":
        cert = [s.get("certified") for s in results.fold_summaries]
        floors = [s.get("certifiable_sensitivity_floor") for s in results.fold_summaries]
        floors = [f for f in floors if f is not None]
        mean_floor = float(np.mean(floors)) if floors else float("nan")
        print(
            f"Certification: requested floor {decision_cfg.sensitivity_floor:.2f}; "
            f"certified folds {sum(bool(c) for c in cert)}/{len(cert)}; "
            f"mean certifiable floor {mean_floor:.3f} "
            "(honest: internal positives are few -> see ThyroidXL external calibration)"
        )

    out_dir = Path(cfg.get("logging", {}).get("output_dir", "outputs/track1"))
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": str(args.config),
        "encoder": {
            "name": encoder.name,
            "adaptation": encoder.adaptation,
            "weights": encoder.load_info.source,
        },
        "auc": asdict(auc),
        "decision_metrics": asdict(dm),
        "fold_summaries": results.fold_summaries,
        "oof": [
            {"nodule_id": nid, "score": float(s), "label": int(y), "action": a}
            for nid, s, y, a in zip(
                results.nodule_ids, results.oof_scores, results.oof_labels, results.oof_actions
            )
        ],
    }
    (out_dir / "cv_results.json").write_text(json.dumps(payload, indent=2))
    np.savez(
        out_dir / "attention.npz",
        **{nid: att for nid, att in zip(results.nodule_ids, results.attention)},
    )
    print(f"\nSaved OOF predictions + attention to {out_dir}/")

    if wandb is not None:
        wandb.log(
            {
                "auc": auc.auc,
                "auc_ci_low": auc.ci_low,
                "auc_ci_high": auc.ci_high,
                "sensitivity": dm.sensitivity,
                "specificity": dm.specificity,
                "nnb": dm.number_needed_to_biopsy,
                "abstention": dm.abstention_rate,
            }
        )
        wandb.finish()


if __name__ == "__main__":
    main()
