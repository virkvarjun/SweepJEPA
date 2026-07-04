#!/usr/bin/env python3
"""Stage D + the pre-registered Track 1 vs Track 2 comparison.

All arms share the US foundation backbone (matched backbone, #6) and the Track 1
MIL + conformal head. Only the *representation* differs:

  * track1_raw / attention        - per-frame US-FM features + gated attention-MIL
  * track1_raw / set_transformer  - same features + set-transformer (strong baseline)
  * track2 / pose_conditioned     - pose-conditioned JEPA tubelet embeddings
  * track2 / pose_free            - identical JEPA with pose conditioning OFF

Pre-registered success: pose-conditioned must beat BOTH pose-free AND
set-transformer on downstream AUC or decision-specificity. Null results are
reported honestly. On synthetic data with random-init backbones the signal is
near chance by construction — this validates the machinery; real weights/data
are required for a real verdict (see docs/STATUS.md).

    PYTHONPATH=. python scripts/run_track2_comparison.py --fast
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.shared.synthetic import SyntheticCineDataset
from src.track1.encoder import USEncoder
from src.track1.pipeline import (
    DecisionConfig,
    TrainConfig,
    cross_validate,
    precompute_embeddings,
)
from src.track2.jepa import PoseConditionedJEPA
from src.track2.stage_c import (
    build_clip_tubelets,
    jepa_embedding_bags,
    pretrain_jepa,
    stack_tubelets,
)


def _evaluate(bags, aggregator, n_splits, seed):
    res = cross_validate(
        bags,
        TrainConfig(aggregator=aggregator, epochs=20, lr=1e-2, hidden_dim=128, seed=seed),
        DecisionConfig(type="rcps", sensitivity_floor=0.9, delta=0.1, calib_fraction=0.4),
        n_splits=n_splits, seed=seed,
    )
    return {
        "auc": round(res.auc.auc, 3),
        "auc_ci": [round(res.auc.ci_low, 3), round(res.auc.ci_high, 3)],
        "specificity": round(res.decision.specificity, 3),
        "sensitivity": round(res.decision.sensitivity, 3),
    }


def _jepa_arm(clips, embed_dim, sc, pose_conditioned, teacher_type, target_space,
              shuffle, epochs, seed):
    jepa = PoseConditionedJEPA(
        embed_dim=embed_dim, depth=sc.get("depth", 4),
        predictor_depth=sc.get("predictor_depth", 2), num_heads=sc.get("num_heads", 6),
        ema_tau=sc.get("ema_tau", 0.996), teacher_type=teacher_type,
        pose_conditioned=pose_conditioned, target_space=target_space,
    )
    tokens, poses = stack_tubelets(clips)
    pretrain_jepa(tokens, poses, jepa, epochs=epochs, batch_size=sc.get("batch_size", 8),
                  lr=sc.get("lr", 1e-3), mask_ratio=sc.get("mask_ratio", 0.75),
                  seed=seed, shuffle_tubelets=shuffle)
    return jepa_embedding_bags(clips, jepa)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/track2.yaml"))
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--extra-arms", action="store_true",
                        help="also run frozen-teacher, input-space, shuffled arms")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    sc = cfg["stage_c"]
    seed = 0
    n_splits = 5

    n_nodules = 40 if args.fast else 192
    n_frames = 8 if args.fast else 16
    image_size = 224 if cfg["encoder"]["name"] != "imagenet_cnn" else 96
    epochs = 8 if args.fast else int(sc.get("epochs", 30))

    encoder = USEncoder(name=cfg["encoder"]["name"], adaptation="frozen")
    print(f"Backbone {encoder.name} dim={encoder.embed_dim} weights={encoder.load_info.source}")

    dataset = SyntheticCineDataset(
        n_nodules=n_nodules, n_malignant=max(4, n_nodules // 11),
        n_frames=n_frames, image_size=image_size, seed=seed,
    )

    print("Caching per-frame features (raw Track 1 representation)...")
    raw_bags = precompute_embeddings(dataset, encoder, device=args.device)
    print("Building tubelets (Track 2 representation)...")
    clips = build_clip_tubelets(
        dataset, encoder, temporal_size=sc["tubelet"]["temporal_size"], device=args.device
    )

    results = {}
    print("Evaluating arms...")
    results["track1_raw / attention"] = _evaluate(raw_bags, "gated_attention", n_splits, seed)
    results["track1_raw / set_transformer"] = _evaluate(raw_bags, "set_transformer", n_splits, seed)

    pose_bags = _jepa_arm(clips, encoder.embed_dim, sc, True, "ema", "latent", False, epochs, seed)
    results["track2 / pose_conditioned"] = _evaluate(pose_bags, "gated_attention", n_splits, seed)

    free_bags = _jepa_arm(clips, encoder.embed_dim, sc, False, "ema", "latent", False, epochs, seed)
    results["track2 / pose_free"] = _evaluate(free_bags, "gated_attention", n_splits, seed)

    if args.extra_arms:
        frozen_bags = _jepa_arm(clips, encoder.embed_dim, sc, True, "frozen", "latent", False, epochs, seed)
        results["track2 / frozen_teacher"] = _evaluate(frozen_bags, "gated_attention", n_splits, seed)
        input_bags = _jepa_arm(clips, encoder.embed_dim, sc, True, "ema", "input", False, epochs, seed)
        results["track2 / input_recon"] = _evaluate(input_bags, "gated_attention", n_splits, seed)
        shuf_bags = _jepa_arm(clips, encoder.embed_dim, sc, True, "ema", "latent", True, epochs, seed)
        results["track2 / shuffled_order"] = _evaluate(shuf_bags, "gated_attention", n_splits, seed)

    print("\n=== Downstream comparison (matched backbone) ===")
    print(f"{'arm':<32} {'AUC':>6} {'CI':>15} {'spec':>6} {'sens':>6}")
    for name, m in results.items():
        ci = f"[{m['auc_ci'][0]:.2f},{m['auc_ci'][1]:.2f}]"
        print(f"{name:<32} {m['auc']:>6.3f} {ci:>15} {m['specificity']:>6.3f} {m['sensitivity']:>6.3f}")

    pc = results["track2 / pose_conditioned"]
    pf = results["track2 / pose_free"]
    st = results["track1_raw / set_transformer"]
    beats_auc = pc["auc"] > pf["auc"] and pc["auc"] > st["auc"]
    beats_spec = pc["specificity"] > pf["specificity"] and pc["specificity"] > st["specificity"]
    verdict = beats_auc or beats_spec
    print("\nPre-registered check: pose-conditioned beats pose-free AND set-transformer")
    print(f"  on AUC: {beats_auc}; on specificity: {beats_spec} -> "
          f"{'SUPPORTED' if verdict else 'NULL (not supported on this synthetic run)'}")

    out_dir = Path("outputs/track2")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(
        json.dumps({"results": results, "verdict_supported": verdict}, indent=2)
    )
    print(f"\nWrote comparison to {out_dir}/comparison.json")


if __name__ == "__main__":
    main()
