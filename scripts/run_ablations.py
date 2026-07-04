#!/usr/bin/env python3
"""Track 1 ablations: backbone x aggregation x decision.

Covers the three Track 1 ablation axes from the proposal:
  * backbone:    US-FM (ViT) vs ImageNet CNN
  * aggregation: attention-MIL vs mean-pool vs set-transformer
  * decision:    RCPS (conformal) vs weighted-Youden

Embeddings are cached once per backbone, then aggregation/decision vary cheaply.
Prints a results table and writes JSON. Use ``--fast`` for a quick synthetic run.

    PYTHONPATH=. python scripts/run_ablations.py --fast
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from src.shared.config import build_decision_config, build_train_config, load_config
from src.shared.synthetic import SyntheticCineDataset
from src.track1.encoder import USEncoder
from src.track1.pipeline import cross_validate, precompute_embeddings

DEFAULT_BACKBONES = ["usfm", "imagenet_cnn"]
DEFAULT_AGGREGATORS = ["gated_attention", "average", "set_transformer"]
DEFAULT_DECISIONS = ["conformal", "youden"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/track1.yaml"))
    parser.add_argument("--fast", action="store_true", help="small synthetic run")
    parser.add_argument("--backbones", nargs="*", default=DEFAULT_BACKBONES)
    parser.add_argument("--aggregators", nargs="*", default=DEFAULT_AGGREGATORS)
    parser.add_argument("--decisions", nargs="*", default=DEFAULT_DECISIONS)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_train = build_train_config(cfg)
    if args.fast:
        base_train = replace(base_train, epochs=8)
        n_nodules, n_frames = 30, 3
    else:
        n_nodules, n_frames = 192, 8

    rows = []
    for backbone in args.backbones:
        # ViT needs 224x224 (fixed positional embeddings); the CNN is size-agnostic.
        image_size = 224 if backbone != "imagenet_cnn" or not args.fast else 96
        dataset = SyntheticCineDataset(
            n_nodules=n_nodules, n_malignant=max(3, n_nodules // 11),
            n_frames=n_frames, image_size=image_size, seed=0,
        )
        encoder = USEncoder(name=backbone, adaptation="frozen")
        print(f"[{backbone}] caching embeddings (dim={encoder.embed_dim})...")
        bags = precompute_embeddings(dataset, encoder, device=args.device)

        for agg in args.aggregators:
            for dtype in args.decisions:
                train_cfg = replace(base_train, aggregator=agg)
                cfg2 = dict(cfg)
                cfg2["decision"] = {**cfg["decision"], "type": dtype}
                dec_cfg = build_decision_config(cfg2)
                res = cross_validate(
                    bags, train_cfg, dec_cfg,
                    n_splits=int(cfg.get("cv", {}).get("n_splits", 5)),
                    seed=0, device=args.device,
                )
                certified = any(s.get("certified") for s in res.fold_summaries) \
                    if dtype == "conformal" else None
                row = {
                    "backbone": backbone,
                    "aggregation": agg,
                    "decision": dtype,
                    "auc": round(res.auc.auc, 3),
                    "auc_ci": [round(res.auc.ci_low, 3), round(res.auc.ci_high, 3)],
                    "sensitivity": round(res.decision.sensitivity, 3),
                    "specificity": round(res.decision.specificity, 3),
                    "abstention": round(res.decision.abstention_rate, 3),
                    "certified": certified,
                }
                rows.append(row)
                print(
                    f"  {agg:>16} / {dtype:<9} "
                    f"AUC {row['auc']:.3f} [{row['auc_ci'][0]:.3f}-{row['auc_ci'][1]:.3f}] "
                    f"sens {row['sensitivity']:.3f} spec {row['specificity']:.3f}"
                )

    out_dir = Path(cfg.get("logging", {}).get("output_dir", "outputs/track1"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ablations.json").write_text(json.dumps({"rows": rows, "train": asdict(base_train)}, indent=2))
    print(f"\nWrote {len(rows)} ablation rows to {out_dir}/ablations.json")


if __name__ == "__main__":
    main()
