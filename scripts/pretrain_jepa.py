#!/usr/bin/env python3
"""Stage C: pose-conditioned V-JEPA pretraining on unlabeled cine tubelets.

Encodes unlabeled cine clips with the (matched) US foundation backbone, partitions
each into pose-annotated tubelets, and pretrains the pose-conditioned JEPA with a
feature-space L1 objective + EMA target. Saves the context encoder for Stage D.

    PYTHONPATH=. python scripts/pretrain_jepa.py --config configs/track2.yaml --fast
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from src.shared.synthetic import SyntheticCineDataset
from src.track1.encoder import USEncoder
from src.track2.jepa import PoseConditionedJEPA
from src.track2.stage_c import build_clip_tubelets, pretrain_jepa, stack_tubelets


def load_config(path: Path) -> dict:
    with Path(path).open() as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/track2.yaml"))
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sc = cfg["stage_c"]
    enc_cfg = cfg["encoder"]
    data = cfg["data"]

    n_clips = 20 if args.fast else int(data.get("synthetic_pretrain_clips", 80))
    n_frames = 8 if args.fast else int(data.get("synthetic_frames", 16))
    image_size = (
        96 if (args.fast and enc_cfg["name"] == "imagenet_cnn") else data.get("frame_size", [224])[0]
    )
    epochs = 8 if args.fast else int(sc.get("epochs", 30))

    encoder = USEncoder(name=enc_cfg["name"], adaptation="frozen")
    print(f"Backbone {encoder.name} dim={encoder.embed_dim} weights={encoder.load_info.source}")

    # Unlabeled cine (labels ignored during pretraining).
    dataset = SyntheticCineDataset(
        n_nodules=n_clips, n_malignant=n_clips // 3, n_frames=n_frames,
        image_size=image_size, seed=0,
    )
    print(f"Building tubelets from {len(dataset)} unlabeled clips...")
    clips = build_clip_tubelets(
        dataset, encoder, temporal_size=sc["tubelet"]["temporal_size"], device=args.device
    )
    tokens, poses = stack_tubelets(clips)
    print(f"Tubelet tensor: {tuple(tokens.shape)} (clips, tubelets, dim)")

    jepa = PoseConditionedJEPA(
        embed_dim=encoder.embed_dim,
        depth=sc.get("depth", 4),
        predictor_depth=sc.get("predictor_depth", 2),
        num_heads=sc.get("num_heads", 6),
        ema_tau=sc.get("ema_tau", 0.996),
        teacher_type=sc.get("teacher_type", "ema"),
        pose_conditioned=sc.get("pose_conditioning", True),
        target_space=sc.get("target_space", "latent"),
    )
    history = pretrain_jepa(
        tokens, poses, jepa, epochs=epochs, batch_size=sc.get("batch_size", 8),
        lr=sc.get("lr", 1e-3), mask_ratio=sc.get("mask_ratio", 0.75),
        seed=sc.get("seed", 0), device=args.device,
    )
    print(
        f"JEPA loss: {history[0]:.4f} -> {history[-1]:.4f} "
        f"({'decreased' if history[-1] < history[0] else 'flat'})"
    )

    out_dir = Path("outputs/track2")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(jepa.state_dict(), out_dir / "jepa.pt")
    print(f"Saved pretrained JEPA to {out_dir}/jepa.pt")


if __name__ == "__main__":
    main()
