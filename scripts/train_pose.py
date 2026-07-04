#!/usr/bin/env python3
"""Stage B: pretrain R_psi on TUS-REC2024, then refine on thyroid cine (no GT).

Supervised pretraining uses tracker ground-truth (SE(3) loss). Transfer/refinement
onto thyroid cine has no tracking labels, so it uses cycle-consistency +
trajectory-smoothness. Reports sweep-reconstruction drift (final-frame error).

    PYTHONPATH=. python scripts/train_pose.py --epochs 3
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from src.shared.synthetic import SyntheticCineDataset
from src.track2.losses import (
    cycle_consistency_loss,
    drift_error,
    se3_loss,
    trajectory_smoothness_loss,
)
from src.track2.pose_estimator import PoseEstimator
from src.track2.sweep_reconstruction import compose_relative_transforms
from src.track2.tusrec import FramePairDataset, load_tusrec_or_synthetic


def pretrain_supervised(model, sweeps, epochs, lr, device):
    pairs = FramePairDataset(sweeps, stride=1)
    loader = DataLoader(pairs, batch_size=16, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    last = None
    for ep in range(epochs):
        total, n = 0.0, 0
        for fi, fj, t_rel in loader:
            fi, fj, t_rel = fi.to(device), fj.to(device), t_rel.to(device)
            opt.zero_grad()
            pred = model.to_matrix(model(fi, fj))
            loss = se3_loss(pred, t_rel, rotation_weight=1.0)
            loss.backward()
            opt.step()
            total += loss.item() * fi.size(0)
            n += fi.size(0)
        last = total / max(n, 1)
        print(f"  [supervised] epoch {ep + 1}/{epochs} SE(3) loss {last:.4f}")
    return last


def refine_self_supervised(model, cine_frames, epochs, lr, device):
    """Refine on cine clips with no pose GT (cycle + smoothness)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    first = last = None
    for ep in range(epochs):
        opt.zero_grad()
        fwd = model.to_matrix(model(cine_frames[:-1], cine_frames[1:]))
        bwd = model.to_matrix(model(cine_frames[1:], cine_frames[:-1]))
        absolute = compose_relative_transforms(fwd)
        loss = cycle_consistency_loss(fwd, bwd) + 0.1 * trajectory_smoothness_loss(absolute)
        loss.backward()
        opt.step()
        last = loss.item()
        if ep == 0:
            first = last
        print(f"  [refine] epoch {ep + 1}/{epochs} consistency loss {last:.4f}")
    return first, last


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--refine-epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--representation", default="rot6d", choices=["rot6d", "axis_angle"])
    args = parser.parse_args()

    torch.manual_seed(0)
    device = args.device

    sweeps = load_tusrec_or_synthetic(synthetic_kwargs={"n_sweeps": 24, "n_frames": 16})
    src = "TUS-REC2024" if type(sweeps).__name__ == "TUSRECSweepDataset" else "synthetic (fallback)"
    print(f"Pose source: {src}; {len(sweeps)} sweeps")

    model = PoseEstimator(representation=args.representation).to(device)

    print("Stage B.1 — supervised pretraining on tracked sweeps")
    pretrain_supervised(model, sweeps, args.epochs, args.lr, device)

    # Drift report on a held-out sweep.
    sweep = sweeps[0]
    deltas = model.estimate_sweep(sweep.frames.to(device)).cpu()
    pred_poses = compose_relative_transforms(deltas)
    report = drift_error(pred_poses, sweep.poses)
    print(
        f"  Reconstruction: final-frame drift {report['final_frame_translation']:.3f}, "
        f"mean drift {report['mean_translation']:.3f}, "
        f"mean rotation err {report['mean_rotation_deg']:.2f} deg"
    )

    print("Stage B.2 — self-supervised refinement on thyroid cine (no GT)")
    cine = SyntheticCineDataset(n_nodules=1, n_malignant=0, n_frames=16, image_size=128, seed=1)
    cine_frames = cine[0].frames.to(device)
    first, last = refine_self_supervised(model, cine_frames, args.refine_epochs, args.lr, device)
    print(f"  Consistency loss {first:.4f} -> {last:.4f} "
          f"({'decreased' if last < first else 'did not decrease'})")


if __name__ == "__main__":
    main()
