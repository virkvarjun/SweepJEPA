"""Losses for sensorless pose recovery (Stage B / R_psi).

Two regimes:

* **Supervised** (TUS-REC2024, tracker GT): SE(3) loss = translation error +
  geodesic rotation error between predicted and ground-truth relative transforms.
* **Self-supervised refinement** (thyroid cine, no tracker): cycle-consistency
  (composing i->j and j->i should return to identity) and trajectory-smoothness
  (freehand sweeps move smoothly, so second differences of the pose should be
  small) let R_psi transfer to thyroid clips without pose labels.
"""

from __future__ import annotations

import torch

from src.track2.sweep_reconstruction import matrix_to_axis_angle, se3_inverse


def translation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L2 translation error between (..., 4, 4) transforms."""
    return (pred[..., :3, 3] - target[..., :3, 3]).norm(dim=-1).mean()


def geodesic_rotation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Geodesic angle (radians) between predicted and target rotations.

    ``theta = arccos((tr(R_pred^T R_target) - 1) / 2)``.
    """
    r_pred = pred[..., :3, :3]
    r_target = target[..., :3, :3]
    rel = r_pred.transpose(-1, -2) @ r_target
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.acos(cos).mean()


def se3_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    rotation_weight: float = 1.0,
) -> torch.Tensor:
    """Combined SE(3) supervised loss = translation + weighted geodesic rotation."""
    return translation_loss(pred, target) + rotation_weight * geodesic_rotation_loss(
        pred, target
    )


def cycle_consistency_loss(
    forward: torch.Tensor, backward: torch.Tensor
) -> torch.Tensor:
    """Composing forward (i->j) then backward (j->i) should give identity.

    Args:
        forward: (..., 4, 4) predicted transform i->j.
        backward: (..., 4, 4) predicted transform j->i.
    """
    composed = forward @ backward
    eye = torch.eye(4, device=forward.device, dtype=forward.dtype).expand_as(composed)
    return (composed - eye).abs().mean()


def trajectory_smoothness_loss(poses_or_deltas: torch.Tensor) -> torch.Tensor:
    """Penalise jerk: the second difference of the pose trajectory.

    Accepts either absolute poses ``(N, 4, 4)`` (uses translation + rotation) or
    a flat delta sequence ``(N, C)``. Freehand probe motion is smooth, so large
    second differences are unphysical.
    """
    if poses_or_deltas.dim() >= 3 and poses_or_deltas.shape[-2:] == (4, 4):
        trans = poses_or_deltas[..., :3, 3]  # (N, 3)
        aa = matrix_to_axis_angle(poses_or_deltas[..., :3, :3])  # (N, 3)
        seq = torch.cat([trans, aa], dim=-1)  # (N, 6)
    else:
        seq = poses_or_deltas
    if seq.shape[0] < 3:
        return seq.new_zeros(())
    second_diff = seq[2:] - 2.0 * seq[1:-1] + seq[:-2]
    return second_diff.abs().mean()


def drift_error(pred_poses: torch.Tensor, gt_poses: torch.Tensor) -> dict:
    """Reconstruction error metrics for a composed sweep (report card for R_psi).

    Args:
        pred_poses: (N, 4, 4) composed absolute poses from predicted deltas.
        gt_poses: (N, 4, 4) ground-truth absolute poses.

    Returns:
        dict with ``final_frame_translation`` (drift at the last frame),
        ``mean_translation`` (avg over frames), and ``mean_rotation_deg``.
    """
    # Align both to identity at frame 0 to measure accumulated drift.
    pred = se3_inverse(pred_poses[:1]) @ pred_poses
    gt = se3_inverse(gt_poses[:1]) @ gt_poses
    trans_err = (pred[:, :3, 3] - gt[:, :3, 3]).norm(dim=-1)
    rot_err = geodesic_rotation_loss(pred, gt) * 180.0 / torch.pi
    return {
        "final_frame_translation": float(trans_err[-1].item()),
        "mean_translation": float(trans_err.mean().item()),
        "mean_rotation_deg": float(rot_err.item()),
    }
