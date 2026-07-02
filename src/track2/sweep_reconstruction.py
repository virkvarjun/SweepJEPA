"""Compose inter-frame SE(3) poses into a pseudo-volume and pose graph."""

from __future__ import annotations

import torch


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle (B, 3) to rotation matrices (B, 3, 3)."""
    theta = torch.norm(axis_angle, dim=-1, keepdim=True).clamp(min=1e-8)
    k = axis_angle / theta
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zeros = torch.zeros_like(kx)
    K = torch.stack(
        [
            zeros, -kz, ky,
            kz, zeros, -kx,
            -ky, kx, zeros,
        ],
        dim=-1,
    ).view(*axis_angle.shape[:-1], 3, 3)
    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    eye = eye.expand(*axis_angle.shape[:-1], 3, 3)
    sin_t = torch.sin(theta)[..., None]
    cos_t = torch.cos(theta)[..., None]
    return eye + sin_t * K + (1 - cos_t) * (K @ K)


def se3_from_6dof(delta: torch.Tensor) -> torch.Tensor:
    """Build 4x4 homogeneous transforms from 6-DOF (translation + axis-angle)."""
    t = delta[..., :3]
    r = axis_angle_to_matrix(delta[..., 3:])
    b = delta.shape[:-1]
    transform = torch.zeros(*b, 4, 4, device=delta.device, dtype=delta.dtype)
    transform[..., :3, :3] = r
    transform[..., :3, 3] = t
    transform[..., 3, 3] = 1.0
    return transform


def compose_sweep_poses(deltas: torch.Tensor) -> torch.Tensor:
    """Compose consecutive relative poses into absolute frame poses.

    Args:
        deltas: (B, N-1, 6) inter-frame transforms delta_{i->i+1}.

    Returns:
        (B, N, 4, 4) absolute pose graph (identity at frame 0).
    """
    transforms = se3_from_6dof(deltas)
    b, n_minus_1 = transforms.shape[:2]
    poses = [torch.eye(4, device=deltas.device, dtype=deltas.dtype).expand(b, 4, 4)]
    cumulative = poses[0]
    for i in range(n_minus_1):
        cumulative = cumulative @ transforms[:, i]
        poses.append(cumulative)
    return torch.stack(poses, dim=1)
