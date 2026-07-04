"""Compose inter-frame SE(3) poses into a pseudo-volume and pose graph.

Stage B of Track 2: given per-frame relative poses (from the tracker on TUS-REC,
or estimated by R_psi on thyroid cine), compose them into an absolute pose graph
and a 3D pseudo-volume, then partition the sweep into spatio-temporal tubelets
that feed the pose-conditioned JEPA.
"""

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


def compose_relative_transforms(relatives: torch.Tensor) -> torch.Tensor:
    """Compose (N-1, 4, 4) relative transforms into (N, 4, 4) absolute poses.

    Identity at frame 0; ``T_k = T_{k-1} @ delta_{k-1->k}``. The matrix analogue
    of :func:`compose_sweep_poses` (which takes 6-DOF vectors).
    """
    device, dtype = relatives.device, relatives.dtype
    poses = [torch.eye(4, device=device, dtype=dtype)]
    for i in range(relatives.shape[0]):
        poses.append(poses[-1] @ relatives[i])
    return torch.stack(poses, dim=0)


def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert a 6D continuous rotation (Zhou et al., 2019) to (..., 3, 3).

    The 6D representation is continuous and singularity-free, which regresses far
    better than axis-angle for pose networks. The two 3-vectors are
    Gram-Schmidt-orthonormalised into the first two columns; the third is their
    cross product.
    """
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1, eps=1e-8)
    a2_proj = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(a2_proj, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # columns


def se3_from_rot6d(pose9: torch.Tensor) -> torch.Tensor:
    """Build (..., 4, 4) transforms from 9-DOF (translation 3 + rot6d 6)."""
    t = pose9[..., :3]
    r = rot6d_to_matrix(pose9[..., 3:])
    b = pose9.shape[:-1]
    transform = torch.zeros(*b, 4, 4, device=pose9.device, dtype=pose9.dtype)
    transform[..., :3, :3] = r
    transform[..., :3, 3] = t
    transform[..., 3, 3] = 1.0
    return transform


def se3_inverse(transform: torch.Tensor) -> torch.Tensor:
    """Inverse of an SE(3) transform (..., 4, 4) via ``R^T`` and ``-R^T t``."""
    r = transform[..., :3, :3]
    t = transform[..., :3, 3]
    r_inv = r.transpose(-1, -2)
    out = torch.zeros_like(transform)
    out[..., :3, :3] = r_inv
    out[..., :3, 3] = -(r_inv @ t.unsqueeze(-1)).squeeze(-1)
    out[..., 3, 3] = 1.0
    return out


def relative_pose(pose_i: torch.Tensor, pose_j: torch.Tensor) -> torch.Tensor:
    """Relative transform mapping frame i's coordinates into frame j: ``T_i^-1 T_j``."""
    return se3_inverse(pose_i) @ pose_j


def pose_graph_edges(poses: torch.Tensor) -> torch.Tensor:
    """Consecutive relative transforms of an absolute pose graph.

    Args:
        poses: (..., N, 4, 4) absolute poses.

    Returns:
        (..., N-1, 4, 4) relatives ``delta_{i->i+1} = T_i^-1 T_{i+1}``.
    """
    return relative_pose(poses[..., :-1, :, :], poses[..., 1:, :, :])


def partition_temporal_tubelets(
    features: torch.Tensor,
    poses: torch.Tensor,
    temporal_size: int = 2,
):
    """Group an ordered frame sequence into temporal tubelets.

    Args:
        features: (N, D) per-frame embeddings (from the US foundation encoder).
        poses: (N, 4, 4) absolute frame poses aligned with ``features``.
        temporal_size: number of frames per tubelet; a trailing remainder shorter
            than ``temporal_size`` still forms a (smaller) final tubelet.

    Returns:
        ``(tokens, tubelet_poses, deltas)`` where
          * ``tokens`` (T, D) is the mean-pooled feature per tubelet,
          * ``tubelet_poses`` (T, 4, 4) is the pose of each tubelet's first frame,
          * ``deltas`` (T, 6-ish) placeholder-free relative pose of each tubelet to
            the previous one as a flat [translation(3), log-rot(3)] is returned by
            :func:`relative_pose_vector`.

    Note: this is the *temporal* tubelet partition over pooled per-frame tokens.
    Full spatio-temporal tubelets (spatial patch tokens x time) are a refinement
    tracked in docs/STATUS.md.
    """
    n = features.shape[0]
    if n != poses.shape[0]:
        raise ValueError("features and poses must share the frame dimension")
    tokens = []
    tubelet_poses = []
    for start in range(0, n, temporal_size):
        end = min(start + temporal_size, n)
        tokens.append(features[start:end].mean(dim=0))
        tubelet_poses.append(poses[start])
    tokens = torch.stack(tokens, dim=0)
    tubelet_poses = torch.stack(tubelet_poses, dim=0)
    deltas = relative_pose_vector(tubelet_poses)
    return tokens, tubelet_poses, deltas


def relative_pose_vector(poses: torch.Tensor) -> torch.Tensor:
    """Per-step relative pose as a flat 6-vector [translation(3), axis-angle(3)].

    Element 0 is zero (no predecessor). Used as the JEPA "action" conditioning.
    """
    n = poses.shape[0]
    out = torch.zeros(n, 6, device=poses.device, dtype=poses.dtype)
    if n > 1:
        rel = relative_pose(poses[:-1], poses[1:])  # (N-1, 4, 4)
        out[1:, :3] = rel[:, :3, 3]
        out[1:, 3:] = matrix_to_axis_angle(rel[:, :3, :3])
    return out


def matrix_to_axis_angle(r: torch.Tensor) -> torch.Tensor:
    """Rotation matrices (..., 3, 3) -> axis-angle (..., 3)."""
    cos = ((r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2]) - 1.0) * 0.5
    cos = cos.clamp(-1.0, 1.0)
    theta = torch.acos(cos)
    axis = torch.stack(
        [
            r[..., 2, 1] - r[..., 1, 2],
            r[..., 0, 2] - r[..., 2, 0],
            r[..., 1, 0] - r[..., 0, 1],
        ],
        dim=-1,
    )
    sin = torch.sin(theta).unsqueeze(-1)
    axis = torch.where(sin.abs() < 1e-6, torch.zeros_like(axis), axis / (2.0 * sin))
    return axis * theta.unsqueeze(-1)


def build_pseudo_volume(poses: torch.Tensor, frame_size: float = 1.0) -> dict:
    """Place frame planes in 3D by their poses; return a releasable pose graph.

    Produces the axis-aligned extent and each frame plane's four corner points in
    world space — enough to release reconstructed pose graphs / pseudo-volumes
    (Milestone 5) and to visualise the sweep. Frames are unit squares in their
    local xy-plane, sized by ``frame_size``.

    Returns a dict with ``corners`` (N, 4, 3), ``centers`` (N, 3), and
    ``extent`` (2, 3) [min, max].
    """
    h = frame_size * 0.5
    local = torch.tensor(
        [[-h, -h, 0.0], [h, -h, 0.0], [h, h, 0.0], [-h, h, 0.0]],
        device=poses.device, dtype=poses.dtype,
    )  # (4, 3)
    local_h = torch.cat([local, torch.ones(4, 1, dtype=poses.dtype, device=poses.device)], dim=-1)
    corners = torch.einsum("nij,kj->nki", poses, local_h)[..., :3]  # (N, 4, 3)
    centers = poses[:, :3, 3]
    pts = corners.reshape(-1, 3)
    extent = torch.stack([pts.min(dim=0).values, pts.max(dim=0).values], dim=0)
    return {"corners": corners, "centers": centers, "extent": extent}
