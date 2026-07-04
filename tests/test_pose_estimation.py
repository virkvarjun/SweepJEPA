"""R_psi geometry: rot6d round-trips, relative pose, tubelets, pose losses."""

from __future__ import annotations

import math

import torch

from src.track2.losses import (
    cycle_consistency_loss,
    drift_error,
    geodesic_rotation_loss,
    se3_loss,
    trajectory_smoothness_loss,
    translation_loss,
)
from src.track2.pose_estimator import PoseEstimator
from src.track2.sweep_reconstruction import (
    axis_angle_to_matrix,
    build_pseudo_volume,
    compose_relative_transforms,
    compose_sweep_poses,
    partition_temporal_tubelets,
    pose_graph_edges,
    relative_pose,
    rot6d_to_matrix,
    se3_from_rot6d,
    se3_inverse,
)


def test_rot6d_is_valid_rotation():
    r = rot6d_to_matrix(torch.randn(8, 6))
    eye = torch.eye(3).expand(8, 3, 3)
    assert torch.allclose(r @ r.transpose(-1, -2), eye, atol=1e-5)
    assert torch.allclose(torch.linalg.det(r), torch.ones(8), atol=1e-5)


def test_rot6d_identity():
    ident6d = torch.tensor([[1.0, 0, 0, 0, 1.0, 0]])
    assert torch.allclose(rot6d_to_matrix(ident6d)[0], torch.eye(3), atol=1e-6)


def test_se3_inverse_roundtrip():
    t = se3_from_rot6d(torch.randn(5, 9))
    inv = se3_inverse(t)
    eye = torch.eye(4).expand(5, 4, 4)
    assert torch.allclose(t @ inv, eye, atol=1e-5)


def test_relative_pose_consistency():
    poses = compose_sweep_poses(0.05 * torch.randn(1, 6, 6))[0]  # (7,4,4)
    rel = relative_pose(poses[2], poses[5])
    # T_2 @ rel == T_5
    assert torch.allclose(poses[2] @ rel, poses[5], atol=1e-5)


def test_pose_graph_edges_recompose():
    poses = compose_sweep_poses(0.05 * torch.randn(1, 5, 6))[0]  # (6,4,4)
    edges = pose_graph_edges(poses)
    recomposed = compose_relative_transforms(edges)
    assert torch.allclose(recomposed, poses, atol=1e-5)


def test_compose_relative_matrices_matches_6dof():
    deltas6 = 0.05 * torch.randn(1, 4, 6)
    from src.track2.sweep_reconstruction import se3_from_6dof

    absolute_6dof = compose_sweep_poses(deltas6)[0]
    absolute_mat = compose_relative_transforms(se3_from_6dof(deltas6[0]))
    assert torch.allclose(absolute_6dof, absolute_mat, atol=1e-5)


def test_partition_temporal_tubelets():
    feats = torch.randn(7, 16)
    poses = compose_sweep_poses(0.05 * torch.randn(1, 6, 6))[0]
    tokens, tposes, deltas = partition_temporal_tubelets(feats, poses, temporal_size=2)
    assert tokens.shape == (4, 16)  # ceil(7/2)
    assert tposes.shape == (4, 4, 4)
    assert deltas.shape == (4, 6)
    assert torch.allclose(deltas[0], torch.zeros(6), atol=1e-6)  # first has no predecessor


def test_pseudo_volume_shapes():
    poses = compose_sweep_poses(0.1 * torch.randn(1, 9, 6))[0]  # (10,4,4)
    vol = build_pseudo_volume(poses, frame_size=1.0)
    assert vol["corners"].shape == (10, 4, 3)
    assert vol["centers"].shape == (10, 3)
    assert vol["extent"].shape == (2, 3)
    assert (vol["extent"][1] >= vol["extent"][0]).all()


def test_pose_losses_zero_at_ground_truth():
    t = se3_from_rot6d(torch.randn(4, 9))
    assert translation_loss(t, t).item() < 1e-6
    # acos is ill-conditioned near 1, so float32 self-loss is ~1e-4, not exactly 0.
    assert geodesic_rotation_loss(t, t).item() < 1e-3
    assert se3_loss(t, t).item() < 1e-3


def test_geodesic_known_angle():
    ident = torch.eye(4).unsqueeze(0)
    rotated = ident.clone()
    rotated[0, :3, :3] = axis_angle_to_matrix(torch.tensor([[0.0, 0.0, math.pi / 2]]))[0]
    assert abs(geodesic_rotation_loss(ident, rotated).item() - math.pi / 2) < 1e-4


def test_cycle_consistency():
    t = se3_from_rot6d(torch.randn(3, 9))
    assert cycle_consistency_loss(t, se3_inverse(t)).item() < 1e-5


def test_trajectory_smoothness_penalises_jerk():
    smooth = compose_sweep_poses(
        0.02 * torch.arange(1, 6).float().view(1, 5, 1) * torch.ones(1, 5, 6)
    )[0]
    jerky = compose_sweep_poses(0.3 * torch.randn(1, 5, 6))[0]
    assert trajectory_smoothness_loss(smooth) < trajectory_smoothness_loss(jerky)


def test_pose_estimator_forward_and_matrix():
    for rep, rot_dim in [("rot6d", 6), ("axis_angle", 3)]:
        model = PoseEstimator(representation=rep)
        fi, fj = torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64)
        vec = model(fi, fj)
        assert vec.shape == (2, 3 + rot_dim)
        mat = model.to_matrix(vec)
        assert mat.shape == (2, 4, 4)
        assert torch.allclose(mat[:, 3, :], torch.tensor([0.0, 0, 0, 1.0]).expand(2, 4), atol=1e-6)


def test_pose_estimator_identity_init_rot6d():
    # rot6d head is initialised to identity rotation.
    model = PoseEstimator(representation="rot6d").eval()
    with torch.no_grad():
        vec = model(torch.zeros(1, 3, 32, 32), torch.zeros(1, 3, 32, 32))
    r = model.to_matrix(vec)[0, :3, :3]
    assert torch.allclose(r, torch.eye(3), atol=1e-5)


def test_estimate_sweep_and_drift():
    model = PoseEstimator().eval()
    frames = torch.randn(8, 3, 48, 48)
    deltas = model.estimate_sweep(frames)
    assert deltas.shape == (7, 4, 4)
    poses = compose_relative_transforms(deltas)
    report = drift_error(poses, poses)  # against itself -> ~zero drift
    assert report["final_frame_translation"] < 1e-4
