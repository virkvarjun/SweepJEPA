"""SE(3) geometry: rotation validity, round-trips, pose composition."""

from __future__ import annotations

import math

import torch

from src.track2.sweep_reconstruction import (
    axis_angle_to_matrix,
    compose_sweep_poses,
    se3_from_6dof,
)


def test_zero_axis_angle_is_identity():
    R = axis_angle_to_matrix(torch.zeros(4, 3))
    assert torch.allclose(R, torch.eye(3).expand(4, 3, 3), atol=1e-6)


def test_rotations_are_orthonormal_det_one():
    aa = torch.randn(10, 3)
    R = axis_angle_to_matrix(aa)
    eye = torch.eye(3).expand(10, 3, 3)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-5)
    dets = torch.linalg.det(R)
    assert torch.allclose(dets, torch.ones(10), atol=1e-5)


def test_known_90deg_rotation_about_z():
    aa = torch.tensor([[0.0, 0.0, math.pi / 2]])
    R = axis_angle_to_matrix(aa)[0]
    x_axis = torch.tensor([1.0, 0.0, 0.0])
    # +90deg about z sends +x to +y.
    assert torch.allclose(R @ x_axis, torch.tensor([0.0, 1.0, 0.0]), atol=1e-5)


def test_se3_structure():
    delta = torch.randn(3, 6)
    T = se3_from_6dof(delta)
    assert T.shape == (3, 4, 4)
    assert torch.allclose(T[:, 3, :], torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(3, 4), atol=1e-6)
    assert torch.allclose(T[:, :3, 3], delta[:, :3], atol=1e-6)


def test_compose_zero_deltas_gives_identities():
    deltas = torch.zeros(2, 5, 6)
    poses = compose_sweep_poses(deltas)
    assert poses.shape == (2, 6, 4, 4)
    assert torch.allclose(poses, torch.eye(4).expand(2, 6, 4, 4), atol=1e-6)


def test_compose_accumulates_translation():
    # Pure +x translation of 0.1 per step accumulates linearly.
    deltas = torch.zeros(1, 3, 6)
    deltas[..., 0] = 0.1
    poses = compose_sweep_poses(deltas)
    xs = poses[0, :, 0, 3]
    assert torch.allclose(xs, torch.tensor([0.0, 0.1, 0.2, 0.3]), atol=1e-5)
