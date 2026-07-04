"""Stage B: sensorless inter-frame pose estimation R_psi.

Regresses the relative rigid transform ``delta_{i->j}`` in SE(3) from a frame
pair. Pretrained supervised on optically-tracked sweeps (TUS-REC2024), then
refined on thyroid cine (no tracker) with cycle + trajectory-smoothness losses.

The rotation is regressed in the continuous **6D representation** (Zhou et al.,
2019) by default — it is singularity-free and trains far better than axis-angle —
and converted to a matrix for losses/composition. An ``axis_angle`` mode is kept
for ablation and matches the 6-DOF convention used elsewhere.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.track2.sweep_reconstruction import (
    rot6d_to_matrix,
    se3_from_6dof,
    se3_from_rot6d,
)


def _conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class PoseEstimator(nn.Module):
    """Regress ``delta_{i->j}`` in SE(3) from a pair of consecutive frames.

    Args:
        feature_dim: width of the pooled feature before the pose head.
        representation: ``"rot6d"`` (default) or ``"axis_angle"``.
    """

    def __init__(self, feature_dim: int = 512, representation: str = "rot6d") -> None:
        super().__init__()
        if representation not in ("rot6d", "axis_angle"):
            raise ValueError("representation must be 'rot6d' or 'axis_angle'")
        self.representation = representation
        self.rot_dim = 6 if representation == "rot6d" else 3

        self.encoder = nn.Sequential(
            _conv_block(6, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            _conv_block(128, 256),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, feature_dim),
            nn.ReLU(inplace=True),
        )
        self.trans_head = nn.Linear(feature_dim, 3)
        self.rot_head = nn.Linear(feature_dim, self.rot_dim)
        # Initialise rotation head to predict identity (rot6d -> [1,0,0,0,1,0]).
        if representation == "rot6d":
            nn.init.zeros_(self.rot_head.weight)
            with torch.no_grad():
                self.rot_head.bias.copy_(torch.tensor([1.0, 0, 0, 0, 1.0, 0]))
        else:
            nn.init.zeros_(self.rot_head.weight)
            nn.init.zeros_(self.rot_head.bias)

    def forward(self, frame_i: torch.Tensor, frame_j: torch.Tensor) -> torch.Tensor:
        """Return the raw pose vector ``[translation(3), rotation(rot_dim)]``."""
        feat = self.encoder(torch.cat([frame_i, frame_j], dim=1))
        return torch.cat([self.trans_head(feat), self.rot_head(feat)], dim=-1)

    def to_matrix(self, pose_vec: torch.Tensor) -> torch.Tensor:
        """Convert a predicted pose vector to a (..., 4, 4) SE(3) transform."""
        if self.representation == "rot6d":
            return se3_from_rot6d(pose_vec)
        return se3_from_6dof(pose_vec)

    def predict_matrix(self, frame_i: torch.Tensor, frame_j: torch.Tensor) -> torch.Tensor:
        return self.to_matrix(self.forward(frame_i, frame_j))

    @torch.no_grad()
    def estimate_sweep(self, frames: torch.Tensor) -> torch.Tensor:
        """Relative transforms along a whole sweep.

        Args:
            frames: (N, 3, H, W) ordered sweep frames.

        Returns:
            (N-1, 4, 4) consecutive relative transforms ``delta_{i->i+1}``.
        """
        return self.predict_matrix(frames[:-1], frames[1:])


def rotation_from_vector(pose_vec: torch.Tensor, representation: str) -> torch.Tensor:
    """Standalone rotation-matrix extraction (for tests / external callers)."""
    if representation == "rot6d":
        return rot6d_to_matrix(pose_vec[..., 3:])
    from src.track2.sweep_reconstruction import axis_angle_to_matrix

    return axis_angle_to_matrix(pose_vec[..., 3:])
