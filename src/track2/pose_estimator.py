"""Stage B: sensorless inter-frame pose estimation R_psi."""

from __future__ import annotations

import torch
import torch.nn as nn


class PoseEstimator(nn.Module):
    """Regress relative rigid transform delta_{i->j} in SE(3) from frame pairs.

    Pretrained on externally tracked sweeps; refined on thyroid clips with
    cycle and trajectory-smoothness consistency losses.
    """

    def __init__(self, feature_dim: int = 512) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, feature_dim),
            nn.ReLU(inplace=True),
        )
        # 6-DOF: translation (3) + rotation as axis-angle (3)
        self.head = nn.Linear(feature_dim, 6)

    def forward(self, frame_i: torch.Tensor, frame_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_i, frame_j: (B, 3, H, W) consecutive cine frames.

        Returns:
            (B, 6) relative pose delta_{i->j}.
        """
        pair = torch.cat([frame_i, frame_j], dim=1)
        return self.head(self.encoder(pair))
