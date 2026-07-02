"""Stage C: pose-conditioned joint-embedding predictive architecture."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseConditionedJEPA(nn.Module):
    """Predict masked spatio-temporal tubelet latents conditioned on probe pose.

    L_JEPA = (1/|M|) sum_j || P_phi(E_theta(x), delta_{x->y}, m_j) - sg[E_bar(y_j)] ||_1
    """

    def __init__(
        self,
        embed_dim: int = 768,
        pose_dim: int = 6,
        ema_tau: float = 0.996,
    ) -> None:
        super().__init__()
        self.ema_tau = ema_tau

        self.context_encoder = nn.Linear(embed_dim, embed_dim)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.pose_embed = nn.Linear(pose_dim, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    @torch.no_grad()
    def update_ema(self) -> None:
        for ema_p, ctx_p in zip(
            self.target_encoder.parameters(),
            self.context_encoder.parameters(),
        ):
            ema_p.data.mul_(self.ema_tau).add_(ctx_p.data, alpha=1.0 - self.ema_tau)

    def forward(
        self,
        context: torch.Tensor,
        targets: torch.Tensor,
        delta_pose: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            context: (B, D) visible tubelet embeddings x.
            targets: (B, M, D) masked target embeddings y.
            delta_pose: (B, 6) relative pose delta_{x->y} from Stage B.

        Returns:
            Scalar L1 JEPA loss in feature space.
        """
        ctx_z = self.context_encoder(context)
        pose_z = self.pose_embed(delta_pose)

        with torch.no_grad():
            target_z = self.target_encoder(targets)

        m = self.mask_token.expand(targets.size(0), targets.size(1), -1)
        pred_input = torch.cat(
            [ctx_z.unsqueeze(1).expand(-1, targets.size(1), -1) + pose_z.unsqueeze(1), m],
            dim=-1,
        )
        pred_z = self.predictor(pred_input)
        return F.l1_loss(pred_z, target_z)
