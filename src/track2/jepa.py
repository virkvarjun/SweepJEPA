"""Stage C: pose-conditioned joint-embedding predictive architecture (V-JEPA).

Predicts masked spatio-temporal tubelet latents conditioned on relative probe
pose, in **feature space** (never pixels — US speckle is stochastic; non-negotiable
#4). The "action" conditioning the predictor is the relative probe pose Δ∈SE(3)
between context and target — this pose conditioning is the novelty (#5).

    L_JEPA = (1/|M|) Σ_{j∈M} || P_φ(E_θ(x), Δ_{x→y}, m_j) − sg[Ē_θ(y_j)] ||_1

Toggles wired for the decisive ablations:
  * ``pose_conditioned``: add the pose "action" to target queries, or not.
  * ``teacher_type``: ``"ema"`` (momentum target) vs ``"frozen"`` (SALT-style
    fixed teacher) — the US-JEPA-motivated EMA-vs-frozen ablation.
  * ``target_space``: ``"latent"`` (JEPA) vs ``"input"`` (pixel/token recon
    baseline, matched masking) — the latent-vs-reconstruction ablation.

The token dimension is the US foundation-model feature dim, so Stage C sees the
same backbone as Track 1 (matched backbone/compute, #6) and only tubelet + pose
conditioning differ.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class _TokenTransformer(nn.Module):
    """Pre-LN transformer encoder over a token sequence."""

    def __init__(self, dim: int, depth: int, heads: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.net = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        return self.net(x, src_key_padding_mask=key_padding_mask)


def random_tubelet_mask(
    batch: int,
    n_tubelets: int,
    mask_ratio: float = 0.75,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Boolean target mask ``(B, T)`` — True = masked target, False = visible.

    Guarantees at least one visible and one target token per row so neither the
    context encoder nor the loss degenerates.
    """
    k = int(round(mask_ratio * n_tubelets))
    k = min(max(k, 1), n_tubelets - 1)
    mask = torch.zeros(batch, n_tubelets, dtype=torch.bool)
    for b in range(batch):
        perm = torch.randperm(n_tubelets, generator=generator)
        mask[b, perm[:k]] = True
    return mask


class PoseConditionedJEPA(nn.Module):
    """Pose-conditioned V-JEPA over tubelet tokens (feature space)."""

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 4,
        predictor_depth: int = 2,
        num_heads: int = 6,
        pose_dim: int = 6,
        ema_tau: float = 0.996,
        teacher_type: str = "ema",
        pose_conditioned: bool = True,
        target_space: str = "latent",
        max_tubelets: int = 128,
    ) -> None:
        super().__init__()
        if teacher_type not in ("ema", "frozen"):
            raise ValueError("teacher_type must be 'ema' or 'frozen'")
        if target_space not in ("latent", "input"):
            raise ValueError("target_space must be 'latent' or 'input'")
        self.embed_dim = embed_dim
        self.ema_tau = ema_tau
        self.teacher_type = teacher_type
        self.pose_conditioned = pose_conditioned
        self.target_space = target_space

        self.context_encoder = _TokenTransformer(embed_dim, depth, num_heads)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = _TokenTransformer(embed_dim, predictor_depth, num_heads)
        self.pred_proj = nn.Linear(embed_dim, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tubelets, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.pose_embed = nn.Linear(pose_dim, embed_dim)

    @torch.no_grad()
    def update_ema(self) -> None:
        """Momentum update of the target encoder (no-op for a frozen teacher)."""
        if self.teacher_type != "ema":
            return
        for tp, cp in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            tp.data.mul_(self.ema_tau).add_(cp.data, alpha=1.0 - self.ema_tau)

    def encode_context(self, tokens: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        """Encode only the visible tokens (targets hidden from attention)."""
        t = tokens.shape[1]
        x = tokens + self.pos_embed[:, :t]
        return self.context_encoder(x, key_padding_mask=target_mask)

    def forward(
        self,
        tokens: torch.Tensor,
        target_mask: torch.Tensor,
        poses: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            tokens: ``(B, T, D)`` tubelet tokens (US-FM features per tubelet).
            target_mask: ``(B, T)`` bool, True = masked target.
            poses: ``(B, T, pose_dim)`` relative probe pose per tubelet (the action).

        Returns:
            Scalar feature-space L1 JEPA loss over masked tubelets.
        """
        b, t, d = tokens.shape
        pos = self.pos_embed[:, :t]

        ctx = self.encode_context(tokens, target_mask)  # (B, T, D)

        # Target latents (stop-grad): teacher on the full, unmasked sequence, or
        # the raw input tokens for the reconstruction-baseline ablation.
        with torch.no_grad():
            if self.target_space == "latent":
                target = self.target_encoder(tokens + pos)
            else:
                target = tokens

        # Predictor input: visible positions carry context; masked positions carry
        # the mask token, plus positional and (optionally) pose conditioning.
        mask_tokens = self.mask_token.expand(b, t, d)
        pred_in = torch.where(target_mask.unsqueeze(-1), mask_tokens, ctx)
        pred_in = pred_in + pos
        if self.pose_conditioned:
            pred_in = pred_in + self.pose_embed(poses)
        pred = self.pred_proj(self.predictor(pred_in))

        # L1 over masked positions only.
        per_token = F.l1_loss(pred, target, reduction="none").mean(dim=-1)  # (B, T)
        denom = target_mask.float().sum().clamp(min=1.0)
        return (per_token * target_mask.float()).sum() / denom


def build_jepa(**kwargs) -> PoseConditionedJEPA:
    """Factory mirroring the config knobs (kept for backward compatibility)."""
    return PoseConditionedJEPA(**kwargs)
