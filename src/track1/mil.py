"""Bag aggregators over per-frame embeddings (mask-aware).

All aggregators share the signature ``forward(bag, mask=None) -> (logits, alpha)``
with ``bag`` of shape ``(B, N, D)``, an optional ``(B, N)`` boolean validity mask
(True = real frame), and outputs ``logits (B, 1)`` and per-frame weights
``alpha (B, N)``. This common interface lets the aggregation ablation
(attention-MIL vs mean-pool vs set-transformer) swap one for another with no
other pipeline change. Supervision is always bag-level (non-negotiable #2).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = -1e9


def _apply_mask(weights: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return weights
    return weights.masked_fill(~mask, NEG_INF)


class GatedAttentionMIL(nn.Module):
    """Gated attention-MIL pooling (Ilse et al., 2018) with bag-level supervision.

    Attention weights ``alpha_i`` expose which frames drive each prediction,
    replacing label broadcasting to every frame.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.25) -> None:
        super().__init__()
        self.attention_v = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Dropout(dropout)
        )
        self.attention_u = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Sigmoid(), nn.Dropout(dropout)
        )
        self.attention_w = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(input_dim, 1)

    def forward(
        self, bag: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a_v = self.attention_v(bag)
        a_u = self.attention_u(bag)
        weights = self.attention_w(a_v * a_u).squeeze(-1)  # (B, N)
        alpha = F.softmax(_apply_mask(weights, mask), dim=-1)
        z = torch.sum(alpha.unsqueeze(-1) * bag, dim=1)
        logits = self.classifier(z)
        return logits, alpha


class MeanPoolMIL(nn.Module):
    """Masked mean-pool baseline aggregator."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.25) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, bag: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            z = bag.mean(dim=1)
            alpha = bag.new_full(bag.shape[:2], 1.0 / bag.shape[1])
        else:
            m = mask.unsqueeze(-1).to(bag.dtype)
            counts = m.sum(dim=1).clamp(min=1.0)
            z = (bag * m).sum(dim=1) / counts
            alpha = mask.to(bag.dtype) / mask.sum(dim=1, keepdim=True).clamp(min=1).to(bag.dtype)
        return self.classifier(z), alpha


class SetTransformerMIL(nn.Module):
    """Set-transformer aggregator: self-attention over frames + attention pool.

    A permutation-equivariant encoder (``TransformerEncoder``) followed by pooling
    by multihead attention (PMA) with a single learned seed. This is the stronger
    aggregation ablation arm that Track 2's pose-conditioned model must beat.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.seed = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pool = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(
        self, bag: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.proj(bag)
        key_padding = (~mask) if mask is not None else None
        h = self.encoder(h, src_key_padding_mask=key_padding)
        q = self.seed.expand(h.size(0), -1, -1)
        pooled, attn = self.pool(
            q, h, h, key_padding_mask=key_padding, need_weights=True
        )
        logits = self.classifier(pooled.squeeze(1))
        alpha = attn.squeeze(1)  # (B, N)
        return logits, alpha


AGGREGATORS = {
    "gated_attention": GatedAttentionMIL,
    "average": MeanPoolMIL,
    "mean": MeanPoolMIL,
    "set_transformer": SetTransformerMIL,
}


def build_aggregator(name: str, input_dim: int, **kwargs) -> nn.Module:
    """Factory for the aggregation ablation."""
    if name not in AGGREGATORS:
        raise ValueError(f"unknown aggregator {name!r}; choose from {sorted(AGGREGATORS)}")
    return AGGREGATORS[name](input_dim=input_dim, **kwargs)
