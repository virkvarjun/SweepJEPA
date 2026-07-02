"""Gated attention-based multiple-instance learning head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttentionMIL(nn.Module):
    """Attention-MIL pooling with bag-level supervision.

    Attention weights alpha_i expose which frames drive each prediction,
    replacing label broadcasting to every frame.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.25) -> None:
        super().__init__()
        self.attention_v = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.attention_u = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )
        self.attention_w = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, bag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            bag: (B, N, D) per-frame embeddings.

        Returns:
            logits: (B, 1) nodule-level malignancy score z.
            alpha: (B, N) normalized attention weights.
        """
        a_v = self.attention_v(bag)
        a_u = self.attention_u(bag)
        weights = self.attention_w(a_v * a_u).squeeze(-1)
        alpha = F.softmax(weights, dim=-1)
        z = torch.sum(alpha.unsqueeze(-1) * bag, dim=1)
        logits = self.classifier(z)
        return logits, alpha
