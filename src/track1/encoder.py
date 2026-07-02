"""Domain-pretrained ultrasound foundation encoder (ViT)."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

EncoderName = Literal["usfm", "usf_mae", "ultrafedfm"]


class USEncoder(nn.Module):
    """Per-frame feature extractor from a US foundation model.

    Supports frozen inference or LoRA adaptation so the 192-nodule label
    budget is not spent re-learning ultrasound texture.
    """

    def __init__(
        self,
        name: EncoderName = "usfm",
        embed_dim: int = 768,
        freeze: bool = True,
        lora_rank: int = 0,
    ) -> None:
        super().__init__()
        self.name = name
        self.embed_dim = embed_dim
        # Placeholder backbone; swap with USFM / USF-MAE / UltraFedFM weights.
        self.backbone = nn.Identity()
        self.freeze = freeze
        self.lora_rank = lora_rank

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (B, N, 3, H, W) cine frames per nodule bag.

        Returns:
            (B, N, D) per-frame embeddings h_i.
        """
        b, n, c, h, w = frames.shape
        flat = frames.view(b * n, c, h, w)
        features = self.backbone(flat)
        if features.ndim == 4:
            features = features.mean(dim=(-2, -1))
        return features.view(b, n, -1)
