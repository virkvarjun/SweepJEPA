"""Domain-pretrained ultrasound foundation encoder (ViT) with LoRA adaptation.

Track 1's backbone is a domain-pretrained US foundation model (USFM / USF-MAE /
UltraFedFM), loaded as a timm ViT with released weights when available. With only
17 malignant nodules, the label budget must not be spent re-learning ultrasound
texture (non-negotiable #7), so the backbone is one of:

* **frozen** — features only, no gradient;
* **lora** — base frozen, low-rank adapters on attention ``qkv``/``proj`` trained;
* **full** — everything trainable (baseline / ablation).

An ImageNet CNN arm (``imagenet_cnn``) is included as the backbone ablation
(US-FM vs ImageNet-CNN, matching the Cine-CNNTrans MobileNet baseline).

Released US-FM checkpoints are gated; until they land locally, backbones are
built with the correct architecture but random init (``pretrained=False``) so the
whole pipeline is testable offline. Pass ``weights_path`` (or set the per-model
env var) to load real weights; ``load_info`` reports how many keys matched.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import timm
import torch
import torch.nn as nn

Adaptation = str  # "frozen" | "lora" | "full"


@dataclass(frozen=True)
class BackboneSpec:
    timm_name: str
    embed_dim: int
    weights_env: str | None  # env var pointing at a local checkpoint
    imagenet_pretrained: bool  # whether timm has ImageNet weights for this arch


# Registry. The three US foundation models share the ViT-B/16 architecture; only
# their weights differ (that is the point — matched backbone across arms). The
# ImageNet arms exist for the backbone ablation.
BACKBONES: dict[str, BackboneSpec] = {
    "usfm": BackboneSpec("vit_base_patch16_224", 768, "USFM_WEIGHTS", False),
    "usf_mae": BackboneSpec("vit_base_patch16_224", 768, "USF_MAE_WEIGHTS", False),
    "ultrafedfm": BackboneSpec("vit_base_patch16_224", 768, "ULTRAFEDFM_WEIGHTS", False),
    "imagenet_vit": BackboneSpec("vit_base_patch16_224", 768, None, True),
    "imagenet_cnn": BackboneSpec("mobilenetv2_100", 1280, None, True),
}

EncoderName = str


@dataclass
class LoadInfo:
    """Outcome of a weight-loading attempt (surfaced for honest reporting)."""

    source: str  # "checkpoint:<path>" | "imagenet" | "random_init"
    matched_keys: int
    missing_keys: int
    unexpected_keys: int


# --------------------------------------------------------------------------- #
# Minimal LoRA (no external peft dependency)
# --------------------------------------------------------------------------- #
class LoRALinear(nn.Module):
    """Wrap a frozen ``nn.Linear`` with a trainable low-rank update.

    ``y = W x + (alpha / r) * (B A) x``, with ``B`` zero-initialised so the
    adapter is identity at start.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.scaling = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = (x @ self.lora_a.t()) @ self.lora_b.t()
        return self.base(x) + self.scaling * delta


def _inject_lora(
    model: nn.Module, rank: int, alpha: float, targets=("qkv", "proj")
) -> int:
    """Replace attention ``qkv``/``proj`` Linears with LoRA-wrapped versions.

    Returns the number of layers adapted.
    """
    replaced = 0
    for _module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in targets:
                setattr(module, child_name, LoRALinear(child, rank, alpha))
                replaced += 1
    return replaced


# --------------------------------------------------------------------------- #
# Weight loading
# --------------------------------------------------------------------------- #
def _clean_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    for key in ("model", "state_dict", "model_state_dict", "teacher", "target_encoder"):
        if key in state and isinstance(state[key], dict):
            state = state[key]
            break
    cleaned = {}
    for k, v in state.items():
        nk = k
        for prefix in ("module.", "backbone.", "encoder.", "model."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v
    return cleaned


def _load_weights(
    backbone: nn.Module, spec: BackboneSpec, weights_path: str | None
) -> LoadInfo:
    path = weights_path or (os.environ.get(spec.weights_env) if spec.weights_env else None)
    if path and os.path.exists(path):
        raw = torch.load(path, map_location="cpu")
        state = _clean_state_dict(raw)
        model_keys = set(backbone.state_dict().keys())
        result = backbone.load_state_dict(state, strict=False)
        matched = len(model_keys) - len(set(result.missing_keys))
        return LoadInfo(
            source=f"checkpoint:{path}",
            matched_keys=matched,
            missing_keys=len(result.missing_keys),
            unexpected_keys=len(result.unexpected_keys),
        )
    return LoadInfo(source="random_init", matched_keys=0, missing_keys=0, unexpected_keys=0)


class USEncoder(nn.Module):
    """Per-frame feature extractor from a US foundation model.

    ``forward`` maps ``(B, N, 3, H, W)`` cine bags to ``(B, N, D)`` per-frame
    embeddings. Adaptation is chosen from ``freeze`` / ``lora_rank`` for backward
    compatibility, or set explicitly via ``adaptation``.
    """

    def __init__(
        self,
        name: EncoderName = "usfm",
        embed_dim: int | None = None,
        freeze: bool = True,
        lora_rank: int = 0,
        lora_alpha: float = 16.0,
        adaptation: Adaptation | None = None,
        pretrained: bool = False,
        weights_path: str | None = None,
    ) -> None:
        super().__init__()
        if name not in BACKBONES:
            raise ValueError(f"unknown backbone {name!r}; choose from {sorted(BACKBONES)}")
        spec = BACKBONES[name]
        self.name = name
        self.embed_dim = embed_dim or spec.embed_dim

        # Resolve adaptation mode.
        if adaptation is None:
            if not freeze:
                adaptation = "full"
            elif lora_rank > 0:
                adaptation = "lora"
            else:
                adaptation = "frozen"
        self.adaptation = adaptation

        use_imagenet = pretrained and spec.imagenet_pretrained
        self.backbone = timm.create_model(
            spec.timm_name, pretrained=use_imagenet, num_classes=0
        )
        if self.backbone.num_features != self.embed_dim:
            self.embed_dim = self.backbone.num_features

        # Load domain weights if provided; otherwise note the source honestly.
        if use_imagenet:
            self.load_info = LoadInfo("imagenet", self.embed_dim, 0, 0)
        else:
            self.load_info = _load_weights(self.backbone, spec, weights_path)

        self.lora_layers = 0
        if adaptation == "frozen":
            for p in self.backbone.parameters():
                p.requires_grad = False
        elif adaptation == "lora":
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.lora_layers = _inject_lora(self.backbone, lora_rank, lora_alpha)
            if self.lora_layers == 0:
                # CNN backbones have no attention qkv/proj; LoRA is ViT-oriented.
                raise ValueError(
                    f"backbone {name!r} exposes no qkv/proj Linear layers to adapt; "
                    "use adaptation='full' or a ViT backbone for LoRA"
                )
        elif adaptation == "full":
            for p in self.backbone.parameters():
                p.requires_grad = True
        else:
            raise ValueError(f"unknown adaptation {adaptation!r}")

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: ``(B, N, 3, H, W)`` cine frames per nodule bag.

        Returns:
            ``(B, N, D)`` per-frame embeddings ``h_i``.
        """
        if frames.ndim != 5:
            raise ValueError(f"expected (B, N, 3, H, W), got {tuple(frames.shape)}")
        b, n, c, h, w = frames.shape
        flat = frames.reshape(b * n, c, h, w)
        features = self.backbone(flat)
        if features.ndim == 4:  # safety: pool any spatial map
            features = features.mean(dim=(-2, -1))
        return features.view(b, n, -1)
