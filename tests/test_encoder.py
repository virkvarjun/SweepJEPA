"""US encoder: forward shapes, adaptation modes, LoRA parameter accounting."""

from __future__ import annotations

import pytest
import torch

from src.track1.encoder import BACKBONES, LoRALinear, USEncoder


def _frames(b=2, n=3, size=224):
    return torch.randn(b, n, 3, size, size)


def test_forward_shape_vit():
    enc = USEncoder(name="usfm", adaptation="frozen").eval()
    with torch.no_grad():
        out = enc(_frames())
    assert out.shape == (2, 3, enc.embed_dim)
    assert enc.embed_dim == 768


def test_forward_shape_cnn():
    enc = USEncoder(name="imagenet_cnn", adaptation="frozen").eval()
    with torch.no_grad():
        out = enc(_frames())
    assert out.shape == (2, 3, 1280)


def test_frozen_has_no_trainable_params():
    enc = USEncoder(name="usfm", adaptation="frozen")
    assert enc.num_trainable() == 0


def test_full_trains_everything():
    enc = USEncoder(name="usfm", adaptation="full")
    total = sum(p.numel() for p in enc.parameters())
    assert enc.num_trainable() == total


def test_lora_only_adapters_trainable():
    enc = USEncoder(name="usfm", adaptation="lora", lora_rank=8)
    assert enc.lora_layers > 0
    # Every trainable parameter belongs to a LoRA module.
    lora_param_ids = {
        id(p)
        for m in enc.modules()
        if isinstance(m, LoRALinear)
        for p in (m.lora_a, m.lora_b)
    }
    for p in enc.trainable_parameters():
        assert id(p) in lora_param_ids
    # And LoRA is far cheaper than full fine-tuning.
    total = sum(p.numel() for p in enc.parameters())
    assert 0 < enc.num_trainable() < 0.05 * total


def test_lora_identity_at_init():
    # B is zero-initialised, so a fresh LoRA layer equals its base linear.
    base = torch.nn.Linear(16, 32)
    lora = LoRALinear(base, rank=4, alpha=8.0)
    x = torch.randn(5, 16)
    assert torch.allclose(lora(x), base(x), atol=1e-6)


def test_lora_rejected_on_cnn():
    with pytest.raises(ValueError):
        USEncoder(name="imagenet_cnn", adaptation="lora", lora_rank=8)


def test_backward_compat_freeze_flag():
    # Old call style (freeze/lora_rank) still resolves the right adaptation.
    assert USEncoder(name="usfm", freeze=True, lora_rank=0).adaptation == "frozen"
    assert USEncoder(name="usfm", freeze=True, lora_rank=4).adaptation == "lora"
    assert USEncoder(name="usfm", freeze=False).adaptation == "full"


def test_unknown_backbone_rejected():
    with pytest.raises(ValueError):
        USEncoder(name="does_not_exist")


def test_all_backbones_report_source():
    for name in BACKBONES:
        enc = USEncoder(name=name, adaptation="frozen")
        assert enc.load_info.source in {"random_init", "imagenet"}
