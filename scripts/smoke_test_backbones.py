#!/usr/bin/env python3
"""Smoke-test every registered backbone on synthetic cine frames.

Instantiates each backbone in every adaptation mode it supports, runs a forward
pass on a synthetic bag, and prints the embedding dim, output shape, weight
source, and trainable-parameter count. Exits non-zero if any backbone fails.

    PYTHONPATH=. python scripts/smoke_test_backbones.py
"""

from __future__ import annotations

import sys

import torch

from src.shared.synthetic import make_cine_bag
from src.track1.encoder import BACKBONES, USEncoder


def _adaptations_for(name: str):
    modes = ["frozen", "full"]
    # LoRA only applies to attention-based (ViT) backbones.
    if BACKBONES[name].timm_name.startswith("vit"):
        modes.insert(1, "lora")
    return modes


def main() -> int:
    bag = make_cine_bag(label=1, patient_id="p0", nodule_id="n0", n_frames=4, seed=0)
    frames = bag.frames.unsqueeze(0)  # (1, N, 3, H, W)

    failures = []
    for name in BACKBONES:
        for mode in _adaptations_for(name):
            try:
                kwargs = {"adaptation": mode}
                if mode == "lora":
                    kwargs["lora_rank"] = 8
                enc = USEncoder(name=name, **kwargs).eval()
                with torch.no_grad():
                    out = enc(frames)
                assert out.shape == (1, 4, enc.embed_dim), out.shape
                print(
                    f"[ok] {name:>12} / {mode:<6} "
                    f"dim={enc.embed_dim:<4} out={tuple(out.shape)} "
                    f"src={enc.load_info.source:<12} "
                    f"trainable={enc.num_trainable():,} lora_layers={enc.lora_layers}"
                )
            except Exception as e:  # noqa: BLE001 - smoke test reports all failures
                failures.append((name, mode, repr(e)))
                print(f"[FAIL] {name} / {mode}: {e!r}")

    if failures:
        print(f"\n{len(failures)} backbone/mode combos failed.")
        return 1
    print("\nAll backbones passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
