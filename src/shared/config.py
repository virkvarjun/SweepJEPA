"""Config loading + object construction shared by the Track 1 entry points.

Keeps ``train_track1`` / ``eval`` / ``run_ablations`` thin: they load one YAML and
ask here for the dataset, encoder, and typed pipeline configs. Every experiment
is reproducible from ``(config, seed)``.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.shared.data import ThyroidCineDataset
from src.shared.synthetic import SyntheticCineDataset, SyntheticStaticDataset
from src.track1.encoder import USEncoder
from src.track1.pipeline import DecisionConfig, TrainConfig


def load_config(path: str | Path) -> dict:
    with Path(path).open() as f:
        return yaml.safe_load(f)


def build_encoder(cfg: dict) -> USEncoder:
    enc = cfg["encoder"]
    lora = enc.get("lora", {}) or {}
    lora_enabled = bool(lora.get("enabled", False))
    return USEncoder(
        name=enc["name"],
        freeze=enc.get("freeze", True),
        lora_rank=int(lora.get("rank", 8)) if lora_enabled else 0,
        lora_alpha=float(lora.get("alpha", 16)),
        weights_path=enc.get("weights_path"),
    )


def build_cine_dataset(cfg: dict, split: str = "train"):
    """Return a dataset of ``NoduleBag`` — synthetic or the real manifest."""
    data = cfg["data"]
    frame = data.get("frame_size", [224, 224])[0]
    manifest = Path(data.get("manifest", "")) if data.get("manifest") else None
    if not data.get("synthetic", True) and manifest and manifest.exists():
        return ThyroidCineDataset.from_manifest(
            manifest, root=Path(data["root"]), split=split, image_size=frame
        )
    seed = cfg.get("cv", {}).get("seed", 0)
    # Synthetic AIMI-shaped data. Size is overridable for fast smoke runs.
    return SyntheticCineDataset(
        n_nodules=int(data.get("synthetic_nodules", 192)),
        n_malignant=int(data.get("synthetic_malignant", 17)),
        n_frames=int(data.get("synthetic_frames", 16)),
        image_size=frame,
        seed=seed,
    )


def build_external_static_dataset(cfg: dict, seed: int = 0):
    """ThyroidXL-shaped external per-image test set (synthetic placeholder)."""
    frame = cfg["data"].get("frame_size", [224, 224])[0]
    return SyntheticStaticDataset(n_images=400, image_size=frame, seed=seed)


def build_train_config(cfg: dict) -> TrainConfig:
    tr = cfg.get("training", {})
    mil = cfg.get("mil", {})
    loss = cfg.get("loss", {})
    agg = mil.get("type", "gated_attention")
    return TrainConfig(
        aggregator=agg,
        hidden_dim=int(mil.get("hidden_dim", 256)),
        dropout=float(mil.get("dropout", 0.25)),
        lr=float(tr.get("lr", 1e-3)),
        weight_decay=float(tr.get("weight_decay", 1e-4)),
        epochs=int(tr.get("epochs", 20)),
        batch_size=int(tr.get("batch_size", 8)),
        focal_alpha=float(loss.get("alpha", 0.25)),
        focal_gamma=float(loss.get("gamma", 2.0)),
        oversample=bool(loss.get("minority_oversample", True)),
        seed=int(tr.get("seed", 0)),
    )


def build_decision_config(cfg: dict) -> DecisionConfig:
    dec = cfg.get("decision", {})
    dtype = dec.get("type", "conformal")
    # config uses the clinical name "conformal"; the pipeline uses "rcps".
    pipeline_type = "rcps" if dtype in ("conformal", "rcps") else "youden"
    return DecisionConfig(
        type=pipeline_type,
        sensitivity_floor=float(dec.get("sensitivity_floor", 0.95)),
        delta=float(dec.get("delta", 0.1)),
        abstention_target=float(dec.get("abstention_target", 0.2)),
        calib_fraction=float(dec.get("calib_fraction", 0.4)),
        sensitivity_weight=float(dec.get("sensitivity_weight", 2.0)),
    )
