#!/usr/bin/env python3
"""Train Track 1: US encoder + attention-MIL + conformal decision."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.track1.conformal import ConformalDecisionLayer
from src.track1.encoder import USEncoder
from src.track1.mil import GatedAttentionMIL


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/track1.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    encoder_cfg = cfg["encoder"]
    mil_cfg = cfg["mil"]
    decision_cfg = cfg["decision"]

    encoder = USEncoder(
        name=encoder_cfg["name"],
        freeze=encoder_cfg["freeze"],
        lora_rank=encoder_cfg["lora"]["rank"] if encoder_cfg["lora"]["enabled"] else 0,
    )
    mil = GatedAttentionMIL(
        input_dim=encoder.embed_dim,
        hidden_dim=mil_cfg["hidden_dim"],
        dropout=mil_cfg["dropout"],
    )
    decision = ConformalDecisionLayer(
        sensitivity_floor=decision_cfg["sensitivity_floor"],
    )

    print(f"Track 1 pipeline ready: {encoder.name} + {mil_cfg['type']} + {decision_cfg['type']}")
    print(f"External test: {cfg['evaluation']['external_test']}")


if __name__ == "__main__":
    main()
