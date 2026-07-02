#!/usr/bin/env python3
"""Stage B+C: pose recovery and pose-conditioned JEPA pretraining."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.track2.jepa import PoseConditionedJEPA
from src.track2.pose_estimator import PoseEstimator


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/track2.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    stage_c = cfg["stage_c"]

    pose_estimator = PoseEstimator()
    jepa = PoseConditionedJEPA(
        ema_tau=stage_c["ema_tau"],
    )

    print("Track 2 pretraining pipeline ready:")
    print(f"  Stage B: {cfg['stage_b']['pose_estimator']}")
    print(f"  Stage C: {stage_c['architecture']} (mask_ratio={stage_c['mask_ratio']})")
    print(f"  Stage D reuses Track 1 head: {cfg['stage_d']['reuse_track1_head']}")


if __name__ == "__main__":
    main()
