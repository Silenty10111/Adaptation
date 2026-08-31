#!/usr/bin/env python3
"""Backfill left/right planned gait heatmaps for an existing batch.

Historical batch files do not contain measured joint trajectories or the
final probe-selected plan.  This command reconstructs a plan from the saved
description and final forward axis, and labels the generated metadata
accordingly.  New ``batch_test.py`` runs render the executed final plan
directly and do not need reconstruction.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from adaptation.gait import compute_adaptive_plan
from scripts._batch_plot import render_gait_visuals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", type=Path, help="existing batch_results/<run> directory")
    parser.add_argument("--robots", nargs="*", default=None, help="optional robot directory names")
    parser.add_argument("--limit", type=int, default=0, help="maximum robots; 0 means all")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    batch_dir = args.batch_dir.resolve()
    if not batch_dir.is_dir():
        raise SystemExit(f"batch directory not found: {batch_dir}")
    requested = set(args.robots or [])
    robot_dirs = sorted(
        path for path in batch_dir.iterdir()
        if path.is_dir() and (path / "robot_description.json").exists()
        and (path / "trajectory.json").exists()
        and (not requested or path.name in requested)
    )
    if args.limit > 0:
        robot_dirs = robot_dirs[:args.limit]
    png_dir = batch_dir / "png"
    png_dir.mkdir(exist_ok=True)
    completed = 0
    for robot_dir in robot_dirs:
        heatmap = robot_dir / "gait_heatmap.png"
        phase = robot_dir / "gait_phase.png"
        if not args.overwrite and heatmap.exists():
            print(f"[Skip] {robot_dir.name}: heatmap already exists")
            continue
        try:
            description = json.loads(
                (robot_dir / "robot_description.json").read_text(encoding="utf-8")
            )
            trajectory = json.loads(
                (robot_dir / "trajectory.json").read_text(encoding="utf-8")
            )
            axis = trajectory.get("forward_axis", [1.0, 0.0])
            plan = compute_adaptive_plan(description, {}, forced_axis=axis)
            duration = float(trajectory.get(
                "original_sample_count", len(trajectory.get("samples", []))
            )) / 60.0
            data = {
                "robot_name": robot_dir.name,
                "description": description,
                "com_trail": trajectory.get("samples", []),
                "gait_duration_s": max(duration, 2.0 / max(
                    float(plan.get("cpg", {}).get("frequency_hz", 0.85)), 1e-6
                )),
                "gait_plan": plan,
                "gait_plan_source": (
                    "reconstructed_from_saved_description_and_axis"
                ),
            }
            generated = render_gait_visuals(data, robot_dir / "trajectory.png")
            # Older renderer versions created a separate stance/swing plot.
            # It is intentionally removed: the requested output is now only
            # the left/right leg-angle heatmap.
            if phase.exists():
                phase.unlink()
            stale_flat_phase = png_dir / f"{robot_dir.name}__gait_phase.png"
            if stale_flat_phase.exists():
                stale_flat_phase.unlink()
            for source in generated:
                if source.suffix.lower() == ".png":
                    shutil.copy2(
                        source,
                        png_dir / f"{robot_dir.name}__{source.stem}.png",
                    )
            completed += 1
            print(f"[OK] {robot_dir.name}: {heatmap.name}")
        except Exception as exc:
            print(f"[FAIL] {robot_dir.name}: {exc}")
    print(f"[Done] generated gait visuals for {completed}/{len(robot_dirs)} robots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
