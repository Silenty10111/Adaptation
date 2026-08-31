#!/usr/bin/env python3
"""Ordered, simulation-in-the-loop locomotion validation and auto-tuning.

Validation order is fixed: standard hexapod -> amputated hexapods -> arbitrary
generated morphologies.  Every accepted result must pass a long confirmation;
failed plans are refined along their measured steady displacement direction.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from reexec import maybe_reexec

maybe_reexec(
    os.environ.get("ISAAC_PYTHON", "/data/conda/envs/unitree-rl/bin/python"),
    os.environ.get("ISAAC_LD_LIBRARY_PATH", "/data/conda/envs/unitree-rl/lib"),
)

from adaptation.autotune import (  # noqa: E402
    generate_candidates, refine_axis_from_trajectory, score_metrics,
)
from adaptation.sim import _RobotSimCtx  # noqa: E402
from adaptation.stability import evaluate_ssm  # noqa: E402
from adaptation.validation import evaluate_trajectory  # noqa: E402


AMPUTATION_NAMES = [
    *(f"missing_leg_{i}" for i in range(6)),
    "missing_legs_01", "missing_legs_12", "missing_legs_34", "missing_legs_45",
    "missing_legs_03", "missing_legs_14", "missing_legs_25",
    "missing_legs_05", "missing_legs_23", "missing_legs_02", "missing_legs_35",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "standard", "amputated", "arbitrary"), default="all")
    parser.add_argument("--probe-steps", type=int, default=720)
    parser.add_argument("--validation-steps", type=int, default=1200)
    parser.add_argument("--max-candidates", type=int, default=18)
    parser.add_argument("--axis-refinements", type=int, default=2)
    parser.add_argument("--amputation-root", type=Path,
                        default=REPO_ROOT / "validation_results" / "dev_amputation_fixed")
    parser.add_argument("--arbitrary-root", type=Path,
                        default=REPO_ROOT / "batch_results" / "20260712_162341")
    parser.add_argument("--arbitrary-count", type=int, default=5)
    parser.add_argument("--robots", default="",
                        help="可选逗号分隔机器人名称，只复验指定案例。")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def measure(ctx, plan, steps):
    trail, axis, yaw, diagnostics = ctx.run_episode(
        plan, steps, return_yaw_stats=True, return_diagnostics=True,
        diagnostic_stride=5,
    )
    metrics = evaluate_trajectory(
        trail, axis, gait_frequency_hz=float(plan.get("cpg", {}).get("frequency_hz", 0.85))
    )
    metrics["episode_diagnostics"] = diagnostics
    return trail, metrics, yaw


def validate_one(name, asset_dir: Path, urdf_name: str, args):
    description = json.loads((asset_dir / "robot_description.json").read_text())
    ssm = evaluate_ssm(description, threshold=0.05)
    candidates = generate_candidates(description, args.max_candidates)
    best = None
    ranked = []
    history = []

    with _RobotSimCtx(description, asset_dir / urdf_name, use_gpu=True) as ctx:
        for candidate in candidates:
            trail, probe, _ = measure(ctx, candidate.plan, args.probe_steps)
            history.append({"name": candidate.name, "phase": "probe", "metrics": probe})
            rank = score_metrics(probe)
            ranked.append((rank, candidate))
            if best is None or rank > best[0]:
                best = (rank, candidate.name, candidate.plan, trail, probe)
            if probe["passed"]:
                long_trail, confirmed, _ = measure(ctx, candidate.plan, args.validation_steps)
                history.append({"name": candidate.name, "phase": "confirm", "metrics": confirmed})
                if confirmed["passed"]:
                    best = (score_metrics(confirmed), candidate.name, candidate.plan, long_trail, confirmed)
                    break
                if score_metrics(confirmed) > best[0]:
                    best = (score_metrics(confirmed), candidate.name, candidate.plan, long_trail, confirmed)

        # Very asymmetric crawls can spend most of a short probe settling.
        # Confirm the strongest probe candidates and the reverse-axis tail even
        # when none passed early; this catches viable slow forward/backward gaits.
        if not best[4]["passed"]:
            already_confirmed = {
                item["name"] for item in history if item["phase"] == "confirm"
            }
            ordered = [item[1] for item in sorted(ranked, key=lambda x: x[0], reverse=True)[:4]]
            ordered.extend(candidates[-3:])
            seen = set()
            for candidate in ordered:
                if candidate.name in seen or candidate.name in already_confirmed:
                    continue
                seen.add(candidate.name)
                long_trail, confirmed, _ = measure(ctx, candidate.plan, args.validation_steps)
                history.append({"name": candidate.name, "phase": "confirm", "metrics": confirmed})
                if score_metrics(confirmed) > best[0]:
                    best = (score_metrics(confirmed), candidate.name, candidate.plan, long_trail, confirmed)
                if confirmed["passed"]:
                    break

        _, best_name, best_plan, best_trail, best_metrics = best

        # Closed-loop axis calibration.  Rebuild the complete plan along the
        # direction the robot actually sustained, not just a static PCA axis.
        for iteration in range(args.axis_refinements):
            if best_metrics["passed"]:
                break
            refined = refine_axis_from_trajectory(description, best_plan, best_trail)
            trail, metrics, _ = measure(ctx, refined, args.validation_steps)
            tag = f"{best_name}_axis_refine_{iteration + 1}"
            history.append({"name": tag, "phase": "confirm", "metrics": metrics})
            if score_metrics(metrics) > score_metrics(best_metrics):
                best_name, best_plan, best_trail, best_metrics = tag, refined, trail, metrics

        # Straight but slow plans get one bounded cadence/amplitude rescue.
        if not best_metrics["passed"] and float(best_metrics.get("forward_speed", 0.0)) > 0.010:
            rescue = copy.deepcopy(best_plan)
            rescue["cpg"]["frequency_hz"] = 1.05
            amps = rescue["topology"].get("per_leg_stride_amplitudes", {})
            rescue["_per_amp_override"] = {k: min(0.90, float(v) * 1.35) for k, v in amps.items()}
            trail, metrics, _ = measure(ctx, rescue, args.validation_steps)
            history.append({"name": f"{best_name}_speed_rescue", "phase": "confirm", "metrics": metrics})
            if score_metrics(metrics) > score_metrics(best_metrics):
                best_name, best_plan, best_trail, best_metrics = (
                    f"{best_name}_speed_rescue", rescue, trail, metrics
                )

    row = {
        "robot": name,
        "num_legs": description.get("num_legs"),
        "ssm": ssm["ssm"],
        "passed": best_metrics["passed"],
        "selected_plan": best_name,
        "final_axis": best_plan.get("final_forward_axis"),
        "metrics": best_metrics,
        "history": history,
    }
    print(f"[{'PASS' if row['passed'] else 'FAIL'}] {name}: {best_name} "
          f"v={best_metrics.get('forward_speed', 0):.3f}m/s "
          f"drift={best_metrics.get('drift_ratio', 0):.3f} "
          f"cv={best_metrics.get('speed_cv', 0):.3f}", flush=True)
    return row


def arbitrary_assets(root: Path, count: int):
    summary = json.loads((root / "summary.json").read_text())
    names = [r["robot"] for r in summary if r.get("status") == "ok" and r["robot"] != "robot_ref_standard"]
    for name in names[:count]:
        yield name, root / name, "robot.urdf"


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or (REPO_ROOT / "validation_results" / timestamp / "summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    assets = []
    if args.stage in ("all", "standard"):
        assets.append(("standard_hexapod", REPO_ROOT / "robot_assets" / "standard_hexapod", "generated_robot.urdf"))
    if args.stage in ("all", "amputated"):
        assets.extend((name, args.amputation_root / name, "robot.urdf") for name in AMPUTATION_NAMES)
    if args.stage in ("all", "arbitrary"):
        assets.extend(arbitrary_assets(args.arbitrary_root, args.arbitrary_count))
    selected = {name.strip() for name in args.robots.split(",") if name.strip()}
    if selected:
        assets = [item for item in assets if item[0] in selected]

    results = []
    for name, directory, urdf in assets:
        results.append(validate_one(name, directory, urdf, args))
        output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    passed = sum(bool(row["passed"]) for row in results)
    print(f"[Summary] {passed}/{len(results)} passed -> {output}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
