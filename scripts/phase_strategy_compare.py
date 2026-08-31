#!/usr/bin/env python3
"""Reproducible Isaac Gym comparison for gait phase strategies.

Results are written directly under ``batch_results``.  Metrics unavailable
from the current simulator API are emitted as null with an explicit reason;
the script never substitutes synthetic collision, slip, or recovery values.
"""

from __future__ import annotations

import argparse
import copy
import html
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from reexec import maybe_reexec

TARGET_PYTHON = os.environ.get("ISAAC_PYTHON", "/data/conda/envs/unitree-rl/bin/python")
TARGET_LD_PATH = os.environ.get("ISAAC_LD_LIBRARY_PATH", "/data/conda/envs/unitree-rl/lib")
PLOT_PYTHON = os.environ.get("ADAPTATION_PYTHON", "/data/conda/envs/Adaptation/bin/python")
PLOT_SCRIPT = REPO_ROOT / "scripts" / "_batch_plot.py"
maybe_reexec(TARGET_PYTHON, TARGET_LD_PATH)

from adaptation.gait import compute_adaptive_plan
from adaptation.autotune import refine_axis_from_trajectory
from adaptation.morphology import amputate_legs
from adaptation.phase import leg_phase_state, resolve_duty_factors, resolve_phase_offsets
from adaptation.sim import _RobotSimCtx
from adaptation.stability import compute_projected_com_xy, compute_ssm, evaluate_ssm
from adaptation.validation import evaluate_trajectory
from scripts.generate_urdf import build_urdf


STRATEGIES: List[Tuple[str, str, float]] = [
    ("binary", "binary", 1.0),
    ("hildebrand", "hildebrand", 1.0),
    ("uniform_wave", "uniform_wave", 1.0),
    ("balanced_wave", "balanced_wave", 1.0),
    ("geometry_wave_1.0", "geometry_wave", 1.0),
    ("geometry_wave_1.25", "geometry_wave", 1.25),
    ("geometry_wave_1.5", "geometry_wave", 1.5),
    ("adaptive_wave", "adaptive_wave", 1.0),
]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_generated_cases(leg_count: int, limit: int = 1) -> List[Tuple[Path, Path]]:
    candidates = sorted(
        REPO_ROOT.glob("batch_results/*/robot_*/robot_description.json"),
        reverse=True,
    )
    matches: List[Tuple[Path, Path]] = []
    seen_seeds = set()
    for description_path in candidates:
        if "robot_ref_standard" in description_path.parts:
            continue
        description = _load(description_path)
        if int(description.get("num_legs", -1)) != leg_count:
            continue
        seed = description.get("seed")
        if seed in seen_seeds:
            continue
        for name in ("robot.urdf", "generated_robot.urdf"):
            urdf_path = description_path.parent / name
            if urdf_path.exists():
                matches.append((description_path, urdf_path))
                seen_seeds.add(seed)
                break
        if len(matches) >= limit:
            return matches
    if not matches:
        raise FileNotFoundError(f"no generated {leg_count}-leg description/URDF pair found")
    return matches


def build_cases_from_root(models_root: Path, limit: int = 0) -> List[dict]:
    """Build arbitrary-morphology cases from a generated batch directory."""
    root = models_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"models root not found: {root}")
    cases: List[dict] = []
    for description_path in sorted(root.rglob("robot_description.json")):
        urdf_path = next(
            (
                description_path.parent / filename
                for filename in ("robot.urdf", "generated_robot.urdf")
                if (description_path.parent / filename).exists()
            ),
            None,
        )
        if urdf_path is None:
            continue
        relative = description_path.relative_to(root)
        name = relative.parts[0] if len(relative.parts) > 1 else description_path.parent.name
        description = _load(description_path)
        cases.append({
            "name": name,
            "description_path": description_path,
            "urdf_path": urdf_path,
            "state": {},
            "scenario_tags": ["arbitrary_morphology", "elongated_many_leg"],
            "scenario_semantics": (
                f"generated elongated arbitrary morphology; "
                f"{int(description.get('num_legs', 0))} legs"
            ),
        })
        if limit > 0 and len(cases) >= limit:
            break
    if not cases:
        raise FileNotFoundError(
            f"no robot_description.json + robot.urdf pairs found under {root}"
        )
    return cases


def _absolute_meshes(description: dict, source_directory: Path) -> dict:
    derived = copy.deepcopy(description)
    for link in derived.get("links", []):
        mesh = Path(str(link.get("mesh_path", "")))
        if mesh and not mesh.is_absolute():
            link["mesh_path"] = str((source_directory / mesh).resolve())
    return derived


def _write_derived_asset(name: str, description: dict) -> Tuple[Path, Path]:
    description_path = REPO_ROOT / "batch_results" / f"phase_compare_{name}.json"
    urdf_path = REPO_ROOT / "batch_results" / f"phase_compare_{name}.urdf"
    description_path.write_text(json.dumps(description, indent=2), encoding="utf-8")
    build_urdf(description, urdf_path, description_path)
    return description_path, urdf_path


def build_cases(random_seed_count: int = 5) -> List[dict]:
    standard_description_path = (
        REPO_ROOT / "robot_assets" / "standard_hexapod" / "robot_description.json"
    )
    standard_urdf_path = (
        REPO_ROOT / "robot_assets" / "standard_hexapod" / "generated_robot.urdf"
    )
    standard = _load(standard_description_path)
    cases = [{
        "name": "standard_6",
        "description_path": standard_description_path,
        "urdf_path": standard_urdf_path,
        "state": {},
        "scenario_tags": ["standard_hexapod"],
        "scenario_semantics": "regular standard hexapod",
    }]
    random_seed_count = max(int(random_seed_count), 4)
    pools = {count: _find_generated_cases(count, random_seed_count) for count in (4, 6, 7, 10)}
    chosen: List[Tuple[int, Path, Path]] = []
    round_index = 0
    while len(chosen) < random_seed_count:
        made_progress = False
        for count in (4, 6, 7, 10):
            if round_index < len(pools[count]) and len(chosen) < random_seed_count:
                description_path, urdf_path = pools[count][round_index]
                chosen.append((count, description_path, urdf_path))
                made_progress = True
        if not made_progress:
            break
        round_index += 1
    for count, description_path, urdf_path in chosen:
        seed = _load(description_path).get("seed", "unknown")
        cases.append({
            "name": f"generated_{count}_seed{seed}",
            "description_path": description_path,
            "urdf_path": urdf_path,
            "state": {},
            "scenario_tags": ["arbitrary_morphology", f"random_{count}_leg"],
            "scenario_semantics": (
                "irregular/asymmetric generated morphology" if count == 7
                else "generated arbitrary morphology"
            ),
        })

    cases.append({
        "name": "standard_locked_leg_1",
        "description_path": standard_description_path,
        "urdf_path": standard_urdf_path,
        "state": {"locked_leg_ids": [1]},
        "scenario_tags": ["locked_leg"],
        "scenario_semantics": "leg 1 remains physical and is held as passive group-C support",
    })

    absolute_standard = _absolute_meshes(standard, standard_description_path.parent)
    amputated = amputate_legs(absolute_standard, [1])
    missing_description_path, missing_urdf_path = _write_derived_asset(
        "standard_missing_leg_1", amputated
    )
    cases.append({
        "name": "standard_missing_leg_1",
        "description_path": missing_description_path,
        "urdf_path": missing_urdf_path,
        "state": {},
        "scenario_tags": ["single_leg_missing", "unequal_left_right_leg_count"],
        "scenario_semantics": "leg 1 physically removed and remaining legs renumbered",
    })

    double_amputated = amputate_legs(absolute_standard, [1, 4])
    double_description_path, double_urdf_path = _write_derived_asset(
        "standard_missing_legs_1_4", double_amputated
    )
    cases.append({
        "name": "standard_missing_legs_1_4",
        "description_path": double_description_path,
        "urdf_path": double_urdf_path,
        "state": {},
        "scenario_tags": ["double_leg_missing"],
        "scenario_semantics": "original legs 1 and 4 physically removed; remaining legs renumbered",
    })

    offset_description = _absolute_meshes(standard, standard_description_path.parent)
    for link in offset_description.get("links", []):
        if link.get("role") == "trunk":
            center = list(link["mass_properties"].get("center_mass", [0.0, 0.0, 0.0]))
            center[0] += 0.08
            center[1] += 0.04
            link["mass_properties"]["center_mass"] = center
            break
    offset_description_path, offset_urdf_path = _write_derived_asset(
        "standard_com_offset", offset_description
    )
    cases.append({
        "name": "standard_com_offset",
        "description_path": offset_description_path,
        "urdf_path": offset_urdf_path,
        "state": {},
        "scenario_tags": ["center_of_mass_offset"],
        "scenario_semantics": "trunk inertial CoM shifted by (+0.08,+0.04) m",
    })
    return cases


def _maximum_simultaneous_swing(plan: dict, samples: int = 1440) -> int:
    active = [int(value) for value in plan.get("cpg", {}).get("active_leg_ids", [])]
    offsets = resolve_phase_offsets(plan, active)
    duties, _ = resolve_duty_factors(plan, active)
    return max(
        sum(
            not leg_phase_state(TWO_PI * index / samples, leg_id, offsets, duties).is_stance
            for leg_id in active
        )
        for index in range(samples)
    ) if active else 0


def _convex_hull(points: List[List[float]]) -> np.ndarray:
    """Return a deterministic CCW 2-D monotonic-chain hull."""
    unique = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(unique) <= 2:
        return np.asarray(unique, dtype=float)

    def cross(origin, first, second):
        return ((first[0] - origin[0]) * (second[1] - origin[1])
                - (first[1] - origin[1]) * (second[0] - origin[0]))

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _scheduled_ssm_stats(description: dict, plan: dict, samples: int = 1440) -> dict:
    """Compute SSM over the commanded contact schedule at nominal foot poses."""
    feet = {
        int(link["leg_id"]): list(link["default_world_origin"][:2])
        for link in description.get("links", [])
        if link.get("role") == "foot" and link.get("leg_id") is not None
    }
    active = [int(value) for value in plan.get("cpg", {}).get("active_leg_ids", [])]
    passive = {
        int(value) for value in plan.get("topology", {}).get("groups", {}).get("group_c", [])
    }
    offsets = resolve_phase_offsets(plan, active)
    duties, _ = resolve_duty_factors(plan, active)
    com = compute_projected_com_xy(description)
    margins = []
    for index in range(samples):
        phase = TWO_PI * index / samples
        support = set(passive)
        support.update(
            leg_id for leg_id in active
            if leg_phase_state(phase, leg_id, offsets, duties).is_stance
        )
        hull = _convex_hull([feet[leg_id] for leg_id in support if leg_id in feet])
        margins.append(compute_ssm(hull, com))
    return {
        "minimum": float(np.min(margins)) if margins else None,
        "p05": float(np.percentile(margins, 5.0)) if margins else None,
    }


TWO_PI = 2.0 * math.pi


def _episode_record(
    case: dict,
    description: dict,
    strategy_label: str,
    phase_strategy: str,
    wave_count: float,
    trail: list,
    forward_axis: list,
    yaw_stats: dict,
    episode_diagnostics: dict,
    plan: dict,
) -> dict:
    frequency = float(plan.get("cpg", {}).get("frequency_hz", 0.85))
    metrics = evaluate_trajectory(
        trail, forward_axis, gait_frequency_hz=frequency
    )
    episode_diagnostics = dict(episode_diagnostics)
    if not metrics.get("passed", False) and episode_diagnostics.get("failure_reason") is None:
        terminal_time = max(len(trail) - 1, 0) / 60.0
        episode_diagnostics["failure_reason"] = "unknown"
        episode_diagnostics["first_failure_time_s"] = terminal_time
        events = list(episode_diagnostics.get("failure_events", []))
        events.append({"time_s": terminal_time, "reason": "unknown"})
        episode_diagnostics["failure_events"] = events
    static_ssm = evaluate_ssm(description).get("ssm")
    scheduled_ssm = _scheduled_ssm_stats(description, plan)
    return {
        "case": case["name"],
        "scenario_semantics": case["scenario_semantics"],
        "scenario_tags": case.get("scenario_tags", []),
        "seed": description.get("seed"),
        "leg_count": int(description.get("num_legs", 0)),
        "strategy": strategy_label,
        "phase_strategy": phase_strategy,
        "selected_phase_strategy": plan.get("cpg", {}).get(
            "selected_phase_strategy", phase_strategy
        ),
        "wave_count": wave_count,
        "duty_factor": plan.get("cpg", {}).get("duty_factor"),
        "forward_speed": metrics.get("forward_speed"),
        "lateral_displacement_ratio": metrics.get("drift_ratio"),
        "lateral_speed": metrics.get("lateral_speed"),
        "yaw_tracking_error": episode_diagnostics.get("yaw_tracking_rmse"),
        "static_ssm": static_ssm,
        "minimum_dynamic_ssm": episode_diagnostics.get("ssm_min"),
        "dynamic_ssm_p05": episode_diagnostics.get("ssm_p05"),
        "ssm_metric_basis": "Isaac rigid contacts and measured terminal-body positions",
        "scheduled_ssm_min_proxy": scheduled_ssm["minimum"],
        "scheduled_ssm_p05_proxy": scheduled_ssm["p05"],
        "maximum_simultaneous_swing_legs": episode_diagnostics.get(
            "max_simultaneous_swing_legs", _maximum_simultaneous_swing(plan)
        ),
        "foot_or_leg_collision_count": (
            (episode_diagnostics.get("leg_body_collision_count") or 0)
            + (episode_diagnostics.get("leg_leg_collision_count") or 0)
            if episode_diagnostics.get("leg_body_collision_count") is not None
            and episode_diagnostics.get("leg_leg_collision_count") is not None
            else None
        ),
        "slip_rate": episode_diagnostics.get("foot_slip_ratio"),
        "post_change_recovery_time": None,
        "trajectory_success": bool(metrics.get("passed", False)),
        "safety_event_free": episode_diagnostics.get("failure_reason") is None,
        "success": bool(
            metrics.get("passed", False)
            and episode_diagnostics.get("failure_reason") is None
        ),
        "trajectory_metrics": metrics,
        "episode_diagnostics": episode_diagnostics,
        "gait_selection": plan.get("cpg", {}).get("selection_diagnostics"),
        "unsupported_metrics": {
            "post_change_recovery_time": "scenario starts with the changed morphology; no online switch occurs",
        },
    }


def _aggregate(records: List[dict]) -> Dict[str, dict]:
    """Aggregate measured experiment outputs by strategy without imputation."""
    summary: Dict[str, dict] = {}
    for strategy in sorted({str(record["strategy"]) for record in records}):
        rows = [record for record in records if record["strategy"] == strategy]
        def mean_available(field: str):
            values = [float(row[field]) for row in rows if row.get(field) is not None]
            return float(np.mean(values)) if values else None
        reasons: Dict[str, int] = {}
        for row in rows:
            reason = row.get("episode_diagnostics", {}).get("failure_reason") or "none"
            reasons[str(reason)] = reasons.get(str(reason), 0) + 1
        summary[strategy] = {
            "episodes": len(rows),
            "success_count": sum(bool(row.get("success")) for row in rows),
            "success_rate": (
                sum(bool(row.get("success")) for row in rows) / len(rows)
                if rows else None
            ),
            "failure_reason_distribution": reasons,
            "mean_forward_speed": mean_available("forward_speed"),
            "mean_lateral_displacement_ratio": mean_available(
                "lateral_displacement_ratio"
            ),
            "mean_yaw_tracking_error": mean_available("yaw_tracking_error"),
            "mean_actual_ssm_min": mean_available("minimum_dynamic_ssm"),
            "mean_actual_ssm_p05": mean_available("dynamic_ssm_p05"),
            "mean_slip_ratio": mean_available("slip_rate"),
            "mean_roll_rmse": (
                float(np.mean([
                    row["episode_diagnostics"]["roll_rmse"] for row in rows
                    if row.get("episode_diagnostics", {}).get("roll_rmse") is not None
                ])) if any(
                    row.get("episode_diagnostics", {}).get("roll_rmse") is not None
                    for row in rows
                ) else None
            ),
            "mean_pitch_rmse": (
                float(np.mean([
                    row["episode_diagnostics"]["pitch_rmse"] for row in rows
                    if row.get("episode_diagnostics", {}).get("pitch_rmse") is not None
                ])) if any(
                    row.get("episode_diagnostics", {}).get("pitch_rmse") is not None
                    for row in rows
                ) else None
            ),
            "total_leg_body_collisions": sum(
                int(row["episode_diagnostics"]["leg_body_collision_count"])
                for row in rows
                if row.get("episode_diagnostics", {}).get("leg_body_collision_count") is not None
            ),
            "total_leg_leg_collisions": sum(
                int(row["episode_diagnostics"]["leg_leg_collision_count"])
                for row in rows
                if row.get("episode_diagnostics", {}).get("leg_leg_collision_count") is not None
            ),
            "mean_peak_torque_ratio": (
                float(np.mean([
                    row["episode_diagnostics"]["peak_joint_torque_ratio"] for row in rows
                    if row.get("episode_diagnostics", {}).get("peak_joint_torque_ratio") is not None
                ])) if any(
                    row.get("episode_diagnostics", {}).get("peak_joint_torque_ratio") is not None
                    for row in rows
                ) else None
            ),
            "recovery_time": None,
            "recovery_time_unavailable_reason": (
                "no online morphology/plan change was applied during these episodes"
            ),
        }
    return summary


def _selection_key(record: dict) -> tuple:
    """Rank fresh confirmations while keeping safety separate from trajectory."""
    diagnostics = record.get("episode_diagnostics", {})
    trajectory = record.get("trajectory_metrics", {})
    ssm_p05 = diagnostics.get("ssm_p05")
    slip = diagnostics.get("foot_slip_ratio")
    speed_cv = trajectory.get("speed_cv")
    return (
        int(bool(record.get("success"))),
        int(bool(record.get("trajectory_success"))),
        int(diagnostics.get("failure_reason") is None),
        float(ssm_p05) if ssm_p05 is not None else -1e9,
        -float(slip) if slip is not None else -1e9,
        -float(speed_cv) if speed_cv is not None else -1e9,
        float(record.get("forward_speed") or -1e9),
    )


def _selected_by_case(records: List[dict]) -> Dict[str, dict]:
    selected: Dict[str, dict] = {}
    for case_name in dict.fromkeys(str(row["case"]) for row in records):
        candidates = [row for row in records if str(row["case"]) == case_name]
        if not candidates:
            continue
        best = max(candidates, key=_selection_key)
        selected[case_name] = {
            "strategy": best.get("strategy"),
            "selected_phase_strategy": best.get("selected_phase_strategy"),
            "trajectory_success": bool(best.get("trajectory_success")),
            "safety_event_free": bool(best.get("safety_event_free")),
            "success": bool(best.get("success")),
            "forward_speed": best.get("forward_speed"),
            "lateral_displacement_ratio": best.get("lateral_displacement_ratio"),
            "speed_cv": best.get("trajectory_metrics", {}).get("speed_cv"),
            "failure_reason": best.get("episode_diagnostics", {}).get("failure_reason"),
            "selection_basis": (
                "fresh confirmation; full success, trajectory acceptance, "
                "event-free safety, SSM p05, slip, speed CV, then speed"
            ),
        }
    return selected


def _safe_component(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_"
                   for character in str(value))


def _write_episode_artifacts(
    *, case: dict, description: dict, plan: dict, record: dict,
    trail: list, forward_axis: list, artifacts_root: Path, report_root: Path,
) -> Dict[str, str]:
    """Persist one auditable episode and render trajectory plus gait heatmap."""
    episode_dir = (
        artifacts_root / _safe_component(case["name"])
        / _safe_component(record["strategy"])
    )
    episode_dir.mkdir(parents=True, exist_ok=True)
    plan_path = episode_dir / "gait_plan.json"
    trajectory_data_path = episode_dir / "trajectory_data.json"
    input_path = episode_dir / "visualization_input.json"
    trajectory_path = episode_dir / "trajectory.png"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    trajectory_data_path.write_text(json.dumps({
        "case": case["name"],
        "strategy": record["strategy"],
        "forward_axis": forward_axis,
        "samples": trail,
        "trajectory_metrics": record["trajectory_metrics"],
        "episode_diagnostics": record["episode_diagnostics"],
    }, indent=2), encoding="utf-8")

    axis = np.asarray(forward_axis, dtype=float)[:2]
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    lateral = np.array([-axis[1], axis[0]], dtype=float)
    samples = np.asarray(trail, dtype=float)
    displacement = (
        samples[-1, :2] - samples[0, :2]
        if samples.ndim == 2 and len(samples) > 1 else np.zeros(2, dtype=float)
    )
    static_result = evaluate_ssm(description)
    plot_data = {
        "robot_name": f"{case['name']} — {record['strategy']}",
        "description": description,
        "ssm_result": static_result,
        "com_trail": trail,
        "forward_axis": forward_axis,
        "fwd_dist": float(np.dot(displacement, axis)),
        "lat_dist": float(np.dot(displacement, lateral)),
        "metrics": record["trajectory_metrics"],
        "out_path": str(trajectory_path.resolve()),
        "gait_plan": plan,
        "gait_plan_source": "executed_phase_strategy_comparison_plan",
        "gait_duration_s": max(len(trail) / 60.0, 1.0),
    }
    input_path.write_text(json.dumps(plot_data), encoding="utf-8")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        f"{REPO_ROOT}:{environment['PYTHONPATH']}"
        if environment.get("PYTHONPATH") else str(REPO_ROOT)
    )
    completed = subprocess.run(
        [PLOT_PYTHON, str(PLOT_SCRIPT), str(input_path)],
        capture_output=True, text=True, env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"plot rendering failed for {case['name']}/{record['strategy']}: "
            f"{completed.stderr[-800:]}"
        )
    heatmap_path = episode_dir / "gait_heatmap.png"
    if not trajectory_path.exists() or not heatmap_path.exists():
        raise RuntimeError(
            f"renderer did not create all images for {case['name']}/{record['strategy']}"
        )

    def relative(path: Path) -> str:
        return path.resolve().relative_to(report_root.resolve()).as_posix()

    return {
        "trajectory_plot": relative(trajectory_path),
        "gait_heatmap": relative(heatmap_path),
        "gait_visualization_data": relative(episode_dir / "gait_visualization.json"),
        "gait_plan": relative(plan_path),
        "trajectory_data": relative(trajectory_data_path),
    }


def _write_html_report(output: Path, document: dict) -> Path:
    """Write a compact per-robot, per-strategy visual comparison gallery."""
    report_path = output.parent / "report.html"
    records = document.get("records", [])
    cases = list(dict.fromkeys(str(record["case"]) for record in records))
    sections = []
    def metric(value) -> str:
        return "—" if value is None else f"{float(value):.4f}"

    summary_rows = []
    for strategy, values in document.get("strategy_summary", {}).items():
        summary_rows.append(
            "<tr>"
            f"<td>{html.escape(str(strategy))}</td>"
            f"<td>{int(values.get('episodes', 0))}</td>"
            f"<td>{int(values.get('success_count', 0))}</td>"
            f"<td>{metric(values.get('mean_forward_speed'))}</td>"
            f"<td>{metric(values.get('mean_lateral_displacement_ratio'))}</td>"
            f"<td>{metric(values.get('mean_yaw_tracking_error'))}</td>"
            f"<td>{metric(values.get('mean_actual_ssm_min'))}</td>"
            f"<td>{metric(values.get('mean_slip_ratio'))}</td>"
            "</tr>"
        )
    summary_table = (
        '<table><thead><tr><th>Strategy</th><th>Episodes</th><th>Success</th>'
        '<th>Mean speed (m/s)</th><th>Mean drift</th><th>Mean yaw RMSE</th>'
        '<th>Mean actual SSM min</th><th>Mean slip</th></tr></thead><tbody>'
        f'{"".join(summary_rows)}</tbody></table>'
    )

    for case_name in cases:
        cards = []
        selected_strategy = document.get("selected_by_case", {}).get(
            case_name, {},
        ).get("strategy")
        for record in [row for row in records if row["case"] == case_name]:
            artifacts = record.get("artifacts", {})
            heatmap = html.escape(str(artifacts.get("gait_heatmap", "")))
            trajectory = html.escape(str(artifacts.get("trajectory_plot", "")))
            speed = record.get("forward_speed")
            drift = record.get("lateral_displacement_ratio")
            cards.append(
                '<article class="card">'
                f'<h3>{"✓ SELECTED · " if record["strategy"] == selected_strategy else ""}'
                f'{html.escape(str(record["strategy"]))}</h3>'
                f'<p>selected={html.escape(str(record.get("selected_phase_strategy")))}; '
                f'v={metric(speed)} m/s; drift={metric(drift)}; '
                f'success={bool(record.get("success"))}; '
                f'failure={html.escape(str(record.get("episode_diagnostics", {}).get("failure_reason")))}</p>'
                f'<a href="{heatmap}"><img src="{heatmap}" alt="gait heatmap"></a>'
                f'<a href="{trajectory}"><img src="{trajectory}" alt="trajectory"></a>'
                '</article>'
            )
        sections.append(
            f'<section><h2>{html.escape(case_name)}</h2>'
            f'<div class="grid">{"".join(cards)}</div></section>'
        )
    report_path.write_text(
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Phase strategy comparison</title><style>'
        'body{font-family:Arial,sans-serif;margin:20px;background:#f4f5f7;color:#222}'
        '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}'
        '.card{background:white;padding:12px;border-radius:8px;box-shadow:0 1px 5px #bbb}'
        '.card img{width:100%;margin-top:8px;border:1px solid #ddd}'
        'h2{margin-top:32px}p{font-size:13px}'
        'table{border-collapse:collapse;width:100%;background:white}'
        'th,td{border:1px solid #ccc;padding:7px;text-align:right}'
        'th:first-child,td:first-child{text-align:left}</style></head><body>'
        '<h1>多足机器人相位策略批量对比</h1>'
        '<p>热力图为执行计划的关节目标，不是实测 DOF；轨迹指标来自 Isaac Gym。</p>'
        f'{summary_table}{"".join(sections)}</body></html>',
        encoding="utf-8",
    )
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument(
        "--calibrate-course", action="store_true",
        help=(
            "run a separate short probe, learn its realised course while "
            "preserving the motor plan, then evaluate a fresh episode"
        ),
    )
    parser.add_argument("--calibration-probe-steps", type=int, default=360)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--diagnostic-stride", type=int, default=5)
    parser.add_argument("--duty-factor", type=float, default=None)
    parser.add_argument("--frequency-hz", type=float, default=None)
    parser.add_argument("--wave-direction", type=float, default=1.0)
    parser.add_argument("--stride-direction", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument(
        "--swing-sign-mode",
        choices=("legacy_side", "kinematic_jacobian"),
        default="legacy_side",
    )
    parser.add_argument("--stance-calibration-blend", type=float, default=0.35)
    parser.add_argument("--touchdown-settle-steps", type=int, default=25)
    parser.add_argument("--touchdown-vertical-reset", type=float, default=0.35)
    parser.add_argument("--reset-maximum-stance-extension", type=float, default=0.06)
    parser.add_argument("--disable-reset-contact-calibration", action="store_true")
    parser.add_argument("--amplitude-scale", type=float, default=1.0)
    parser.add_argument("--flip-axis", action="store_true")
    parser.add_argument("--minimum-support-count", type=int, default=None)
    parser.add_argument("--maximum-swing", type=int, default=None)
    parser.add_argument("--stance-search-ratio", type=float, default=None)
    parser.add_argument("--latch-stance-search", action="store_true")
    parser.add_argument("--emergency-support-count", type=int, default=None)
    parser.add_argument("--emergency-contact-recovery", action="store_true")
    parser.add_argument("--disable-emergency-contact-recovery", action="store_true")
    parser.add_argument(
        "--disable-contact-feedback", action="store_true",
        help="run the legacy open-loop executor for an ablation baseline",
    )
    parser.add_argument(
        "--phase-gate", action="store_true",
        help="also delay lift-off from measured support count (conservative crawl mode)",
    )
    parser.add_argument("--random-seed-count", type=int, default=5)
    parser.add_argument(
        "--models-root", type=Path,
        help="use every generated robot_description.json + robot.urdf below this root",
    )
    parser.add_argument("--model-limit", type=int, default=0)
    parser.add_argument(
        "--strategies", nargs="*", default=None,
        choices=[item[0] for item in STRATEGIES],
        help="strategy labels to run; default runs every registered strategy",
    )
    parser.add_argument("--cases", nargs="*", default=None,
                        help="optional case names returned by build_cases()")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "batch_results" / "phase_strategy_comparison.json",
    )
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    args = parser.parse_args()

    requested = set(args.cases or [])
    source_cases = (
        build_cases_from_root(args.models_root, args.model_limit)
        if args.models_root is not None else build_cases(args.random_seed_count)
    )
    cases = [case for case in source_cases if not requested or case["name"] in requested]
    if requested - {case["name"] for case in cases}:
        raise SystemExit(f"unknown/unavailable cases: {sorted(requested - {case['name'] for case in cases})}")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts_root = (
        args.artifacts_dir.resolve()
        if args.artifacts_dir is not None else output.parent / "artifacts"
    )
    artifacts_root.mkdir(parents=True, exist_ok=True)
    document = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "steps": args.steps,
        "use_gpu": not args.cpu,
        "default_strategy_changed": False,
        "records": [],
    }
    for case in cases:
        description = _load(case["description_path"])
        print(f"[PhaseCompare] case={case['name']} legs={description.get('num_legs')}")
        strategies = [
            item for item in STRATEGIES
            if not args.strategies or item[0] in set(args.strategies)
        ]
        for label, strategy, wave_count in strategies:
            state = copy.deepcopy(case["state"])
            state["cpg"] = {
                "phase_strategy": strategy,
                "wave_count": wave_count,
                "wave_direction": args.wave_direction,
                "stride_direction": args.stride_direction,
                "swing_sign_mode": args.swing_sign_mode,
                "stance_calibration_blend": args.stance_calibration_blend,
                "touchdown_settle_steps": args.touchdown_settle_steps,
                "touchdown_vertical_reset": args.touchdown_vertical_reset,
                "reset_maximum_stance_extension": args.reset_maximum_stance_extension,
            }
            if args.disable_reset_contact_calibration:
                state["cpg"]["reset_contact_calibration"] = False
            if args.duty_factor is not None:
                state["cpg"]["duty_factor"] = args.duty_factor
            if args.frequency_hz is not None:
                state["cpg"]["frequency_hz"] = args.frequency_hz
            feedback_overrides = {
                "enabled": not args.disable_contact_feedback,
                "gate_liftoff": args.phase_gate,
            }
            if args.minimum_support_count is not None:
                feedback_overrides["minimum_support_count"] = args.minimum_support_count
            if args.maximum_swing is not None:
                feedback_overrides["maximum_simultaneous_swing"] = args.maximum_swing
            if args.stance_search_ratio is not None:
                feedback_overrides["stance_search_ratio"] = args.stance_search_ratio
            if args.latch_stance_search:
                feedback_overrides["latch_stance_search"] = True
            if args.emergency_support_count is not None:
                feedback_overrides["emergency_support_count"] = args.emergency_support_count
            if args.emergency_contact_recovery:
                feedback_overrides["emergency_contact_recovery"] = True
            if args.disable_emergency_contact_recovery:
                feedback_overrides["emergency_contact_recovery"] = False
            state["cpg"]["contact_feedback"] = feedback_overrides
            if strategy == "adaptive_wave":
                state["cpg"]["selector"] = {"samples_per_cycle": 360}
            plan = compute_adaptive_plan(description, state)
            if args.flip_axis:
                original_axis = np.asarray(
                    plan.get("final_forward_axis", [1.0, 0.0]), dtype=float,
                )
                plan = compute_adaptive_plan(
                    description, state, forced_axis=(-original_axis).tolist(),
                )
            if not math.isclose(args.amplitude_scale, 1.0):
                amplitudes = plan.get("topology", {}).get(
                    "per_leg_stride_amplitudes", {}
                )
                plan["_per_amp_override"] = {
                    str(leg_id): float(np.clip(
                        float(amplitude) * args.amplitude_scale, 0.10, 1.20,
                    ))
                    for leg_id, amplitude in amplitudes.items()
                }
            probe_summary = None
            if args.calibrate_course:
                # PhysX retains solver/contact state across an in-place actor
                # reset.  Probe and confirmation therefore use separately
                # created simulators; otherwise some asymmetric robots reverse
                # direction on the second nominally identical episode.
                with _RobotSimCtx(
                    description, case["urdf_path"], use_gpu=not args.cpu,
                ) as probe_ctx:
                    probe_trail, probe_axis, _ = probe_ctx.run_episode(
                        plan, args.calibration_probe_steps,
                        return_yaw_stats=False,
                    )
                probe_summary = evaluate_trajectory(
                    probe_trail, probe_axis,
                    gait_frequency_hz=float(
                        plan.get("cpg", {}).get("frequency_hz", 0.85)
                    ),
                )
                plan = refine_axis_from_trajectory(
                    description, plan, probe_trail,
                )
            with _RobotSimCtx(
                description, case["urdf_path"], use_gpu=not args.cpu,
            ) as confirmation_ctx:
                trail, axis, yaw_stats, episode_diagnostics = confirmation_ctx.run_episode(
                    plan, args.steps, return_yaw_stats=True,
                    return_diagnostics=True,
                    diagnostic_stride=args.diagnostic_stride,
                )
            record = _episode_record(
                case, description, label, strategy, wave_count,
                trail, axis, yaw_stats, episode_diagnostics, plan,
            )
            if probe_summary is not None:
                record["course_calibration"] = {
                    "probe_steps": int(args.calibration_probe_steps),
                    "probe_metrics_on_original_axis": probe_summary,
                    "axis_calibration": plan.get("axis_calibration"),
                    "confirmation_is_fresh_episode": True,
                    "physics_context_reused": False,
                }
            record["artifacts"] = _write_episode_artifacts(
                case=case, description=description, plan=plan, record=record,
                trail=trail, forward_axis=axis,
                artifacts_root=artifacts_root, report_root=output.parent,
            )
            document["records"].append(record)
            document["strategy_summary"] = _aggregate(document["records"])
            document["selected_by_case"] = _selected_by_case(document["records"])
            output.write_text(json.dumps(document, indent=2), encoding="utf-8")
            print(
                f"  {label}: v={record['forward_speed']} "
                f"drift={record['lateral_displacement_ratio']} success={record['success']}"
            )
    report_path = _write_html_report(output, document)
    print(f"[PhaseCompare] wrote {output}")
    print(f"[PhaseCompare] report {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
