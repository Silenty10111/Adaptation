"""Simulation-in-the-loop candidate generation and gait selection."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from adaptation.gait import compute_adaptive_plan


@dataclass
class CandidatePlan:
    name: str
    plan: Dict


def _unit(vector) -> np.ndarray:
    value = np.asarray(vector, dtype=float)[:2]
    return value / max(float(np.linalg.norm(value)), 1e-9)


def _trunk_axis(description: Dict) -> np.ndarray:
    polygon = np.asarray(description.get("trunk_polygon_xy", []), dtype=float)
    if len(polygon) < 3:
        return np.array([1.0, 0.0])
    centered = polygon - polygon.mean(axis=0)
    _, vectors = np.linalg.eigh(centered.T @ centered)
    axis = _unit(vectors[:, -1])
    return axis if axis[0] >= 0.0 else -axis


def generate_candidates(description: Dict, max_candidates: int = 18) -> List[CandidatePlan]:
    """Build deterministic axis/controller candidates from robot geometry."""
    base = compute_adaptive_plan(description, {})
    trunk = _trunk_axis(description)
    lateral = np.array([-trunk[1], trunk[0]])
    planned = _unit(base.get("final_forward_axis", trunk))
    raw_axes = [planned, trunk, lateral, -lateral,
                _unit(trunk + lateral), _unit(trunk - lateral), -trunk]
    axes: List[np.ndarray] = []
    for axis in raw_axes:
        # A walking axis is directed: +x and -x are distinct candidates.  Only
        # collapse vectors pointing in the same direction.
        if not any(float(np.dot(axis, old)) > 0.9995 for old in axes):
            axes.append(axis)

    result: List[CandidatePlan] = []
    for axis_idx, axis in enumerate(axes):
        plan = base if axis_idx == 0 else compute_adaptive_plan(
            description, {}, forced_axis=axis.tolist()
        )
        result.append(CandidatePlan(f"axis_{axis_idx}_alternating", plan))
        legacy = copy.deepcopy(plan)
        legacy["cpg"]["mode"] = "legacy_sine"
        result.append(CandidatePlan(f"axis_{axis_idx}_legacy", legacy))
        swapped = copy.deepcopy(plan)
        group_a = set(swapped["topology"]["groups"].get("group_a", []))
        group_b = set(swapped["topology"]["groups"].get("group_b", []))
        swapped["cpg"]["phase_offsets"] = {
            str(lid): float(0.0 if lid in group_b else np.pi)
            for lid in sorted(group_a | group_b)
        }
        result.append(CandidatePlan(f"axis_{axis_idx}_phase_swap", swapped))
        if len(result) >= max_candidates:
            break
    return result[:max_candidates]


def score_metrics(metrics: Dict[str, object]) -> float:
    """Rank failed probes while strongly preferring accepted candidates."""
    if metrics.get("passed"):
        return 1000.0 + float(metrics.get("forward_speed", 0.0))
    forward = float(metrics.get("forward_speed", 0.0))
    lateral = abs(float(metrics.get("lateral_speed", 0.0)))
    course = float(metrics.get("heading_error_deg", 180.0))
    speed_cv = float(metrics.get("speed_cv", 10.0))
    height = float(metrics.get("mean_height", 0.0))
    return (8.0 * forward - 5.0 * lateral - 0.02 * course
            - 0.5 * speed_cv + 2.0 * min(height, 0.5))


def refine_axis_from_trajectory(description: Dict, plan: Dict, trajectory) -> Dict:
    """Re-plan along the steady displacement measured by a simulation probe."""
    samples = np.asarray(trajectory, dtype=float)
    if samples.ndim != 2 or len(samples) < 10:
        return copy.deepcopy(plan)
    start = min(int(len(samples) * 0.2), len(samples) - 2)
    displacement = samples[-1, :2] - samples[start, :2]
    if float(np.linalg.norm(displacement)) < 1e-4:
        return copy.deepcopy(plan)
    refined = compute_adaptive_plan(description, {}, forced_axis=_unit(displacement).tolist())
    refined["cpg"]["mode"] = plan.get("cpg", {}).get("mode", refined["cpg"]["mode"])
    return refined
