"""Build auditable planned-gait raster data without an Isaac Gym dependency."""

from __future__ import annotations

import math
import re
from typing import Dict, Mapping, Sequence

import numpy as np

from .phase import leg_phase_state, resolve_duty_factors, resolve_phase_offsets


DEFAULT_SWING_AMPLITUDE = 0.32
DEFAULT_STANCE_LIFT_RATIO = 0.05
DEFAULT_SWING_LIFT_RATIO = 0.78
DEFAULT_STANCE_DROP_RATIO = 0.90
DEFAULT_SWING_DROP_RATIO = 0.38


def _foot_positions(description: Mapping) -> Dict[int, np.ndarray]:
    return {
        int(link["leg_id"]): np.asarray(link["default_world_origin"], dtype=float)[:2]
        for link in description.get("links", [])
        if link.get("role") == "foot" and link.get("leg_id") is not None
    }


def _joint_ranges(description: Mapping) -> Dict[int, Dict[str, float]]:
    ranges: Dict[int, Dict[str, float]] = {}
    pattern = re.compile(r"^leg_(\d+)_(lift|swing|drop)$")
    for joint in description.get("joints", []):
        match = pattern.match(str(joint.get("name", "")))
        if match is None:
            continue
        limit = joint.get("limit", {})
        try:
            joint_range = float(limit["upper"]) - float(limit["lower"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(joint_range) and joint_range > 0.0:
            ranges.setdefault(int(match.group(1)), {})[match.group(2)] = joint_range
    return ranges


def build_planned_gait_raster(
    description: Mapping,
    gait_plan: Mapping,
    *,
    duration_s: float = 4.0,
    sample_rate_hz: float = 60.0,
    maximum_samples: int = 600,
) -> Dict[str, object]:
    """Return phase and planned joint-target heatmap data.

    The angular values are controller targets reconstructed from the same
    phase offsets, duty factors, joint ranges and trajectory coordinates as
    the executor.  They are not measured DOF states.
    """
    feet = _foot_positions(description)
    physical_leg_ids = sorted(feet)
    topology = gait_plan.get("topology", {}) if isinstance(gait_plan, Mapping) else {}
    groups = topology.get("groups", {}) if isinstance(topology, Mapping) else {}
    cpg = gait_plan.get("cpg", {}) if isinstance(gait_plan, Mapping) else {}
    group_a = {int(value) for value in groups.get("group_a", [])}
    group_b = {int(value) for value in groups.get("group_b", [])}
    passive = {int(value) for value in groups.get("group_c", [])} & set(physical_leg_ids)
    configured_active = cpg.get("active_leg_ids", []) if isinstance(cpg, Mapping) else []
    active = {
        int(value) for value in configured_active
        if int(value) in feet and int(value) not in passive
    }
    if not active:
        active = ((group_a | group_b) & set(physical_leg_ids)) - passive

    frequency = max(float(cpg.get("frequency_hz", 0.85)), 1e-6)
    duration = max(float(duration_s), 1.0 / frequency)
    requested_samples = max(int(round(duration * max(sample_rate_hz, 1.0))), 2)
    sample_count = min(requested_samples, max(int(maximum_samples), 2))
    time_s = np.linspace(0.0, duration, sample_count, endpoint=False, dtype=float)
    offsets = resolve_phase_offsets(gait_plan, active)
    duties, _ = resolve_duty_factors(gait_plan, active)
    amplitudes = {
        int(key): float(value)
        for key, value in topology.get("per_leg_stride_amplitudes", {}).items()
    }
    joint_ranges = _joint_ranges(description)

    forward = np.asarray(gait_plan.get("final_forward_axis", [1.0, 0.0]), dtype=float)[:2]
    norm = float(np.linalg.norm(forward))
    forward = forward / norm if norm > 1e-12 else np.array([1.0, 0.0])
    lateral = np.array([-forward[1], forward[0]], dtype=float)

    row_count = len(physical_leg_ids)
    phase_fraction = np.zeros((row_count, sample_count), dtype=float)
    contact_state = np.full((row_count, sample_count), 2, dtype=np.int8)
    swing_delta = np.zeros((row_count, sample_count), dtype=float)
    lift_delta = np.zeros((row_count, sample_count), dtype=float)
    drop_delta = np.zeros((row_count, sample_count), dtype=float)
    group_labels = []
    side_labels = []
    foot_center = (
        np.mean(np.asarray(list(feet.values()), dtype=float), axis=0)
        if feet else np.zeros(2, dtype=float)
    )

    for row, leg_id in enumerate(physical_leg_ids):
        lateral_position = float(np.dot(feet[leg_id] - foot_center, lateral))
        side_labels.append("left" if lateral_position >= 0.0 else "right")
        if leg_id in passive:
            group_labels.append("C/passive")
            contact_state[row, :] = 0
            continue
        if leg_id not in active:
            group_labels.append("inactive")
            continue
        group_labels.append("A" if leg_id in group_a else ("B" if leg_id in group_b else "active"))
        side_sign = 1.0 if float(np.dot(feet[leg_id], lateral)) > 0.0 else -1.0
        stride_scale = float(amplitudes.get(leg_id, 1.0))
        swing_range = joint_ranges.get(leg_id, {}).get("swing", 1.0)
        lift_range = joint_ranges.get(leg_id, {}).get("lift", 1.0)
        drop_range = joint_ranges.get(leg_id, {}).get("drop", 1.0)
        for column, current_time in enumerate(time_s):
            state = leg_phase_state(
                2.0 * math.pi * frequency * float(current_time),
                leg_id, offsets, duties,
            )
            phase_fraction[row, column] = state.normalized_phase
            contact_state[row, column] = 0 if state.is_stance else 1
            swing_delta[row, column] = (
                DEFAULT_SWING_AMPLITUDE * stride_scale * side_sign
                * state.fore_aft * swing_range
            )
            lift_delta[row, column] = (
                (DEFAULT_SWING_LIFT_RATIO - DEFAULT_STANCE_LIFT_RATIO)
                * state.lift * lift_range
            )
            drop_delta[row, column] = (
                (DEFAULT_SWING_DROP_RATIO - DEFAULT_STANCE_DROP_RATIO)
                * state.lift * drop_range
            )

    return {
        "data_scope": "planned_controller_targets_not_measured_joint_states",
        "leg_ids": physical_leg_ids,
        "group_labels": group_labels,
        "side_labels": side_labels,
        "time_s": time_s.tolist(),
        "frequency_hz": frequency,
        "phase_strategy": cpg.get("phase_strategy", "binary"),
        "selected_phase_strategy": cpg.get(
            "selected_phase_strategy", cpg.get("phase_strategy", "binary")
        ),
        "phase_offsets_rad": {str(key): float(value) for key, value in offsets.items()},
        "duty_factors": {str(key): float(value) for key, value in duties.items()},
        "phase_fraction": phase_fraction.tolist(),
        "contact_state": contact_state.tolist(),
        "contact_state_codes": {"0": "stance", "1": "swing", "2": "inactive"},
        "swing_joint_target_delta_rad": swing_delta.tolist(),
        "lift_joint_target_delta_rad": lift_delta.tolist(),
        "drop_joint_target_delta_rad": drop_delta.tolist(),
    }
