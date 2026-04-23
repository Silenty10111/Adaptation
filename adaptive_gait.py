#!/usr/bin/env python3
"""Adaptive virtual-body axis estimation and grouped gait planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import LineString, MultiPoint, Point, Polygon


EPS = 1e-9


@dataclass
class LegState:
    leg_id: int
    foot_xy: np.ndarray
    foot_z: float
    hip_xy: np.ndarray
    swing_limit: float
    lift_axis_xy: np.ndarray
    torque_weight: float = 1.0
    valid: bool = True
    locked: bool = False
    phase: str = "stance"
    upcoming_stance: bool = False
    swing_vector_xy: Optional[np.ndarray] = None


def normalize(vector: Sequence[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(array))
    if norm < EPS:
        raise ValueError(f"Cannot normalize near-zero vector: {vector}")
    return array / norm


def perpendicular_xy(vector: Sequence[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    return np.array([-array[1], array[0]], dtype=float)


def to_float_list(values: Sequence[float]) -> List[float]:
    return [float(value) for value in values]


def _link_by_role(description: Mapping[str, object], role: str) -> Dict[int, Mapping[str, object]]:
    result: Dict[int, Mapping[str, object]] = {}
    for link in description.get("links", []):
        if link.get("role") == role and link.get("leg_id") is not None:
            result[int(link["leg_id"])] = link
    return result


def _joint_by_name(description: Mapping[str, object], name: str) -> Optional[Mapping[str, object]]:
    for joint in description.get("joints", []):
        if joint.get("name") == name:
            return joint
    return None


def estimate_center_of_mass_xy(description: Mapping[str, object]) -> np.ndarray:
    weighted_sum = np.zeros(2, dtype=float)
    total_mass = 0.0
    for link in description.get("links", []):
        mass_props = link.get("mass_properties", {})
        mass = float(mass_props.get("mass", 0.0))
        if mass <= 0.0:
            continue
        origin = np.asarray(link.get("default_world_origin", [0.0, 0.0, 0.0]), dtype=float)
        center_mass = np.asarray(mass_props.get("center_mass", [0.0, 0.0, 0.0]), dtype=float)
        world_center = origin + center_mass
        weighted_sum += world_center[:2] * mass
        total_mass += mass

    if total_mass <= EPS:
        return np.zeros(2, dtype=float)
    return weighted_sum / total_mass


def extract_leg_states(
    description: Mapping[str, object],
    state_overrides: Optional[Mapping[str, object]] = None,
) -> List[LegState]:
    foot_links = _link_by_role(description, "foot")
    hip_links = _link_by_role(description, "joint_sphere")
    overrides = state_overrides or {}
    locked_ids = {int(value) for value in overrides.get("locked_leg_ids", [])}
    missing_ids = {int(value) for value in overrides.get("missing_leg_ids", [])}
    phases = {int(key): str(value) for key, value in overrides.get("phases", {}).items()}
    upcoming = {int(value) for value in overrides.get("upcoming_stance_leg_ids", [])}
    torque_weights = {int(key): float(value) for key, value in overrides.get("torque_weights", {}).items()}
    swing_vectors = {
        int(key): np.asarray(value, dtype=float)[:2]
        for key, value in overrides.get("swing_vectors", {}).items()
    }

    leg_states: List[LegState] = []
    num_legs = int(description.get("num_legs", 0))
    for leg_id in range(num_legs):
        foot_link = foot_links.get(leg_id)
        hip_link = hip_links.get(leg_id)
        swing_joint = _joint_by_name(description, f"leg_{leg_id}_swing")
        lift_joint = _joint_by_name(description, f"leg_{leg_id}_lift")
        valid = foot_link is not None and hip_link is not None and swing_joint is not None and lift_joint is not None

        foot_origin = np.asarray((foot_link or {}).get("default_world_origin", [0.0, 0.0, 0.0]), dtype=float)
        hip_origin = np.asarray((hip_link or {}).get("default_world_origin", [0.0, 0.0, 0.0]), dtype=float)
        swing_limit = 0.0
        if swing_joint is not None:
            joint_limit = swing_joint.get("limit", {})
            swing_limit = float(joint_limit.get("upper", 0.0)) - float(joint_limit.get("lower", 0.0))

        lift_axis = np.asarray((lift_joint or {}).get("axis", [1.0, 0.0, 0.0]), dtype=float)[:2]
        if np.linalg.norm(lift_axis) < EPS:
            lift_axis = np.array([1.0, 0.0], dtype=float)

        leg_states.append(
            LegState(
                leg_id=leg_id,
                foot_xy=foot_origin[:2],
                foot_z=float(foot_origin[2]),
                hip_xy=hip_origin[:2],
                swing_limit=max(swing_limit, 0.0),
                lift_axis_xy=normalize(lift_axis),
                torque_weight=torque_weights.get(leg_id, 1.0),
                valid=valid and leg_id not in missing_ids,
                locked=leg_id in locked_ids,
                phase=phases.get(leg_id, "stance"),
                upcoming_stance=leg_id in upcoming,
                swing_vector_xy=swing_vectors.get(leg_id),
            )
        )

    return leg_states


def select_support_legs(legs: Sequence[LegState], foot_height_epsilon: float = 0.025) -> Tuple[List[LegState], List[LegState]]:
    active_legs = [leg for leg in legs if leg.valid and not leg.locked]
    if not active_legs:
        return [], []

    min_z = min(leg.foot_z for leg in active_legs)
    stance = []
    near_stance = []
    for leg in active_legs:
        in_contact = leg.phase == "stance" or leg.foot_z <= min_z + foot_height_epsilon
        if in_contact:
            stance.append(leg)
        elif leg.upcoming_stance or leg.phase in {"pre_stance", "drop"}:
            near_stance.append(leg)
    return stance, near_stance


def compute_support_statistics(stance_legs: Sequence[LegState], near_stance_legs: Sequence[LegState]) -> Tuple[np.ndarray, np.ndarray]:
    weighted_points: List[np.ndarray] = []
    weights: List[float] = []
    for leg in stance_legs:
        weighted_points.append(leg.foot_xy)
        weights.append(1.0)
    for leg in near_stance_legs:
        weighted_points.append(leg.foot_xy)
        weights.append(0.35)

    if not weighted_points:
        return np.zeros(2, dtype=float), np.array([1.0, 0.0], dtype=float)

    points = np.asarray(weighted_points, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    support_center = (points * weight_array[:, None]).sum(axis=0) / weight_array.sum()
    centered = points - support_center
    covariance = (centered.T * weight_array).dot(centered) / max(float(weight_array.sum()), 1.0)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal_axis = normalize(eigenvectors[:, int(np.argmax(eigenvalues))])
    return support_center, principal_axis


def compute_support_polygon(stance_legs: Sequence[LegState]) -> Polygon:
    if not stance_legs:
        return Point(0.0, 0.0).buffer(0.02)

    points = [tuple(leg.foot_xy.tolist()) for leg in stance_legs]
    hull = MultiPoint(points).convex_hull
    if isinstance(hull, Polygon):
        return hull
    if isinstance(hull, LineString):
        return hull.buffer(0.01, cap_style=2)
    return hull.buffer(0.02)


def _longest_linestring(geometry) -> Optional[LineString]:
    if geometry.is_empty:
        return None
    if isinstance(geometry, LineString):
        return geometry
    if hasattr(geometry, "geoms"):
        segments = [geom for geom in geometry.geoms if isinstance(geom, LineString) and geom.length > EPS]
        if not segments:
            return None
        return max(segments, key=lambda segment: segment.length)
    return None


def approximate_medial_axis(polygon: Polygon, support_center: np.ndarray, axis: np.ndarray, samples: int = 21) -> List[List[float]]:
    if polygon.is_empty:
        return [to_float_list(support_center)]

    coords = np.asarray(polygon.exterior.coords[:-1], dtype=float)
    projections = coords.dot(axis)
    start = float(np.min(projections))
    end = float(np.max(projections))
    lateral = normalize(perpendicular_xy(axis))
    half_span = max(float(np.max(np.abs(coords.dot(lateral)))), 0.2) + 0.2

    medial_points: List[List[float]] = []
    for offset in np.linspace(start, end, max(samples, 3)):
        center_point = axis * offset
        cross_section = LineString(
            [
                tuple((center_point - lateral * half_span).tolist()),
                tuple((center_point + lateral * half_span).tolist()),
            ]
        )
        intersection = polygon.intersection(cross_section)
        segment = _longest_linestring(intersection)
        if segment is None:
            continue
        segment_coords = np.asarray(segment.coords, dtype=float)
        midpoint = 0.5 * (segment_coords[0] + segment_coords[-1])
        medial_points.append(to_float_list(midpoint))

    if not medial_points:
        return [to_float_list(support_center)]
    return medial_points


def build_default_swing_vectors(
    legs: Sequence[LegState],
    support_center: np.ndarray,
    forward_axis: np.ndarray,
    trunk_polygon_xy: Sequence[Sequence[float]],
) -> Dict[int, np.ndarray]:
    trunk_points = np.asarray(trunk_polygon_xy, dtype=float)
    if len(trunk_points) == 0:
        body_extent = 0.2
    else:
        body_extent = max(float(np.ptp(trunk_points.dot(forward_axis))), 0.2)
    step_length = 0.18 * body_extent + 0.03
    lateral_axis = normalize(perpendicular_xy(forward_axis))

    ordered = sorted(
        [leg for leg in legs if leg.valid and not leg.locked],
        key=lambda leg: float(np.arctan2(leg.foot_xy[1] - support_center[1], leg.foot_xy[0] - support_center[0])),
    )
    swing_vectors: Dict[int, np.ndarray] = {}
    for index, leg in enumerate(ordered):
        side = np.sign(float(np.dot(leg.foot_xy - support_center, lateral_axis))) or 1.0
        group_bias = 1.0 if index % 2 == 0 else 0.85
        swing_vectors[leg.leg_id] = forward_axis * (step_length * group_bias) - lateral_axis * side * step_length * 0.12
    return swing_vectors


def choose_head_tail_direction(
    axis: np.ndarray,
    legs: Sequence[LegState],
    default_swings: Mapping[int, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    scores: Dict[str, float] = {"positive": 0.0, "negative": 0.0}
    candidate_axes = {"positive": axis, "negative": -axis}
    drive_sums: Dict[str, np.ndarray] = {"positive": np.zeros(2, dtype=float), "negative": np.zeros(2, dtype=float)}

    for label, candidate_axis in candidate_axes.items():
        for leg in legs:
            if not leg.valid or leg.locked:
                continue
            swing = leg.swing_vector_xy if leg.swing_vector_xy is not None else default_swings.get(leg.leg_id)
            if swing is None:
                swing = leg.hip_xy - leg.foot_xy
            amplitude = float(np.linalg.norm(swing))
            if amplitude < EPS:
                continue
            unit_swing = swing / amplitude
            propulsive_gain = max(float(np.dot(candidate_axis, unit_swing)), 0.0)
            if propulsive_gain <= 0.0:
                continue
            phase_gain = 1.15 if leg.phase in {"swing", "lift", "pre_stance", "drop"} else 0.75
            range_gain = 1.0 + min(leg.swing_limit, 1.2)
            contribution = propulsive_gain * amplitude * phase_gain * range_gain
            drive_sums[label] += candidate_axis * contribution
            scores[label] += contribution

    best_label = "positive" if scores["positive"] >= scores["negative"] else "negative"
    return candidate_axes[best_label], drive_sums[best_label], scores


def compute_translational_compensation(
    com_xy: np.ndarray,
    support_center: np.ndarray,
    torque_weights: Sequence[float],
    regularization: float = 0.25,
    max_translation: float = 0.12,
) -> np.ndarray:
    delta = com_xy - support_center
    avg_weight = float(np.mean(np.asarray(torque_weights, dtype=float))) if torque_weights else 1.0
    penalty = np.eye(2, dtype=float) * max(avg_weight, 0.1)
    system = np.eye(2, dtype=float) + regularization * penalty
    translation = -np.linalg.solve(system, delta)
    norm = float(np.linalg.norm(translation))
    if norm > max_translation:
        translation *= max_translation / norm
    return translation


def build_grouped_topology(
    legs: Sequence[LegState],
    forward_axis: np.ndarray,
) -> Dict[str, object]:
    lateral_axis = normalize(perpendicular_xy(forward_axis))
    active_legs = [leg for leg in legs if leg.valid and not leg.locked]
    ordered = sorted(
        active_legs,
        key=lambda leg: (
            float(np.dot(leg.foot_xy, forward_axis)),
            float(np.dot(leg.foot_xy, lateral_axis)),
        ),
    )

    groups = {"group_a": [], "group_b": []}
    for index, leg in enumerate(ordered):
        group_name = "group_a" if index % 2 == 0 else "group_b"
        groups[group_name].append(int(leg.leg_id))

    inhibition_rules = []
    zeroed_edges = []
    for leg in legs:
        if leg.valid and not leg.locked:
            continue
        inhibition_rules.append(
            {
                "leg_id": int(leg.leg_id),
                "reason": "locked" if leg.locked else "missing_or_invalid",
                "in_degree": 0.0,
                "out_degree": 0.0,
            }
        )
        zeroed_edges.append([int(leg.leg_id), 0.0, 0.0])

    phase_offsets = {
        "group_a": 0.0,
        "group_b": float(np.pi),
    }
    return {
        "groups": groups,
        "phase_offsets": phase_offsets,
        "inhibition_rules": inhibition_rules,
        "coupling_matrix_zeroed_edges": zeroed_edges,
    }


def compute_adaptive_plan(
    description: Mapping[str, object],
    state_overrides: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    legs = extract_leg_states(description, state_overrides)
    support_legs, near_stance_legs = select_support_legs(legs)
    support_center, initial_axis = compute_support_statistics(support_legs, near_stance_legs)
    support_polygon = compute_support_polygon(support_legs)
    safety_corridor = approximate_medial_axis(support_polygon, support_center, initial_axis)

    default_swings = build_default_swing_vectors(
        legs,
        support_center,
        initial_axis,
        description.get("trunk_polygon_xy", []),
    )
    forward_axis, drive_vector, direction_scores = choose_head_tail_direction(initial_axis, legs, default_swings)
    com_xy = np.asarray(state_overrides.get("com_xy"), dtype=float) if state_overrides and "com_xy" in state_overrides else estimate_center_of_mass_xy(description)
    torque_weights = [leg.torque_weight for leg in legs if leg.valid and not leg.locked]
    compensation = compute_translational_compensation(com_xy, support_center, torque_weights)
    topology = build_grouped_topology(legs, forward_axis)

    planned_swings = {}
    for leg in legs:
        if not leg.valid or leg.locked:
            continue
        swing = leg.swing_vector_xy if leg.swing_vector_xy is not None else default_swings.get(leg.leg_id, np.zeros(2, dtype=float))
        effective = max(float(np.dot(forward_axis, swing)), 0.0)
        planned_swings[str(leg.leg_id)] = {
            "swing_vector_xy": to_float_list(swing),
            "effective_drive_along_forward": float(effective),
        }

    return {
        "support_center_xy": to_float_list(support_center),
        "projected_com_xy": to_float_list(com_xy),
        "initial_virtual_forward_axis": to_float_list(initial_axis),
        "final_forward_axis": to_float_list(forward_axis),
        "drive_resultant_xy": to_float_list(drive_vector),
        "direction_scores": {key: float(value) for key, value in direction_scores.items()},
        "support_polygon_xy": [to_float_list(point) for point in np.asarray(support_polygon.exterior.coords[:-1], dtype=float)],
        "safety_corridor_xy": safety_corridor,
        "translational_compensation_xy": to_float_list(compensation),
        "planned_swings": planned_swings,
        "support_leg_ids": [int(leg.leg_id) for leg in support_legs],
        "near_stance_leg_ids": [int(leg.leg_id) for leg in near_stance_legs],
        "topology": topology,
    }