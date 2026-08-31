"""Static full-cycle gait feasibility checks and constrained candidate selection.

All metrics in this module are kinematic/static *proxies*.  No function claims
to predict dynamic speed, collision impulse, slip, or actuator feasibility.
The stance convention is shared with :mod:`adaptation.phase`.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .phase import (
    DEFAULT_DUTY_FACTOR,
    TWO_PI,
    binary_phase_offsets,
    blend_phase_offsets,
    circular_distance,
    geometry_wave_phase_offsets,
    leg_phase_state,
    resolve_duty_factors,
    resolve_phase_offsets,
    wrap_2pi,
)
from .stability import compute_ssm


@dataclass(frozen=True)
class GaitFeasibilityConfig:
    """Hard constraints used while sampling a complete commanded cycle."""

    samples_per_cycle: int = 360
    minimum_ssm: float = 0.0
    minimum_stance_count: int = 3
    allow_dynamic_support: bool = False
    maximum_phase_jump: float = math.pi
    minimum_adjacent_swing_spacing: float = 0.08


@dataclass(frozen=True)
class CandidateSearchConfig:
    """Finite experimental search space; not a biological optimality claim."""

    wave_counts: Tuple[float, ...] = (0.5, 1.0, 1.25, 1.5, 2.0)
    global_phase_origins: Tuple[float, ...] = (
        0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0,
    )
    lateral_phase_offsets: Tuple[float, ...] = (
        math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0,
    )
    duty_factors: Tuple[float, ...] = (0.50, 0.60, 0.70, 0.80)
    wave_direction: float = 1.0


@dataclass(frozen=True)
class CandidateScoreWeights:
    """Weights for explicitly named static/kinematic proxy metrics."""

    ssm_p05: float = 5.0
    ssm_minimum: float = 2.0
    propulsion_proxy: float = 1.0
    left_right_imbalance: float = 1.5
    yaw_moment_proxy: float = 1.5
    simultaneous_swing: float = 0.5
    collision_risk: float = 3.0
    phase_jump: float = 0.5
    binary_change: float = 0.2


@dataclass
class GaitCandidate:
    """One deterministic phase/duty candidate."""

    name: str
    strategy: str
    phase_offsets: Dict[int, float]
    duty_factors: Dict[int, float]
    wave_count: float = 1.0
    global_phase_origin: float = 0.0
    lateral_phase_offset: float = math.pi
    stride_scale_factor: float = 1.0
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class GaitFeasibilityResult:
    """Full-cycle hard-constraint result and static proxy metrics."""

    ssm_min: float
    ssm_p05: float
    ssm_mean: float
    minimum_stance_count: int
    maximum_swing_count: int
    phase_jump_cost: float
    feasible: bool
    rejection_reasons: List[str]
    nominal_collision_risk_count: int
    propulsion_proxy: float
    left_right_support_imbalance: float
    yaw_moment_proxy: float
    binary_change_cost: float
    initial_stance_leg_ids: List[int]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class GaitSelectionResult:
    """Selected candidate plus auditable screening results."""

    candidate: GaitCandidate
    feasibility: GaitFeasibilityResult
    score: Optional[float]
    selection_mode: str
    evaluated_candidates: int
    feasible_candidates: int
    candidate_summaries: List[Dict[str, object]] = field(default_factory=list)


def _normalized_axis(axis: Sequence[float]) -> np.ndarray:
    vector = np.asarray(axis[:2], dtype=float)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else np.array([1.0, 0.0], dtype=float)


def _convex_hull(points: Sequence[Sequence[float]]) -> np.ndarray:
    unique = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(unique) <= 2:
        return np.asarray(unique, dtype=float)

    def cross(origin, first, second):
        return ((first[0] - origin[0]) * (second[1] - origin[1])
                - (first[1] - origin[1]) * (second[0] - origin[0]))

    lower: List[Tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: List[Tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _adjacent_pairs(
    feet: Mapping[int, np.ndarray], center: np.ndarray,
) -> List[Tuple[int, int]]:
    if len(feet) < 2:
        return []
    ordered = sorted(
        feet,
        key=lambda leg_id: (
            math.atan2(feet[leg_id][1] - center[1], feet[leg_id][0] - center[0]),
            leg_id,
        ),
    )
    if len(ordered) == 2:
        return [(ordered[0], ordered[1])]
    return [
        (ordered[index], ordered[(index + 1) % len(ordered)])
        for index in range(len(ordered))
    ]


def _candidate_duties(
    active_leg_ids: Iterable[int], duty_factor: float,
) -> Dict[int, float]:
    return {int(leg_id): float(duty_factor) for leg_id in active_leg_ids}


def evaluate_gait_candidate(
    *,
    active_leg_ids: Iterable[int],
    foot_positions: Mapping[int, Sequence[float]],
    com_xy: Sequence[float],
    forward_axis: Sequence[float],
    candidate: GaitCandidate,
    missing_leg_ids: Iterable[int] = (),
    locked_leg_ids: Iterable[int] = (),
    passive_leg_ids: Iterable[int] = (),
    previous_phase_offsets: Optional[Mapping[int, float]] = None,
    binary_reference_offsets: Optional[Mapping[int, float]] = None,
    config: GaitFeasibilityConfig = GaitFeasibilityConfig(),
) -> GaitFeasibilityResult:
    """Sample one full cycle using exactly the executor's ``q < D`` rule."""
    missing = {int(value) for value in missing_leg_ids}
    locked = {int(value) for value in locked_leg_ids}
    passive = {int(value) for value in passive_leg_ids} | locked
    requested_active = [int(value) for value in active_leg_ids]
    active = sorted(set(requested_active) - missing - locked)
    feet = {
        int(leg_id): np.asarray(position, dtype=float)[:2]
        for leg_id, position in foot_positions.items()
    }
    com = np.asarray(com_xy[:2], dtype=float)
    forward = _normalized_axis(forward_axis)
    lateral = np.array([-forward[1], forward[0]], dtype=float)
    reasons: List[str] = []

    candidate_ids = {int(value) for value in candidate.phase_offsets}
    duty_ids = {int(value) for value in candidate.duty_factors}
    if set(requested_active) & missing or candidate_ids & missing or duty_ids & missing:
        reasons.append("missing_leg_access")
    if set(requested_active) & locked or candidate_ids & locked or duty_ids & locked:
        reasons.append("locked_leg_active")
    if any(leg_id not in feet for leg_id in set(active) | candidate_ids):
        reasons.append("missing_foot_geometry")
    if any(leg_id not in candidate_ids for leg_id in active):
        reasons.append("missing_phase_offset")
    if any(leg_id not in duty_ids for leg_id in active):
        reasons.append("missing_duty_factor")
    values = list(candidate.phase_offsets.values()) + list(candidate.duty_factors.values())
    if not all(math.isfinite(float(value)) for value in values):
        reasons.append("non_finite_phase_or_duty")
    if any(not 0.0 < float(candidate.duty_factors.get(leg_id, 0.0)) < 1.0 for leg_id in active):
        reasons.append("invalid_duty_factor")

    phase_jump_values = []
    if previous_phase_offsets is not None:
        for leg_id in active:
            if leg_id in previous_phase_offsets and leg_id in candidate.phase_offsets:
                phase_jump_values.append(circular_distance(
                    previous_phase_offsets[leg_id], candidate.phase_offsets[leg_id]
                ))
        if phase_jump_values and max(phase_jump_values) > config.maximum_phase_jump + 1e-12:
            reasons.append("phase_jump_limit")
    phase_jump_cost = float(np.mean(phase_jump_values)) if phase_jump_values else 0.0

    binary_change_values = []
    if binary_reference_offsets is not None:
        for leg_id in active:
            if leg_id in binary_reference_offsets and leg_id in candidate.phase_offsets:
                binary_change_values.append(circular_distance(
                    binary_reference_offsets[leg_id], candidate.phase_offsets[leg_id]
                ))
    binary_change_cost = (
        float(np.mean(binary_change_values)) if binary_change_values else 0.0
    )

    active_feet = {leg_id: feet[leg_id] for leg_id in active if leg_id in feet}
    adjacency = _adjacent_pairs(active_feet, com)
    close_pairs = {
        tuple(sorted((a, b)))
        for a, b in adjacency
        if float(np.linalg.norm(feet[a] - feet[b]))
        < config.minimum_adjacent_swing_spacing
    }

    ssm_values: List[float] = []
    stance_counts: List[int] = []
    swing_counts: List[int] = []
    imbalance_values: List[float] = []
    yaw_proxy_values: List[float] = []
    collision_risk_count = 0
    degenerate = False
    initial_stance: List[int] = []
    sample_count = max(int(config.samples_per_cycle), 4)
    for sample_index in range(sample_count):
        base_phase = TWO_PI * sample_index / sample_count
        stance = set(passive)
        swing = set()
        for leg_id in active:
            if leg_id not in candidate.phase_offsets or leg_id not in candidate.duty_factors:
                continue
            state = leg_phase_state(
                base_phase, leg_id, candidate.phase_offsets, candidate.duty_factors
            )
            (stance if state.is_stance else swing).add(leg_id)
        if sample_index == 0:
            initial_stance = sorted(stance)
        stance_points = [feet[leg_id] for leg_id in stance if leg_id in feet]
        hull = _convex_hull(stance_points)
        if len(hull) < 3:
            degenerate = True
            ssm_values.append(0.0)
        else:
            ssm_values.append(float(compute_ssm(hull, com)))
        stance_counts.append(len(stance))
        swing_counts.append(len(swing))
        left = sum(
            1 for leg_id in stance if leg_id in feet
            and float(np.dot(feet[leg_id] - com, lateral)) >= 0.0
        )
        right = sum(1 for leg_id in stance if leg_id in feet) - left
        imbalance_values.append(abs(left - right) / max(left + right, 1))
        yaw_proxy_values.append(abs(sum(
            float((feet[leg_id][0] - com[0]) * forward[1]
                  - (feet[leg_id][1] - com[1]) * forward[0])
            for leg_id in stance if leg_id in feet
        )))
        collision_risk_count += sum(
            1 for pair in close_pairs if pair[0] in swing and pair[1] in swing
        )

    ssm_min = float(np.min(ssm_values)) if ssm_values else float("-inf")
    ssm_p05 = float(np.percentile(ssm_values, 5.0)) if ssm_values else float("-inf")
    ssm_mean = float(np.mean(ssm_values)) if ssm_values else float("-inf")
    minimum_stance = min(stance_counts) if stance_counts else 0
    maximum_swing = max(swing_counts) if swing_counts else len(active)
    if degenerate and not config.allow_dynamic_support:
        reasons.append("degenerate_support_polygon")
    if minimum_stance < config.minimum_stance_count and not config.allow_dynamic_support:
        reasons.append("insufficient_commanded_stance")
    if ssm_min < config.minimum_ssm:
        reasons.append("ssm_below_threshold")
    if collision_risk_count > 0:
        reasons.append("nominal_adjacent_swing_collision_risk")
    reasons = list(dict.fromkeys(reasons))

    mean_duty = (
        float(np.mean([candidate.duty_factors[leg_id] for leg_id in active]))
        if active and all(leg_id in candidate.duty_factors for leg_id in active)
        else 0.0
    )
    # Named proxy only: rewards having both support time and recovery time.
    propulsion_proxy = len(active) * 4.0 * mean_duty * (1.0 - mean_duty)
    return GaitFeasibilityResult(
        ssm_min=ssm_min,
        ssm_p05=ssm_p05,
        ssm_mean=ssm_mean,
        minimum_stance_count=minimum_stance,
        maximum_swing_count=maximum_swing,
        phase_jump_cost=phase_jump_cost,
        feasible=not reasons,
        rejection_reasons=reasons,
        nominal_collision_risk_count=collision_risk_count,
        propulsion_proxy=float(propulsion_proxy),
        left_right_support_imbalance=float(np.mean(imbalance_values)),
        yaw_moment_proxy=float(np.mean(yaw_proxy_values)),
        binary_change_cost=binary_change_cost,
        initial_stance_leg_ids=initial_stance,
    )


def generate_gait_candidates(
    *,
    active_leg_ids: Iterable[int],
    foot_positions: Mapping[int, Sequence[float]],
    forward_axis: Sequence[float],
    groups: Mapping[str, Sequence[int]],
    search: CandidateSearchConfig = CandidateSearchConfig(),
) -> List[GaitCandidate]:
    """Generate deterministic binary, Hildebrand and geometry candidates."""
    active = sorted({int(value) for value in active_leg_ids})
    candidates: List[GaitCandidate] = []
    base_duty = 0.60
    binary = binary_phase_offsets(groups, active)
    candidates.append(GaitCandidate(
        "binary", "binary", binary, _candidate_duties(active, base_duty)
    ))
    hildebrand = geometry_wave_phase_offsets(
        foot_positions, forward_axis, active_leg_ids=active,
        wave_count=1.0, lateral_phase_lag=math.pi, use_rank_positions=True,
    )
    candidates.append(GaitCandidate(
        "hildebrand", "hildebrand", hildebrand,
        _candidate_duties(active, base_duty),
    ))
    for wave_count in search.wave_counts:
        for origin in search.global_phase_origins:
            for lateral_offset in search.lateral_phase_offsets:
                raw = geometry_wave_phase_offsets(
                    foot_positions,
                    forward_axis,
                    active_leg_ids=active,
                    wave_count=float(wave_count),
                    wave_direction=search.wave_direction,
                    lateral_phase_lag=float(lateral_offset),
                )
                shifted = {
                    leg_id: wrap_2pi(offset + float(origin))
                    for leg_id, offset in raw.items()
                }
                for duty in search.duty_factors:
                    name = (
                        f"geometry_wave:w={wave_count:g}:o={origin:.6f}:"
                        f"lr={lateral_offset:.6f}:d={duty:.3f}"
                    )
                    candidates.append(GaitCandidate(
                        name=name,
                        strategy="geometry_wave",
                        phase_offsets=shifted,
                        duty_factors=_candidate_duties(active, float(duty)),
                        wave_count=float(wave_count),
                        global_phase_origin=float(origin),
                        lateral_phase_offset=float(lateral_offset),
                    ))
    return candidates


def score_gait_candidate(
    result: GaitFeasibilityResult,
    weights: CandidateScoreWeights = CandidateScoreWeights(),
) -> float:
    """Score feasible static proxy metrics; raises for rejected candidates."""
    if not result.feasible:
        raise ValueError("cannot score an infeasible gait candidate")
    return float(
        weights.ssm_p05 * result.ssm_p05
        + weights.ssm_minimum * result.ssm_min
        + weights.propulsion_proxy * result.propulsion_proxy
        - weights.left_right_imbalance * result.left_right_support_imbalance
        - weights.yaw_moment_proxy * result.yaw_moment_proxy
        - weights.simultaneous_swing * result.maximum_swing_count
        - weights.collision_risk * result.nominal_collision_risk_count
        - weights.phase_jump * result.phase_jump_cost
        - weights.binary_change * result.binary_change_cost
    )


def candidate_from_plan(plan: Mapping, active_leg_ids: Iterable[int]) -> GaitCandidate:
    """Convert a backward-compatible gait plan to a fully resolved candidate."""
    active = sorted({int(value) for value in active_leg_ids})
    offsets = resolve_phase_offsets(plan, active)
    duties, _ = resolve_duty_factors(plan, active)
    cpg = plan.get("cpg", {}) if isinstance(plan, Mapping) else {}
    return GaitCandidate(
        name="current_plan",
        strategy=str(cpg.get("phase_strategy", "binary")),
        phase_offsets=offsets,
        duty_factors=duties,
        wave_count=float(cpg.get("wave_count", 1.0)),
    )


def select_gait_candidate(
    *,
    active_leg_ids: Iterable[int],
    foot_positions: Mapping[int, Sequence[float]],
    com_xy: Sequence[float],
    forward_axis: Sequence[float],
    groups: Mapping[str, Sequence[int]],
    missing_leg_ids: Iterable[int] = (),
    locked_leg_ids: Iterable[int] = (),
    passive_leg_ids: Iterable[int] = (),
    current_candidate: Optional[GaitCandidate] = None,
    feasibility_config: GaitFeasibilityConfig = GaitFeasibilityConfig(),
    search_config: CandidateSearchConfig = CandidateSearchConfig(),
    score_weights: CandidateScoreWeights = CandidateScoreWeights(),
) -> GaitSelectionResult:
    """Hard-filter candidates, score feasible ones, and apply safe fallbacks."""
    active = sorted({int(value) for value in active_leg_ids})
    binary_reference = binary_phase_offsets(groups, active)
    candidates = generate_gait_candidates(
        active_leg_ids=active,
        foot_positions=foot_positions,
        forward_axis=forward_axis,
        groups=groups,
        search=search_config,
    )
    if current_candidate is not None:
        candidates.insert(0, current_candidate)
    previous = current_candidate.phase_offsets if current_candidate is not None else None
    evaluated: List[Tuple[GaitCandidate, GaitFeasibilityResult, Optional[float]]] = []
    for candidate in candidates:
        result = evaluate_gait_candidate(
            active_leg_ids=active,
            foot_positions=foot_positions,
            com_xy=com_xy,
            forward_axis=forward_axis,
            candidate=candidate,
            missing_leg_ids=missing_leg_ids,
            locked_leg_ids=locked_leg_ids,
            passive_leg_ids=passive_leg_ids,
            previous_phase_offsets=previous,
            binary_reference_offsets=binary_reference,
            config=feasibility_config,
        )
        score = score_gait_candidate(result, score_weights) if result.feasible else None
        evaluated.append((candidate, result, score))

    feasible = [item for item in evaluated if item[1].feasible]
    summaries = [
        {
            "name": candidate.name,
            "strategy": candidate.strategy,
            "feasible": result.feasible,
            "score": score,
            "rejection_reasons": result.rejection_reasons,
            "ssm_min": result.ssm_min,
            "ssm_p05": result.ssm_p05,
        }
        for candidate, result, score in evaluated
    ]
    if feasible:
        chosen = max(
            feasible,
            key=lambda item: (float(item[2]), -candidates.index(item[0])),
        )
        return GaitSelectionResult(
            chosen[0], chosen[1], chosen[2], "scored_feasible_candidate",
            len(evaluated), len(feasible), summaries,
        )

    # Fallback 1 is already represented by current_candidate above.  If it is
    # infeasible, construct a deterministic high-duty/half-stride binary plan.
    conservative = GaitCandidate(
        name="conservative_binary",
        strategy="binary",
        phase_offsets=binary_reference,
        duty_factors=_candidate_duties(active, 0.80),
        stride_scale_factor=0.50,
        metadata={"fallback": "increased_duty_reduced_stride"},
    )
    conservative_result = evaluate_gait_candidate(
        active_leg_ids=active,
        foot_positions=foot_positions,
        com_xy=com_xy,
        forward_axis=forward_axis,
        candidate=conservative,
        missing_leg_ids=missing_leg_ids,
        locked_leg_ids=locked_leg_ids,
        passive_leg_ids=passive_leg_ids,
        previous_phase_offsets=previous,
        binary_reference_offsets=binary_reference,
        config=feasibility_config,
    )
    if conservative_result.feasible:
        return GaitSelectionResult(
            conservative, conservative_result,
            score_gait_candidate(conservative_result, score_weights),
            "conservative_fallback", len(evaluated) + 1, 1, summaries,
        )

    # Last resort: command no active swing.  It is still checked and its
    # static infeasibility remains visible rather than being hidden.
    valid_feet = sorted(
        int(leg_id) for leg_id in foot_positions
        if int(leg_id) not in {int(value) for value in missing_leg_ids}
    )
    stop = GaitCandidate(
        name="all_stance_stop",
        strategy="stop",
        phase_offsets={},
        duty_factors={},
        stride_scale_factor=0.0,
        metadata={"fallback": "all_stance_stop", "passive_leg_ids": valid_feet},
    )
    stop_result = evaluate_gait_candidate(
        active_leg_ids=[],
        foot_positions=foot_positions,
        com_xy=com_xy,
        forward_axis=forward_axis,
        candidate=stop,
        missing_leg_ids=missing_leg_ids,
        locked_leg_ids=locked_leg_ids,
        passive_leg_ids=valid_feet,
        config=feasibility_config,
    )
    summaries.append({
        "name": conservative.name,
        "strategy": conservative.strategy,
        "feasible": conservative_result.feasible,
        "score": None,
        "rejection_reasons": conservative_result.rejection_reasons,
        "ssm_min": conservative_result.ssm_min,
        "ssm_p05": conservative_result.ssm_p05,
    })
    return GaitSelectionResult(
        stop, stop_result, None, "all_stance_stop_fallback",
        len(evaluated) + 2, int(stop_result.feasible), summaries,
    )


def blend_and_validate_phase_switch(
    *,
    previous: GaitCandidate,
    target: GaitCandidate,
    alpha: float,
    active_leg_ids: Iterable[int],
    foot_positions: Mapping[int, Sequence[float]],
    com_xy: Sequence[float],
    forward_axis: Sequence[float],
    missing_leg_ids: Iterable[int] = (),
    locked_leg_ids: Iterable[int] = (),
    passive_leg_ids: Iterable[int] = (),
    config: GaitFeasibilityConfig = GaitFeasibilityConfig(),
) -> Tuple[GaitCandidate, GaitFeasibilityResult]:
    """Blend surviving legs on shortest arcs and re-check the intermediate plan."""
    missing = {int(value) for value in missing_leg_ids}
    locked = {int(value) for value in locked_leg_ids}
    active = sorted(
        {int(value) for value in active_leg_ids} - missing - locked
    )
    blended_offsets = blend_phase_offsets(
        {leg_id: previous.phase_offsets[leg_id]
         for leg_id in active if leg_id in previous.phase_offsets},
        {leg_id: target.phase_offsets[leg_id]
         for leg_id in active if leg_id in target.phase_offsets},
        alpha,
    )
    blend = float(np.clip(alpha, 0.0, 1.0))
    blended_duties = {
        leg_id: (
            (1.0 - blend) * previous.duty_factors.get(leg_id, DEFAULT_DUTY_FACTOR)
            + blend * target.duty_factors.get(leg_id, DEFAULT_DUTY_FACTOR)
        )
        for leg_id in active
    }
    candidate = GaitCandidate(
        name=f"phase_transition:{blend:.3f}",
        strategy=target.strategy,
        phase_offsets=blended_offsets,
        duty_factors=blended_duties,
        wave_count=target.wave_count,
        global_phase_origin=target.global_phase_origin,
        lateral_phase_offset=target.lateral_phase_offset,
        stride_scale_factor=(
            (1.0 - blend) * previous.stride_scale_factor
            + blend * target.stride_scale_factor
        ),
        metadata={"transition_alpha": blend},
    )
    result = evaluate_gait_candidate(
        active_leg_ids=active,
        foot_positions=foot_positions,
        com_xy=com_xy,
        forward_axis=forward_axis,
        candidate=candidate,
        missing_leg_ids=missing,
        locked_leg_ids=locked,
        passive_leg_ids=passive_leg_ids,
        previous_phase_offsets=previous.phase_offsets,
        config=config,
    )
    return candidate, result


def apply_candidate_to_plan(
    plan: Mapping, candidate: GaitCandidate,
    active_leg_ids: Iterable[int], locked_leg_ids: Iterable[int] = (),
) -> Dict:
    """Return a backward-compatible plan copy containing a checked candidate."""
    result = copy.deepcopy(dict(plan))
    active = sorted({int(value) for value in active_leg_ids})
    locked = {int(value) for value in locked_leg_ids}
    cpg = result.setdefault("cpg", {})
    cpg["phase_strategy"] = candidate.strategy
    cpg["active_leg_ids"] = [leg_id for leg_id in active if leg_id not in locked]
    cpg["phase_offsets"] = {
        str(leg_id): float(candidate.phase_offsets[leg_id])
        for leg_id in cpg["active_leg_ids"] if leg_id in candidate.phase_offsets
    }
    cpg["per_leg_duty_factors"] = {
        str(leg_id): float(candidate.duty_factors[leg_id])
        for leg_id in cpg["active_leg_ids"] if leg_id in candidate.duty_factors
    }
    if cpg["per_leg_duty_factors"]:
        cpg["duty_factor"] = float(np.mean(list(cpg["per_leg_duty_factors"].values())))
    cpg["wave_count"] = float(candidate.wave_count)
    cpg["global_phase_origin"] = float(candidate.global_phase_origin)
    cpg["lateral_phase_lag"] = float(candidate.lateral_phase_offset)
    topology = result.setdefault("topology", {})
    groups = topology.setdefault("groups", {})
    groups["group_a"] = [value for value in groups.get("group_a", []) if value not in locked]
    groups["group_b"] = [value for value in groups.get("group_b", []) if value not in locked]
    groups["group_c"] = sorted(set(groups.get("group_c", [])) | locked)
    amplitudes = topology.setdefault("per_leg_stride_amplitudes", {})
    for leg_id in cpg["active_leg_ids"]:
        key = str(leg_id)
        amplitudes[key] = float(amplitudes.get(key, 1.0)) * candidate.stride_scale_factor
    if candidate.strategy == "stop":
        valid = sorted(set(active) | locked)
        groups["group_a"] = []
        groups["group_b"] = []
        groups["group_c"] = valid
    return result
