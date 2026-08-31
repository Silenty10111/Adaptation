"""Shared gait phase scheduling utilities.

This module is intentionally independent of Isaac Gym.  It owns the phase,
duty-factor and periodic trajectory conventions used by planners, simulators
and MPC contact scheduling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np


TWO_PI = 2.0 * math.pi
DEFAULT_DUTY_FACTOR = 0.60
DUTY_FACTOR_BOUNDS = (0.35, 0.85)
PHASE_STRATEGIES = (
    "binary", "hildebrand", "geometry_wave", "uniform_wave", "balanced_wave",
    "adaptive_wave",
)


def wrap_2pi(angle: float) -> float:
    """Normalize an angle to the half-open interval ``[0, 2*pi)``."""
    wrapped = math.fmod(float(angle), TWO_PI)
    if wrapped < 0.0:
        wrapped += TWO_PI
    return 0.0 if math.isclose(wrapped, TWO_PI, abs_tol=1e-14) else wrapped


def circular_distance(a: float, b: float) -> float:
    """Return the absolute shortest distance between two circular angles."""
    return abs(math.atan2(math.sin(float(a) - float(b)), math.cos(float(a) - float(b))))


def circular_mean(angles: Sequence[float], weights: Optional[Sequence[float]] = None) -> float:
    """Compute a unit-circle mean, including values straddling 0 and 2*pi."""
    if not angles:
        raise ValueError("circular_mean requires at least one angle")
    values = np.asarray(angles, dtype=float)
    if weights is None:
        weight_array = np.ones(len(values), dtype=float)
    else:
        weight_array = np.asarray(weights, dtype=float)
        if weight_array.shape != values.shape:
            raise ValueError("weights must have the same length as angles")
        if np.any(weight_array < 0.0):
            raise ValueError("circular weights must be non-negative")
    if float(np.sum(weight_array)) <= 0.0:
        raise ValueError("circular weights must have a positive sum")
    x = float(np.sum(weight_array * np.cos(values)))
    y = float(np.sum(weight_array * np.sin(values)))
    if math.hypot(x, y) < 1e-12:
        # Antipodal data has no unique mean.  Use the first sample so the
        # fallback remains deterministic instead of depending on round-off.
        return wrap_2pi(float(values[0]))
    return wrap_2pi(math.atan2(y, x))


def circular_interpolate(start: float, end: float, alpha: float) -> float:
    """Interpolate along the shortest circular arc from ``start`` to ``end``."""
    blend = float(np.clip(alpha, 0.0, 1.0))
    delta = math.atan2(math.sin(float(end) - float(start)), math.cos(float(end) - float(start)))
    return wrap_2pi(float(start) + blend * delta)


def smoothstep5(value: float) -> float:
    """Quintic smoothstep on ``[0, 1]`` with zero first/second derivatives."""
    t = float(np.clip(value, 0.0, 1.0))
    return t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)


def clip_duty_factor(
    value: float,
    bounds: Tuple[float, float] = DUTY_FACTOR_BOUNDS,
) -> Tuple[float, bool]:
    """Clip a duty factor and report whether clipping occurred."""
    lower, upper = float(bounds[0]), float(bounds[1])
    if lower >= upper:
        raise ValueError("invalid duty-factor bounds")
    raw = float(value)
    clipped = float(np.clip(raw, lower, upper))
    return clipped, not math.isclose(raw, clipped, rel_tol=0.0, abs_tol=1e-12)


def _integer_keyed(mapping: object) -> Dict[int, float]:
    if not isinstance(mapping, Mapping):
        return {}
    result: Dict[int, float] = {}
    for key, value in mapping.items():
        try:
            result[int(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def resolve_phase_offsets(plan: Mapping, leg_ids: Iterable[int]) -> Dict[int, float]:
    """Resolve per-leg offsets, falling back to legacy A/B groups per leg.

    Explicit ``cpg.phase_offsets`` always wins.  A leg missing from that field
    falls back to 0 for group A, pi for group B, and 0 for any other active leg.
    Passive group-C legs are normally excluded by the caller.
    """
    cpg = plan.get("cpg", {}) if isinstance(plan, Mapping) else {}
    explicit = _integer_keyed(cpg.get("phase_offsets", {}) if isinstance(cpg, Mapping) else {})
    topology = plan.get("topology", {}) if isinstance(plan, Mapping) else {}
    groups = topology.get("groups", {}) if isinstance(topology, Mapping) else {}
    group_b = {int(v) for v in groups.get("group_b", [])}
    return {
        int(leg_id): wrap_2pi(
            explicit[int(leg_id)] if int(leg_id) in explicit
            else (math.pi if int(leg_id) in group_b else 0.0)
        )
        for leg_id in leg_ids
    }


def resolve_duty_factors(
    plan: Mapping,
    leg_ids: Iterable[int],
    default: float = DEFAULT_DUTY_FACTOR,
) -> Tuple[Dict[int, float], List[str]]:
    """Resolve global/per-leg duty factors and return clipping diagnostics."""
    cpg = plan.get("cpg", {}) if isinstance(plan, Mapping) else {}
    if not isinstance(cpg, Mapping):
        cpg = {}
    raw_global = float(cpg.get("duty_factor", default))
    global_duty, global_clipped = clip_duty_factor(raw_global)
    diagnostics: List[str] = []
    if global_clipped:
        diagnostics.append(
            f"cpg.duty_factor clipped from {raw_global:.6g} to {global_duty:.6g}"
        )
    per_leg = _integer_keyed(cpg.get("per_leg_duty_factors", {}))
    resolved: Dict[int, float] = {}
    for leg_id_raw in leg_ids:
        leg_id = int(leg_id_raw)
        raw = per_leg.get(leg_id, global_duty)
        duty, clipped = clip_duty_factor(raw)
        resolved[leg_id] = duty
        if clipped:
            diagnostics.append(
                f"cpg.per_leg_duty_factors[{leg_id}] clipped from {raw:.6g} to {duty:.6g}"
            )
    return resolved, diagnostics


@dataclass(frozen=True)
class PhaseState:
    """Periodic phase state and continuous joint-trajectory coordinates."""

    phase: float
    normalized_phase: float
    duty_factor: float
    is_stance: bool
    stance_progress: float
    swing_progress: float
    fore_aft: float
    lift: float


@dataclass(frozen=True)
class ContactPhaseGateConfig:
    """Bounds for contact-aware admission of nominal swing phases.

    The oscillator remains the timing source.  This gate only delays lift-off
    when the measured support set is too small.  Touchdown hysteresis remains
    in the executor because it requires a history of confirmed airborne
    samples.  Keeping lift-off admission independent of Isaac Gym makes the
    safety logic deterministic and unit-testable.
    """

    minimum_support_count: int = 3
    maximum_simultaneous_swing: int = 3
    swing_start_window: float = 0.35
    early_touchdown_progress: float = 0.35


def contact_aware_swing_set(
    nominal_states: Mapping[int, PhaseState],
    actual_contact_leg_ids: Optional[Iterable[int]],
    previous_swing_leg_ids: Iterable[int] = (),
    config: ContactPhaseGateConfig = ContactPhaseGateConfig(),
) -> Set[int]:
    """Choose which nominal swing legs may actually leave the ground.

    With contact telemetry available, a new swing is admitted only when the
    remaining measured contacts satisfy ``minimum_support_count``.  A swing
    already in progress is allowed to finish; the caller is responsible for
    touchdown hysteresis.  If telemetry is unavailable, only the simultaneous-
    swing bound is applied so legacy non-Isaac callers remain usable.
    """
    states = {int(leg_id): state for leg_id, state in nominal_states.items()}
    previous = {int(leg_id) for leg_id in previous_swing_leg_ids}
    contacts = (
        None if actual_contact_leg_ids is None
        else {int(leg_id) for leg_id in actual_contact_leg_ids}
    )
    maximum_swing = max(int(config.maximum_simultaneous_swing), 0)
    minimum_support = max(int(config.minimum_support_count), 0)

    admitted: Set[int] = set()
    for leg_id in sorted(previous):
        state = states.get(leg_id)
        if state is None or state.is_stance:
            continue
        admitted.add(leg_id)

    if len(admitted) >= maximum_swing:
        return set(sorted(admitted)[:maximum_swing])

    candidates = sorted(
        (
            (state.swing_progress, leg_id)
            for leg_id, state in states.items()
            if not state.is_stance
            and leg_id not in admitted
            and state.swing_progress <= float(config.swing_start_window)
        ),
        key=lambda item: (item[0], item[1]),
    )
    for _, leg_id in candidates:
        if len(admitted) >= maximum_swing:
            break
        if contacts is not None:
            remaining_contacts = len(contacts - admitted - {leg_id})
            if remaining_contacts < minimum_support:
                continue
        admitted.add(leg_id)
    return admitted


def phase_state(phase: float, duty_factor: float) -> PhaseState:
    """Evaluate stance/swing state and a C2 fore-aft periodic trajectory.

    During stance the foot command moves slowly from front (+1) to rear (-1).
    During swing it returns from rear to front.  Position and fore-aft velocity
    are continuous at lift-off and the cycle boundary.  The lift coordinate is
    zero in stance and follows a smooth bell during swing.
    """
    duty, _ = clip_duty_factor(duty_factor)
    wrapped = wrap_2pi(phase)
    q = wrapped / TWO_PI
    if q < duty:
        stance_progress = q / max(duty, 1e-12)
        fore_aft = 1.0 - 2.0 * smoothstep5(stance_progress)
        return PhaseState(
            wrapped, q, duty, True, stance_progress, 0.0,
            float(fore_aft), 0.0,
        )
    swing_progress = (q - duty) / max(1.0 - duty, 1e-12)
    fore_aft = -1.0 + 2.0 * smoothstep5(swing_progress)
    # smoothstep bell is C2 at both joins, unlike sin(pi*s)^2 whose second
    # derivative jumps against the constant stance segment.
    lift = 4.0 * smoothstep5(swing_progress) * smoothstep5(1.0 - swing_progress)
    return PhaseState(
        wrapped, q, duty, False, 1.0, swing_progress,
        float(fore_aft), float(lift),
    )


def leg_phase_state(
    base_phase: float,
    leg_id: int,
    phase_offsets: Mapping[int, float],
    duty_factors: Mapping[int, float],
) -> PhaseState:
    """Evaluate one leg using resolved per-leg phase and duty dictionaries."""
    return phase_state(
        float(base_phase) + float(phase_offsets.get(int(leg_id), 0.0)),
        float(duty_factors.get(int(leg_id), DEFAULT_DUTY_FACTOR)),
    )


def _normalized_axis(axis: Sequence[float]) -> np.ndarray:
    vector = np.asarray(axis[:2], dtype=float)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else np.array([1.0, 0.0], dtype=float)


def _active_geometry(
    foot_positions: Mapping[int, Sequence[float]],
    active_leg_ids: Optional[Iterable[int]],
    missing_leg_ids: Iterable[int],
    locked_leg_ids: Iterable[int],
) -> Dict[int, np.ndarray]:
    missing = {int(v) for v in missing_leg_ids}
    locked = {int(v) for v in locked_leg_ids}
    requested = (
        {int(v) for v in active_leg_ids}
        if active_leg_ids is not None else {int(v) for v in foot_positions}
    )
    return {
        int(leg_id): np.asarray(position, dtype=float)[:2]
        for leg_id, position in foot_positions.items()
        if int(leg_id) in requested and int(leg_id) not in missing | locked
    }


def _side_chains(
    feet: Mapping[int, np.ndarray], forward_axis: np.ndarray,
) -> Tuple[List[int], List[int], Dict[int, float]]:
    if not feet:
        return [], [], {}
    lateral_axis = np.array([-forward_axis[1], forward_axis[0]], dtype=float)
    center = np.mean(np.asarray(list(feet.values()), dtype=float), axis=0)
    lateral = {
        leg_id: float(np.dot(position - center, lateral_axis))
        for leg_id, position in feet.items()
    }
    tolerance = max(max((abs(v) for v in lateral.values()), default=0.0) * 1e-8, 1e-10)
    positive = [leg_id for leg_id, value in lateral.items() if value > tolerance]
    negative = [leg_id for leg_id, value in lateral.items() if value < -tolerance]
    centerline = [leg_id for leg_id, value in lateral.items() if abs(value) <= tolerance]
    # Assign centerline legs deterministically to the shorter chain.  Geometry
    # coordinates are the primary keys; leg ID is only a final exact-tie break.
    centerline.sort(key=lambda lid: (
        -float(np.dot(feet[lid] - center, forward_axis)),
        float(feet[lid][0]), float(feet[lid][1]), lid,
    ))
    for leg_id in centerline:
        (positive if len(positive) <= len(negative) else negative).append(leg_id)
    return positive, negative, lateral


def geometry_wave_phase_offsets(
    foot_positions: Mapping[int, Sequence[float]],
    forward_axis: Sequence[float],
    active_leg_ids: Optional[Iterable[int]] = None,
    missing_leg_ids: Iterable[int] = (),
    locked_leg_ids: Iterable[int] = (),
    wave_count: float = 1.0,
    wave_direction: float = 1.0,
    lateral_phase_lag: float = math.pi,
    use_rank_positions: bool = False,
) -> Dict[int, float]:
    """Build a continuous phase field from directed foot geometry.

    ``use_rank_positions`` implements the regular-chain Hildebrand research
    mode.  The default uses normalized longitudinal coordinates and therefore
    retains irregular physical spacing.  Missing and locked legs are excluded
    before coordinates are normalized.
    """
    feet = _active_geometry(
        foot_positions, active_leg_ids, missing_leg_ids, locked_leg_ids,
    )
    if not feet:
        return {}
    forward = _normalized_axis(forward_axis)
    center = np.mean(np.asarray(list(feet.values()), dtype=float), axis=0)
    positive, negative, _ = _side_chains(feet, forward)
    offsets: Dict[int, float] = {}
    direction_sign = 1.0 if float(wave_direction) >= 0.0 else -1.0

    for side_index, chain in enumerate((positive, negative)):
        if not chain:
            continue
        ordered = sorted(chain, key=lambda lid: (
            -float(np.dot(feet[lid] - center, forward)),
            float(feet[lid][0]), float(feet[lid][1]), lid,
        ))
        longitudinal = {
            leg_id: float(np.dot(feet[leg_id] - center, forward))
            for leg_id in ordered
        }
        x_max = max(longitudinal.values())
        x_min = min(longitudinal.values())
        span = x_max - x_min
        for rank, leg_id in enumerate(ordered):
            if len(ordered) <= 1:
                normalized = 0.0
            elif use_rank_positions or span <= 1e-12:
                normalized = rank / float(len(ordered) - 1)
            else:
                normalized = (x_max - longitudinal[leg_id]) / span
            side_offset = 0.0 if side_index == 0 or not positive or not negative else lateral_phase_lag
            offsets[leg_id] = wrap_2pi(
                direction_sign * TWO_PI * float(wave_count) * normalized + side_offset
            )
    return offsets


def binary_phase_offsets(
    groups: Mapping[str, Sequence[int]], active_leg_ids: Iterable[int],
) -> Dict[int, float]:
    """Return the backward-compatible 0/pi group phase map."""
    group_b = {int(v) for v in groups.get("group_b", [])}
    return {
        int(leg_id): (math.pi if int(leg_id) in group_b else 0.0)
        for leg_id in active_leg_ids
    }


def uniform_geometry_wave_phase_offsets(
    foot_positions: Mapping[int, Sequence[float]],
    forward_axis: Sequence[float],
    active_leg_ids: Optional[Iterable[int]] = None,
    missing_leg_ids: Iterable[int] = (),
    locked_leg_ids: Iterable[int] = (),
    wave_direction: float = 1.0,
) -> Dict[int, float]:
    """Spread every active leg uniformly over one geometry-ordered cycle.

    The legacy two-chain wave can align several irregularly spaced legs at the
    same phase.  This schedule sorts physical front-to-rear stations and then
    lateral position, but assigns a unique equally spaced phase to each leg.
    Consequently an ``N``-leg robot has at most roughly ``N*(1-duty)`` legs in
    swing instead of a morphology-dependent clustered peak.
    """
    feet = _active_geometry(
        foot_positions, active_leg_ids, missing_leg_ids, locked_leg_ids,
    )
    if not feet:
        return {}
    forward = _normalized_axis(forward_axis)
    lateral = np.array([-forward[1], forward[0]], dtype=float)
    center = np.mean(np.asarray(list(feet.values()), dtype=float), axis=0)
    ordered = sorted(feet, key=lambda leg_id: (
        -float(np.dot(feet[leg_id] - center, forward)),
        -float(np.dot(feet[leg_id] - center, lateral)),
        float(feet[leg_id][0]), float(feet[leg_id][1]), leg_id,
    ))
    direction = 1.0 if float(wave_direction) >= 0.0 else -1.0
    count = len(ordered)
    return {
        int(leg_id): wrap_2pi(direction * TWO_PI * rank / count)
        for rank, leg_id in enumerate(ordered)
    }


def balanced_geometry_wave_phase_offsets(
    foot_positions: Mapping[int, Sequence[float]],
    forward_axis: Sequence[float],
    active_leg_ids: Optional[Iterable[int]] = None,
    missing_leg_ids: Iterable[int] = (),
    locked_leg_ids: Iterable[int] = (),
    wave_direction: float = 1.0,
    lateral_phase_lag: float = math.pi,
) -> Dict[int, float]:
    """Uniformly phase each body side and offset the opposite side by pi."""
    feet = _active_geometry(
        foot_positions, active_leg_ids, missing_leg_ids, locked_leg_ids,
    )
    if not feet:
        return {}
    forward = _normalized_axis(forward_axis)
    center = np.mean(np.asarray(list(feet.values()), dtype=float), axis=0)
    positive, negative, _ = _side_chains(feet, forward)
    direction = 1.0 if float(wave_direction) >= 0.0 else -1.0
    result: Dict[int, float] = {}
    for side_index, chain in enumerate((positive, negative)):
        ordered = sorted(chain, key=lambda leg_id: (
            -float(np.dot(feet[leg_id] - center, forward)),
            float(feet[leg_id][0]), float(feet[leg_id][1]), leg_id,
        ))
        side_lag = 0.0 if side_index == 0 else float(lateral_phase_lag)
        for rank, leg_id in enumerate(ordered):
            result[int(leg_id)] = wrap_2pi(
                direction * TWO_PI * rank / max(len(ordered), 1) + side_lag
            )
    return result


def build_phase_offsets(
    strategy: str,
    foot_positions: Mapping[int, Sequence[float]],
    forward_axis: Sequence[float],
    groups: Mapping[str, Sequence[int]],
    active_leg_ids: Iterable[int],
    missing_leg_ids: Iterable[int] = (),
    locked_leg_ids: Iterable[int] = (),
    wave_count: float = 1.0,
    wave_direction: float = 1.0,
    lateral_phase_lag: float = math.pi,
    stability_margin: Optional[float] = None,
) -> Tuple[Dict[int, float], float]:
    """Build offsets for binary, Hildebrand, geometry or adaptive wave modes.

    The adaptive mode is deliberately conservative and experimental: when no
    wave count is supplied it selects 1.0/1.25/1.5 from active leg count, with
    no claim that those values are universally optimal.
    """
    normalized_strategy = str(strategy or "binary").strip().lower()
    if normalized_strategy not in PHASE_STRATEGIES:
        raise ValueError(
            f"unsupported phase_strategy={strategy!r}; expected one of {PHASE_STRATEGIES}"
        )
    missing = {int(v) for v in missing_leg_ids}
    locked = {int(v) for v in locked_leg_ids}
    active = [int(v) for v in active_leg_ids if int(v) not in missing | locked]
    selected_wave_count = float(wave_count)
    if normalized_strategy == "adaptive_wave":
        if not math.isfinite(selected_wave_count) or selected_wave_count <= 0.0:
            selected_wave_count = 1.0 if len(active) <= 6 else (1.25 if len(active) <= 8 else 1.5)
        # Low SSM keeps the conservative one-wave schedule.  This is a policy
        # fallback, not a claim of biological optimality.
        if stability_margin is not None and float(stability_margin) < 0.03:
            selected_wave_count = min(selected_wave_count, 1.0)
        normalized_strategy = "geometry_wave"
    if normalized_strategy == "binary":
        return binary_phase_offsets(groups, active), selected_wave_count
    if normalized_strategy == "uniform_wave":
        return uniform_geometry_wave_phase_offsets(
            foot_positions=foot_positions,
            forward_axis=forward_axis,
            active_leg_ids=active,
            missing_leg_ids=missing,
            locked_leg_ids=locked,
            wave_direction=wave_direction,
        ), 1.0
    if normalized_strategy == "balanced_wave":
        return balanced_geometry_wave_phase_offsets(
            foot_positions=foot_positions,
            forward_axis=forward_axis,
            active_leg_ids=active,
            missing_leg_ids=missing,
            locked_leg_ids=locked,
            wave_direction=wave_direction,
            lateral_phase_lag=lateral_phase_lag,
        ), 1.0
    return geometry_wave_phase_offsets(
        foot_positions=foot_positions,
        forward_axis=forward_axis,
        active_leg_ids=active,
        missing_leg_ids=missing,
        locked_leg_ids=locked,
        wave_count=selected_wave_count,
        wave_direction=wave_direction,
        lateral_phase_lag=lateral_phase_lag,
        use_rank_positions=(normalized_strategy == "hildebrand"),
    ), selected_wave_count


def blend_phase_offsets(
    previous: Mapping[int, float], current: Mapping[int, float], alpha: float,
) -> Dict[int, float]:
    """Blend surviving leg offsets circularly during an online plan switch."""
    result: Dict[int, float] = {}
    for leg_id, target in current.items():
        old = previous.get(int(leg_id), target)
        result[int(leg_id)] = circular_interpolate(old, target, alpha)
    return result


def yaw_rate_error(omega_z: float, omega_z_ref: float) -> float:
    """Return yaw-rate tracking error without suppressing commanded turns."""
    return float(omega_z) - float(omega_z_ref)
