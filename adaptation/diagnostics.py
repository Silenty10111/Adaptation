"""Episode-level locomotion diagnostics and first-root-cause classification.

The accumulator has no Isaac Gym dependency.  Simulation frontends translate
available engine telemetry into :class:`EpisodeStepTelemetry`; unavailable
measurements remain ``None`` and are accompanied by a reason in the output.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from .stability import compute_ssm


FAILURE_PRIORITY = (
    "numerical_failure",
    "invalid_initial_pose",
    "insufficient_contact",
    "support_polygon_failure",
    "body_collision",
    "self_collision",
    "joint_limit",
    "torque_saturation",
    "fall_roll",
    "fall_pitch",
    "reverse_motion",
    "insufficient_forward_motion",
    "excessive_lateral_drift",
    "excessive_yaw_error",
    "unknown",
)


def contact_support_ssm(
    contact_positions: Mapping[int, Sequence[float]],
    com_xy: Sequence[float],
) -> float:
    """Compute a CCW contact-hull SSM, returning 0 for degenerate support."""
    unique = sorted({
        (float(position[0]), float(position[1]))
        for position in contact_positions.values()
    })
    if len(unique) < 3:
        return 0.0

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
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=float)
    return float(compute_ssm(hull, np.asarray(com_xy[:2], dtype=float)))


@dataclass(frozen=True)
class EpisodeDiagnosticConfig:
    """Thresholds for online events and terminal failure classification."""

    minimum_contact_count: int = 2
    minimum_body_height: float = 0.12
    invalid_initial_pose_window_s: float = 0.25
    fall_roll_rad: float = math.radians(45.0)
    fall_pitch_rad: float = math.radians(45.0)
    support_ssm_threshold: float = -0.005
    joint_limit_ratio: float = 1.0
    torque_saturation_ratio: float = 0.98
    contact_failure_persistence_s: float = 0.10
    minimum_forward_displacement: float = 0.02
    reverse_displacement_threshold: float = 0.02
    maximum_drift_ratio: float = 0.30
    maximum_yaw_rmse_rad: float = math.radians(20.0)
    slip_speed_threshold: float = 0.05


@dataclass
class EpisodeStepTelemetry:
    """Measurements for one simulation step; optional means not measured."""

    time_s: float
    body_position: Optional[Sequence[float]] = None
    body_rpy: Optional[Sequence[float]] = None
    yaw_reference: float = 0.0
    commanded_stance_leg_ids: Sequence[int] = field(default_factory=tuple)
    actual_contact_leg_ids: Optional[Sequence[int]] = None
    contact_foot_speed: Optional[Mapping[int, float]] = None
    leg_body_collision_count: Optional[int] = None
    leg_leg_collision_count: Optional[int] = None
    joint_position_ratio: Optional[float] = None
    joint_velocity_ratio: Optional[float] = None
    joint_torque_ratio: Optional[float] = None
    actual_contact_ssm: Optional[float] = None


@dataclass
class EpisodeDiagnostics:
    """Serializable diagnostic record requested by experiment frontends."""

    failure_reason: Optional[str]
    first_failure_time_s: Optional[float]
    fell: bool
    fall_time_s: Optional[float]
    forward_displacement: float
    lateral_displacement: float
    drift_ratio: float
    yaw_tracking_rmse: Optional[float]
    roll_rmse: Optional[float]
    pitch_rmse: Optional[float]
    min_body_height: Optional[float]
    commanded_stance_leg_ids: List[int]
    actual_contact_leg_ids: Optional[List[int]]
    minimum_actual_contact_count: Optional[int]
    contact_mismatch_ratio: Optional[float]
    foot_slip_ratio: Optional[float]
    leg_body_collision_count: Optional[int]
    leg_leg_collision_count: Optional[int]
    peak_joint_position_ratio: Optional[float]
    peak_joint_velocity_ratio: Optional[float]
    peak_joint_torque_ratio: Optional[float]
    ssm_min: Optional[float]
    ssm_p05: Optional[float]
    max_simultaneous_swing_legs: int
    measurement_unavailable_reasons: Dict[str, str]
    failure_events: List[Dict[str, object]]

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)


class EpisodeDiagnosticAccumulator:
    """Accumulate telemetry and preserve the first observed root cause.

    Time ordering dominates classification.  If several failures first occur
    in the same step, :data:`FAILURE_PRIORITY` resolves the tie.
    """

    def __init__(
        self,
        forward_axis: Sequence[float],
        active_leg_count: int,
        config: EpisodeDiagnosticConfig = EpisodeDiagnosticConfig(),
    ) -> None:
        axis = np.asarray(forward_axis[:2], dtype=float)
        norm = float(np.linalg.norm(axis))
        self._forward = axis / norm if norm > 1e-12 else np.array([1.0, 0.0])
        self._lateral = np.array([-self._forward[1], self._forward[0]])
        self._active_leg_count = max(int(active_leg_count), 0)
        self._config = config
        self._initial_xy: Optional[np.ndarray] = None
        self._latest_xy: Optional[np.ndarray] = None
        self._roll: List[float] = []
        self._pitch: List[float] = []
        self._yaw_errors: List[float] = []
        self._heights: List[float] = []
        self._ssm: List[float] = []
        self._contact_counts: List[int] = []
        self._mismatch_numerator = 0
        self._mismatch_denominator = 0
        self._slip_samples = 0
        self._contact_speed_samples = 0
        self._leg_body_collisions: Optional[int] = None
        self._leg_leg_collisions: Optional[int] = None
        self._peak_position: Optional[float] = None
        self._peak_velocity: Optional[float] = None
        self._peak_torque: Optional[float] = None
        self._latest_commanded: List[int] = []
        self._latest_actual: Optional[List[int]] = None
        self._maximum_swing = 0
        self._failure_reason: Optional[str] = None
        self._first_failure_time: Optional[float] = None
        self._fell = False
        self._fall_time: Optional[float] = None
        self._events: List[Dict[str, object]] = []
        self._recorded_event_reasons = set()
        self._pending_failure_since: Dict[str, float] = {}
        self._last_time = 0.0
        self._samples = 0
        self._unavailable: Dict[str, str] = {}

    def mark_unavailable(self, field_name: str, reason: str) -> None:
        """Record why an optional measurement could not be collected."""
        self._unavailable[str(field_name)] = str(reason)

    @staticmethod
    def _finite(values: Optional[Sequence[float]]) -> bool:
        return values is None or bool(np.all(np.isfinite(np.asarray(values, dtype=float))))

    def _record_failures(self, time_s: float, reasons: Sequence[str]) -> None:
        ordered = [reason for reason in FAILURE_PRIORITY if reason in set(reasons)]
        for reason in ordered:
            if reason not in self._recorded_event_reasons:
                self._events.append({"time_s": float(time_s), "reason": reason})
                self._recorded_event_reasons.add(reason)
        if self._failure_reason is None and ordered:
            self._failure_reason = ordered[0]
            self._first_failure_time = float(time_s)

    @staticmethod
    def _update_peak(previous: Optional[float], current: Optional[float]) -> Optional[float]:
        if current is None or not math.isfinite(float(current)):
            return previous
        return max(previous if previous is not None else 0.0, abs(float(current)))

    def update(self, telemetry: EpisodeStepTelemetry) -> None:
        """Consume one time-ordered simulation sample."""
        time_s = float(telemetry.time_s)
        self._last_time = time_s
        self._samples += 1
        reasons: List[str] = []

        if not (
            math.isfinite(time_s)
            and self._finite(telemetry.body_position)
            and self._finite(telemetry.body_rpy)
        ):
            reasons.append("numerical_failure")

        if telemetry.body_position is not None and self._finite(telemetry.body_position):
            position = np.asarray(telemetry.body_position, dtype=float)
            if len(position) >= 2:
                if self._initial_xy is None:
                    self._initial_xy = position[:2].copy()
                self._latest_xy = position[:2].copy()
            if len(position) >= 3:
                height = float(position[2])
                self._heights.append(height)
                if (
                    time_s <= self._config.invalid_initial_pose_window_s
                    and height < self._config.minimum_body_height
                ):
                    reasons.append("invalid_initial_pose")

        roll = pitch = yaw = 0.0
        if telemetry.body_rpy is not None and self._finite(telemetry.body_rpy):
            rpy = list(telemetry.body_rpy)
            if len(rpy) >= 3:
                roll, pitch, yaw = map(float, rpy[:3])
                self._roll.append(roll)
                self._pitch.append(pitch)
                yaw_error = math.atan2(
                    math.sin(yaw - float(telemetry.yaw_reference)),
                    math.cos(yaw - float(telemetry.yaw_reference)),
                )
                self._yaw_errors.append(yaw_error)
                if time_s <= self._config.invalid_initial_pose_window_s and (
                    abs(roll) >= self._config.fall_roll_rad
                    or abs(pitch) >= self._config.fall_pitch_rad
                ):
                    reasons.append("invalid_initial_pose")
                if abs(roll) >= self._config.fall_roll_rad:
                    reasons.append("fall_roll")
                if abs(pitch) >= self._config.fall_pitch_rad:
                    reasons.append("fall_pitch")
                if ("fall_roll" in reasons or "fall_pitch" in reasons) and not self._fell:
                    self._fell = True
                    self._fall_time = time_s

        commanded = {int(value) for value in telemetry.commanded_stance_leg_ids}
        self._latest_commanded = sorted(commanded)
        self._maximum_swing = max(
            self._maximum_swing,
            max(self._active_leg_count - len(commanded), 0),
        )
        if telemetry.actual_contact_leg_ids is not None:
            actual = {int(value) for value in telemetry.actual_contact_leg_ids}
            self._latest_actual = sorted(actual)
            self._contact_counts.append(len(actual))
            union = commanded | actual
            self._mismatch_numerator += len(commanded ^ actual)
            self._mismatch_denominator += max(len(union), 1)
            if len(actual) < self._config.minimum_contact_count:
                reasons.append("insufficient_contact")
            elif len(actual) < 3:
                reasons.append("support_polygon_failure")
            if telemetry.contact_foot_speed is not None:
                for leg_id in actual:
                    speed = telemetry.contact_foot_speed.get(leg_id)
                    if speed is None or not math.isfinite(float(speed)):
                        continue
                    self._contact_speed_samples += 1
                    if float(speed) > self._config.slip_speed_threshold:
                        self._slip_samples += 1

        if telemetry.actual_contact_ssm is not None:
            ssm = float(telemetry.actual_contact_ssm)
            if math.isfinite(ssm):
                self._ssm.append(ssm)
                if ssm < self._config.support_ssm_threshold:
                    reasons.append("support_polygon_failure")
            else:
                reasons.append("numerical_failure")

        if telemetry.leg_body_collision_count is not None:
            count = max(int(telemetry.leg_body_collision_count), 0)
            self._leg_body_collisions = (self._leg_body_collisions or 0) + count
            if count > 0:
                reasons.append("body_collision")
        if telemetry.leg_leg_collision_count is not None:
            count = max(int(telemetry.leg_leg_collision_count), 0)
            self._leg_leg_collisions = (self._leg_leg_collisions or 0) + count
            if count > 0:
                reasons.append("self_collision")

        self._peak_position = self._update_peak(
            self._peak_position, telemetry.joint_position_ratio
        )
        self._peak_velocity = self._update_peak(
            self._peak_velocity, telemetry.joint_velocity_ratio
        )
        self._peak_torque = self._update_peak(
            self._peak_torque, telemetry.joint_torque_ratio
        )
        if (
            telemetry.joint_position_ratio is not None
            and float(telemetry.joint_position_ratio) >= self._config.joint_limit_ratio
        ):
            reasons.append("joint_limit")
        if (
            telemetry.joint_torque_ratio is not None
            and float(telemetry.joint_torque_ratio) >= self._config.torque_saturation_ratio
        ):
            reasons.append("torque_saturation")

        # A one-frame two-contact state is common during a valid touchdown.
        # Require persistence for contact/support failures, but preserve the
        # onset timestamp once the condition is confirmed.
        delayed = {"insufficient_contact", "support_polygon_failure"}
        self._record_failures(time_s, [reason for reason in reasons if reason not in delayed])
        for reason in delayed:
            if reason in reasons:
                onset = self._pending_failure_since.setdefault(reason, time_s)
                if time_s - onset >= self._config.contact_failure_persistence_s:
                    self._record_failures(onset, [reason])
            else:
                self._pending_failure_since.pop(reason, None)

    def _displacements(self) -> tuple[float, float]:
        if self._initial_xy is None or self._latest_xy is None:
            return 0.0, 0.0
        delta = self._latest_xy - self._initial_xy
        return float(np.dot(delta, self._forward)), float(np.dot(delta, self._lateral))

    @staticmethod
    def _rmse(values: Sequence[float]) -> float:
        return float(np.sqrt(np.mean(np.square(values)))) if values else 0.0

    def finalize(self) -> EpisodeDiagnostics:
        """Apply terminal motion checks and return the complete record."""
        forward, lateral = self._displacements()
        drift = abs(lateral) / max(abs(forward), 1e-9)
        yaw_rmse = self._rmse(self._yaw_errors)
        terminal: List[str] = []
        if self._samples == 0:
            terminal.append("numerical_failure")
        elif forward < -self._config.reverse_displacement_threshold:
            terminal.append("reverse_motion")
        elif forward < self._config.minimum_forward_displacement:
            terminal.append("insufficient_forward_motion")
        elif drift > self._config.maximum_drift_ratio:
            terminal.append("excessive_lateral_drift")
        elif yaw_rmse > self._config.maximum_yaw_rmse_rad:
            terminal.append("excessive_yaw_error")
        self._record_failures(self._last_time, terminal)

        if not self._contact_counts:
            self._unavailable.setdefault(
                "actual_contact_leg_ids", "actual contacts were not measured"
            )
            self._unavailable.setdefault(
                "minimum_actual_contact_count", "actual contacts were not measured"
            )
            self._unavailable.setdefault(
                "contact_mismatch_ratio", "actual contacts were not measured"
            )
        if not self._contact_speed_samples:
            self._unavailable.setdefault(
                "foot_slip_ratio", "contacting-foot velocities were not measured"
            )
        if self._leg_body_collisions is None:
            self._unavailable.setdefault(
                "leg_body_collision_count", "rigid contact pairs were not measured"
            )
        if self._leg_leg_collisions is None:
            self._unavailable.setdefault(
                "leg_leg_collision_count", "rigid contact pairs were not measured"
            )
        if self._peak_position is None:
            self._unavailable.setdefault(
                "peak_joint_position_ratio", "joint positions or limits were not measured"
            )
        if self._peak_velocity is None:
            self._unavailable.setdefault(
                "peak_joint_velocity_ratio", "joint velocities or limits were not measured"
            )
        if self._peak_torque is None:
            self._unavailable.setdefault(
                "peak_joint_torque_ratio", "joint torques or effort limits were not measured"
            )
        if not self._ssm:
            self._unavailable.setdefault(
                "ssm_min", "actual contact support polygons were not measured"
            )
            self._unavailable.setdefault(
                "ssm_p05", "actual contact support polygons were not measured"
            )

        return EpisodeDiagnostics(
            failure_reason=self._failure_reason,
            first_failure_time_s=self._first_failure_time,
            fell=self._fell,
            fall_time_s=self._fall_time,
            forward_displacement=forward,
            lateral_displacement=lateral,
            drift_ratio=float(drift),
            yaw_tracking_rmse=yaw_rmse,
            roll_rmse=self._rmse(self._roll),
            pitch_rmse=self._rmse(self._pitch),
            min_body_height=min(self._heights) if self._heights else None,
            commanded_stance_leg_ids=self._latest_commanded,
            actual_contact_leg_ids=self._latest_actual,
            minimum_actual_contact_count=(
                min(self._contact_counts) if self._contact_counts else None
            ),
            contact_mismatch_ratio=(
                self._mismatch_numerator / self._mismatch_denominator
                if self._mismatch_denominator else None
            ),
            foot_slip_ratio=(
                self._slip_samples / self._contact_speed_samples
                if self._contact_speed_samples else None
            ),
            leg_body_collision_count=self._leg_body_collisions,
            leg_leg_collision_count=self._leg_leg_collisions,
            peak_joint_position_ratio=self._peak_position,
            peak_joint_velocity_ratio=self._peak_velocity,
            peak_joint_torque_ratio=self._peak_torque,
            ssm_min=min(self._ssm) if self._ssm else None,
            ssm_p05=float(np.percentile(self._ssm, 5.0)) if self._ssm else None,
            max_simultaneous_swing_legs=self._maximum_swing,
            measurement_unavailable_reasons=dict(self._unavailable),
            failure_events=list(self._events),
        )


def diagnose_trajectory_only(
    samples: Sequence[Sequence[float]],
    forward_axis: Sequence[float],
    dt: float = 1.0 / 60.0,
    yaw_reference: float = 0.0,
    active_leg_count: int = 0,
    config: EpisodeDiagnosticConfig = EpisodeDiagnosticConfig(),
) -> Dict[str, object]:
    """Classify trajectory-only paths while leaving engine telemetry null.

    Samples follow the project convention ``[x, y, yaw?, z?]``.  Roll and
    pitch cannot be reconstructed from that convention and are therefore not
    fabricated; their unavailable status is explicit in the returned mapping.
    """
    accumulator = EpisodeDiagnosticAccumulator(
        forward_axis, active_leg_count=active_leg_count, config=config
    )
    accumulator.mark_unavailable(
        "roll_rmse", "trajectory samples do not contain body roll"
    )
    accumulator.mark_unavailable(
        "pitch_rmse", "trajectory samples do not contain body pitch"
    )
    has_yaw = any(len(sample) >= 3 for sample in samples)
    has_height = any(len(sample) >= 4 for sample in samples)
    if not has_yaw:
        accumulator.mark_unavailable(
            "yaw_tracking_rmse", "trajectory samples do not contain body yaw"
        )
    if not has_height:
        accumulator.mark_unavailable(
            "min_body_height", "trajectory samples do not contain body height"
        )
    for index, sample in enumerate(samples):
        values = list(sample)
        if len(values) < 2:
            accumulator.update(EpisodeStepTelemetry(
                time_s=index * dt, body_position=[float("nan")] * 3,
            ))
            continue
        body_position = [float(values[0]), float(values[1])]
        if len(values) >= 4:
            body_position.append(float(values[3]))
        body_rpy = (
            [0.0, 0.0, float(values[2])] if len(values) >= 3 else None
        )
        accumulator.update(EpisodeStepTelemetry(
            time_s=index * dt,
            body_position=body_position,
            body_rpy=body_rpy,
            yaw_reference=yaw_reference,
        ))
    record = accumulator.finalize().to_dict()
    # Zeros above are placeholders needed only for the accumulator input; the
    # output must remain null when roll/pitch were not measured.
    record["roll_rmse"] = None
    record["pitch_rmse"] = None
    if not has_yaw:
        record["yaw_tracking_rmse"] = None
    if not has_height:
        record["min_body_height"] = None
    return record
