"""Common locomotion metrics and acceptance criteria.

The original experiment scripts mostly judged a gait from its final displacement.
That can label a curved, oscillatory or briefly moving robot as successful.  This
module deliberately keeps the evaluator independent from Isaac Gym so the same
definition is used by unit tests, probes and long batch runs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Sequence

import numpy as np


@dataclass(frozen=True)
class LocomotionCriteria:
    min_forward_speed: float = 0.020
    max_lateral_speed: float = 0.020
    max_drift_ratio: float = 0.25
    max_heading_error_deg: float = 12.0
    max_speed_cv: float = 0.35
    max_height_std: float = 0.035
    min_mean_height: float = 0.20


DEFAULT_CRITERIA = LocomotionCriteria()


def evaluate_trajectory(
    samples: Sequence[Sequence[float]],
    forward_axis: Sequence[float],
    dt: float = 1.0 / 60.0,
    warmup_fraction: float = 0.20,
    gait_frequency_hz: float = 0.85,
    criteria: LocomotionCriteria = DEFAULT_CRITERIA,
) -> Dict[str, object]:
    """Evaluate straight, steady locomotion from ``[x, y, yaw, z?]`` samples."""
    arr = np.asarray(samples, dtype=float)
    if arr.ndim != 2 or len(arr) < 3 or arr.shape[1] < 2:
        return {
            "passed": False,
            "reason": "insufficient_samples",
            "criteria": asdict(criteria),
        }

    fwd = np.asarray(forward_axis[:2], dtype=float)
    norm = float(np.linalg.norm(fwd))
    if norm < 1e-9:
        return {
            "passed": False,
            "reason": "invalid_forward_axis",
            "criteria": asdict(criteria),
        }
    fwd /= norm
    lat = np.array([-fwd[1], fwd[0]], dtype=float)

    start = min(max(int(len(arr) * warmup_fraction), 0), len(arr) - 3)
    steady = arr[start:]
    delta = np.diff(steady[:, :2], axis=0)
    vf = delta @ fwd / dt
    vl = delta @ lat / dt

    # A gait is periodic, so compare cycle-sized moving averages instead of raw
    # frame velocities.  Raw contact impulses would otherwise dominate the CV.
    # Compare cycle-averaged speeds.  Legged locomotion necessarily has
    # within-cycle contact ripple; "steady" means that consecutive gait cycles
    # have a consistent mean velocity.
    cycle_samples = int(round(1.0 / max(gait_frequency_hz * dt, 1e-9)))
    window = max(3, min(cycle_samples, len(vf) // 4))
    kernel = np.ones(window, dtype=float) / window
    vf_smooth = np.convolve(vf, kernel, mode="valid")
    vl_smooth = np.convolve(vl, kernel, mode="valid")

    total_dt = max((len(steady) - 1) * dt, dt)
    displacement = steady[-1, :2] - steady[0, :2]
    forward_speed = float(np.dot(displacement, fwd) / total_dt)
    lateral_speed = float(np.dot(displacement, lat) / total_dt)
    mean_abs_forward_speed = float(np.mean(np.abs(vf_smooth)))
    speed_std = float(np.std(vf_smooth))
    speed_cv = speed_std / max(mean_abs_forward_speed, criteria.min_forward_speed)
    drift_ratio = abs(lateral_speed) / max(abs(forward_speed), 1e-6)

    # Course error, not chassis-yaw error: arbitrary morphologies may walk
    # sideways while keeping the trunk orientation fixed.
    heading_error_deg = float(abs(np.degrees(np.arctan2(lateral_speed, forward_speed))))
    body_heading_error_deg = 0.0
    yaw_std_deg = 0.0
    if arr.shape[1] >= 3:
        planned_yaw = float(np.arctan2(fwd[1], fwd[0]))
        yaw = np.unwrap(steady[:, 2])
        err = np.arctan2(np.sin(yaw - planned_yaw), np.cos(yaw - planned_yaw))
        body_heading_error_deg = float(np.degrees(np.mean(np.abs(err))))
        yaw_std_deg = float(np.degrees(np.std(err)))

    height_std = 0.0
    mean_height = 0.0
    if arr.shape[1] >= 4:
        height_std = float(np.std(steady[:, 3]))
        mean_height = float(np.mean(steady[:, 3]))

    checks = {
        "forward_speed": forward_speed >= criteria.min_forward_speed,
        "lateral_speed": abs(lateral_speed) <= criteria.max_lateral_speed,
        "drift_ratio": drift_ratio <= criteria.max_drift_ratio,
        "heading_error": heading_error_deg <= criteria.max_heading_error_deg,
        "speed_cv": speed_cv <= criteria.max_speed_cv,
        "height_std": height_std <= criteria.max_height_std,
        "mean_height": arr.shape[1] < 4 or mean_height >= criteria.min_mean_height,
    }
    return {
        "passed": bool(all(checks.values())),
        "reason": "accepted" if all(checks.values()) else "criteria_failed",
        "checks": checks,
        "criteria": asdict(criteria),
        "forward_speed": forward_speed,
        "lateral_speed": lateral_speed,
        "drift_ratio": float(drift_ratio),
        "heading_error_deg": heading_error_deg,
        "body_heading_error_deg": body_heading_error_deg,
        "yaw_std_deg": yaw_std_deg,
        "speed_mean_abs": mean_abs_forward_speed,
        "speed_std": speed_std,
        "speed_cv": float(speed_cv),
        "height_std": height_std,
        "mean_height": mean_height,
        "steady_samples": int(len(steady)),
    }
