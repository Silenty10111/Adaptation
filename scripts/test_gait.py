#!/usr/bin/env python3
"""Adaptive gait test in Isaac Gym with forward-direction visualisation and grouped locomotion.

Loads the generated URDF, computes the adaptive plan (via plan_gait / adaptation.gait),
then executes an alternating group-phase cyclic gait while rendering on-screen arrows for:

    - final_forward_axis  (orange)
    - support_polygon_xy  (green, when SSM > 0)
    - projected_com_xy    (cyan cross)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from adaptation.diagnostics import diagnose_trajectory_only
from adaptation.phase import (
    DEFAULT_DUTY_FACTOR,
    leg_phase_state,
    phase_state,
    resolve_duty_factors,
    resolve_phase_offsets,
)


# ===========================================================================
# USER HYPERPARAMETER — change this one path to select the robot to visualize.
#
# Accepted values:
#   1. A robot directory containing robot_description.json and robot.urdf
#      (or generated_robot.urdf).
#   2. A direct path to a .urdf file.
#   3. A direct path to robot_description.json.
#
# Example for a robot from the 500-robot batch:
# ROBOT_MODEL_PATH = REPO_ROOT / "batch_results/20260818_131804/robot_00_seed7"
# ===========================================================================
REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOT_MODEL_PATH = REPO_ROOT / "batch_results/20260818_131804/robot_05_seed350292"

# Viewer gait defaults.  These match the controller used by the validated
# long-horizon simulations rather than the older symmetric sine executor.
DEFAULT_SWING_RATIO_AMPLITUDE = 0.32
# A 0.10 rad lift produces about 3--5 cm unloaded foot clearance on the
# generated morphology family.  Larger values look dramatic but leave too
# little tripod support and caused progressive yaw in long-horizon tests.
SWING_LIFT_DELTA_RAD = 0.10
DEFAULT_GAIT_EXECUTOR = "step_cycle"


TARGET_PYTHON = "/data/conda/envs/unitree-rl/bin/python"
TARGET_LD_PATH = "/data/conda/envs/unitree-rl/lib"
ASSET_DIR_NAME = "robot_assets"
DEFAULT_VARIANTS_DIR = "variants"
DEFAULT_VARIANT_URDF = "robot.urdf"
STANDARD_SUBDIR = "standard_hexapod"


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def maybe_reexec_in_unitree_env() -> bool:
    if os.environ.get("TEST_GAIT_REEXEC") == "1":
        return False
    if sys.executable == TARGET_PYTHON:
        return False
    if not Path(TARGET_PYTHON).exists():
        return False
    env = dict(os.environ)
    ld_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{TARGET_LD_PATH}:{ld_path}" if ld_path else TARGET_LD_PATH
    # Add project root to PYTHONPATH so adaptation package is importable after re-exec
    project_root = str(Path(__file__).resolve().parent.parent)
    py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{project_root}:{py_path}" if py_path else project_root
    env["TEST_GAIT_REEXEC"] = "1"
    print("[INFO] Auto-switching to unitree-rl environment.")
    os.execvpe(TARGET_PYTHON, [TARGET_PYTHON, *sys.argv], env)
    return True


def load_gymapi():
    try:
        from isaacgym import gymapi  # type: ignore
        return gymapi
    except ModuleNotFoundError:
        maybe_reexec_in_unitree_env()
        print("[ERROR] Cannot import isaacgym. Run with unitree-rl python.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Adaptive plan
# ---------------------------------------------------------------------------

def compute_plan(description: dict, state: dict) -> dict:
    try:
        from adaptation.gait import compute_adaptive_plan
        return compute_adaptive_plan(description, state)
    except (ModuleNotFoundError, ImportError):
        num_legs = int(description.get("num_legs", 0))
        ids = list(range(num_legs))
        return {
            "support_center_xy": [0.0, 0.0],
            "projected_com_xy": [0.0, 0.0],
            "initial_virtual_forward_axis": [1.0, 0.0],
            "final_forward_axis": [1.0, 0.0],
            "drive_resultant_xy": [0.0, 0.0],
            "direction_scores": {"positive": 0.0, "negative": 0.0},
            "support_polygon_xy": [],
            "safety_corridor_xy": [],
            "translational_compensation_xy": [0.0, 0.0],
            "planned_swings": {},
            "support_leg_ids": ids,
            "near_stance_leg_ids": [],
            "topology": {
                "groups": {"group_a": ids[::2], "group_b": ids[1::2]},
                "phase_offsets": {"group_a": 0.0, "group_b": float(np.pi)},
                "inhibition_rules": [],
                "coupling_matrix_zeroed_edges": [],
            },
        }


def load_description(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_relative_path(path: Path) -> Path:
    """Resolve user paths relative to the repository, independent of cwd."""
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    repo_candidate = (REPO_ROOT / path).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return (Path.cwd() / path).resolve()


def resolve_robot_model(model_path: Path) -> tuple[Path, Path]:
    """Resolve one model hyperparameter into description and URDF paths.

    ``model_path`` may be a robot directory, a URDF, or a JSON description.
    Resolution is intentionally strict: a typo must not silently load a
    fallback robot and produce a misleading visualization.
    """
    model_path = _repo_relative_path(Path(model_path))
    if not model_path.exists():
        raise FileNotFoundError(f"robot model path does not exist: {model_path}")

    if model_path.is_dir():
        description_path = model_path / "robot_description.json"
        explicit_urdf = None
    elif model_path.suffix.lower() == ".urdf":
        description_path = model_path.parent / "robot_description.json"
        explicit_urdf = model_path
    elif model_path.suffix.lower() == ".json":
        description_path = model_path
        explicit_urdf = None
    else:
        raise ValueError(
            "ROBOT_MODEL_PATH must be a robot directory, a .urdf file, "
            f"or robot_description.json: {model_path}"
        )

    if not description_path.is_file():
        raise FileNotFoundError(
            f"robot description not found next to model: {description_path}"
        )

    urdf_candidates: List[Path] = []
    if explicit_urdf is not None:
        urdf_candidates.append(explicit_urdf)
    else:
        # Prefer the two file names used by batch and standard robot assets.
        urdf_candidates.extend([
            description_path.parent / "robot.urdf",
            description_path.parent / "generated_robot.urdf",
        ])
        try:
            description = load_description(description_path)
            declared = description.get("urdf_path")
            if declared:
                declared_path = Path(str(declared)).expanduser()
                if declared_path.is_absolute():
                    urdf_candidates.append(declared_path)
                else:
                    urdf_candidates.extend([
                        REPO_ROOT / declared_path,
                        description_path.parent / declared_path,
                        description_path.parent / declared_path.name,
                    ])
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(
                f"cannot read robot description {description_path}: {exc}"
            ) from exc
        urdf_candidates.extend(sorted(description_path.parent.glob("*.urdf")))

    urdf_path = next(
        (candidate.resolve() for candidate in urdf_candidates if candidate.is_file()),
        None,
    )
    if urdf_path is None:
        checked = ", ".join(str(path) for path in dict.fromkeys(urdf_candidates))
        raise FileNotFoundError(
            f"no URDF found for {description_path}; checked: {checked}"
        )
    return description_path.resolve(), urdf_path


def resolve_asset_paths(description_path: Path, urdf_path: Path) -> tuple[Path, Path]:
    if urdf_path.exists() and description_path.exists():
        return description_path, urdf_path

    assets_root = Path(ASSET_DIR_NAME)
    # 1) Prefer latest variant URDF if present.
    variants_dir = assets_root / DEFAULT_VARIANTS_DIR
    if variants_dir.exists():
        candidates = list(variants_dir.glob(f"*/{DEFAULT_VARIANT_URDF}"))
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            candidate_urdf = candidates[0]
            fallback_desc = assets_root / "robot_description.json"
            if fallback_desc.exists():
                return fallback_desc, candidate_urdf

    # 2) Fall back to the standard hexapod assets if present.
    standard_desc = assets_root / STANDARD_SUBDIR / "robot_description.json"
    standard_urdf = assets_root / STANDARD_SUBDIR / "generated_robot.urdf"
    if standard_desc.exists() and standard_urdf.exists():
        return standard_desc, standard_urdf

    return description_path, urdf_path


# ---------------------------------------------------------------------------
# Joint helpers
# ---------------------------------------------------------------------------

def ratio_to_joint(lower: float, upper: float, ratio: float) -> float:
    return float(lower + max(0.0, min(1.0, ratio)) * (upper - lower))


def resolve_joint_triplets(
    gym, env, actor, description: dict,
) -> Dict[int, dict]:
    dof_props = gym.get_actor_dof_properties(env, actor)
    lower = np.asarray(dof_props["lower"], dtype=np.float32)
    upper = np.asarray(dof_props["upper"], dtype=np.float32)
    mids = 0.5 * (
        np.where(np.isfinite(lower), lower, -0.5)
        + np.where(np.isfinite(upper), upper, 0.5)
    )
    names = gym.get_actor_dof_names(env, actor)
    name_to_idx = {n: i for i, n in enumerate(names)}

    triplets: Dict[int, dict] = {}
    for leg_id in range(int(description.get("num_legs", 0))):
        lift_n = f"leg_{leg_id}_lift"
        swing_n = f"leg_{leg_id}_swing"
        drop_n = f"leg_{leg_id}_drop"
        if lift_n not in name_to_idx or swing_n not in name_to_idx or drop_n not in name_to_idx:
            continue
        triplets[leg_id] = {
            "lift_idx": name_to_idx[lift_n],
            "swing_idx": name_to_idx[swing_n],
            "drop_idx": name_to_idx[drop_n],
            "lift_lower": float(lower[name_to_idx[lift_n]]),
            "lift_upper": float(upper[name_to_idx[lift_n]]),
            "swing_lower": float(lower[name_to_idx[swing_n]]),
            "swing_upper": float(upper[name_to_idx[swing_n]]),
            "drop_lower": float(lower[name_to_idx[drop_n]]),
            "drop_upper": float(upper[name_to_idx[drop_n]]),
            "swing_mid": float(mids[name_to_idx[swing_n]]),
        }
    return triplets


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def quintic_smoothstep(edge0: float, edge1: float, x: float) -> float:
    """C²-continuous smoothstep. Both 1st and 2nd derivatives vanish at edges.

    Uses quintic polynomial: t³(10 − 15t + 6t²) instead of Hermite t²(3 − 2t),
    which only guarantees C¹ (velocity) continuity.  The quintic form eliminates
    acceleration discontinuities → no joint jerk spikes at stance↔swing boundaries.
    """
    t = max(0.0, min(1.0, (x - edge0) / max(edge1 - edge0, 1e-9)))
    return t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)

# Keep the old Hermite name as an alias for the quintic — all call sites below
# now use the C²-continuous version.
smoothstep = quintic_smoothstep


def _swing_lift_profile(swing_progress: float) -> float:
    """Bell-shaped lift profile: 0 at stance boundaries, peaks at mid-swing.

    swing_progress ∈ [0, 1] — 0/1 = stance, 0.5 = peak swing.
    Uses sin(π·s) which has zero 1st-derivative at both ends (soft lift-off / touch-down).
    """
    if swing_progress <= 0.0 or swing_progress >= 1.0:
        return 0.0
    return float(math.sin(math.pi * swing_progress))


def gait_wave(phase_rad: float, duty_factor: float) -> tuple[float, float]:
    """Backward-compatible wrapper around the shared periodic scheduler."""
    state = phase_state(phase_rad, duty_factor)
    return state.fore_aft, state.lift


def quat_to_euler(w: float, x: float, y: float, z: float) -> tuple:
    """Convert quaternion (w,x,y,z) to roll, pitch, yaw in radians."""
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)
    return roll, pitch, yaw


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def draw_arrow(gym, viewer, env, gymapi, origin: List[float], direction: List[float],
               length: float, z: float, rgb: List[float]) -> None:
    ox, oy = float(origin[0]), float(origin[1])
    fx, fy = float(direction[0]), float(direction[1])
    norm = math.hypot(fx, fy)
    if norm < 1e-9:
        return
    fx, fy = fx / norm, fy / norm

    ex, ey = ox + fx * length, oy + fy * length
    head = length * 0.20
    spread = 0.45
    px, py = -fy, fx
    left = (ex - fx * head + px * head * spread, ey - fy * head + py * head * spread)
    right = (ex - fx * head - px * head * spread, ey - fy * head - py * head * spread)

    verts = np.array([
        [ox, oy, z, ex, ey, z],
        [ex, ey, z, left[0], left[1], z],
        [ex, ey, z, right[0], right[1], z],
    ], dtype=np.float32)
    colors = np.array([rgb, rgb, rgb], dtype=np.float32)
    gym.add_lines(viewer, env, 3, verts, colors)


def draw_cross(gym, viewer, env, gymapi, cx: float, cy: float,
               size: float, z: float, rgb: List[float]) -> None:
    verts = np.array([
        [cx - size, cy, z, cx + size, cy, z],
        [cx, cy - size, z, cx, cy + size, z],
    ], dtype=np.float32)
    colors = np.array([rgb, rgb], dtype=np.float32)
    gym.add_lines(viewer, env, 2, verts, colors)


def draw_trail(gym, viewer, env, points: List[List[float]],
               z: float = 0.018, rgb: Optional[List[float]] = None,
               max_segments: int = 600) -> None:
    """Draw a decimated world-frame body trail so motion remains obvious."""
    if len(points) < 2:
        return
    color = rgb or [0.85, 0.05, 0.85]
    stride = max(1, math.ceil((len(points) - 1) / max_segments))
    sampled = points[::stride]
    if sampled[-1] is not points[-1]:
        sampled.append(points[-1])
    verts = np.array([
        [a[0], a[1], z, b[0], b[1], z]
        for a, b in zip(sampled[:-1], sampled[1:])
    ], dtype=np.float32)
    colors = np.tile(np.asarray(color, dtype=np.float32), (len(verts), 1))
    gym.add_lines(viewer, env, len(verts), verts, colors)


def get_body_xy(gym, env, actor, gymapi) -> List[float]:
    states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)
    if states is None or len(states) == 0:
        return [0.0, 0.0]
    p = states["pose"]["p"][0]
    return [float(p["x"]), float(p["y"])]


def get_body_z(gym, env, actor, gymapi) -> float:
    """Return Z of the actor's root body."""
    states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)
    if states is None or len(states) == 0:
        return 0.0
    return float(states["pose"]["p"][0]["z"])


def get_body_attitude(gym, env, actor, gymapi) -> tuple:
    """Return (roll, pitch, yaw) of the actor's root body in radians."""
    states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)
    if states is None or len(states) == 0:
        return (0.0, 0.0, 0.0)
    r = states["pose"]["r"][0]
    return quat_to_euler(float(r["w"]), float(r["x"]), float(r["y"]), float(r["z"]))


def get_body_yaw_rate(gym, env, actor, gymapi) -> float:
    """Return Z angular velocity (yaw rate, rad/s) of the actor's root body."""
    states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)
    if states is None or len(states) == 0:
        return 0.0
    try:
        return float(states["vel"]["angular"][0]["z"])
    except (KeyError, IndexError, TypeError):
        return 0.0


def apply_online_yaw_correction(
    per_leg_amps: Dict[str, float],
    plan: dict,
    yaw_error_rad: float,
    yaw_error_integral: float = 0.0,
    kp: float = 1.0,
    ki: float = 0.3,
) -> Tuple[Dict[str, float], float]:
    """Adjust per-leg stride amplitudes to counteract body yaw error.

    Uses the physics-based lever-arm model from the gait plan.
    yaw_error_rad > 0 (body yawed CCW from planned) → need CW torque → reduce
    right-side amplitude, increase left-side.

    Returns (corrected_amplitudes, updated_integral).
    """
    yaw_levers = {
        int(k): float(v)
        for k, v in plan.get("yaw_balance", {}).get("yaw_levers", {}).items()
    }
    if not yaw_levers:
        return dict(per_leg_amps), yaw_error_integral

    max_lever = max(abs(v) for v in yaw_levers.values())
    if max_lever < 1e-9:
        return dict(per_leg_amps), yaw_error_integral

    # PI control: yaw_signal = Kp * error + Ki * integral(error)
    # Saturate yaw_signal to [-1, 1] to stay within correction range.
    # Threshold of 0.3 rad (~17°) for full correction.
    yaw_p = float(np.clip(yaw_error_rad / 0.3, -1.0, 1.0))
    yaw_i = float(np.clip(yaw_error_integral / 0.3, -1.0, 1.0))
    yaw_signal = float(np.clip(kp * yaw_p + ki * yaw_i, -1.0, 1.0))

    corrected: Dict[str, float] = {}
    for lid_str, amp in per_leg_amps.items():
        lever = yaw_levers.get(int(lid_str), 0.0)
        s = 1.0 - yaw_signal * lever / max_lever
        s = float(np.clip(s, 0.40, 1.60))
        corrected[lid_str] = float(np.clip(float(amp) * s, 0.15, 0.85))

    return corrected, yaw_error_integral


# ---------------------------------------------------------------------------
# Gait control
# ---------------------------------------------------------------------------

def foot_xy_map(description: dict) -> Dict[int, np.ndarray]:
    mapping: Dict[int, np.ndarray] = {}
    for link in description.get("links", []):
        if link.get("role") != "foot" or link.get("leg_id") is None:
            continue
        origin = np.asarray(link.get("default_world_origin", [0.0, 0.0, 0.0]), dtype=float)
        mapping[int(link["leg_id"])] = origin[:2]
    return mapping


def gait_stance_weights(
    leg_ids: List[int], gait_plan: dict, sim_time: float, gait_freq: float,
    phase_offsets: Dict[int, float], duty_factors: Dict[int, float],
) -> Dict[int, float]:
    """Return 1 for grounded legs and 0 at maximum swing clearance."""
    groups = gait_plan.get("topology", {}).get("groups", {})
    group_c = set(groups.get("group_c", []))
    phase = 2.0 * math.pi * max(gait_freq, 0.02) * sim_time
    weights: Dict[int, float] = {}
    for leg_id in leg_ids:
        if leg_id in group_c:
            weights[leg_id] = 1.0
            continue
        state = leg_phase_state(phase, leg_id, phase_offsets, duty_factors)
        weights[leg_id] = 1.0 - state.lift
    return weights


def add_leveling_bias(
    targets: np.ndarray, triplets: Dict[int, dict],
    level_bias: Dict[int, float], stance_weights: Dict[int, float],
    lower: np.ndarray, upper: np.ndarray,
) -> np.ndarray:
    """Apply gentle roll/pitch correction mainly through stance legs."""
    adjusted = targets.copy()
    for leg_id, joints in triplets.items():
        bias = level_bias.get(leg_id, 0.0) * stance_weights.get(leg_id, 1.0)
        adjusted[joints["lift_idx"]] += (
            0.25 * bias * (joints["lift_upper"] - joints["lift_lower"])
        )
        adjusted[joints["drop_idx"]] += (
            0.65 * bias * (joints["drop_upper"] - joints["drop_lower"])
        )
    return np.clip(adjusted, lower, upper)


def build_gait_targets(
    description: dict,
    gait_plan: dict,
    triplets: Dict[int, dict],
    defaults: np.ndarray,
    sim_time: float,
    gait_freq: float,
    swing_amp: float,
    stance_lift: float,
    swing_lift: float,
    stance_drop: float,
    swing_drop: float,
    *,
    per_leg_stride_amplitudes: Optional[Dict[str, float]] = None,
    touchdown_ramp: Optional[Dict[int, int]] = None,
    touchdown_ramp_steps: int = 25,
    gait_mode: str = "alternating",
    duty_factor: float = DEFAULT_DUTY_FACTOR,
    phase_offsets: Optional[Dict[int, float]] = None,
    per_leg_duty_factors: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    """
    构造每步的关节目标位置。

    使用 quintic smoothstep + 钟形 lift 曲线确保 C² 连续足端轨迹。

    per_leg_stride_amplitudes : {str(leg_id): amplitude} — CPG 每腿步幅缩放因子
    touchdown_ramp             : {leg_id: steps_since_touchdown} — 落地淡入状态字典
                                 由调用方维护，本函数会就地更新脚从摆动→支撑时的计数器
    touchdown_ramp_steps       : 淡入动作持续的仿真步数（默认 25 步 ≈ 0.42s）
    """
    targets = defaults.copy()
    phase = 2.0 * np.pi * max(gait_freq, 0.02) * sim_time

    group_a = list(gait_plan.get("topology", {}).get("groups", {}).get("group_a", []))
    group_b = list(gait_plan.get("topology", {}).get("groups", {}).get("group_b", []))
    group_c = list(gait_plan.get("topology", {}).get("groups", {}).get("group_c", []))
    active_swing_legs = set(group_a) | set(group_b)   # group_c stays in stance

    forward_axis = np.asarray(gait_plan.get("final_forward_axis", [1.0, 0.0]), dtype=float)
    forward_axis = forward_axis / max(float(np.linalg.norm(forward_axis)), 1e-9)

    fmap = foot_xy_map(description)

    # Lateral axis: 90° CCW from forward axis (points to the "left" of the robot)
    lateral_axis = np.array([-forward_axis[1], forward_axis[0]], dtype=float)

    amp_map: Dict[str, float] = per_leg_stride_amplitudes or {}
    resolved_offsets = (
        phase_offsets if phase_offsets is not None
        else resolve_phase_offsets(gait_plan, active_swing_legs)
    )
    if per_leg_duty_factors is not None:
        resolved_duties = per_leg_duty_factors
    elif "duty_factor" in gait_plan.get("cpg", {}):
        resolved_duties, _ = resolve_duty_factors(gait_plan, active_swing_legs)
    else:
        resolved_duties, _ = resolve_duty_factors(
            {"cpg": {"duty_factor": duty_factor}}, active_swing_legs
        )

    for leg_id, joints in triplets.items():
        # ── Group-c passive legs: swing-only micro-push with no lift change ──────
        # Keeps feet on ground (adaptation.stability maintained) but oscillates swing joint
        # slightly to reduce static friction drag and contribute small forward push.
        if leg_id in group_c:
            # Pure stance: hold swing at mid, press feet firmly into ground for lateral support.
            targets[joints["lift_idx"]] = ratio_to_joint(
                joints["lift_lower"], joints["lift_upper"], stance_lift)
            targets[joints["drop_idx"]] = ratio_to_joint(
                joints["drop_lower"], joints["drop_upper"], stance_drop)
            targets[joints["swing_idx"]] = ratio_to_joint(
                joints["swing_lower"], joints["swing_upper"], 0.5)
            continue

        leg_state = leg_phase_state(
            phase, leg_id, resolved_offsets, resolved_duties,
        )
        swing_wave, swing_alpha = leg_state.fore_aft, leg_state.lift

        foot_xy_vec = fmap.get(leg_id, np.zeros(2, dtype=float))
        # Use lateral position (perpendicular to forward) to decide swing rotation sense.
        # +lateral (left side):  positive Z-rotation swings foot forward → dir_sign = +1
        # -lateral (right side): negative Z-rotation swings foot forward → dir_sign = -1
        lateral_pos = float(np.dot(foot_xy_vec, lateral_axis))
        dir_sign = 1.0 if lateral_pos > 0.0 else -1.0

        # ── CPG 每腿步幅解耦 ─────────────────────────────────────────────────
        leg_stride_scale = float(amp_map.get(str(leg_id), 1.0))
        effective_amp = swing_amp * leg_stride_scale

        # The lift target is above the stance target; drop stays close to its
        # neutral support angle.  This prevents the lower link folding under
        # the body and makes the return swing visibly leave the ground.
        lift_r = stance_lift + (swing_lift - stance_lift) * swing_alpha
        drop_r = stance_drop + (swing_drop - stance_drop) * swing_alpha
        swing_r = 0.5 + effective_amp * dir_sign * swing_wave

        # ── 触地淡入滤波（Touchdown ramp-up） ───────────────────────────────
        # A short contact ramp prevents an abrupt full-load transfer when the
        # returning tripod touches down.  It is used by the validated
        # step-cycle controller; the smooth sine executor does not need it.
        if touchdown_ramp is not None:
            is_swing = not leg_state.is_stance
            if not is_swing:
                # 支撑相：若之前处于摆动 → 开始淡入计数
                if leg_id not in touchdown_ramp:
                    # 首次进入支撑或已完成淡入 → 不在字典里，无操作
                    pass
                else:
                    # 正在淡入：计数递增，使用 quintic easing 消除 touchdown 冲击
                    touchdown_ramp[leg_id] += 1
                    ramp_raw = min(touchdown_ramp[leg_id] / touchdown_ramp_steps, 1.0)
                    if ramp_raw >= 1.0:
                        del touchdown_ramp[leg_id]  # 淡入完成
                    else:
                        # Quintic easing: C²-continuous ramp from 0→1 (zero accel at both ends)
                        ramp_progress = smoothstep(0.0, 1.0, ramp_raw)
                        default_lift = ratio_to_joint(joints["lift_lower"], joints["lift_upper"], 0.5)
                        default_drop = ratio_to_joint(joints["drop_lower"], joints["drop_upper"], 0.5)
                        default_swing = joints["swing_lower"] + 0.5 * (joints["swing_upper"] - joints["swing_lower"])
                        tgt_lift = ratio_to_joint(joints["lift_lower"], joints["lift_upper"], lift_r)
                        tgt_drop = ratio_to_joint(joints["drop_lower"], joints["drop_upper"], drop_r)
                        tgt_swing = ratio_to_joint(joints["swing_lower"], joints["swing_upper"], swing_r)
                        targets[joints["lift_idx"]]  = default_lift  + ramp_progress * (tgt_lift  - default_lift)
                        targets[joints["drop_idx"]]  = default_drop  + ramp_progress * (tgt_drop  - default_drop)
                        targets[joints["swing_idx"]] = default_swing + ramp_progress * (tgt_swing - default_swing)
                        continue
            else:
                # 摆动相：标记下一次进入支撑时需要淡入
                touchdown_ramp[leg_id] = 0

        targets[joints["lift_idx"]] = ratio_to_joint(
            joints["lift_lower"], joints["lift_upper"], lift_r,
        )
        targets[joints["drop_idx"]] = ratio_to_joint(
            joints["drop_lower"], joints["drop_upper"], drop_r,
        )
        targets[joints["swing_idx"]] = ratio_to_joint(
            joints["swing_lower"], joints["swing_upper"], swing_r,
        )

    return targets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", type=Path, default=ROBOT_MODEL_PATH,
        help=("Robot directory, URDF, or robot_description.json. The default "
              "is ROBOT_MODEL_PATH at the top of this file."),
    )
    p.add_argument("--description", type=Path,
                   help="Legacy override for robot_description.json.")
    p.add_argument("--urdf", type=Path,
                   help="Legacy override for the URDF path.")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--steps", type=int, default=2400)
    p.add_argument("--hold-steps", type=int, default=300,
                   help="Warm-up stabilisation steps with static standing posture before gait begins.")
    p.add_argument("--stand-only", action="store_true",
                   help="Viewer mode: hold static standing posture indefinitely (no gait).")
    p.add_argument("--compute-device-id", type=int, default=0)
    p.add_argument("--graphics-device-id", type=int, default=0)
    p.add_argument("--cpu-sim", action="store_true")
    p.add_argument("--gpu-pipeline", action="store_true",
                   help="Enable GPU rendering pipeline (disabled by default).")
    p.add_argument("--body-height", type=float, default=0.50)
    p.add_argument(
        "--gait-frequency", type=float,
        help="Override the morphology plan frequency (default: use plan value).",
    )
    p.add_argument(
        "--duty-factor", type=float,
        help="Override stance fraction of each cycle (default: use plan value).",
    )
    p.add_argument(
        "--gait-executor", choices=("smooth_sine", "step_cycle"),
        default=DEFAULT_GAIT_EXECUTOR,
        help=("Joint trajectory executor. step_cycle is the visually distinct "
              "default; smooth_sine is retained for comparison."),
    )
    p.add_argument("--swing-ratio-amplitude", type=float,
                   default=DEFAULT_SWING_RATIO_AMPLITUDE)
    p.add_argument("--swing-lift-ratio", type=float, default=0.78)
    p.add_argument("--stance-lift-ratio", type=float, default=0.05)
    p.add_argument("--swing-drop-ratio", type=float, default=0.38)
    p.add_argument("--stance-drop-ratio", type=float, default=0.90)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        if args.description is not None and args.urdf is not None:
            desc_path = _repo_relative_path(args.description)
            urdf_path = _repo_relative_path(args.urdf)
            if not desc_path.is_file():
                raise FileNotFoundError(f"robot description does not exist: {desc_path}")
            if not urdf_path.is_file():
                raise FileNotFoundError(f"URDF does not exist: {urdf_path}")
        elif args.description is not None:
            desc_path, urdf_path = resolve_robot_model(args.description)
        elif args.urdf is not None:
            desc_path, urdf_path = resolve_robot_model(args.urdf)
        else:
            desc_path, urdf_path = resolve_robot_model(args.model)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] Cannot resolve robot model: {exc}")
        return 2

    print(f"[Model] description = {desc_path}")
    print(f"[Model] URDF        = {urdf_path}")

    gymapi = load_gymapi()
    description = load_description(desc_path)
    gait_plan = compute_plan(description, {})

    # ---- print plan summary -------------------------------------------------
    print("[Plan] Adaptive gait plan summary")
    print(f"  support_center_xy          = {gait_plan['support_center_xy']}")
    print(f"  projected_com_xy           = {gait_plan['projected_com_xy']}")
    print(f"  initial_virtual_forward_axis = {gait_plan['initial_virtual_forward_axis']}")
    print(f"  final_forward_axis         = {gait_plan['final_forward_axis']}")
    print(f"  direction_scores           = +{gait_plan['direction_scores']['positive']:.4f} "
          f"/ -{gait_plan['direction_scores']['negative']:.4f}")
    print(f"  drive_resultant_xy         = {gait_plan['drive_resultant_xy']}")
    topo = gait_plan["topology"]
    cpg = gait_plan.get("cpg", {})
    gait_mode = str(cpg.get("mode", "alternating"))
    gait_frequency = float(
        args.gait_frequency
        if args.gait_frequency is not None
        else cpg.get("frequency_hz", 0.85)
    )
    execution_plan = dict(gait_plan)
    execution_cpg = dict(cpg)
    if args.duty_factor is not None:
        execution_cpg["duty_factor"] = args.duty_factor
    execution_plan["cpg"] = execution_cpg
    phase_offsets = resolve_phase_offsets(execution_plan, topo["groups"]["group_a"] + topo["groups"]["group_b"])
    per_leg_duty_factors, duty_diagnostics = resolve_duty_factors(
        execution_plan, topo["groups"]["group_a"] + topo["groups"]["group_b"],
    )
    duty_factor = float(
        next(iter(per_leg_duty_factors.values()), DEFAULT_DUTY_FACTOR)
    )
    print(f"  group_a ({len(topo['groups']['group_a'])} legs): {topo['groups']['group_a']}")
    print(f"  group_b ({len(topo['groups']['group_b'])} legs): {topo['groups']['group_b']}")
    if topo["inhibition_rules"]:
        for r in topo["inhibition_rules"]:
            print(f"  [INHIBIT] leg {r['leg_id']}: {r['reason']}")
    print(f"  cpg                        = {gait_mode}, "
          f"{gait_frequency:.2f} Hz, duty={duty_factor:.2f}")
    if args.gait_executor != "step_cycle":
        print("  [Phase] smooth_sine is deprecated; using unified duty-factor executor")
    print("  trajectory executor        = unified_phase")
    for message in duty_diagnostics:
        print(f"  [Phase] {message}")

    # ---- Isaac Gym setup ----------------------------------------------------
    gym = gymapi.acquire_gym()

    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    use_gpu = not args.cpu_sim
    sim_params.use_gpu_pipeline = bool(args.gpu_pipeline)
    sim_params.physx.use_gpu = use_gpu
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 2

    graphics_id = -1 if args.headless else args.graphics_device_id
    sim = gym.create_sim(args.compute_device_id, graphics_id, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        if use_gpu:
            sim_params.physx.use_gpu = False
            sim = gym.create_sim(0, graphics_id, gymapi.SIM_PHYSX, sim_params)
        if sim is None:
            print("[ERROR] Failed to create sim.")
            return 1

    viewer = None
    try:
        # Ground
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 1.8
        plane_params.dynamic_friction = 1.6
        plane_params.restitution = 0.0
        gym.add_ground(sim, plane_params)

        # Asset
        urdf_path = urdf_path.resolve()
        if not urdf_path.exists():
            print(f"[ERROR] URDF not found: {urdf_path}")
            return 1

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False
        asset_options.flip_visual_attachments = False
        # MUST be True: hip sphere meshes overlap the body box — see test_standard_gait.py comment.
        asset_options.collapse_fixed_joints = True

        asset_dir = str(urdf_path.parent)
        asset = gym.load_asset(sim, asset_dir, urdf_path.name, asset_options)
        if asset is None:
            print(f"[ERROR] Failed to load asset: {urdf_path}")
            return 1

        # Environment & actor
        env = gym.create_env(sim, gymapi.Vec3(-3.0, -3.0, 0.0),
                             gymapi.Vec3(3.0, 3.0, 2.0), 1)
        if env is None:
            print("[ERROR] Failed to create env.")
            return 1

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, float(args.body_height))
        actor = gym.create_actor(env, asset, pose, "test_gait_robot", 0, 1)
        if actor < 0:
            print("[ERROR] Failed to create actor.")
            return 1

        # ---- Per-joint DOF configuration (differentiated by joint role) -----
        dof_props = gym.get_actor_dof_properties(env, actor)
        dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        dof_names = gym.get_asset_dof_names(asset)
        dof_count = len(dof_names)

        # PD gains — keep original stiffness for weight support; smoothness comes from
        # the C²-continuous trajectory (quintic smoothstep + bell lift profile), not
        # from looser gains.
        dof_props["stiffness"].fill(200.0)
        dof_props["damping"].fill(20.0)
        if "effort" in dof_props.dtype.names:
            dof_props["effort"].fill(1000.0)
        if "armature" in dof_props.dtype.names:
            dof_props["armature"].fill(0.01)

        for idx, name in enumerate(dof_names):
            if "_swing" in name:
                dof_props["stiffness"][idx] = 100.0
                dof_props["damping"][idx] = 10.0
                if "effort" in dof_props.dtype.names:
                    dof_props["effort"][idx] = 1000.0
            elif "_drop" in name:
                dof_props["stiffness"][idx] = 250.0
                dof_props["damping"][idx] = 25.0
                if "effort" in dof_props.dtype.names:
                    dof_props["effort"][idx] = 200.0
            elif "_lift" in name:
                dof_props["stiffness"][idx] = 200.0
                dof_props["damping"][idx] = 20.0
                if "effort" in dof_props.dtype.names:
                    dof_props["effort"][idx] = 1000.0

        gym.set_actor_dof_properties(env, actor, dof_props)

        lower = np.asarray(dof_props["lower"], dtype=np.float32)
        upper = np.asarray(dof_props["upper"], dtype=np.float32)
        default_targets = 0.5 * (
            np.where(np.isfinite(lower), lower, -0.5)
            + np.where(np.isfinite(upper), upper, 0.5)
        ).astype(np.float32)

        triplets = resolve_joint_triplets(gym, env, actor, description)
        fmap = foot_xy_map(description)

        # group_c passive legs keep default friction — they serve as lateral anchors.
        group_c_topo = topo.get("groups", {}).get("group_c", [])
        if group_c_topo:
            print(f"[Friction] group_c legs (lateral anchors, default friction): {group_c_topo}")

        # Heuristic: if description foot z-positions (at joint angles=0) are already
        # at ~body_height depth, use neutral joint angles (angle=0) as stand targets.
        # Otherwise use the standard hexapod ratios (lift≈min, drop≈max).
        foot_z_vals = [
            float(lnk["default_world_origin"][2])
            for lnk in description.get("links", [])
            if lnk.get("role") == "foot" and lnk.get("leg_id") is not None
        ]
        mean_foot_z = float(np.mean(foot_z_vals)) if foot_z_vals else -0.35
        # If neutral pose already lands feet within 0.12 m of ground → use angle=0
        feet_at_ground = abs(mean_foot_z + args.body_height) < 0.12
        if feet_at_ground:
            print(f"[Stand] Neutral-joint stand pose (foot_z≈{mean_foot_z:.3f},"
                  f" body_h={args.body_height:.2f})")

        stand_targets = default_targets.copy()
        for leg_id, joints in triplets.items():
            if feet_at_ground:
                # angle=0 for lift and drop so feet stay at neutral ground position
                lift_r = max(0.0, min(1.0,
                    (0.0 - joints["lift_lower"]) / max(joints["lift_upper"] - joints["lift_lower"], 1e-9)))
                drop_r = max(0.0, min(1.0,
                    (0.0 - joints["drop_lower"]) / max(joints["drop_upper"] - joints["drop_lower"], 1e-9)))
            else:
                lift_r = 0.05    # standard hexapod: lift near minimum to reach down
                drop_r = 0.995   # standard hexapod: drop near maximum to reach down
            stand_targets[joints["lift_idx"]] = ratio_to_joint(
                joints["lift_lower"], joints["lift_upper"], lift_r,
            )
            stand_targets[joints["drop_idx"]] = ratio_to_joint(
                joints["drop_lower"], joints["drop_upper"], drop_r,
            )
            stand_targets[joints["swing_idx"]] = ratio_to_joint(
                joints["swing_lower"], joints["swing_upper"], 0.50,
            )

        finite_lower = np.where(np.isfinite(lower), lower, -1e9)
        finite_upper = np.where(np.isfinite(upper), upper, 1e9)
        stand_targets = np.clip(stand_targets, finite_lower, finite_upper)

        dof_states = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
        dof_states["pos"] = stand_targets
        dof_states["vel"].fill(0.0)
        gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)
        # Preserve an exactly level articulated pose.  Warm-up contact forces
        # can otherwise leave a small initial yaw/roll that the gait amplifies.
        initial_rb_states = np.copy(
            gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL),
        )

        print(f"[OK] Asset loaded: {dof_count} DOFs, "
              f"{gym.get_asset_rigid_body_count(asset)} bodies")
        print(f"[OK] Standing at body height {args.body_height:.2f} m")

        # ---- Warm-up: static hold-stabilisation ------------------------------
        hold_steps = max(args.hold_steps, 0)
        if hold_steps > 0:
            print(f"[HOLD] Stabilising for {hold_steps} steps ...")
            # Sag controller: gently push drop joints down until feet reach ground.
            # Active for ALL robots including feet_at_ground ones — the detection
            # threshold (0.12m) may still leave feet 0.028–0.084m above ground.
            for _ in range(hold_steps):
                dof_states_pos = gym.get_actor_dof_states(
                    env, actor, gymapi.STATE_POS,
                )
                joint_pos = np.asarray(dof_states_pos["pos"], dtype=np.float32)
                for idx, name in enumerate(dof_names):
                    sag = stand_targets[idx] - joint_pos[idx]
                    if "_drop" in name and sag > 0.004:
                        stand_targets[idx] = min(
                            finite_upper[idx],
                            stand_targets[idx] + min(0.012, 0.22 * sag),
                        )
                    elif "_lift" in name and sag > 0.004:
                        stand_targets[idx] = max(
                            finite_lower[idx],
                            stand_targets[idx] - min(0.008, 0.16 * sag),
                        )
                stand_targets = np.clip(stand_targets, finite_lower, finite_upper)
                gym.set_actor_dof_position_targets(env, actor, stand_targets)
                gym.simulate(sim)
                gym.fetch_results(sim, True)
            print("[HOLD] Stabilisation complete.")

        # Return to the level origin after sag calibration, then allow the
        # calibrated stance to settle with per-leg roll/pitch compensation.
        dof_states = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
        dof_states["pos"] = stand_targets
        dof_states["vel"].fill(0.0)
        gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)
        rb_reset = np.copy(initial_rb_states)
        rb_reset["pose"]["p"][0]["x"] = 0.0
        rb_reset["pose"]["p"][0]["y"] = 0.0
        rb_reset["pose"]["p"][0]["z"] = float(args.body_height)
        rb_reset["pose"]["r"][0]["x"] = 0.0
        rb_reset["pose"]["r"][0]["y"] = 0.0
        rb_reset["pose"]["r"][0]["z"] = 0.0
        rb_reset["pose"]["r"][0]["w"] = 1.0
        rb_reset["vel"]["linear"].fill(0.0)
        rb_reset["vel"]["angular"].fill(0.0)
        gym.set_actor_rigid_body_states(env, actor, rb_reset, gymapi.STATE_ALL)

        x_extent = max((abs(float(v[0])) for v in fmap.values()), default=1.0)
        y_extent = max((abs(float(v[1])) for v in fmap.values()), default=1.0)
        level_bias = {leg_id: 0.0 for leg_id in triplets}
        for _ in range(90):
            level_targets = stand_targets.copy()
            for leg_id, joints in triplets.items():
                bias = level_bias.get(leg_id, 0.0)
                level_targets[joints["lift_idx"]] += (
                    0.25 * bias * (joints["lift_upper"] - joints["lift_lower"])
                )
                level_targets[joints["drop_idx"]] += (
                    0.65 * bias * (joints["drop_upper"] - joints["drop_lower"])
                )
            level_targets = np.clip(level_targets, finite_lower, finite_upper)
            gym.set_actor_dof_position_targets(env, actor, level_targets)
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            roll, pitch, _ = get_body_attitude(gym, env, actor, gymapi)
            for leg_id, foot in fmap.items():
                tilt = (roll * float(foot[1]) / y_extent
                        - pitch * float(foot[0]) / x_extent)
                raw = float(np.clip(0.70 * tilt, -0.18, 0.18))
                level_bias[leg_id] = 0.85 * level_bias.get(leg_id, 0.0) + 0.15 * raw
        print("[LEVEL] Pose reset and 90-step active levelling complete.")

        print(f"[OK] Gait: topology={gait_mode}, executor=unified_phase, "
              f"{gait_frequency:.2f} Hz, duty={duty_factor:.2f}")

        # ---- Simulation loop -------------------------------------------------
        forward_axis = gait_plan.get("final_forward_axis", [1.0, 0.0])
        lateral_axis = np.array([-forward_axis[1], forward_axis[0]], dtype=float)
        group_a = topo["groups"]["group_a"]
        group_b = topo["groups"]["group_b"]

        # ── 为形态自适应地计算 stance/swing lift/drop 比率 ─────────────────
        if feet_at_ground and triplets:
            fj = next(iter(triplets.values()))
            lift_range = max(fj["lift_upper"] - fj["lift_lower"], 1e-9)
            drop_range = max(fj["drop_upper"] - fj["drop_lower"], 1e-9)
            # Match the batch-validated controller: neutral joints support the
            # body and a positive lift change raises the returning leg.  The
            # previous -0.25 rad target moved the foot down, while drop=0.90
            # folded the lower link underneath the body.
            _stance_lift = max(0.0, min(1.0, (0.0 - fj["lift_lower"]) / lift_range))
            _swing_lift = max(0.0, min(
                1.0, (SWING_LIFT_DELTA_RAD - fj["lift_lower"]) / lift_range,
            ))
            _stance_drop = max(0.0, min(1.0, (0.0 - fj["drop_lower"]) / drop_range))
            _swing_drop = _stance_drop
            print(f"[Stand] Morphology-adapted lift/drop ratios: "
                  f"stance_lift={_stance_lift:.3f}, swing_lift={_swing_lift:.3f}, "
                  f"stance_drop={_stance_drop:.3f}, swing_drop={_swing_drop:.3f}")
            print(f"[Clearance] lift joint: 0.000 -> "
                  f"{SWING_LIFT_DELTA_RAD:+.3f} rad during swing")
        else:
            _stance_lift = args.stance_lift_ratio
            _swing_lift = max(args.swing_lift_ratio, _stance_lift + 0.08)
            _stance_drop = args.stance_drop_ratio
            _swing_drop = args.swing_drop_ratio

        # CPG 每腿步幅缩放因子（来自 adaptation.gait 的 per_leg_stride_amplitudes）
        per_leg_stride_amplitudes: Dict[str, float] = {
            str(k): float(v)
            for k, v in topo.get("per_leg_stride_amplitudes", {}).items()
        }
        _base_amplitudes = dict(per_leg_stride_amplitudes)  # reference for online correction
        if per_leg_stride_amplitudes:
            print(f"[OK] Per-leg stride amplitudes: { {int(k): round(v, 3) for k, v in per_leg_stride_amplitudes.items()} }")
        else:
            print("[INFO] No per_leg_stride_amplitudes found; using uniform swing_amp.")

        # ── Body height compensator (slow integral term) ────────────────────
        _body_height_target = args.body_height
        _height_integral = 0.0
        _height_kI_lift = 0.25  # integral gain for lift ratio
        _height_kI_drop = 0.40  # integral gain for drop ratio (more mechanical advantage)
        _stance_lift_effective = _stance_lift
        _stance_drop_effective = _stance_drop
        _lift_ratio_min = 0.0 if feet_at_ground else 0.05
        # Once the body is under load the drop joint needs moderate extension
        # to preserve ground contact; 0.55 is the validated controller bound.
        _drop_ratio_min = 0.55
        # ── Yaw PI controller state ─────────────────────────────────────────
        _yaw_error_integral = 0.0

        # 触地淡入状态：{leg_id: steps_since_touchdown}，在摆动→支撑切换时初始化为 0
        touchdown_ramp: Dict[int, int] = {}

        com_trail: List[List[float]] = []   # [x, y, body_yaw]
        yaw_history: List[float] = []        # full body-yaw time series
        correction_log: List[dict] = []      # record of correction activations
        sim_time = 0.0
        dt = sim_params.dt
        steps_per_cycle = max(1, int(round(1.0 / (gait_frequency * dt))))
        # planned heading from gait plan (constant)
        # A morphology may translate sideways; do not force its chassis to
        # rotate onto the translation axis.
        planned_yaw_rad = float(gait_plan.get("body_yaw_target", 0.0))

        if not args.headless:
            viewer = gym.create_viewer(sim, gymapi.CameraProperties())
            if viewer is None:
                print("[ERROR] Failed to create viewer.")
                return 1
            cam_pos = gymapi.Vec3(2.4, 1.8, 1.4)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.35)
            gym.viewer_camera_look_at(viewer, env, cam_pos, cam_target)

            heading_deg = math.degrees(math.atan2(forward_axis[1], forward_axis[0]))
            stand_only = getattr(args, "stand_only", False)
            mode_label = "[STAND-ONLY]" if stand_only else "[Gait]"
            print(f"[OK] Viewer open — heading = {heading_deg:.1f}° — {mode_label} — close window to exit.")
            frame_count = 0
            while not gym.query_viewer_has_closed(viewer):
                gym.clear_lines(viewer)
                frame_count += 1

                if stand_only:
                    # ── Static standing: just hold the stand targets ─────
                    gym.set_actor_dof_position_targets(env, actor, stand_targets)
                else:
                    targets = build_gait_targets(
                        description, gait_plan, triplets, stand_targets.copy(), sim_time,
                        gait_frequency, args.swing_ratio_amplitude,
                        _stance_lift_effective, _swing_lift,
                        _stance_drop_effective, _swing_drop,
                        per_leg_stride_amplitudes=per_leg_stride_amplitudes,
                        touchdown_ramp=touchdown_ramp,
                        gait_mode="unified_phase",
                        duty_factor=duty_factor,
                        phase_offsets=phase_offsets,
                        per_leg_duty_factors=per_leg_duty_factors,
                    )
                stance_weights = gait_stance_weights(
                    list(triplets), gait_plan, sim_time, gait_frequency,
                    phase_offsets, per_leg_duty_factors,
                )
                targets = add_leveling_bias(
                    targets if not stand_only else stand_targets,
                    triplets, level_bias, stance_weights,
                    finite_lower, finite_upper,
                )
                gym.set_actor_dof_position_targets(env, actor, targets)
                gym.simulate(sim)
                gym.fetch_results(sim, True)
                gym.step_graphics(sim)

                body_xy = get_body_xy(gym, env, actor, gymapi)
                _, _, body_yaw = get_body_attitude(gym, env, actor, gymapi)
                com_trail.append([body_xy[0], body_xy[1], body_yaw])
                yaw_history.append(body_yaw)
                body_roll, body_pitch, _ = get_body_attitude(gym, env, actor, gymapi)
                for leg_id, foot in fmap.items():
                    tilt = (body_roll * float(foot[1]) / y_extent
                            - body_pitch * float(foot[0]) / x_extent)
                    raw = float(np.clip(0.70 * tilt, -0.18, 0.18))
                    level_bias[leg_id] = (
                        0.90 * level_bias.get(leg_id, 0.0) + 0.10 * raw
                    )

                # ── Body height compensation (every 120 steps) ─────────────
                if frame_count % 120 == 0:
                    body_z = get_body_z(gym, env, actor, gymapi)
                    z_error = _body_height_target - body_z
                    _height_integral += 0.03 * z_error  # faster accumulation
                    _height_integral = float(np.clip(_height_integral, -0.15, 0.35))
                    _stance_lift_effective = _stance_lift - _height_kI_lift * _height_integral
                    _stance_lift_effective = float(np.clip(
                        _stance_lift_effective, _lift_ratio_min, 0.95,
                    ))
                    _stance_drop_effective = _stance_drop - _height_kI_drop * _height_integral
                    _stance_drop_effective = float(np.clip(
                        _stance_drop_effective, _drop_ratio_min, 1.0,
                    ))

                if frame_count % steps_per_cycle == 0 and len(com_trail) > 1:
                    cycle_start = np.asarray(
                        com_trail[max(0, len(com_trail) - steps_per_cycle)][:2], dtype=float,
                    )
                    cycle_delta = np.asarray(body_xy, dtype=float) - cycle_start
                    cycle_fwd = float(np.dot(cycle_delta, forward_axis))
                    cycle_lat = float(np.dot(cycle_delta, lateral_axis))
                    cycle_z = get_body_z(gym, env, actor, gymapi)
                    print(f"[Cycle {frame_count // steps_per_cycle:02d}] "
                          f"forward={cycle_fwd:+.4f} m, lateral={cycle_lat:+.4f} m, "
                          f"body_z={cycle_z:.3f} m")

                # ── Online yaw correction (every 60 steps ≈ 1 s) ─────────
                # Primary: proportional feedback on absolute body yaw error.
                # Secondary: local drift in body frame for fine-tuning.
                if frame_count % 60 == 0:
                    # ── Absolute yaw error (wrapped to [-π, π]) ──────────────
                    yaw_error = body_yaw - planned_yaw_rad
                    yaw_error = float(np.arctan2(np.sin(yaw_error), np.cos(yaw_error)))
                    yaw_err_deg = math.degrees(yaw_error)

                    # ── Local drift in body frame ────────────────────────────
                    if len(com_trail) >= 60:
                        recent = np.array(com_trail[-60:], dtype=float)
                        disp = recent[-1, :2] - recent[0, :2]
                        avg_yaw = float(np.mean(recent[:, 2]))
                        cos_y = math.cos(avg_yaw)
                        sin_y = math.sin(avg_yaw)
                        body_fwd = np.array([cos_y, sin_y], dtype=float)
                        body_lat = np.array([-sin_y, cos_y], dtype=float)
                        fwd_disp = float(np.dot(disp, body_fwd))
                        lat_disp = float(np.dot(disp, body_lat))
                        drift_ratio = abs(lat_disp) / max(abs(fwd_disp), 0.01)
                    else:
                        lat_disp = 0.0
                        fwd_disp = 0.0
                        drift_ratio = 0.0

                    world_delta = (
                        np.asarray(com_trail[-1][:2], dtype=float)
                        - np.asarray(com_trail[0][:2], dtype=float)
                    )
                    cross_track_error = float(np.dot(world_delta, lateral_axis))
                    cross_track_velocity = 0.0
                    if len(com_trail) >= 61:
                        cross_track_velocity = float(np.dot(
                            np.asarray(com_trail[-1][:2], dtype=float)
                            - np.asarray(com_trail[-61][:2], dtype=float),
                            lateral_axis,
                        ))
                    desired_yaw_offset = float(np.clip(
                        -2.0 * cross_track_error - 0.8 * cross_track_velocity,
                        -0.40, 0.40,
                    ))
                    control_error = yaw_error - desired_yaw_offset

                    # ── Combined correction (PI on yaw error) ──────────────
                    # Heading plus cross-track feedback: asymmetric robots may
                    # keep a small body yaw yet steadily leave the planned line.
                    if abs(control_error) > 0.0087 or drift_ratio > 0.05:
                        _yaw_error_integral += 0.02 * control_error
                        _yaw_error_integral = float(np.clip(_yaw_error_integral, -0.5, 0.5))
                        per_leg_stride_amplitudes, _yaw_error_integral = apply_online_yaw_correction(
                            _base_amplitudes, gait_plan, control_error,
                            yaw_error_integral=_yaw_error_integral, kp=1.0, ki=0.3)
                        correction_log.append({
                            "frame": frame_count,
                            "yaw_error_deg": yaw_err_deg,
                            "yaw_integral": _yaw_error_integral,
                            "drift_ratio": drift_ratio,
                            "lat_disp": lat_disp,
                            "fwd_disp": fwd_disp,
                            "cross_track_error": cross_track_error,
                        })

                    if frame_count % 1200 == 0:
                        world_lat_disp = float(np.dot(disp, lateral_axis))
                        world_fwd_disp = float(np.dot(disp, forward_axis))
                        print(f"[YawFB] f={frame_count}  "
                              f"bodyYaw={math.degrees(body_yaw):+.2f}°  "
                              f"yawErr={yaw_err_deg:+.2f}°  "
                              f"drift(body) lat={lat_disp:+.3f}m fwd={fwd_disp:+.3f}m ratio={drift_ratio:.3f}  "
                              f"drift(world) lat={world_lat_disp:+.3f}m fwd={world_fwd_disp:+.3f}m")

                # 1. Forward-direction arrow (orange, 1.5 m)
                draw_arrow(gym, viewer, env, gymapi, body_xy, forward_axis,
                           length=1.5, z=0.012, rgb=[1.0, 0.45, 0.0])

                # Magenta: actual world-frame body trajectory.  Blue scale
                # marks stay fixed on the ground every 0.25 m, making motion
                # over a single gait cycle perceptible in the viewer.
                draw_trail(gym, viewer, env, com_trail)
                trail_origin = np.asarray(com_trail[0][:2], dtype=float)
                for tick_i in range(-4, 25):
                    tick_xy = trail_origin + 0.25 * tick_i * np.asarray(forward_axis)
                    major = (tick_i % 4 == 0)
                    draw_cross(
                        gym, viewer, env, gymapi,
                        float(tick_xy[0]), float(tick_xy[1]),
                        size=0.055 if major else 0.022,
                        z=0.003,
                        rgb=[0.10, 0.25, 0.80] if major else [0.35, 0.45, 0.65],
                    )

                # 2. Per-leg foot-position markers coloured by group & phase
                phase_now = 2.0 * np.pi * gait_frequency * sim_time
                for leg_id, joints in triplets.items():
                    marker_state = leg_phase_state(
                        phase_now, leg_id, phase_offsets, per_leg_duty_factors,
                    )
                    in_swing = not marker_state.is_stance
                    ft_xy = fmap.get(leg_id, np.zeros(2, dtype=float))
                    fx_w = body_xy[0] + ft_xy[0]
                    fy_w = body_xy[1] + ft_xy[1]

                    if in_swing:
                        rgb_pt = [0.55, 0.55, 0.55]    # grey = swing
                    elif leg_id in group_a:
                        rgb_pt = [0.10, 0.40, 0.90]    # blue = group_a stance
                    else:
                        rgb_pt = [0.90, 0.25, 0.20]    # red = group_b stance

                    draw_cross(gym, viewer, env, gymapi, fx_w, fy_w,
                               size=0.04, z=0.006, rgb=rgb_pt)

                # 3. Support polygon (green)
                poly = gait_plan.get("support_polygon_xy", [])
                if len(poly) >= 3:
                    p_arr = np.array(poly, dtype=float)
                    com_xy_arr = np.array(
                        gait_plan.get("projected_com_xy", [0.0, 0.0]), dtype=float,
                    )
                    p_off = p_arr + np.array(body_xy) - com_xy_arr
                    n = len(p_off)
                    verts_poly = np.zeros((n, 6), dtype=np.float32)
                    colors_poly = np.tile([0.0, 0.85, 0.0], (n, 1)).astype(np.float32)
                    for i in range(n):
                        a = p_off[i]
                        b = p_off[(i + 1) % n]
                        verts_poly[i] = [a[0], a[1], 0.004, b[0], b[1], 0.004]
                    gym.add_lines(viewer, env, n, verts_poly, colors_poly)

                    # 4. CoM projection (cyan cross)
                    draw_cross(gym, viewer, env, gymapi, body_xy[0], body_xy[1],
                               size=0.06, z=0.004, rgb=[0.0, 1.0, 1.0])

                # 5. Status print every 200 frames (actual body yaw vs planned heading)
                if frame_count % 200 == 1:
                    yaw_err_deg = math.degrees(body_yaw - planned_yaw_rad)
                    print(f"[{frame_count:5d}] plannedHdg={math.degrees(planned_yaw_rad):.1f}°  "
                          f"bodyYaw={math.degrees(body_yaw):+.2f}°  "
                          f"yawErr={yaw_err_deg:+.2f}°  "
                          f"body_xy=[{body_xy[0]:.3f}, {body_xy[1]:.3f}]")

                gym.draw_viewer(viewer, sim, True)
                gym.sync_frame_time(sim)
                sim_time += dt

            # ── Viewer closed — print summary ─────────────────────────────
            trail_arr = np.array(com_trail, dtype=float)
            if len(trail_arr) > 1:
                displacement = trail_arr[-1, :2] - trail_arr[0, :2]
                fwd_dist = float(np.dot(displacement, forward_axis))
                lat_dist = float(np.dot(displacement, lateral_axis))
                initial_yaw = float(trail_arr[0, 2])
                final_yaw = float(trail_arr[-1, 2])
                yaw_drift_deg = math.degrees(final_yaw - initial_yaw)
                print(f"\n[Motion summary after {frame_count} steps ({sim_time:.1f}s)]")
                print(f"  start (x,y,yaw) = [{trail_arr[0,0]:.3f}, {trail_arr[0,1]:.3f}, {math.degrees(initial_yaw):+.2f}°]")
                print(f"  end   (x,y,yaw) = [{trail_arr[-1,0]:.3f}, {trail_arr[-1,1]:.3f}, {math.degrees(final_yaw):+.2f}°]")
                print(f"  forward dist     = {fwd_dist:+.4f} m")
                print(f"  lateral drift    = {lat_dist:+.4f} m")
                print(f"  yaw drift        = {yaw_drift_deg:+.2f}°")
                print(f"  drift/fwd ratio  = {abs(lat_dist)/max(abs(fwd_dist),0.01)*100:.2f}%")
                if len(yaw_history) > 10:
                    ya = np.array(yaw_history)
                    print(f"  yaw range        = [{math.degrees(float(ya.min())):+.2f}°, {math.degrees(float(ya.max())):+.2f}°]")
                    print(f"  yaw std          = {math.degrees(float(ya.std())):.2f}°")
                if correction_log:
                    print(f"  corrections      = {len(correction_log)} activations")
                    avg_ratio = float(np.mean([c['drift_ratio'] for c in correction_log]))
                    print(f"  avg drift ratio  = {avg_ratio:.4f}")
        else:
            for step in range(max(args.steps, 1)):
                targets = build_gait_targets(
                    description, gait_plan, triplets, stand_targets.copy(), sim_time,
                    gait_frequency, args.swing_ratio_amplitude,
                    _stance_lift_effective, _swing_lift,
                    _stance_drop_effective, _swing_drop,
                    per_leg_stride_amplitudes=per_leg_stride_amplitudes,
                    touchdown_ramp=touchdown_ramp,
                    gait_mode="unified_phase",
                    duty_factor=duty_factor,
                    phase_offsets=phase_offsets,
                    per_leg_duty_factors=per_leg_duty_factors,
                )
                stance_weights = gait_stance_weights(
                    list(triplets), gait_plan, sim_time, gait_frequency,
                    phase_offsets, per_leg_duty_factors,
                )
                targets = add_leveling_bias(
                    targets, triplets, level_bias, stance_weights,
                    finite_lower, finite_upper,
                )
                gym.set_actor_dof_position_targets(env, actor, targets)
                gym.simulate(sim)
                gym.fetch_results(sim, True)

                body_xy = get_body_xy(gym, env, actor, gymapi)
                _, _, body_yaw = get_body_attitude(gym, env, actor, gymapi)
                com_trail.append([body_xy[0], body_xy[1], body_yaw])
                yaw_history.append(body_yaw)
                body_roll, body_pitch, _ = get_body_attitude(gym, env, actor, gymapi)
                for leg_id, foot in fmap.items():
                    tilt = (body_roll * float(foot[1]) / y_extent
                            - body_pitch * float(foot[0]) / x_extent)
                    raw = float(np.clip(0.70 * tilt, -0.18, 0.18))
                    level_bias[leg_id] = (
                        0.90 * level_bias.get(leg_id, 0.0) + 0.10 * raw
                    )

                # ── Body height compensation (every 120 steps) ─────────────
                if (step + 1) % 120 == 0:
                    body_z = get_body_z(gym, env, actor, gymapi)
                    z_error = _body_height_target - body_z
                    _height_integral += 0.03 * z_error
                    _height_integral = float(np.clip(_height_integral, -0.15, 0.35))
                    _stance_lift_effective = _stance_lift - _height_kI_lift * _height_integral
                    _stance_lift_effective = float(np.clip(
                        _stance_lift_effective, _lift_ratio_min, 0.95,
                    ))
                    _stance_drop_effective = _stance_drop - _height_kI_drop * _height_integral
                    _stance_drop_effective = float(np.clip(
                        _stance_drop_effective, _drop_ratio_min, 1.0,
                    ))

                if (step + 1) % steps_per_cycle == 0 and len(com_trail) > 1:
                    cycle_start = np.asarray(
                        com_trail[max(0, len(com_trail) - steps_per_cycle)][:2], dtype=float,
                    )
                    cycle_delta = np.asarray(body_xy, dtype=float) - cycle_start
                    cycle_fwd = float(np.dot(cycle_delta, forward_axis))
                    cycle_lat = float(np.dot(cycle_delta, lateral_axis))
                    cycle_z = get_body_z(gym, env, actor, gymapi)
                    print(f"[Cycle {(step + 1) // steps_per_cycle:02d}] "
                          f"forward={cycle_fwd:+.4f} m, lateral={cycle_lat:+.4f} m, "
                          f"body_z={cycle_z:.3f} m")

                # ── Online yaw correction (every 60 steps ≈ 1 s) ─────────
                # Primary: proportional feedback on absolute body yaw error.
                # Secondary: local drift in body frame for fine-tuning.
                if (step + 1) % 60 == 0:
                    # ── Absolute yaw error (wrapped to [-π, π]) ──────────────
                    yaw_error = body_yaw - planned_yaw_rad
                    yaw_error = float(np.arctan2(np.sin(yaw_error), np.cos(yaw_error)))
                    yaw_err_deg = math.degrees(yaw_error)

                    # ── Local drift in body frame ────────────────────────────
                    if len(com_trail) >= 60:
                        recent = np.array(com_trail[-60:], dtype=float)
                        disp = recent[-1, :2] - recent[0, :2]
                        avg_yaw = float(np.mean(recent[:, 2]))
                        cos_y = math.cos(avg_yaw)
                        sin_y = math.sin(avg_yaw)
                        body_fwd = np.array([cos_y, sin_y], dtype=float)
                        body_lat = np.array([-sin_y, cos_y], dtype=float)
                        fwd_disp = float(np.dot(disp, body_fwd))
                        lat_disp = float(np.dot(disp, body_lat))
                        drift_ratio = abs(lat_disp) / max(abs(fwd_disp), 0.01)
                    else:
                        lat_disp = 0.0
                        fwd_disp = 0.0
                        drift_ratio = 0.0

                    world_delta = (
                        np.asarray(com_trail[-1][:2], dtype=float)
                        - np.asarray(com_trail[0][:2], dtype=float)
                    )
                    cross_track_error = float(np.dot(world_delta, lateral_axis))
                    cross_track_velocity = 0.0
                    if len(com_trail) >= 61:
                        cross_track_velocity = float(np.dot(
                            np.asarray(com_trail[-1][:2], dtype=float)
                            - np.asarray(com_trail[-61][:2], dtype=float),
                            lateral_axis,
                        ))
                    desired_yaw_offset = float(np.clip(
                        -2.0 * cross_track_error - 0.8 * cross_track_velocity,
                        -0.40, 0.40,
                    ))
                    control_error = yaw_error - desired_yaw_offset

                    # ── Combined correction (PI on yaw error) ──────────────
                    if abs(control_error) > 0.0087 or drift_ratio > 0.05:
                        _yaw_error_integral += 0.02 * control_error
                        _yaw_error_integral = float(np.clip(_yaw_error_integral, -0.5, 0.5))
                        per_leg_stride_amplitudes, _yaw_error_integral = apply_online_yaw_correction(
                            _base_amplitudes, gait_plan, control_error,
                            yaw_error_integral=_yaw_error_integral, kp=1.0, ki=0.3)
                        if (step + 1) % 2400 == 0:
                            print(f"[YawFB] f={step+1}  "
                                  f"bodyYaw={math.degrees(body_yaw):+.2f}°  "
                                  f"yawErr={yaw_err_deg:+.2f}°  "
                                  f"drift(body) lat={lat_disp:+.3f}m fwd={fwd_disp:+.3f}m ratio={drift_ratio:.3f}")

                if (step + 1) % 400 == 0:
                    roll, pitch, _ = get_body_attitude(gym, env, actor, gymapi)
                    body_z = get_body_z(gym, env, actor, gymapi)
                    print(f"[{step + 1:5d}/{args.steps}] "
                          f"z={body_z:.3f} xy=[{body_xy[0]:.4f},{body_xy[1]:.4f}] "
                          f"roll={math.degrees(roll):.1f}° pitch={math.degrees(pitch):.1f}°")

                sim_time += dt

            # Headless summary
            trail = np.array(com_trail, dtype=float)
            if len(trail) > 1:
                displacement = trail[-1, :2] - trail[0, :2]
                forward = np.array(forward_axis, dtype=float)
                forward = forward / max(float(np.linalg.norm(forward)), 1e-9)
                fwd_dist = float(np.dot(displacement, forward))
                lat_dist = float(np.dot(
                    displacement, np.array([-forward[1], forward[0]], dtype=float),
                ))
                initial_yaw = float(trail[0, 2])
                final_yaw = float(trail[-1, 2])
                yaw_drift_deg = math.degrees(final_yaw - initial_yaw)

                print(f"\n[Motion summary after {args.steps} steps]")
                print(f"  start (x,y,yaw) = [{trail[0,0]:.3f}, {trail[0,1]:.3f}, {math.degrees(initial_yaw):+.2f}°]")
                print(f"  end   (x,y,yaw) = [{trail[-1,0]:.3f}, {trail[-1,1]:.3f}, {math.degrees(final_yaw):+.2f}°]")
                print(f"  forward dist     = {fwd_dist:+.4f} m")
                print(f"  lateral drift    = {lat_dist:+.4f} m")
                print(f"  yaw drift        = {yaw_drift_deg:+.2f}°")
                print(f"  drift/fwd ratio  = {abs(lat_dist)/max(abs(fwd_dist),0.01)*100:.2f}%")
                # Yaw oscillation stats
                if len(yaw_history) > 10:
                    ya = np.array(yaw_history)
                    print(f"  yaw range        = [{math.degrees(float(ya.min())):+.2f}°, {math.degrees(float(ya.max())):+.2f}°]")
                    print(f"  yaw std          = {math.degrees(float(ya.std())):.2f}°")
                if correction_log:
                    print(f"  corrections      = {len(correction_log)} activations")
                    avg_ratio = float(np.mean([c['drift_ratio'] for c in correction_log]))
                    print(f"  avg drift ratio  = {avg_ratio:.4f}")

        episode_diagnostics = diagnose_trajectory_only(
            com_trail, forward_axis, active_leg_count=len(triplets),
        )
        print("[Episode diagnostics: trajectory-only viewer entry]")
        print(json.dumps(episode_diagnostics, indent=2, ensure_ascii=False))
        print("[OK] Simulation finished.")
        return 0

    finally:
        if viewer is not None:
            gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)


if __name__ == "__main__":
    raise SystemExit(main())
