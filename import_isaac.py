#!/usr/bin/env python3
"""Load the generated URDF in Isaac Gym and run a macro gait demo."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np

# stability module is in the same directory; import lazily so the file
# still runs without it (warning only).
try:
    from stability import (
        compute_ssm,
        compute_support_polygon_xy,
        compute_projected_com_xy,
    )
    _HAS_STABILITY = True
except ImportError:
    _HAS_STABILITY = False


TARGET_PYTHON = "/data/conda/envs/unitree-rl/bin/python"
TARGET_LD_PATH = "/data/conda/envs/unitree-rl/lib"
ASSET_DIR_NAME = "robot_assets"


def maybe_reexec_with_runtime_env() -> None:
    if os.environ.get("IMPORT_ISAAC_REEXEC") == "1":
        return

    env = dict(os.environ)
    ld_path = env.get("LD_LIBRARY_PATH", "")
    paths = [part for part in ld_path.split(":") if part]
    if TARGET_LD_PATH not in paths:
        env["LD_LIBRARY_PATH"] = f"{TARGET_LD_PATH}:{ld_path}" if ld_path else TARGET_LD_PATH
        env["IMPORT_ISAAC_REEXEC"] = "1"
        print("[INFO] 补充 LD_LIBRARY_PATH 后重启进程以加载 Isaac Gym 动态库。")
        os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def maybe_reexec_in_unitree_python() -> None:
    if os.environ.get("IMPORT_ISAAC_PY_REEXEC") == "1":
        return
    if sys.executable == TARGET_PYTHON:
        return
    if not Path(TARGET_PYTHON).exists():
        return

    env = dict(os.environ)
    ld_path = env.get("LD_LIBRARY_PATH", "")
    if TARGET_LD_PATH not in [part for part in ld_path.split(":") if part]:
        env["LD_LIBRARY_PATH"] = f"{TARGET_LD_PATH}:{ld_path}" if ld_path else TARGET_LD_PATH
    env["IMPORT_ISAAC_PY_REEXEC"] = "1"
    print(f"[INFO] 当前解释器不含 isaacgym，自动切换到: {TARGET_PYTHON}")
    os.execvpe(TARGET_PYTHON, [TARGET_PYTHON, *sys.argv], env)


def build_fallback_plan(description: Dict[str, object], reason: str) -> Dict[str, object]:
    num_legs = int(description.get("num_legs", 0))
    leg_ids = list(range(max(num_legs, 0)))
    group_a = [leg_id for index, leg_id in enumerate(leg_ids) if index % 2 == 0]
    group_b = [leg_id for index, leg_id in enumerate(leg_ids) if index % 2 == 1]
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
        "support_leg_ids": leg_ids,
        "near_stance_leg_ids": [],
        "topology": {
            "groups": {"group_a": group_a, "group_b": group_b},
            "phase_offsets": {"group_a": 0.0, "group_b": float(np.pi)},
            "inhibition_rules": [
                {
                    "leg_id": -1,
                    "reason": reason,
                    "in_degree": 0.0,
                    "out_degree": 0.0,
                }
            ],
            "coupling_matrix_zeroed_edges": [],
        },
    }


def compute_plan_with_fallback(description: Dict[str, object]) -> Dict[str, object]:
    try:
        from adaptive_gait import compute_adaptive_plan

        return compute_adaptive_plan(description)
    except ModuleNotFoundError as exc:
        missing_name = getattr(exc, "name", "unknown")
        print(
            "[WARN] 无法导入自适应规划依赖，切换为简化分组规划。"
            f" missing module: {missing_name}"
        )
        print("[WARN] 若需要完整阶段一/二几何规划，请在当前环境安装 shapely。")
        return build_fallback_plan(description, reason=f"missing_module:{missing_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--description", type=Path, default=Path(ASSET_DIR_NAME) / "robot_description.json")
    parser.add_argument("--urdf", type=Path, default=Path(ASSET_DIR_NAME) / "generated_robot.urdf")
    parser.add_argument("--steps", type=int, default=1200, help="Headless mode simulation steps.")
    parser.add_argument("--headless", action="store_true", help="Run without viewer.")
    parser.add_argument("--compute-device-id", type=int, default=0, help="CUDA device for physics compute.")
    parser.add_argument("--graphics-device-id", type=int, default=0, help="CUDA device for rendering.")
    parser.add_argument("--cpu-sim", action="store_true", help="Force CPU simulation.")
    parser.add_argument("--gpu-pipeline", action="store_true", help="Enable GPU pipeline if supported.")
    parser.add_argument("--stiffness", type=float, default=2200.0, help="Joint position-control stiffness.")
    parser.add_argument("--damping", type=float, default=320.0, help="Joint position-control damping.")
    parser.add_argument("--effort", type=float, default=25000.0, help="Default per-joint effort limit.")
    parser.add_argument("--body-height", type=float, default=0.72, help="Initial base height above ground.")
    parser.add_argument("--gait-frequency", type=float, default=0.85, help="Cycle frequency in Hz.")
    parser.add_argument("--swing-ratio-amplitude", type=float, default=0.26, help="Swing joint ratio amplitude around neutral.")
    parser.add_argument("--swing-lift-ratio", type=float, default=0.78, help="Lift joint ratio during swing phase.")
    parser.add_argument("--stance-lift-ratio", type=float, default=0.54, help="Lift joint ratio during stance phase.")
    parser.add_argument("--swing-drop-ratio", type=float, default=0.38, help="Drop joint ratio during swing phase.")
    parser.add_argument("--stance-drop-ratio", type=float, default=0.90, help="Drop joint ratio during stance phase.")
    return parser.parse_args()


def load_description(description_path: Path) -> Dict[str, object]:
    return json.loads(description_path.read_text(encoding="utf-8"))


def np_clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class MacroLegController:
    """Map macro actions onto generated URDF joint names."""

    def __init__(self, leg_count: int) -> None:
        self.leg_count = leg_count
        self.pending_targets: Dict[str, float] = {}

    def _joint_name(self, leg_id: int, action: str) -> str:
        if leg_id < 0 or leg_id >= self.leg_count:
            raise IndexError(f"leg_id out of range: {leg_id}")
        return f"leg_{leg_id}_{action}"

    def lift_leg(self, leg_id: int, amount: float = 0.60) -> None:
        self.pending_targets[self._joint_name(leg_id, "lift")] = float(amount)

    def swing_leg(self, leg_id: int, target_pos: List[float]) -> None:
        if len(target_pos) != 3:
            raise ValueError("target_pos must be a 3D vector [x, y, z].")
        swing_angle = math.atan2(float(target_pos[1]), max(abs(float(target_pos[0])), 1e-6))
        self.pending_targets[self._joint_name(leg_id, "swing")] = float(np_clip(swing_angle, -0.55, 0.55))

    def drop_leg(self, leg_id: int, extension: float = 0.75) -> None:
        self.pending_targets[self._joint_name(leg_id, "drop")] = float(np_clip(extension, -0.10, 1.10))

    def flush(self) -> Dict[str, float]:
        targets = dict(self.pending_targets)
        self.pending_targets.clear()
        return targets


def load_gymapi():
    maybe_reexec_with_runtime_env()
    try:
        from isaacgym import gymapi

        return gymapi
    except ModuleNotFoundError as exc:
        maybe_reexec_in_unitree_python()
        raise RuntimeError(
            "未找到 isaacgym 包。请使用 unitree-rl 环境运行，例如:\n"
            f"LD_LIBRARY_PATH={TARGET_LD_PATH} {TARGET_PYTHON} import_isaac.py\n"
            f"当前解释器: {sys.executable}"
        ) from exc


def build_dof_targets(
    gym,
    env,
    actor,
    gymapi,
    named_targets: Dict[str, float],
) -> tuple[np.ndarray, list[str]]:
    dof_props = gym.get_actor_dof_properties(env, actor)
    lower = np.asarray(dof_props["lower"], dtype=np.float32)
    upper = np.asarray(dof_props["upper"], dtype=np.float32)
    mids = 0.5 * (np.where(np.isfinite(lower), lower, -0.5) + np.where(np.isfinite(upper), upper, 0.5))
    targets = mids.astype(np.float32)

    dof_names = gym.get_actor_dof_names(env, actor)
    name_to_index = {name: idx for idx, name in enumerate(dof_names)}

    missing_names: list[str] = []
    for joint_name, value in named_targets.items():
        index = name_to_index.get(joint_name)
        if index is None:
            missing_names.append(joint_name)
            continue
        lo = lower[index] if np.isfinite(lower[index]) else -1e9
        hi = upper[index] if np.isfinite(upper[index]) else 1e9
        targets[index] = float(np_clip(float(value), float(lo), float(hi)))

    return targets, missing_names


def clamp_ratio(value: float) -> float:
    return float(np_clip(float(value), 0.0, 1.0))


def ratio_to_joint(lower: float, upper: float, ratio: float) -> float:
    bounded = clamp_ratio(ratio)
    return float(lower + bounded * (upper - lower))


def foot_xy_map(description: Dict[str, object]) -> Dict[int, np.ndarray]:
    mapping: Dict[int, np.ndarray] = {}
    for link in description.get("links", []):
        if link.get("role") != "foot" or link.get("leg_id") is None:
            continue
        leg_id = int(link["leg_id"])
        origin = np.asarray(link.get("default_world_origin", [0.0, 0.0, 0.0]), dtype=float)
        mapping[leg_id] = origin[:2]
    return mapping



def estimate_drop_torque_margin(description: Dict[str, object], gait_plan: Dict[str, object]) -> Dict[str, float]:
    total_mass = 0.0
    trunk_mass = 0.0
    for link in description.get("links", []):
        mass = float(link.get("mass_properties", {}).get("mass", 0.0))
        total_mass += mass
        if link.get("role") == "trunk":
            trunk_mass += mass

    leg_count = int(description.get("num_legs", 0))
    active_support = max(2, leg_count // 2)
    load_per_leg = (total_mass * 9.81) / float(active_support)

    hip_xy: Dict[int, np.ndarray] = {}
    foot_xy: Dict[int, np.ndarray] = {}
    for link in description.get("links", []):
        leg_id = link.get("leg_id")
        if leg_id is None:
            continue
        origin = np.asarray(link.get("default_world_origin", [0.0, 0.0, 0.0]), dtype=float)
        if link.get("role") == "joint_sphere" and str(link.get("name", "")).endswith("_hip"):
            hip_xy[int(leg_id)] = origin[:2]
        elif link.get("role") == "foot":
            foot_xy[int(leg_id)] = origin[:2]

    lever_arms = []
    for leg_id, hip in hip_xy.items():
        foot = foot_xy.get(leg_id)
        if foot is None:
            continue
        lever_arms.append(float(np.linalg.norm(foot - hip)))

    mean_lever = float(np.mean(lever_arms)) if lever_arms else 0.20
    estimated_required = load_per_leg * mean_lever

    drop_efforts = []
    for joint in description.get("joints", []):
        name = str(joint.get("name", ""))
        if not name.endswith("_drop"):
            continue
        limit = joint.get("limit", {})
        if "effort" in limit:
            drop_efforts.append(float(limit["effort"]))
    available_drop = float(np.mean(drop_efforts)) if drop_efforts else 0.0
    margin = available_drop / max(estimated_required, 1e-6)

    # Use stability module when available; fall back to plan values otherwise.
    if _HAS_STABILITY:
        polygon_xy = compute_support_polygon_xy(description)
        com_xy_arr = compute_projected_com_xy(description)
    else:
        polygon_xy = np.asarray(gait_plan.get("support_polygon_xy", []), dtype=float)
        com_xy_arr = np.asarray(gait_plan.get("projected_com_xy", [0.0, 0.0]), dtype=float)
    static_margin = compute_ssm(polygon_xy, com_xy_arr) if _HAS_STABILITY else 0.0
    if not _HAS_STABILITY:
        print("[WARN] stability.py not found; SSM will report 0.0.")

    return {
        "total_mass": float(total_mass),
        "trunk_mass": float(trunk_mass),
        "trunk_mass_ratio": float(trunk_mass / max(total_mass, 1e-6)),
        "estimated_drop_required_torque": float(estimated_required),
        "mean_drop_effort_limit": float(available_drop),
        "drop_torque_margin_ratio": float(margin),
        "static_margin_xy": float(static_margin),
    }


def print_diagnostics(metrics: Dict[str, float]) -> None:
    print("Dynamics diagnostics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    if metrics["static_margin_xy"] < 0.0:
        print("[DIAG] 质心投影在支撑域外，静稳定性存在风险。")
    else:
        print("[DIAG] 质心投影位于支撑域内，静稳定性基本满足。")

    if metrics["drop_torque_margin_ratio"] < 1.0:
        print("[DIAG] 估算关节力矩裕度不足，腿部承载可能吃紧。")
    elif metrics["drop_torque_margin_ratio"] < 1.5:
        print("[DIAG] 估算关节力矩裕度偏紧，建议降低机身质量或增大关节 effort。")
    else:
        print("[DIAG] 估算关节力矩裕度充足。")

    if metrics["trunk_mass_ratio"] > 0.75:
        print("[DIAG] 躯干质量占比过高，动态步态下更易出现腿部过载。")


# ---------------------------------------------------------------------------
# Ground-plane visualisation: forward-direction arrow
# ---------------------------------------------------------------------------

def get_actor_body_xy(gym, env, actor, gymapi) -> List[float]:
    """Return the XY position of the actor's root body (base_link) in world frame."""
    states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)
    if states is None or len(states) == 0:
        return [0.0, 0.0]
    p = states["pose"]["p"][0]
    return [float(p["x"]), float(p["y"])]


def draw_forward_direction_line(
    gym,
    viewer,
    env,
    gymapi,
    origin_xy: List[float],
    forward_axis: List[float],
    length: float = 1.2,
    z: float = 0.008,
) -> None:
    """Render an orange arrow on the ground plane showing the robot's forward direction.

    Draws three debug line segments each frame:
      1. Main shaft from *origin_xy* in *forward_axis* direction.
      2 & 3. Two arrowhead lines meeting at the shaft tip.

    Parameters
    ----------
    gym            Isaac Gym gym object.
    viewer         Isaac Gym viewer object (must not be None).
    env            Environment handle.
    gymapi         gymapi module reference.
    origin_xy      [x, y] start of the arrow (typically robot body XY).
    forward_axis   [fx, fy] unit-ish forward direction vector.
    length         Total arrow shaft length in metres (default 1.2).
    z              Height above ground for the line (default 0.008 m).
    """
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    fx, fy = float(forward_axis[0]), float(forward_axis[1])
    norm = math.sqrt(fx * fx + fy * fy)
    if norm < 1e-9:
        return
    fx, fy = fx / norm, fy / norm

    # Shaft endpoint
    ex, ey = ox + fx * length, oy + fy * length

    # Perpendicular (left-hand side)
    px, py = -fy, fx

    # Arrowhead arms (back 20% of shaft length, spread 35%)
    head = length * 0.20
    spread = 0.35
    left_x  = ex - fx * head + px * head * spread
    left_y  = ey - fy * head + py * head * spread
    right_x = ex - fx * head - px * head * spread
    right_y = ey - fy * head - py * head * spread

    # Vertices: shape (num_lines, 6) — [x1,y1,z1, x2,y2,z2] per row
    verts = np.array(
        [
            [ox, oy, z, ex, ey, z],
            [ex, ey, z, left_x,  left_y,  z],
            [ex, ey, z, right_x, right_y, z],
        ],
        dtype=np.float32,
    )
    # Colors: shape (num_lines, 3) — RGB per row (orange)
    colors = np.array(
        [[1.0, 0.45, 0.0], [1.0, 0.45, 0.0], [1.0, 0.45, 0.0]],
        dtype=np.float32,
    )
    gym.clear_lines(viewer)
    gym.add_lines(viewer, env, 3, verts, colors)


def resolve_joint_triplets(
    gym,
    env,
    actor,
    description: Dict[str, object],
) -> Dict[int, Dict[str, object]]:
    dof_props = gym.get_actor_dof_properties(env, actor)
    lower = np.asarray(dof_props["lower"], dtype=np.float32)
    upper = np.asarray(dof_props["upper"], dtype=np.float32)
    mids = 0.5 * (np.where(np.isfinite(lower), lower, -0.5) + np.where(np.isfinite(upper), upper, 0.5))

    dof_names = gym.get_actor_dof_names(env, actor)
    name_to_index = {name: idx for idx, name in enumerate(dof_names)}

    result: Dict[int, Dict[str, object]] = {}
    for leg_id in range(int(description.get("num_legs", 0))):
        lift_name = f"leg_{leg_id}_lift"
        swing_name = f"leg_{leg_id}_swing"
        drop_name = f"leg_{leg_id}_drop"
        if lift_name not in name_to_index or swing_name not in name_to_index or drop_name not in name_to_index:
            continue
        lift_idx = name_to_index[lift_name]
        swing_idx = name_to_index[swing_name]
        drop_idx = name_to_index[drop_name]
        result[leg_id] = {
            "lift_idx": lift_idx,
            "swing_idx": swing_idx,
            "drop_idx": drop_idx,
            "lift_lower": float(lower[lift_idx]),
            "lift_upper": float(upper[lift_idx]),
            "swing_lower": float(lower[swing_idx]),
            "swing_upper": float(upper[swing_idx]),
            "drop_lower": float(lower[drop_idx]),
            "drop_upper": float(upper[drop_idx]),
            "swing_mid": float(mids[swing_idx]),
        }
    return result


def leg_group_phase(leg_id: int, group_a: List[int], group_b: List[int], base_phase: float) -> float:
    if leg_id in group_b:
        return base_phase + np.pi
    if leg_id in group_a:
        return base_phase
    return base_phase


def build_cyclic_dof_targets(
    description: Dict[str, object],
    gait_plan: Dict[str, object],
    joint_triplets: Dict[int, Dict[str, object]],
    default_targets: np.ndarray,
    sim_time: float,
    args: argparse.Namespace,
) -> np.ndarray:
    targets = default_targets.copy()
    phase = 2.0 * np.pi * max(args.gait_frequency, 0.05) * sim_time
    group_a = list(gait_plan.get("topology", {}).get("groups", {}).get("group_a", []))
    group_b = list(gait_plan.get("topology", {}).get("groups", {}).get("group_b", []))

    forward_axis = np.asarray(gait_plan.get("final_forward_axis", [1.0, 0.0]), dtype=float)
    if np.linalg.norm(forward_axis) < 1e-6:
        forward_axis = np.array([1.0, 0.0], dtype=float)
    forward_axis = forward_axis / np.linalg.norm(forward_axis)
    foot_map = foot_xy_map(description)

    for leg_id, joints in joint_triplets.items():
        leg_phase = leg_group_phase(leg_id, group_a, group_b, phase)
        swing_wave = float(np.sin(leg_phase))
        swing_alpha = max(swing_wave, 0.0)

        foot_xy = foot_map.get(leg_id, np.zeros(2, dtype=float))
        direction_sign = 1.0 if float(np.dot(foot_xy, forward_axis)) >= 0.0 else -1.0

        lift_ratio = args.stance_lift_ratio + (args.swing_lift_ratio - args.stance_lift_ratio) * swing_alpha
        drop_ratio = args.stance_drop_ratio + (args.swing_drop_ratio - args.stance_drop_ratio) * swing_alpha
        swing_center_ratio = 0.5
        swing_ratio = swing_center_ratio + args.swing_ratio_amplitude * direction_sign * swing_wave

        targets[joints["lift_idx"]] = ratio_to_joint(joints["lift_lower"], joints["lift_upper"], lift_ratio)
        targets[joints["drop_idx"]] = ratio_to_joint(joints["drop_lower"], joints["drop_upper"], drop_ratio)
        targets[joints["swing_idx"]] = ratio_to_joint(joints["swing_lower"], joints["swing_upper"], swing_ratio)

    return targets


def configure_actor_dofs(gym, env, actor, gymapi, stiffness: float, damping: float, effort: float) -> None:
    dof_props = gym.get_actor_dof_properties(env, actor)
    dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
    dof_props["stiffness"].fill(float(stiffness))
    dof_props["damping"].fill(float(damping))
    if "effort" in dof_props.dtype.names:
        dof_props["effort"].fill(float(effort))
    gym.set_actor_dof_properties(env, actor, dof_props)


def main() -> None:
    args = parse_args()
    description = load_description(args.description)
    gait_plan = compute_plan_with_fallback(description)
    diagnostics = estimate_drop_torque_margin(description, gait_plan)

    gymapi = load_gymapi()
    gym = gymapi.acquire_gym()

    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2

    use_gpu = not args.cpu_sim
    sim_params.use_gpu_pipeline = bool(use_gpu and args.gpu_pipeline)
    sim_params.physx.use_gpu = use_gpu

    graphics_id = -1 if args.headless else args.graphics_device_id
    sim = gym.create_sim(args.compute_device_id, graphics_id, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        if use_gpu:
            sim_params.use_gpu_pipeline = False
            sim_params.physx.use_gpu = False
            sim = gym.create_sim(0, graphics_id, gymapi.SIM_PHYSX, sim_params)
        if sim is None:
            raise RuntimeError("无法创建 Isaac Gym 仿真。")

    viewer = None
    try:
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 1.8
        plane_params.dynamic_friction = 1.6
        plane_params.restitution = 0.0
        gym.add_ground(sim, plane_params)

        urdf_path = args.urdf.resolve()
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset_options.use_mesh_materials = True

        asset = gym.load_asset(sim, str(urdf_path.parent), urdf_path.name, asset_options)
        if asset is None:
            raise RuntimeError(f"加载 URDF 失败: {urdf_path}")

        env = gym.create_env(sim, gymapi.Vec3(-1.5, -1.5, 0.0), gymapi.Vec3(1.5, 1.5, 2.0), 1)
        if env is None:
            raise RuntimeError("创建环境失败。")

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, float(args.body_height))
        actor = gym.create_actor(env, asset, pose, "generated_robot", 0, 1)
        if actor < 0:
            raise RuntimeError("创建 actor 失败。")

        configure_actor_dofs(gym, env, actor, gymapi, args.stiffness, args.damping, args.effort)
        dof_props = gym.get_actor_dof_properties(env, actor)
        lower = np.asarray(dof_props["lower"], dtype=np.float32)
        upper = np.asarray(dof_props["upper"], dtype=np.float32)
        dof_targets = (0.5 * (np.where(np.isfinite(lower), lower, -0.5) + np.where(np.isfinite(upper), upper, 0.5))).astype(np.float32)
        joint_triplets = resolve_joint_triplets(gym, env, actor, description)
        missing_joints = []

        print("Adaptive gait plan summary:")
        print(json.dumps(gait_plan, indent=2, ensure_ascii=False))
        print_diagnostics(diagnostics)
        print("[INFO] 已启用分组交替周期控制，目标为平地直线推进。")
        if missing_joints:
            print("[WARN] 部分关节名在资产 DOF 中未找到:")
            print(missing_joints)
        print(f"[OK] Loaded in Isaac Gym: {urdf_path}")

        if not args.headless:
            viewer = gym.create_viewer(sim, gymapi.CameraProperties())
            if viewer is None:
                raise RuntimeError("创建 viewer 失败。")
            cam_pos = gymapi.Vec3(2.4, 1.8, 1.4)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.4)
            gym.viewer_camera_look_at(viewer, env, cam_pos, cam_target)
            print("[OK] Viewer 启动，关闭窗口即可退出。")
            forward_axis = list(gait_plan.get("final_forward_axis", [1.0, 0.0]))
            sim_time = 0.0
            while not gym.query_viewer_has_closed(viewer):
                dof_targets = build_cyclic_dof_targets(description, gait_plan, joint_triplets, dof_targets, sim_time, args)
                gym.set_actor_dof_position_targets(env, actor, dof_targets)
                gym.simulate(sim)
                gym.fetch_results(sim, True)
                gym.step_graphics(sim)
                # Update forward-direction arrow every frame
                body_xy = get_actor_body_xy(gym, env, actor, gymapi)
                draw_forward_direction_line(gym, viewer, env, gymapi, body_xy, forward_axis)
                gym.draw_viewer(viewer, sim, True)
                gym.sync_frame_time(sim)
                sim_time += sim_params.dt
        else:
            sim_time = 0.0
            for _ in range(max(args.steps, 1)):
                dof_targets = build_cyclic_dof_targets(description, gait_plan, joint_triplets, dof_targets, sim_time, args)
                gym.set_actor_dof_position_targets(env, actor, dof_targets)
                gym.simulate(sim)
                gym.fetch_results(sim, True)
                sim_time += sim_params.dt
            print("[OK] Headless simulation finished.")
    finally:
        if viewer is not None:
            gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
