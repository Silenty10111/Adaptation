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
        pose.p = gymapi.Vec3(0.0, 0.0, 0.70)
        actor = gym.create_actor(env, asset, pose, "generated_robot", 0, 1)
        if actor < 0:
            raise RuntimeError("创建 actor 失败。")

        configure_actor_dofs(gym, env, actor, gymapi, args.stiffness, args.damping, args.effort)

        controller = MacroLegController(int(description["num_legs"]))
        forward_axis = gait_plan["final_forward_axis"]
        group_a = gait_plan["topology"]["groups"]["group_a"]
        group_b = gait_plan["topology"]["groups"]["group_b"]
        active_group = group_a if group_a else group_b
        step_vector = [forward_axis[0] * 0.18, forward_axis[1] * 0.18, -0.20]

        for leg_id in active_group:
            controller.lift_leg(leg_id)
            controller.swing_leg(leg_id, step_vector)
            controller.drop_leg(leg_id)

        named_targets = controller.flush()
        dof_targets, missing_joints = build_dof_targets(gym, env, actor, gymapi, named_targets)
        gym.set_actor_dof_position_targets(env, actor, dof_targets)

        print("Adaptive gait plan summary:")
        print(json.dumps(gait_plan, indent=2, ensure_ascii=False))
        print("Macro controller targets:")
        print(named_targets)
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
            while not gym.query_viewer_has_closed(viewer):
                gym.set_actor_dof_position_targets(env, actor, dof_targets)
                gym.simulate(sim)
                gym.fetch_results(sim, True)
                gym.step_graphics(sim)
                gym.draw_viewer(viewer, sim, True)
                gym.sync_frame_time(sim)
        else:
            for _ in range(max(args.steps, 1)):
                gym.set_actor_dof_position_targets(env, actor, dof_targets)
                gym.simulate(sim)
                gym.fetch_results(sim, True)
            print("[OK] Headless simulation finished.")
    finally:
        if viewer is not None:
            gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
