#!/usr/bin/env python3
"""Minimal Isaac Gym smoke test for loading the generated robot URDF."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


TARGET_PYTHON = "/data/conda/envs/unitree-rl/bin/python"
TARGET_LD_PATH = "/data/conda/envs/unitree-rl/lib"
GEN_PYTHON = "/data/conda/envs/Adaptation/bin/python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run without opening the Isaac Gym viewer.")
    parser.add_argument("--steps", type=int, default=1200, help="Simulation steps to run in headless mode.")
    parser.add_argument("--compute-device-id", type=int, default=0, help="CUDA device index for physics compute.")
    parser.add_argument("--graphics-device-id", type=int, default=0, help="CUDA device index for rendering.")
    parser.add_argument("--cpu-sim", action="store_true", help="Force CPU simulation instead of GPU.")
    parser.add_argument(
        "--gpu-pipeline",
        action="store_true",
        help="Enable GPU pipeline (disabled by default for stability on some machines).",
    )
    parser.add_argument("--num-robots", type=int, default=20, help="Number of robot instances to spawn.")
    parser.add_argument("--spacing", type=float, default=2.4, help="Spacing between robots in the grid.")
    parser.add_argument(
        "--hold-steps",
        type=int,
        default=500,
        help="Warm-up stabilization steps with reinforced static holding control.",
    )
    parser.add_argument(
        "--torque-scale",
        type=float,
        default=5.0,
        help="Global multiplier for joint effort limits during static holding.",
    )
    parser.add_argument(
        "--effort-cap",
        type=float,
        default=50000.0,
        help="Upper bound for per-joint effort limit (default is intentionally high).",
    )
    parser.add_argument(
        "--proximal-horizontal-ratio",
        type=float,
        default=0.50,
        help="Target ratio in joint range for body-connected leg segment orientation.",
    )
    parser.add_argument(
        "--distal-vertical-ratio",
        type=float,
        default=0.995,
        help="Target ratio in joint range for ground-contact leg segment orientation.",
    )
    parser.add_argument(
        "--adaptive-hold",
        action="store_true",
        help="Enable adaptive target updates during stabilization (enabled by default).",
    )
    parser.add_argument(
        "--no-adaptive-hold",
        action="store_false",
        dest="adaptive_hold",
        help="Disable adaptive target updates during stabilization.",
    )
    parser.add_argument(
        "--no-auto-generate",
        action="store_true",
        help="Disable automatic generation of missing unique variants.",
    )
    parser.set_defaults(adaptive_hold=True)
    return parser.parse_args()

    
def maybe_reexec_in_unitree_env() -> bool:
    if os.environ.get("TEST_GYM_REEXEC") == "1":
        return False

    if sys.executable == TARGET_PYTHON:
        return False

    if not Path(TARGET_PYTHON).exists():
        return False

    env = dict(os.environ)
    ld_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{TARGET_LD_PATH}:{ld_path}" if ld_path else TARGET_LD_PATH
    env["TEST_GYM_REEXEC"] = "1"

    print("[INFO] Auto-switching to Isaac Gym environment: unitree-rl")
    print(f"[INFO] Re-launch with interpreter: {TARGET_PYTHON}")
    os.execvpe(TARGET_PYTHON, [TARGET_PYTHON, *sys.argv], env)
    return True


def load_gymapi():
    try:
        from isaacgym import gymapi

        return gymapi
    except ModuleNotFoundError:
        maybe_reexec_in_unitree_env()
        python_bin = sys.executable
        print("[ERROR] Cannot import 'isaacgym' in current Python environment.")
        print(f"[INFO] Current interpreter: {python_bin}")
        print(f"[INFO] Current Python: {sys.version.split()[0]}")
        print("[INFO] Isaac Gym in this workspace is configured for the 'unitree-rl' env (Python 3.8).")
        print("[INFO] Run with one of these commands:")
        print(
            "  LD_LIBRARY_PATH=/data/conda/envs/unitree-rl/lib "
            "/data/conda/envs/unitree-rl/bin/python test_gym.py"
        )
        print(
            "  env -u LD_LIBRARY_PATH conda run -n unitree-rl env "
            "LD_LIBRARY_PATH=/data/conda/envs/unitree-rl/lib python test_gym.py"
        )
        return None


def initialize_standing_posture(
    gym,
    gymapi,
    env,
    actor,
    asset,
    torque_scale: float,
    effort_cap: float,
    proximal_horizontal_ratio: float,
    distal_vertical_ratio: float,
) -> dict:
    dof_props = gym.get_actor_dof_properties(env, actor)
    dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
    # Reinforced static-hold gains to avoid gravity-induced leg deformation.
    dof_props["stiffness"].fill(1800.0)
    dof_props["damping"].fill(260.0)
    if "effort" in dof_props.dtype.names:
        dof_props["effort"].fill(min(5000.0 * torque_scale, effort_cap))
    if "armature" in dof_props.dtype.names:
        dof_props["armature"].fill(0.08)

    dof_count = len(dof_props["lower"])
    dof_names = gym.get_asset_dof_names(asset)

    # Per-joint reinforcement: prioritize 2nd joint (swing) and extension (drop) joint.
    for idx, name in enumerate(dof_names):
        if "_swing" in name:
            dof_props["stiffness"][idx] = 4000.0
            dof_props["damping"][idx] = 600.0
            if "effort" in dof_props.dtype.names:
                dof_props["effort"][idx] = min(18000.0 * torque_scale, effort_cap)
        elif "_drop" in name:
            dof_props["stiffness"][idx] = 6500.0
            dof_props["damping"][idx] = 900.0
            if "effort" in dof_props.dtype.names:
                dof_props["effort"][idx] = min(30000.0 * torque_scale, effort_cap)
        elif "_lift" in name:
            dof_props["stiffness"][idx] = 2600.0
            dof_props["damping"][idx] = 380.0
            if "effort" in dof_props.dtype.names:
                dof_props["effort"][idx] = min(12000.0 * torque_scale, effort_cap)

    gym.set_actor_dof_properties(env, actor, dof_props)

    lower = np.asarray(dof_props["lower"], dtype=np.float32)
    upper = np.asarray(dof_props["upper"], dtype=np.float32)
    targets = np.zeros(dof_count, dtype=np.float32)

    for idx, name in enumerate(dof_names):
        finite_lower = lower[idx] if np.isfinite(lower[idx]) else -0.5
        finite_upper = upper[idx] if np.isfinite(upper[idx]) else 0.5
        span = finite_upper - finite_lower

        if "_lift" in name:
            # Body-connected segment: keep close to horizontal target.
            targets[idx] = finite_lower + float(np.clip(proximal_horizontal_ratio, 0.0, 1.0)) * span
        elif "_swing" in name:
            # Bias 2nd joint near centerline; do not hard-force vertical geometry.
            targets[idx] = finite_lower + 0.50 * span
        elif "_drop" in name:
            # Ground-contact segment: keep close to vertical target.
            targets[idx] = finite_lower + float(np.clip(distal_vertical_ratio, 0.0, 1.0)) * span
        else:
            targets[idx] = 0.5 * (finite_lower + finite_upper)

    finite_lower = np.where(np.isfinite(lower), lower, -1e9)
    finite_upper = np.where(np.isfinite(upper), upper, 1e9)
    targets = np.clip(targets, finite_lower, finite_upper)

    dof_states = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
    dof_states["pos"] = targets
    dof_states["vel"].fill(0.0)
    gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)
    gym.set_actor_dof_position_targets(env, actor, targets)
    return {
        "targets": targets,
        "names": dof_names,
        "lower": lower,
        "upper": upper,
        "mid": 0.5 * (lower + upper),
    }


def update_stance_targets(controller: dict, joint_pos: np.ndarray) -> np.ndarray:
    targets = controller["targets"].copy()
    names = controller["names"]
    lower = controller["lower"]
    upper = controller["upper"]
    mid = controller["mid"]

    for idx, name in enumerate(names):
        if "_drop" in name:
            sag = targets[idx] - joint_pos[idx]
            if sag > 0.004:
                targets[idx] = min(upper[idx], targets[idx] + min(0.012, 0.22 * sag))
        elif "_lift" in name:
            sag = targets[idx] - joint_pos[idx]
            if sag > 0.004:
                targets[idx] = min(upper[idx], targets[idx] + min(0.008, 0.16 * sag))
        elif "_swing" in name:
            # Softly re-center second joint instead of hard constraints.
            targets[idx] = 0.985 * targets[idx] + 0.015 * mid[idx]

    targets = np.clip(targets, lower, upper)
    controller["targets"] = targets
    return targets


def apply_targets_all(gym, envs: list, actors: list, controllers: list) -> None:
    for env, actor, controller in zip(envs, actors, controllers):
        gym.set_actor_dof_position_targets(env, actor, controller["targets"])


def stabilize_stance(
    gym,
    gymapi,
    sim,
    envs: list,
    actors: list,
    controllers: list,
    steps: int,
    adaptive_hold: bool,
) -> None:
    for _ in range(max(steps, 0)):
        if adaptive_hold:
            for env, actor, controller in zip(envs, actors, controllers):
                dof_states = gym.get_actor_dof_states(env, actor, gymapi.STATE_POS)
                joint_pos = np.asarray(dof_states["pos"], dtype=np.float32)
                update_stance_targets(controller, joint_pos)
                gym.set_actor_dof_position_targets(env, actor, controller["targets"])
        else:
            apply_targets_all(gym, envs, actors, controllers)
        gym.simulate(sim)
        gym.fetch_results(sim, True)


def run_cmd(args: list[str]) -> None:
    subprocess.run(args, check=True)


def run_cmd_ok(args: list[str]) -> bool:
    """Return True on success, False on non-zero exit (e.g. SSM pre-check fail)."""
    return subprocess.run(args).returncode == 0


def clone_generated_assets(repo_root: Path, variant_dir: Path) -> Path:
    src_urdf = repo_root / "robot_assets" / "generated_robot.urdf"
    src_mesh_dir = repo_root / "robot_assets" / "meshes"

    if not src_urdf.exists():
        raise FileNotFoundError(f"URDF not found: {src_urdf}")
    if not src_mesh_dir.exists():
        raise FileNotFoundError(f"Mesh directory not found: {src_mesh_dir}")

    if variant_dir.exists():
        shutil.rmtree(variant_dir)
    (variant_dir / "meshes").mkdir(parents=True, exist_ok=True)

    dst_urdf = variant_dir / "robot.urdf"
    shutil.copy2(src_urdf, dst_urdf)
    for mesh_file in src_mesh_dir.glob("*.stl"):
        shutil.copy2(mesh_file, variant_dir / "meshes" / mesh_file.name)
    return dst_urdf


def parse_variant_index(name: str) -> int | None:
    match = re.match(r"variant_(\d+)_seed\d+", name)
    if not match:
        return None
    return int(match.group(1))


def list_variant_urdfs(variants_root: Path) -> list[Path]:
    return sorted(path for path in variants_root.glob("*/robot.urdf") if path.is_file())


def ensure_unique_variants(repo_root: Path, num_required: int, auto_generate: bool) -> list[Path]:
    variants_root = repo_root / "robot_assets" / "variants"
    variants_root.mkdir(parents=True, exist_ok=True)
    urdfs = list_variant_urdfs(variants_root)

    if len(urdfs) >= num_required:
        return urdfs[:num_required]

    if not auto_generate:
        raise RuntimeError(
            f"Need {num_required} unique variants, but only found {len(urdfs)} in {variants_root}. "
            "Enable auto generation or pre-generate variants first."
        )

    existing_indices = []
    for variant_dir in variants_root.iterdir():
        if variant_dir.is_dir():
            parsed = parse_variant_index(variant_dir.name)
            if parsed is not None:
                existing_indices.append(parsed)

    next_index = max(existing_indices, default=-1) + 1
    needed = num_required - len(urdfs)
    print(f"[INFO] Found {len(urdfs)} unique variants, generating {needed} more...")

    seed_offset = 0   # increments every attempt; next_index increments only on success
    max_attempts = needed * 20
    attempts = 0

    while len(list_variant_urdfs(variants_root)) < num_required and attempts < max_attempts:
        seed = 7 + (next_index + seed_offset) * 17
        name = f"variant_{next_index:02d}_seed{seed}"
        attempts += 1

        ok = run_cmd_ok(
            [
                GEN_PYTHON,
                str(repo_root / "generate_geometry.py"),
                "--robot-name",
                name,
                "--seed",
                str(seed),
                "--leg-placement",
                "random",
            ]
        )
        if not ok:
            print(f"[SKIP] seed={seed} 未通过 SSM 预检，跳过。")
            seed_offset += 1
            continue

        run_cmd([GEN_PYTHON, str(repo_root / "generate_urdf.py")])
        clone_generated_assets(repo_root, variants_root / name)
        next_index += 1
        seed_offset += 1

    urdfs = list_variant_urdfs(variants_root)
    if len(urdfs) < num_required:
        raise RuntimeError(f"Variant generation incomplete: expected {num_required}, got {len(urdfs)}")
    return urdfs[:num_required]


def main() -> int:
    args = parse_args()
    gymapi = load_gymapi()
    if gymapi is None:
        return 1

    gym = gymapi.acquire_gym()

    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    use_gpu = not args.cpu_sim
    sim_params.use_gpu_pipeline = bool(use_gpu and args.gpu_pipeline)
    sim_params.physx.use_gpu = use_gpu
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 2

    graphics_id = -1 if args.headless else args.graphics_device_id
    sim = gym.create_sim(args.compute_device_id, graphics_id, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        if use_gpu:
            print("[WARN] GPU sim creation failed, retrying with CPU sim.")
            sim_params.use_gpu_pipeline = False
            sim_params.physx.use_gpu = False
            sim = gym.create_sim(0, graphics_id, gymapi.SIM_PHYSX, sim_params)
        if sim is None:
            print("[ERROR] Failed to create Isaac Gym sim.")
            return 1

    viewer = None
    try:
        repo_root = Path(__file__).resolve().parent
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 1.8
        plane_params.dynamic_friction = 1.6
        plane_params.restitution = 0.0
        gym.add_ground(sim, plane_params)

        num_robots = max(args.num_robots, 1)
        variant_urdfs = ensure_unique_variants(repo_root, num_robots, auto_generate=not args.no_auto_generate)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset_options.use_mesh_materials = True

        num_per_row = int(np.ceil(np.sqrt(num_robots)))
        env_lower = gymapi.Vec3(-args.spacing * 0.5, -args.spacing * 0.5, 0.0)
        env_upper = gymapi.Vec3(args.spacing * 0.5, args.spacing * 0.5, 1.6)

        assets = []
        for urdf_path in variant_urdfs:
            asset = gym.load_asset(sim, str(urdf_path.parent.resolve()), urdf_path.name, asset_options)
            if asset is None:
                print(f"[ERROR] Failed to load variant asset: {urdf_path}")
                return 1
            assets.append(asset)

        envs = []
        actors = []
        controllers = []
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.70)

        for idx in range(num_robots):
            env = gym.create_env(sim, env_lower, env_upper, num_per_row)
            if env is None:
                print(f"[ERROR] Failed to create env {idx}.")
                return 1

            asset = assets[idx]
            actor = gym.create_actor(env, asset, pose, f"generated_robot_{idx}", idx, 1)
            if actor < 0:
                print(f"[ERROR] Failed to create actor {idx} from asset.")
                return 1

            controller = initialize_standing_posture(
                gym,
                gymapi,
                env,
                actor,
                asset,
                args.torque_scale,
                args.effort_cap,
                args.proximal_horizontal_ratio,
                args.distal_vertical_ratio,
            )
            envs.append(env)
            actors.append(actor)
            controllers.append(controller)

        stabilize_stance(
            gym,
            gymapi,
            sim,
            envs,
            actors,
            controllers,
            args.hold_steps,
            adaptive_hold=args.adaptive_hold,
        )

        dof_count = gym.get_asset_dof_count(assets[0])
        body_count = gym.get_asset_rigid_body_count(assets[0])
        print(f"[OK] Asset loaded. DOFs={dof_count}, Bodies={body_count}")
        print(f"[OK] Spawned robots: {num_robots}, grid: {num_per_row} x {num_per_row}")
        print(f"[OK] Unique variants loaded: {len(variant_urdfs)}")
        print("[OK] Standing posture controller initialized (position control on leg joints).")
        print(
            f"[OK] Static stabilization warm-up steps: {args.hold_steps}, torque scale: {args.torque_scale}, "
            f"effort cap: {args.effort_cap}"
        )
        print(
            f"[OK] Adaptive hold: {args.adaptive_hold}, GPU sim requested: {use_gpu}, "
            f"GPU pipeline: {sim_params.use_gpu_pipeline}"
        )

        if not args.headless:
            viewer = gym.create_viewer(sim, gymapi.CameraProperties())
            if viewer is None:
                print("[ERROR] Failed to create Isaac Gym viewer.")
                return 1

            first_env = envs[0]
            grid_width = float(max(num_per_row - 1, 0)) * args.spacing
            cam_target = gymapi.Vec3(grid_width * 0.5, grid_width * 0.5, 0.42)
            cam_pos = gymapi.Vec3(
                cam_target.x + max(4.5, grid_width * 1.25),
                cam_target.y + max(3.8, grid_width * 0.9),
                max(2.0, grid_width * 0.6),
            )
            gym.viewer_camera_look_at(viewer, first_env, cam_pos, cam_target)

            print("[OK] Viewer started. Close the window to exit.")
            while not gym.query_viewer_has_closed(viewer):
                apply_targets_all(gym, envs, actors, controllers)
                gym.simulate(sim)
                gym.fetch_results(sim, True)
                gym.step_graphics(sim)
                gym.draw_viewer(viewer, sim, True)
                gym.sync_frame_time(sim)
        else:
            for _ in range(max(args.steps, 1)):
                apply_targets_all(gym, envs, actors, controllers)
                gym.simulate(sim)
                gym.fetch_results(sim, True)

        print("[OK] Isaac Gym simulation stepped successfully.")
        return 0
    finally:
        if viewer is not None:
            gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)


if __name__ == "__main__":
    raise SystemExit(main())
