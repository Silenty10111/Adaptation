#!/usr/bin/env python3
"""Minimal Isaac Gym smoke test for loading the generated robot URDF."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


TARGET_PYTHON = "/data/conda/envs/unitree-rl/bin/python"
TARGET_LD_PATH = "/data/conda/envs/unitree-rl/lib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run without opening the Isaac Gym viewer.")
    parser.add_argument("--steps", type=int, default=1200, help="Simulation steps to run in headless mode.")
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

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        print("[ERROR] Failed to create Isaac Gym sim.")
        return 1

    viewer = None
    try:
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        gym.add_ground(sim, plane_params)

        asset_root = Path("robot_assets").resolve()
        asset_file = "generated_robot.urdf"

        if not (asset_root / asset_file).exists():
            print(f"[ERROR] URDF not found: {asset_root / asset_file}")
            return 1

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset_options.use_mesh_materials = True

        asset = gym.load_asset(sim, str(asset_root), asset_file, asset_options)
        if asset is None:
            print("[ERROR] Failed to load asset into Isaac Gym.")
            return 1

        env = gym.create_env(sim, gymapi.Vec3(-1.0, -1.0, 0.0), gymapi.Vec3(1.0, 1.0, 1.5), 1)
        if env is None:
            print("[ERROR] Failed to create env.")
            return 1

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.45)
        actor = gym.create_actor(env, asset, pose, "generated_robot", 0, 1)
        if actor < 0:
            print("[ERROR] Failed to create actor from asset.")
            return 1

        dof_count = gym.get_asset_dof_count(asset)
        body_count = gym.get_asset_rigid_body_count(asset)
        print(f"[OK] Asset loaded. DOFs={dof_count}, Bodies={body_count}")

        if not args.headless:
            viewer = gym.create_viewer(sim, gymapi.CameraProperties())
            if viewer is None:
                print("[ERROR] Failed to create Isaac Gym viewer.")
                return 1

            cam_pos = gymapi.Vec3(2.2, 1.8, 1.4)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.4)
            gym.viewer_camera_look_at(viewer, env, cam_pos, cam_target)

            print("[OK] Viewer started. Close the window to exit.")
            while not gym.query_viewer_has_closed(viewer):
                gym.simulate(sim)
                gym.fetch_results(sim, True)
                gym.step_graphics(sim)
                gym.draw_viewer(viewer, sim, True)
                gym.sync_frame_time(sim)
        else:
            for _ in range(max(args.steps, 1)):
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
