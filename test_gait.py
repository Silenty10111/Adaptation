#!/usr/bin/env python3
"""Adaptive gait test in Isaac Gym with forward-direction visualisation and grouped locomotion.

Loads the generated URDF, computes the adaptive plan (via plan_gait / adaptive_gait),
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
from typing import Dict, List

import numpy as np


TARGET_PYTHON = "/data/conda/envs/unitree-rl/bin/python"
TARGET_LD_PATH = "/data/conda/envs/unitree-rl/lib"
ASSET_DIR_NAME = "robot_assets"


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
        from adaptive_gait import compute_adaptive_plan
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

def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Smooth Hermite interpolation between 0 and 1, zero-derivative at edges."""
    t = max(0.0, min(1.0, (x - edge0) / max(edge1 - edge0, 1e-9)))
    return t * t * (3.0 - 2.0 * t)


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


def get_body_xy(gym, env, actor, gymapi) -> List[float]:
    states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)
    if states is None or len(states) == 0:
        return [0.0, 0.0]
    p = states["pose"]["p"][0]
    return [float(p["x"]), float(p["y"])]


def get_body_attitude(gym, env, actor, gymapi) -> tuple:
    """Return (roll, pitch, yaw) of the actor's root body in radians."""
    states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)
    if states is None or len(states) == 0:
        return (0.0, 0.0, 0.0)
    r = states["pose"]["r"][0]
    return quat_to_euler(float(r["w"]), float(r["x"]), float(r["y"]), float(r["z"]))


# ---------------------------------------------------------------------------
# Gait control
# ---------------------------------------------------------------------------

def leg_group_phase(leg_id: int, group_a: List[int], group_b: List[int],
                    base_phase: float) -> float:
    if leg_id in group_b:
        return base_phase + float(np.pi)
    if leg_id in group_a:
        return base_phase
    return base_phase


def foot_xy_map(description: dict) -> Dict[int, np.ndarray]:
    mapping: Dict[int, np.ndarray] = {}
    for link in description.get("links", []):
        if link.get("role") != "foot" or link.get("leg_id") is None:
            continue
        origin = np.asarray(link.get("default_world_origin", [0.0, 0.0, 0.0]), dtype=float)
        mapping[int(link["leg_id"])] = origin[:2]
    return mapping


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
) -> np.ndarray:
    targets = defaults.copy()
    phase = 2.0 * np.pi * max(gait_freq, 0.02) * sim_time

    group_a = list(gait_plan.get("topology", {}).get("groups", {}).get("group_a", []))
    group_b = list(gait_plan.get("topology", {}).get("groups", {}).get("group_b", []))

    forward_axis = np.asarray(gait_plan.get("final_forward_axis", [1.0, 0.0]), dtype=float)
    forward_axis = forward_axis / max(float(np.linalg.norm(forward_axis)), 1e-9)

    fmap = foot_xy_map(description)

    for leg_id, joints in triplets.items():
        lg_phase = leg_group_phase(leg_id, group_a, group_b, phase)
        swing_wave = float(np.sin(lg_phase))
        # Smoothstep: sin ∈ [-1,1] → smooth transition ∈ [0,1], zero-derivative at stance↔swing
        swing_alpha = smoothstep(-0.05, 0.05, swing_wave)

        foot_xy_vec = fmap.get(leg_id, np.zeros(2, dtype=float))
        dir_sign = 1.0 if float(np.dot(foot_xy_vec, forward_axis)) >= 0.0 else -1.0

        lift_r = stance_lift + (swing_lift - stance_lift) * swing_alpha
        drop_r = stance_drop + (swing_drop - stance_drop) * swing_alpha
        swing_r = 0.5 + swing_amp * dir_sign * swing_wave

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
    p.add_argument("--description", type=Path,
                   default=Path(ASSET_DIR_NAME) / "robot_description.json")
    p.add_argument("--urdf", type=Path,
                   default=Path(ASSET_DIR_NAME) / "generated_robot.urdf")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--steps", type=int, default=2400)
    p.add_argument("--hold-steps", type=int, default=300,
                   help="Warm-up stabilisation steps with static standing posture before gait begins.")
    p.add_argument("--compute-device-id", type=int, default=0)
    p.add_argument("--graphics-device-id", type=int, default=0)
    p.add_argument("--cpu-sim", action="store_true")
    p.add_argument("--gpu-pipeline", action="store_true",
                   help="Enable GPU rendering pipeline (disabled by default).")
    p.add_argument("--body-height", type=float, default=0.50)
    p.add_argument("--gait-frequency", type=float, default=0.85)
    p.add_argument("--swing-ratio-amplitude", type=float, default=0.26)
    p.add_argument("--swing-lift-ratio", type=float, default=0.78)
    p.add_argument("--stance-lift-ratio", type=float, default=0.05)
    p.add_argument("--swing-drop-ratio", type=float, default=0.38)
    p.add_argument("--stance-drop-ratio", type=float, default=0.90)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    gymapi = load_gymapi()

    description = load_description(args.description)
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
    print(f"  group_a ({len(topo['groups']['group_a'])} legs): {topo['groups']['group_a']}")
    print(f"  group_b ({len(topo['groups']['group_b'])} legs): {topo['groups']['group_b']}")
    if topo["inhibition_rules"]:
        for r in topo["inhibition_rules"]:
            print(f"  [INHIBIT] leg {r['leg_id']}: {r['reason']}")

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
        urdf_path = args.urdf.resolve()
        if not urdf_path.exists():
            print(f"[ERROR] URDF not found: {urdf_path}")
            return 1

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False

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

        # Base defaults
        dof_props["stiffness"].fill(40.0)
        dof_props["damping"].fill(2.0)
        if "effort" in dof_props.dtype.names:
            dof_props["effort"].fill(100.0)
        if "armature" in dof_props.dtype.names:
            dof_props["armature"].fill(0.01)

        # Per-joint overrides — same values as test_gym.py verified standing config
        for idx, name in enumerate(dof_names):
            if "_swing" in name:
                dof_props["stiffness"][idx] = 60.0
                dof_props["damping"][idx] = 3.0
                if "effort" in dof_props.dtype.names:
                    dof_props["effort"][idx] = 150.0
            elif "_drop" in name:
                dof_props["stiffness"][idx] = 80.0
                dof_props["damping"][idx] = 4.0
                if "effort" in dof_props.dtype.names:
                    dof_props["effort"][idx] = 200.0
            elif "_lift" in name:
                dof_props["stiffness"][idx] = 50.0
                dof_props["damping"][idx] = 2.5
                if "effort" in dof_props.dtype.names:
                    dof_props["effort"][idx] = 120.0

        gym.set_actor_dof_properties(env, actor, dof_props)

        lower = np.asarray(dof_props["lower"], dtype=np.float32)
        upper = np.asarray(dof_props["upper"], dtype=np.float32)
        default_targets = 0.5 * (
            np.where(np.isfinite(lower), lower, -0.5)
            + np.where(np.isfinite(upper), upper, 0.5)
        ).astype(np.float32)

        triplets = resolve_joint_triplets(gym, env, actor, description)

        # ---- Build standing posture targets (legs angled down to ground) ----
        stand_targets = default_targets.copy()
        for leg_id, joints in triplets.items():
            stand_targets[joints["lift_idx"]] = ratio_to_joint(
                joints["lift_lower"], joints["lift_upper"], 0.05,
            )
            stand_targets[joints["drop_idx"]] = ratio_to_joint(
                joints["drop_lower"], joints["drop_upper"], 0.995,
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

        print(f"[OK] Asset loaded: {dof_count} DOFs, "
              f"{gym.get_asset_rigid_body_count(asset)} bodies")
        print(f"[OK] Standing at body height {args.body_height:.2f} m")

        # ---- Warm-up: static hold-stabilisation ------------------------------
        hold_steps = max(args.hold_steps, 0)
        if hold_steps > 0:
            print(f"[HOLD] Stabilising for {hold_steps} steps ...")
            # Adaptive-sag controller state (reuses test_gym.py logic)
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

        print(f"[OK] Gait: {args.gait_frequency} Hz, "
              f"groups A/B phase-offset = π")

        # ---- Simulation loop -------------------------------------------------
        forward_axis = gait_plan.get("final_forward_axis", [1.0, 0.0])
        group_a = topo["groups"]["group_a"]
        group_b = topo["groups"]["group_b"]

        # Foot positions in world frame from description (offsets from body)
        fmap = foot_xy_map(description)

        com_trail: List[List[float]] = []
        sim_time = 0.0
        dt = sim_params.dt

        if not args.headless:
            viewer = gym.create_viewer(sim, gymapi.CameraProperties())
            if viewer is None:
                print("[ERROR] Failed to create viewer.")
                return 1
            cam_pos = gymapi.Vec3(2.4, 1.8, 1.4)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.35)
            gym.viewer_camera_look_at(viewer, env, cam_pos, cam_target)

            heading_deg = math.degrees(math.atan2(forward_axis[1], forward_axis[0]))
            print(f"[OK] Viewer open — heading = {heading_deg:.1f}° — close window to exit.")
            frame_count = 0
            while not gym.query_viewer_has_closed(viewer):
                gym.clear_lines(viewer)
                frame_count += 1

                targets = build_gait_targets(
                    description, gait_plan, triplets, stand_targets.copy(), sim_time,
                    args.gait_frequency, args.swing_ratio_amplitude,
                    args.stance_lift_ratio, args.swing_lift_ratio,
                    args.stance_drop_ratio, args.swing_drop_ratio,
                )
                gym.set_actor_dof_position_targets(env, actor, targets)
                gym.simulate(sim)
                gym.fetch_results(sim, True)
                gym.step_graphics(sim)

                body_xy = get_body_xy(gym, env, actor, gymapi)

                # 1. Forward-direction arrow (orange, 1.5 m)
                draw_arrow(gym, viewer, env, gymapi, body_xy, forward_axis,
                           length=1.5, z=0.012, rgb=[1.0, 0.45, 0.0])

                # 2. Per-leg foot-position markers coloured by group & phase
                phase_now = 2.0 * np.pi * args.gait_frequency * sim_time
                for leg_id, joints in triplets.items():
                    lg_ph = leg_group_phase(leg_id, group_a, group_b, phase_now)
                    in_swing = float(np.sin(lg_ph)) > 0.0
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

                # 5. Heading angle print every 200 frames
                if frame_count % 200 == 1:
                    hdg = math.degrees(math.atan2(forward_axis[1], forward_axis[0]))
                    print(f"[{frame_count:5d}] heading = {hdg:.1f}°  "
                          f"body_xy = [{body_xy[0]:.3f}, {body_xy[1]:.3f}]")

                gym.draw_viewer(viewer, sim, True)
                gym.sync_frame_time(sim)
                sim_time += dt
        else:
            for step in range(max(args.steps, 1)):
                targets = build_gait_targets(
                    description, gait_plan, triplets, stand_targets.copy(), sim_time,
                    args.gait_frequency, args.swing_ratio_amplitude,
                    args.stance_lift_ratio, args.swing_lift_ratio,
                    args.stance_drop_ratio, args.swing_drop_ratio,
                )
                gym.set_actor_dof_position_targets(env, actor, targets)
                gym.simulate(sim)
                gym.fetch_results(sim, True)

                body_xy = get_body_xy(gym, env, actor, gymapi)
                roll, pitch, _ = get_body_attitude(gym, env, actor, gymapi)
                com_trail.append(body_xy)

                if (step + 1) % 400 == 0:
                    print(f"[{step + 1:5d}/{args.steps}] "
                          f"body_xy = [{body_xy[0]:.4f}, {body_xy[1]:.4f}]  "
                          f"roll = {math.degrees(roll):.1f}°  "
                          f"pitch = {math.degrees(pitch):.1f}°")

                sim_time += dt

            # Headless summary
            trail = np.array(com_trail, dtype=float)
            if len(trail) > 1:
                displacement = trail[-1] - trail[0]
                forward = np.array(forward_axis, dtype=float)
                forward = forward / max(float(np.linalg.norm(forward)), 1e-9)
                fwd_dist = float(np.dot(displacement, forward))
                lat_dist = float(np.dot(
                    displacement, np.array([-forward[1], forward[0]], dtype=float),
                ))
                print(f"\n[Motion summary after {args.steps} steps]")
                print(f"  start       = {trail[0].tolist()}")
                print(f"  end         = {trail[-1].tolist()}")
                print(f"  forward dist  = {fwd_dist:+.4f} m")
                print(f"  lateral drift = {lat_dist:+.4f} m")

        print("[OK] Simulation finished.")
        return 0

    finally:
        if viewer is not None:
            gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)


if __name__ == "__main__":
    raise SystemExit(main())
