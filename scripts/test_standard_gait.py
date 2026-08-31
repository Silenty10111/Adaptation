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
from typing import Dict, List

import numpy as np

from adaptation.phase import leg_phase_state, resolve_duty_factors, resolve_phase_offsets


TARGET_PYTHON = "/data/conda/envs/unitree-rl/bin/python"
TARGET_LD_PATH = "/data/conda/envs/unitree-rl/lib"
ASSET_DIR_NAME = "robot_assets/standard_hexapod"


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


def get_body_pose(gym, env, actor, gymapi):
    """Return (x, y, z) position of root body."""
    states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)
    if states is None or len(states) == 0:
        return (0.0, 0.0, 0.0)
    p = states["pose"]["p"][0]
    return (float(p["x"]), float(p["y"]), float(p["z"]))


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

def diagnose_gait(frame_count: int, triplets: Dict[int, dict], gait_plan: dict, 
                  phase: float, group_a: List[int], group_b: List[int],
                  dof_states = None, dof_names = None) -> None:
    """诊断当前帧的步态执行状态，对比规划 vs 实际。"""
    if frame_count % 120 != 0:
        return
    
    topo = gait_plan.get("topology", {})
    group_c = topo.get("groups", {}).get("group_c", [])
    sw_proj = topo.get("swing_projections", {})
    phase_offsets = resolve_phase_offsets(gait_plan, triplets)
    duty_factors, _ = resolve_duty_factors(gait_plan, triplets)

    print(f"\n[Gait Diag @ frame {frame_count}] 相位={phase/(2*np.pi):.2f} cycles")
    print(f"  group_a (摆腿): {group_a}  group_b (摆腿): {group_b}  group_c (静止): {group_c}")
    
    for leg_id in sorted(triplets.keys()):
        proj = float(sw_proj.get(str(leg_id), 1.0))
        if leg_id in group_c:
            label = "C(passive)"
            state = "静"
            actual_str = ""
            if dof_states is not None and dof_names is not None:
                triplet = triplets[leg_id]
                swing_pos = dof_states["pos"][triplet["swing_idx"]]
                lift_pos  = dof_states["pos"][triplet["lift_idx"]]
                actual_str = f" [实际: swing={swing_pos:.3f}, lift={lift_pos:.3f}]"
            print(f"    leg_{leg_id}: group_{label}, proj={proj:+.2f}{actual_str}")
            continue

        leg_state = leg_phase_state(phase, leg_id, phase_offsets, duty_factors)
        swing_alpha = leg_state.lift
        in_swing = not leg_state.is_stance
        state = "摆" if in_swing else "支"
        group = "A" if leg_id in group_a else "B"
        
        actual_str = ""
        if dof_states is not None and dof_names is not None:
            triplet = triplets[leg_id]
            swing_pos = dof_states["pos"][triplet["swing_idx"]]
            lift_pos = dof_states["pos"][triplet["lift_idx"]]
            actual_str = f" [实际: swing={swing_pos:.3f}, lift={lift_pos:.3f}]"
        
        print(f"    leg_{leg_id}: group_{group}, proj={proj:+.2f}, phase={leg_state.phase/(np.pi):.2f}π, " +
              f"α={swing_alpha:.3f}, state={state}{actual_str}")


def foot_xy_map(description: dict) -> Dict[int, np.ndarray]:
    mapping: Dict[int, np.ndarray] = {}
    for link in description.get("links", []):
        if link.get("role") != "foot" or link.get("leg_id") is None:
            continue
        origin = np.asarray(link.get("default_world_origin", [0.0, 0.0, 0.0]), dtype=float)
        mapping[int(link["leg_id"])] = origin[:2]
    return mapping


def foot_body_map(gym, env, actor, description: dict) -> Dict[int, int]:
    names = gym.get_actor_rigid_body_names(env, actor)
    name_to_idx = {n: i for i, n in enumerate(names)}
    mapping: Dict[int, int] = {}
    for leg_id in range(int(description.get("num_legs", 0))):
        # Fixed joints may be collapsed, so foot link can be missing.
        for body_name in (
            f"leg_{leg_id}_foot",
            f"leg_{leg_id}_lower",
            f"leg_{leg_id}_knee",
        ):
            if body_name in name_to_idx:
                mapping[leg_id] = int(name_to_idx[body_name])
                break
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
    gym=None,
    env=None,
    actor=None,
    dof_props=None,
) -> np.ndarray:
    targets = defaults.copy()
    phase = 2.0 * np.pi * max(gait_freq, 0.02) * sim_time

    topo = gait_plan.get("topology", {})
    group_a = list(topo.get("groups", {}).get("group_a", []))
    group_b = list(topo.get("groups", {}).get("group_b", []))
    # group_c: passive legs with low forward-projection contribution (just stay neutral)
    group_c = list(topo.get("groups", {}).get("group_c", []))
    phase_offsets = resolve_phase_offsets(gait_plan, triplets)
    duty_factors, duty_diagnostics = resolve_duty_factors(gait_plan, triplets)
    if duty_diagnostics and not getattr(build_gait_targets, "_reported_duty_clips", False):
        for diagnostic in duty_diagnostics:
            print(f"[Gait] {diagnostic}")
        build_gait_targets._reported_duty_clips = True
    # Per-leg swing projection onto forward axis (0.0–1.0)
    swing_projections: dict = topo.get("swing_projections", {})

    impedance_cfg = gait_plan.get("impedance", {})
    # Impedance control: blend stiffness between swing (low) and stance (high).
    # Values come FROM gait_plan["impedance"] (adaptation.gait defaults: swing=80, stance=180).
    # Passive compliance: drop (knee) joint uses a LOWER stance target than lift/swing.
    # lift/swing: swing_kp → stance_kp  (full range, structural support)
    # drop:       swing_kp → compliance_kp  (reduced range = softer knee on ground)
    swing_kp_all = float(impedance_cfg.get("swing_kp",   80.0))
    swing_kd_all = float(impedance_cfg.get("swing_kd",    6.0))
    stance_kp_all = float(impedance_cfg.get("stance_kp", 180.0))
    stance_kd_all = float(impedance_cfg.get("stance_kd",  12.0))
    # Drop joint compliance: slightly below stance_kp (default = 60% of range)
    stance_compliance_kp = float(impedance_cfg.get("stance_compliance_kp",
                                                    swing_kp_all + 0.60 * (stance_kp_all - swing_kp_all)))
    stance_compliance_kd = float(impedance_cfg.get("stance_compliance_kd", stance_kd_all))
    # Passive group (group_c): hold at moderate stiffness
    passive_kp = float(impedance_cfg.get("passive_kp", stance_kp_all))
    passive_kd = float(impedance_cfg.get("passive_kd", stance_kd_all))

    forward_axis = np.asarray(gait_plan.get("final_forward_axis", [1.0, 0.0]), dtype=float)
    forward_axis = forward_axis / max(float(np.linalg.norm(forward_axis)), 1e-9)

    fmap = foot_xy_map(description)

    # Lateral axis: 90° CCW from forward axis
    lateral_axis = np.array([-forward_axis[1], forward_axis[0]], dtype=float)

    need_update_props = (gym is not None and env is not None and
                        actor is not None and dof_props is not None)
    if need_update_props:
        updated_props = {name: dof_props[name].copy() for name in dof_props.dtype.names}

    for leg_id, joints in triplets.items():
        foot_xy_vec = fmap.get(leg_id, np.zeros(2, dtype=float))
        # Use lateral position (perpendicular to forward) to decide swing rotation sense.
        # +lateral (left side): negative Z-rotation swings foot forward → dir_sign = -1
        # -lateral (right side): positive Z-rotation swings foot forward → dir_sign = +1
        lateral_pos = float(np.dot(foot_xy_vec, lateral_axis))
        dir_sign = -1.0 if lateral_pos > 0.0 else 1.0

        # ---- PASSIVE group (group_c): low forward projection, stay in neutral ----
        if leg_id in group_c:
            # Neutral swing, slightly raised lift, full drop → stays on ground quietly
            targets[joints["lift_idx"]] = ratio_to_joint(
                joints["lift_lower"], joints["lift_upper"], 0.06,
            )
            targets[joints["drop_idx"]] = ratio_to_joint(
                joints["drop_lower"], joints["drop_upper"], 0.990,
            )
            targets[joints["swing_idx"]] = ratio_to_joint(
                joints["swing_lower"], joints["swing_upper"], 0.50,
            )
            if need_update_props:
                for idx in (joints["lift_idx"], joints["drop_idx"], joints["swing_idx"]):
                    updated_props["stiffness"][idx] = float(passive_kp)
                    updated_props["damping"][idx]   = float(passive_kd)
            continue

        # ---- ACTIVE legs: shared per-leg phase and true duty schedule ------------
        leg_state = leg_phase_state(phase, leg_id, phase_offsets, duty_factors)
        swing_wave = leg_state.fore_aft
        swing_alpha = leg_state.lift

        # Modification 2: project swing amplitude onto per-leg stride axis.
        # swing_projection ≈ 1.0 for laterally-mounted legs,
        #                  ≈ cos(θ) for diagonally-mounted legs,
        #                  ≈ 0      for forward-pointing legs (already in group_c).
        proj = float(swing_projections.get(str(leg_id), 1.0))
        # Scale amplitude by projection, so legs with less forward capacity take
        # smaller steps → robot walks straighter.
        effective_amp = swing_amp * max(proj, 0.0)

        lift_r = stance_lift + (swing_lift - stance_lift) * swing_alpha
        drop_r = stance_drop + (swing_drop - stance_drop) * swing_alpha
        swing_r = 0.5 + effective_amp * dir_sign * swing_wave

        targets[joints["lift_idx"]] = ratio_to_joint(
            joints["lift_lower"], joints["lift_upper"], lift_r,
        )
        targets[joints["drop_idx"]] = ratio_to_joint(
            joints["drop_lower"], joints["drop_upper"], drop_r,
        )
        targets[joints["swing_idx"]] = ratio_to_joint(
            joints["swing_lower"], joints["swing_upper"], swing_r,
        )

        # Modification 1: Smooth passive compliance via blend of swing_alpha.
        # Only the DROP (knee) joint has reduced stance stiffness.
        # lift/swing: blend swing_kp_all → stance_kp_all
        # drop (knee): blend swing_kp_all → stance_compliance_kp (softer)
        if need_update_props:
            t = 1.0 - swing_alpha  # 0 = full swing, 1 = full stance
            kp_ls = swing_kp_all + (stance_kp_all        - swing_kp_all) * t
            kp_d  = swing_kp_all + (stance_compliance_kp - swing_kp_all) * t
            kd_ls = swing_kd_all + (stance_kd_all        - swing_kd_all) * t
            kd_d  = swing_kd_all + (stance_compliance_kd - swing_kd_all) * t
            updated_props["stiffness"][joints["lift_idx"]]  = float(kp_ls)
            updated_props["damping"][joints["lift_idx"]]    = float(kd_ls)
            updated_props["stiffness"][joints["drop_idx"]]  = float(kp_d)
            updated_props["damping"][joints["drop_idx"]]    = float(kd_d)
            updated_props["stiffness"][joints["swing_idx"]] = float(kp_ls)
            updated_props["damping"][joints["swing_idx"]]   = float(kd_ls)

    if need_update_props:
        for k, v in updated_props.items():
            dof_props[k][:] = v
        gym.set_actor_dof_properties(env, actor, dof_props)

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
    p.add_argument("--stance-lift-ratio", type=float, default=0.46)
    p.add_argument("--swing-drop-ratio", type=float, default=0.38)
    p.add_argument("--stance-drop-ratio", type=float, default=0.08)
    p.add_argument("--metrics-out", type=Path, default=None,
                   help="如果指定，将一个 JSON 指标文件写出到此路径（供 search_morphology.py 读取）。")
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
    print(f"  group_c ({len(topo['groups'].get('group_c', []))} passive legs): {topo['groups'].get('group_c', [])}")
    # Print per-leg swing projections
    sw_proj = topo.get("swing_projections", {})
    if sw_proj:
        print("  swing_projections (forward contribution per leg):")
        for lid_s in sorted(sw_proj, key=lambda x: int(x)):
            print(f"    leg_{lid_s}: {sw_proj[lid_s]:+.3f}")
    if topo["inhibition_rules"]:
        for r in topo["inhibition_rules"]:
            print(f"  [INHIBIT] leg {r['leg_id']}: {r['reason']}")
    print()
    print("[CPG] CPG 配置:")
    cpg_cfg = gait_plan.get("cpg", {})
    if cpg_cfg:
        print(f"  frequency_hz: {cpg_cfg.get('frequency_hz')}")
        print(f"  active_leg_ids: {cpg_cfg.get('active_leg_ids')}")
        print(f"  phase_offsets: {cpg_cfg.get('phase_offsets')}")

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

        # 增强的 PID 增益：支撑相需要更高刚度来抵抗重力，摆动相需要响应速度
        # 静态稳定性优化：drop 关节刚度最高（维持足端压地），lift 次之，swing 最低（柔顺误差）
        dof_props["stiffness"].fill(300.0)
        dof_props["damping"].fill(30.0)
        if "effort" in dof_props.dtype.names:
            dof_props["effort"].fill(1200.0)
        if "armature" in dof_props.dtype.names:
            dof_props["armature"].fill(0.01)

        for idx, name in enumerate(dof_names):
            if "_swing" in name:
                # swing 关节：较低刚度允许横向柔顺（类似阻抗控制）
                dof_props["stiffness"][idx] = 120.0
                dof_props["damping"][idx] = 15.0
                if "effort" in dof_props.dtype.names:
                    dof_props["effort"][idx] = 1200.0
            elif "_drop" in name:
                # drop 关节：高刚度维持足端稳定接触
                dof_props["stiffness"][idx] = 400.0
                dof_props["damping"][idx] = 40.0
                if "effort" in dof_props.dtype.names:
                    dof_props["effort"][idx] = 300.0
            elif "_lift" in name:
                # lift 关节：较高刚度维持身体高度
                dof_props["stiffness"][idx] = 300.0
                dof_props["damping"][idx] = 30.0
                if "effort" in dof_props.dtype.names:
                    dof_props["effort"][idx] = 1200.0

        gym.set_actor_dof_properties(env, actor, dof_props)

        print(f"\n[DOF] 关节顺序 ({dof_count} 个):")
        for idx, name in enumerate(dof_names):
            print(f"  [{idx:2d}] {name}")

        lower = np.asarray(dof_props["lower"], dtype=np.float32)
        upper = np.asarray(dof_props["upper"], dtype=np.float32)
        default_targets = 0.5 * (
            np.where(np.isfinite(lower), lower, -0.5)
            + np.where(np.isfinite(upper), upper, 0.5)
        ).astype(np.float32)

        triplets = resolve_joint_triplets(gym, env, actor, description)

        print(f"\n[Triplets] 成功映射的腿数: {len(triplets)}/{int(description.get('num_legs', 0))}")
        for leg_id in sorted(triplets.keys()):
            triplet = triplets[leg_id]
            print(f"  leg_{leg_id}: lift_idx={triplet['lift_idx']:2d} ({dof_names[triplet['lift_idx']]}), " +
                  f"swing_idx={triplet['swing_idx']:2d} ({dof_names[triplet['swing_idx']]}), " +
                  f"drop_idx={triplet['drop_idx']:2d} ({dof_names[triplet['drop_idx']]})")

        # ---- Build standing posture targets (legs angled down to ground) ----
        # 对新的扁平蜘蛛几何：大腿方向近水平（upper_z_weight=0.08），
        # 关节 0 rad 时足端已在设计位置（z≈-0.44m）。
        # lift_ratio=0.46 → lift≈0 rad（维持大腿设计角度）
        # drop_ratio=0.08 → drop≈-0.10 rad（小腿自然下伸角）
        # 旧参数 lift=0.05/drop=0.995 对应 -0.71/0.99 rad，会把腿折叠到空中导致倒地。
        STAND_LIFT_RATIO = args.stance_lift_ratio   # 使用 CLI 参数
        STAND_DROP_RATIO = args.stance_drop_ratio
        stand_targets = default_targets.copy()
        for leg_id, joints in triplets.items():
            stand_targets[joints["lift_idx"]] = ratio_to_joint(
                joints["lift_lower"], joints["lift_upper"], STAND_LIFT_RATIO,
            )
            stand_targets[joints["drop_idx"]] = ratio_to_joint(
                joints["drop_lower"], joints["drop_upper"], STAND_DROP_RATIO,
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
        group_c = topo["groups"].get("group_c", [])

        # Foot positions in world frame from description (offsets from body)
        fmap = foot_xy_map(description)
        foot_body_ids = foot_body_map(gym, env, actor, description)
        
        # 轨迹记录用于诊断前进方向和侧向漂移
        motion_trail: List[List[float]] = []
        # “平滑度”采样列表（忽略预热阶段）
        roll_samples: List[float] = []
        pitch_samples: List[float] = []
        com_z_samples: List[float] = []
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
                    gym=gym, env=env, actor=actor, dof_props=dof_props,
                )
                
                # 诊断当前步态执行状态
                phase = 2.0 * np.pi * max(args.gait_frequency, 0.02) * sim_time
                dof_states_current = gym.get_actor_dof_states(env, actor, gymapi.STATE_POS)
                diagnose_gait(frame_count, triplets, gait_plan, phase, group_a, group_b, 
                             dof_states_current, dof_names)
                
                gym.set_actor_dof_position_targets(env, actor, targets)
                gym.simulate(sim)
                gym.fetch_results(sim, True)
                gym.step_graphics(sim)

                body_states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)
                body_xy = get_body_xy(gym, env, actor, gymapi)

                # 记录运动轨迹用于诊断
                motion_trail.append(body_xy.copy())
                
                # 计算质心 SSM (简化版：假设质心在身体中心)
                com_xy = np.array(body_xy, dtype=float)
                poly = gait_plan.get("support_polygon_xy", [])
                if len(poly) >= 3:
                    p_arr = np.array(poly, dtype=float) + np.array(body_xy)
                    com_xy_static = np.array(gait_plan.get("projected_com_xy", [0.0, 0.0]))
                    com_xy_world = com_xy
                    
                    # 计算质心到每条支撑多边形边的有符号距离
                    n = len(p_arr)
                    distances = []
                    for i in range(n):
                        a = p_arr[i]
                        b = p_arr[(i + 1) % n]
                        edge = b - a
                        edge_len = np.linalg.norm(edge)
                        if edge_len > 1e-9:
                            signed_dist = (edge[0] * (com_xy_world[1] - a[1]) - 
                                          edge[1] * (com_xy_world[0] - a[0])) / edge_len
                            distances.append(signed_dist)
                    
                    if distances:
                        # 兼容多边形顶点顺逆时针：内部对应所有边距离应同号
                        is_inside = all(d <= 1e-5 for d in distances) or all(d >= -1e-5 for d in distances)
                        ssm_value = min(abs(d) for d in distances) if is_inside else -min(abs(d) for d in distances)
                    else:
                        ssm_value = 0.0
                else:
                    ssm_value = 0.0

                # 1. Forward-direction arrow (orange, 1.5 m)
                draw_arrow(gym, viewer, env, gymapi, body_xy, forward_axis,
                           length=1.5, z=0.012, rgb=[1.0, 0.45, 0.0])

                # 2. Per-leg foot-position markers coloured by group & phase
                phase_now = 2.0 * np.pi * args.gait_frequency * sim_time
                phase_offsets = resolve_phase_offsets(gait_plan, triplets)
                duty_factors, _ = resolve_duty_factors(gait_plan, triplets)
                for leg_id, joints in triplets.items():
                    in_swing = not leg_phase_state(
                        phase_now, leg_id, phase_offsets, duty_factors
                    ).is_stance
                    ft_idx = foot_body_ids.get(leg_id)
                    if body_states is not None and ft_idx is not None and ft_idx < len(body_states):
                        p = body_states["pose"]["p"][ft_idx]
                        fx_w = float(p["x"])
                        fy_w = float(p["y"])
                    else:
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
                          f"body_xy = [{body_xy[0]:.3f}, {body_xy[1]:.3f}]  "
                          f"SSM = {ssm_value:.4f} ({'✓ 稳定' if ssm_value > 0 else '✗ 不稳定'})")

                gym.draw_viewer(viewer, sim, True)
                gym.sync_frame_time(sim)
                sim_time += dt
        
        # 运动轨迹分析
        if len(motion_trail) > 1:
            trail = np.array(motion_trail, dtype=float)
            displacement = trail[-1] - trail[0]
            fwd_axis = np.array(forward_axis, dtype=float)
            fwd_axis = fwd_axis / max(float(np.linalg.norm(fwd_axis)), 1e-9)
            fwd_dist = float(np.dot(displacement, fwd_axis))
            lat_dist = float(np.dot(
                displacement, np.array([-fwd_axis[1], fwd_axis[0]], dtype=float),
            ))
            actual_heading = math.atan2(displacement[1], displacement[0])
            planned_heading = math.atan2(fwd_axis[1], fwd_axis[0])
            heading_error = math.degrees(actual_heading - planned_heading)
            
            print(f"\n[运动轨迹分析]")
            print(f"  预期前进方向: {math.degrees(planned_heading):.1f}°")
            print(f"  实际运动方向: {math.degrees(actual_heading):.1f}°")
            print(f"  偏向角度差: {heading_error:+.1f}°")
            print(f"  前进距离: {fwd_dist:+.4f} m")
            print(f"  侧向漂移: {lat_dist:+.4f} m")
            if abs(heading_error) < 5.0:
                print(f"  ✓ 方向控制良好")
            else:
                print(f"  ✗ 方向偏差过大")
        else:
            for step in range(max(args.steps, 1)):
                targets = build_gait_targets(
                    description, gait_plan, triplets, stand_targets.copy(), sim_time,
                    args.gait_frequency, args.swing_ratio_amplitude,
                    args.stance_lift_ratio, args.swing_lift_ratio,
                    args.stance_drop_ratio, args.swing_drop_ratio,
                    gym=gym, env=env, actor=actor, dof_props=dof_props,
                )
                gym.set_actor_dof_position_targets(env, actor, targets)
                gym.simulate(sim)
                gym.fetch_results(sim, True)

                body_xy = get_body_xy(gym, env, actor, gymapi)
                roll, pitch, _ = get_body_attitude(gym, env, actor, gymapi)
                motion_trail.append(body_xy)

                # 平滑度采样：跳过前 1/4 预热步骤
                if step >= args.steps // 4:
                    _, _, bz = get_body_pose(gym, env, actor, gymapi)
                    roll_samples.append(roll)
                    pitch_samples.append(pitch)
                    com_z_samples.append(bz)

                if (step + 1) % 400 == 0:
                    print(f"[{step + 1:5d}/{args.steps}] "
                          f"body_xy = [{body_xy[0]:.4f}, {body_xy[1]:.4f}]  "
                          f"roll = {math.degrees(roll):.1f}°  "
                          f"pitch = {math.degrees(pitch):.1f}°")

                sim_time += dt
            # Headless summary
            trail = np.array(motion_trail, dtype=float)
            if len(trail) > 1:
                displacement = trail[-1] - trail[0]
                forward = np.array(forward_axis, dtype=float)
                forward = forward / max(float(np.linalg.norm(forward)), 1e-9)
                fwd_dist = float(np.dot(displacement, forward))
                lat_dist = float(np.dot(
                    displacement, np.array([-forward[1], forward[0]], dtype=float),
                ))
                actual_heading = math.atan2(displacement[1], displacement[0])
                planned_heading = math.atan2(forward[1], forward[0])
                heading_error = math.degrees(actual_heading - planned_heading)
                print(f"\n[运动轨迹分析 after {args.steps} steps]")
                print(f"  start           = {trail[0].tolist()}")
                print(f"  end             = {trail[-1].tolist()}")
                print(f"  预期方向: {math.degrees(planned_heading):.1f}°  实际方向: {math.degrees(actual_heading):.1f}°")
                print(f"  方向偏差: {heading_error:+.1f}°")
                print(f"  前进距离: {fwd_dist:+.4f} m")
                print(f"  侧向漂移: {lat_dist:+.4f} m")
                print(f"  {'✓ 方向控制良好' if abs(heading_error) < 5.0 else '✗ 方向偏差过大'}")

        # ---- 指标计算与写出 ---------------------------------------------------
        if roll_samples:
            roll_arr  = np.array(roll_samples, dtype=float)
            pitch_arr = np.array(pitch_samples, dtype=float)
            z_arr     = np.array(com_z_samples, dtype=float)
            roll_rmse  = float(np.sqrt(np.mean(roll_arr ** 2)))
            pitch_rmse = float(np.sqrt(np.mean(pitch_arr ** 2)))
            com_z_var  = float(np.var(z_arr))
            com_z_mean = float(np.mean(z_arr))
            print(f"\n[指标]躯干平滑度统计（后 {len(roll_samples)} 帧）")
            print(f"  roll_rmse  = {math.degrees(roll_rmse):.3f}°")
            print(f"  pitch_rmse = {math.degrees(pitch_rmse):.3f}°")
            print(f"  com_z_var  = {com_z_var:.6f} m²")
            print(f"  com_z_mean = {com_z_mean:.4f} m")
        else:
            roll_rmse = pitch_rmse = com_z_var = com_z_mean = 0.0

        if args.metrics_out is not None:
            # 轨迹结果（如果有）
            trail = np.array(motion_trail, dtype=float)
            fwd_dist = lat_dist = heading_error = 0.0
            if len(trail) > 1:
                disp = trail[-1] - trail[0]
                fwd = np.asarray(forward_axis, dtype=float)
                fwd = fwd / max(float(np.linalg.norm(fwd)), 1e-9)
                fwd_dist = float(np.dot(disp, fwd))
                lat_dist = float(np.dot(disp, np.array([-fwd[1], fwd[0]], dtype=float)))
                heading_error = math.degrees(math.atan2(disp[1], disp[0])
                                             - math.atan2(fwd[1], fwd[0]))
            metrics_data = {
                "roll_rmse_rad":  roll_rmse,
                "pitch_rmse_rad": pitch_rmse,
                "com_z_var":      com_z_var,
                "com_z_mean":     com_z_mean,
                "fwd_distance":   fwd_dist,
                "lat_drift":      lat_dist,
                "heading_error_deg": heading_error,
                "steps":          args.steps,
            }
            import json as _json
            Path(args.metrics_out).write_text(
                _json.dumps(metrics_data, indent=2), encoding="utf-8"
            )
            print(f"[指标] 已写出: {args.metrics_out}")

        print("[OK] Simulation finished.")
        return 0

    finally:
        if viewer is not None:
            gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)


if __name__ == "__main__":
    raise SystemExit(main())
