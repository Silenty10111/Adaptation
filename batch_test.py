#!/usr/bin/env python3
"""Batch robot generation + stability check + gait test with trajectory plots.

每次运行会在 OUTPUT_DIR 下新建一个带时间戳的子目录，
对每个机器人生成一张 demo 图（机器人轮廓 + 前进方向直线 + 实际质心轨迹曲线）。

=== 在此处修改批量参数 ====================================================="""

from __future__ import annotations

# ─────────────────────────── 配置区（直接在这里改参数）──────────────────────

# 要生成的机器人数量
NUM_ROBOTS = 500

# 随机种子列表（优先使用此列表；若列表比 NUM_ROBOTS 短则自动补充随机种子）
SEEDS = [7, 42, 137, 256, 512]

# 仿真步数（60 Hz，1200 步 ≈ 20 秒）；作为最短保底步数
SIM_STEPS = 1200

# 最大仿真步数上限（防止无法行走的机器人死循环；7200 步 ≈ 120 秒）
MAX_SIM_STEPS = 7200

# 期望轨迹至少覆盖的前进距离 = MIN_TRAVEL_BODY_LENGTHS × 估算体长
# 达到此目标后仿真提前结束；设为 0 则仅跑 SIM_STEPS 步
MIN_TRAVEL_BODY_LENGTHS = 2.0

# 热身步数
HOLD_STEPS = 300

# 步态参数
GAIT_FREQUENCY       = 0.85
SWING_AMP            = 0.26
SWING_LIFT_RATIO     = 0.78
STANCE_LIFT_RATIO    = 0.05
SWING_DROP_RATIO     = 0.38
STANCE_DROP_RATIO    = 0.90
BODY_HEIGHT          = 0.50

# 静态稳定性阈值（m），小于此值跳过动态仿真
# 提高至 0.03 m 以过滤极不稳定机器人（原 0.0 会导致 4 腿 SSM≈0 机器人
# 进入 PhysX 后立即剧烈翻倒，引发 GPU 内存越界 → 段错误 Exit 139）
SSM_THRESHOLD        = 0.03

# 是否跳过静态稳定性不通过的机器人（True = 跳过不稳定机器人的步态测试）
SKIP_UNSTABLE        = True

# 是否将标准六足（robot_assets/standard_hexapod）作为额外参照机器人加入测试
INCLUDE_STANDARD_HEXAPOD = True

# GPU 加速 — 使用 GPU 物理仿真（需要 NVIDIA GPU + Isaac Gym GPU 版本）
USE_GPU = True

# GPU 并行批量大小 — 每次并行仿真的机器人数
# 最终仿真阶段将机器人按此大小分批，在一个 GPU sim 里同时跑
GPU_BATCH_SIZE = 8

# 输出根目录
OUTPUT_DIR           = "batch_results"

# ── 在线 EKF 偏航修正 ──────────────────────────────────────────────────────
# 启用后，对 baseline 偏航较高的机器人自动尝试 EKF 在线估计修正
USE_ONLINE_EKF       = True
# 触发 EKF 的漂移比阈值 (|lat|/|fwd|)，超过此值启用在线修正
EKF_DRIFT_THRESHOLD  = 0.30
# EKF 仿真步数
EKF_PROBE_STEPS      = 480

# ─────────────────────────── 以下无需修改 ────────────────────────────────────

import os
import sys
from pathlib import Path

TARGET_PYTHON  = os.environ.get("ISAAC_PYTHON", "/data/conda/envs/unitree-rl/bin/python")
TARGET_LD_PATH = os.environ.get("ISAAC_LD_LIBRARY_PATH", "/data/conda/envs/unitree-rl/lib")
REPO_ROOT      = Path(__file__).resolve().parent
GEN_PYTHON     = os.environ.get("ADAPTATION_PYTHON", "/data/conda/envs/Adaptation/bin/python")
GEN_SCRIPT     = REPO_ROOT / "generate_geometry.py"
GEN_URDF       = REPO_ROOT / "generate_urdf.py"
ASSET_ROOT     = REPO_ROOT / "robot_assets"

# ---------------------------------------------------------------------------
# Auto re-exec into unitree-rl environment (must happen before numpy import)
# ---------------------------------------------------------------------------

sys.path.insert(0, str(REPO_ROOT))
from reexec import maybe_reexec

maybe_reexec(TARGET_PYTHON, TARGET_LD_PATH)

# ─── After re-exec (unitree-rl environment) ─────────────────────────────────

import json
import math
import random
import shutil
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Isaac Gym imports (available after re-exec)
# ---------------------------------------------------------------------------

try:
    from isaacgym import gymapi  as _gymapi_mod  # noqa: F401 — ensure importable
    _GYM_AVAILABLE = True
except Exception:
    _GYM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Robot generation helpers
# ---------------------------------------------------------------------------

def generate_robot(seed: int, robot_dir: Path) -> Tuple[Path, Path]:
    """Generate geometry + URDF for one robot into *robot_dir*.

    Returns (description_path, urdf_path).
    """
    robot_dir.mkdir(parents=True, exist_ok=True)
    desc_src  = ASSET_ROOT / "robot_description.json"
    urdf_src  = ASSET_ROOT / "generated_robot.urdf"
    meshes_src = ASSET_ROOT / "meshes"

    # Step 1 — geometry
    cmd_geo = [
        GEN_PYTHON, str(GEN_SCRIPT),
        "--robot-name", f"robot_seed{seed}",
        "--seed", str(seed),
        "--leg-placement", "random",
    ]
    result = subprocess.run(cmd_geo, capture_output=True, text=True, cwd=str(REPO_ROOT))
    if result.returncode != 0 or not desc_src.exists():
        raise RuntimeError(f"generate_geometry failed (seed={seed}):\n{result.stderr[:400]}")

    # Step 2 — URDF
    desc_dst = robot_dir / "robot_description.json"
    urdf_dst = robot_dir / "robot.urdf"
    # Copy description first so generate_urdf reads the right file and
    # the URDF mesh paths are relative to the same directory.
    shutil.copy2(desc_src, desc_dst)
    # Copy meshes
    meshes_dst = robot_dir / "meshes"
    if meshes_src.exists():
        if meshes_dst.exists():
            shutil.rmtree(meshes_dst)
        shutil.copytree(meshes_src, meshes_dst)

    cmd_urdf = [
        GEN_PYTHON, str(GEN_URDF),
        "--description", str(desc_dst),
        "--output", str(urdf_dst),
    ]
    result2 = subprocess.run(cmd_urdf, capture_output=True, text=True, cwd=str(REPO_ROOT))
    if result2.returncode != 0:
        raise RuntimeError(f"generate_urdf failed (seed={seed}):\n{result2.stderr[:400]}")

    # Reload updated description (generate_urdf may patch urdf_path field)
    desc_dst_refreshed = robot_dir / "robot_description.json"
    return desc_dst_refreshed, urdf_dst


# ---------------------------------------------------------------------------
# Static stability check
# ---------------------------------------------------------------------------

def check_stability(description: dict) -> dict:
    from stability import evaluate_ssm
    return evaluate_ssm(description, threshold=SSM_THRESHOLD)


# ---------------------------------------------------------------------------
# Gait simulation — returns com_trail and forward_axis
# ---------------------------------------------------------------------------

def run_gait_sim(
    description: dict,
    urdf_path: Path,
    gait_plan_override: Optional[dict] = None,
    n_max_steps: Optional[int] = None,
    return_yaw_stats: bool = False,
    use_gpu: Optional[bool] = None,
) -> Tuple:
    """Run headless gait simulation.

    Parameters
    ----------
    gait_plan_override : if provided, use this plan instead of calling
        compute_adaptive_plan.  May contain a ``_per_amp_override`` key to
        replace the plan's per_leg_stride_amplitudes.
    n_max_steps : cap on simulation steps (default: max(MAX_SIM_STEPS, SIM_STEPS)).
        Set to a small value (e.g. 360) for a quick probe run.
    return_yaw_stats : if True, also return a dict with yaw-rate statistics
        measured during the simulation.

    Returns
    -------
    (com_trail, forward_axis) normally, or
    (com_trail, forward_axis, yaw_stats) when return_yaw_stats=True.
    """
    from isaacgym import gymapi
    _use_gpu = USE_GPU if use_gpu is None else use_gpu

    # Inline minimal versions of test_gait helpers
    def _load_plan(desc):
        from adaptive_gait import compute_adaptive_plan
        return compute_adaptive_plan(desc, {})

    def _ratio_to_joint(lo, hi, r):
        return float(lo + max(0.0, min(1.0, r)) * (hi - lo))

    def _smoothstep(e0, e1, x):
        t = max(0.0, min(1.0, (x - e0) / max(e1 - e0, 1e-9)))
        return t * t * (3.0 - 2.0 * t)

    def _quat_to_euler(w, x, y, z):
        sinr = 2.0 * (w * x + y * z)
        cosr = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr, cosr)
        sinp = 2.0 * (w * y - z * x)
        pitch = math.asin(max(-1.0, min(1.0, sinp)))
        return roll, pitch

    gait_plan = gait_plan_override if gait_plan_override is not None else _load_plan(description)
    forward_axis = list(gait_plan.get("final_forward_axis", [1.0, 0.0]))
    topo = gait_plan["topology"]
    group_a = topo["groups"]["group_a"]
    group_b = topo["groups"]["group_b"]
    group_c = topo["groups"].get("group_c", [])

    # ---- sim setup ----------------------------------------------------------
    gym = gymapi.acquire_gym()
    sp = gymapi.SimParams()
    sp.up_axis   = gymapi.UP_AXIS_Z
    sp.gravity   = gymapi.Vec3(0.0, 0.0, -9.81)
    sp.dt        = 1.0 / 60.0
    sp.substeps  = 2
    sp.physx.use_gpu = _use_gpu
    sp.physx.num_position_iterations = 8
    sp.physx.num_velocity_iterations = 2

    _gpu_dev = 0 if _use_gpu else -1
    sim = gym.create_sim(0, _gpu_dev, gymapi.SIM_PHYSX, sp)
    if sim is None:
        return [], forward_axis

    try:
        pp = gymapi.PlaneParams()
        pp.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        pp.static_friction  = 1.8
        pp.dynamic_friction = 1.6
        pp.restitution = 0.0
        gym.add_ground(sim, pp)

        ao = gymapi.AssetOptions()
        ao.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
        ao.fix_base_link = False
        ao.collapse_fixed_joints = True
        urdf_path = urdf_path.resolve()
        asset = gym.load_asset(sim, str(urdf_path.parent), urdf_path.name, ao)
        if asset is None:
            return [], forward_axis

        env  = gym.create_env(sim, gymapi.Vec3(-3, -3, 0), gymapi.Vec3(3, 3, 2), 1)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, BODY_HEIGHT)
        actor = gym.create_actor(env, asset, pose, "r", 0, 1)

        # DOF properties
        dof_props  = gym.get_actor_dof_properties(env, actor)
        dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        dof_props["stiffness"].fill(200.0)
        dof_props["damping"].fill(20.0)
        if "effort"   in dof_props.dtype.names: dof_props["effort"].fill(1000.0)
        if "armature" in dof_props.dtype.names: dof_props["armature"].fill(0.01)
        dof_names = gym.get_asset_dof_names(asset)
        for idx, name in enumerate(dof_names):
            if "_swing" in name:
                dof_props["stiffness"][idx] = 100.0; dof_props["damping"][idx] = 10.0
            elif "_drop" in name:
                dof_props["stiffness"][idx] = 250.0; dof_props["damping"][idx] = 25.0
        gym.set_actor_dof_properties(env, actor, dof_props)

        lower = np.where(np.isfinite(dof_props["lower"]), dof_props["lower"], -0.5).astype(np.float32)
        upper = np.where(np.isfinite(dof_props["upper"]), dof_props["upper"],  0.5).astype(np.float32)
        finite_lo = np.where(np.isfinite(dof_props["lower"]), dof_props["lower"], -1e9)
        finite_hi = np.where(np.isfinite(dof_props["upper"]), dof_props["upper"],  1e9)

        name2idx = {n: i for i, n in enumerate(dof_names)}
        triplets: Dict[int, dict] = {}
        for lid in range(int(description.get("num_legs", 0))):
            ln, sn, dn = f"leg_{lid}_lift", f"leg_{lid}_swing", f"leg_{lid}_drop"
            if ln not in name2idx: continue
            triplets[lid] = {
                "lift_idx":   name2idx[ln],  "swing_idx": name2idx[sn],  "drop_idx":  name2idx[dn],
                "lift_lower": float(lower[name2idx[ln]]),  "lift_upper": float(upper[name2idx[ln]]),
                "swing_lower":float(lower[name2idx[sn]]),  "swing_upper":float(upper[name2idx[sn]]),
                "drop_lower": float(lower[name2idx[dn]]),  "drop_upper": float(upper[name2idx[dn]]),
            }

        # --- stand targets ---
        foot_z_vals = [float(lk["default_world_origin"][2])
                       for lk in description.get("links", [])
                       if lk.get("role") == "foot" and lk.get("leg_id") is not None]
        mean_foot_z = float(np.mean(foot_z_vals)) if foot_z_vals else -0.35
        feet_at_ground = abs(mean_foot_z + BODY_HEIGHT) < 0.12

        stand = 0.5 * (lower + upper).astype(np.float32)
        for lid, j in triplets.items():
            if feet_at_ground:
                lr = max(0.0, min(1.0, (0.0 - j["lift_lower"]) / max(j["lift_upper"] - j["lift_lower"], 1e-9)))
                dr = max(0.0, min(1.0, (0.0 - j["drop_lower"]) / max(j["drop_upper"] - j["drop_lower"], 1e-9)))
            else:
                lr, dr = 0.05, 0.995
            stand[j["lift_idx"]]  = _ratio_to_joint(j["lift_lower"],  j["lift_upper"],  lr)
            stand[j["drop_idx"]]  = _ratio_to_joint(j["drop_lower"],  j["drop_upper"],  dr)
            stand[j["swing_idx"]] = _ratio_to_joint(j["swing_lower"], j["swing_upper"], 0.5)
        stand = np.clip(stand, finite_lo, finite_hi)

        ds = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
        ds["pos"] = stand; ds["vel"].fill(0.0)
        gym.set_actor_dof_states(env, actor, ds, gymapi.STATE_ALL)

        # adaptive lift/drop for this morphology
        if feet_at_ground and triplets:
            fj = next(iter(triplets.values()))
            lr = max(fj["lift_upper"] - fj["lift_lower"], 1e-9)
            dr = max(fj["drop_upper"] - fj["drop_lower"], 1e-9)
            _sl  = max(0.0, min(1.0, (0.0  - fj["lift_lower"]) / lr))
            _swl = max(0.0, min(1.0, (+0.25 - fj["lift_lower"]) / lr))
            # For feet_at_ground robots use the angle=0 drop ratio (keeps knee at neutral).
            # STANCE_DROP_RATIO=0.90 is designed for far-extension robots and causes
            # over-bending here.
            _sd  = max(0.0, min(1.0, (0.0  - fj["drop_lower"]) / dr))
            _swd = _sd
        else:
            _sl, _swl, _sd, _swd = STANCE_LIFT_RATIO, SWING_LIFT_RATIO, STANCE_DROP_RATIO, SWING_DROP_RATIO

        # sag controller warm-up
        for _ in range(max(HOLD_STEPS, 0)):
            ds_pos = gym.get_actor_dof_states(env, actor, gymapi.STATE_POS)
            jpos = np.asarray(ds_pos["pos"], dtype=np.float32)
            for idx, name in enumerate(dof_names):
                sag = stand[idx] - jpos[idx]
                if "_drop" in name and sag > 0.004:
                    stand[idx] = min(finite_hi[idx], stand[idx] + min(0.012, 0.22 * sag))
                elif "_lift" in name and sag > 0.004:
                    stand[idx] = max(finite_lo[idx], stand[idx] - min(0.008, 0.16 * sag))
            stand = np.clip(stand, finite_lo, finite_hi)
            gym.set_actor_dof_position_targets(env, actor, stand)
            gym.simulate(sim); gym.fetch_results(sim, True)

        # ── foot lateral position map ──
        fwd = np.asarray(forward_axis, dtype=float)
        fwd = fwd / max(float(np.linalg.norm(fwd)), 1e-9)
        lat = np.array([-fwd[1], fwd[0]], dtype=float)
        fmap: Dict[int, np.ndarray] = {}
        for lk in description.get("links", []):
            if lk.get("role") == "foot" and lk.get("leg_id") is not None:
                fmap[int(lk["leg_id"])] = np.asarray(lk["default_world_origin"], dtype=float)[:2]

        per_amp = {str(k): float(v) for k, v in topo.get("per_leg_stride_amplitudes", {}).items()}
        # Allow per-amplitude override embedded in the plan dict
        if "_per_amp_override" in gait_plan:
            per_amp = {str(k): float(v) for k, v in gait_plan["_per_amp_override"].items()}
        touchdown_ramp: Dict[int, int] = {}
        com_trail: List[List[float]] = []
        yaw_acc: List[float] = []
        sim_time = 0.0
        dt = sp.dt
        phase_now = 0.0

        # Estimate body length from foot bounding box (max axis extent)
        if fmap:
            foot_pts = np.array(list(fmap.values()), dtype=float)
            body_length = float(np.max(foot_pts.max(axis=0) - foot_pts.min(axis=0)))
        else:
            body_length = 0.5  # fallback
        min_travel = body_length * max(MIN_TRAVEL_BODY_LENGTHS, 0.0) if n_max_steps is None else 0.0
        max_steps  = n_max_steps if n_max_steps is not None else max(MAX_SIM_STEPS, SIM_STEPS)

        for step in range(max(max_steps, 1)):
            phase_now = 2.0 * math.pi * max(GAIT_FREQUENCY, 0.02) * sim_time
            targets = stand.copy()

            for lid, j in triplets.items():
                if lid in group_c:
                    targets[j["lift_idx"]]  = _ratio_to_joint(j["lift_lower"], j["lift_upper"], _sl)
                    targets[j["drop_idx"]]  = _ratio_to_joint(j["drop_lower"], j["drop_upper"], _sd)
                    targets[j["swing_idx"]] = _ratio_to_joint(j["swing_lower"], j["swing_upper"], 0.5)
                    continue

                if   lid in group_b: lg_ph = phase_now + math.pi
                elif lid in group_a: lg_ph = phase_now
                else:                lg_ph = 0.0

                sw = float(math.sin(lg_ph))
                alpha = _smoothstep(-0.30, 0.30, sw)
                foot_v = fmap.get(lid, np.zeros(2, dtype=float))
                lat_p = float(np.dot(foot_v, lat))
                dsign = -1.0 if lat_p > 0.0 else 1.0
                eff_amp = SWING_AMP * float(per_amp.get(str(lid), 1.0))

                lr  = _sl  + (_swl - _sl) * alpha
                dr  = _sd  + (_swd - _sd) * alpha
                sr  = 0.5 + eff_amp * dsign * sw

                # touchdown ramp
                is_sw = sw > 0.0
                if not is_sw:
                    if lid in touchdown_ramp:
                        touchdown_ramp[lid] += 1
                        rp = min(touchdown_ramp[lid] / 8, 1.0)
                        if rp >= 1.0:
                            del touchdown_ramp[lid]
                        else:
                            def_l = _ratio_to_joint(j["lift_lower"], j["lift_upper"], 0.5)
                            def_d = _ratio_to_joint(j["drop_lower"], j["drop_upper"], 0.5)
                            def_s = j["swing_lower"] + 0.5 * (j["swing_upper"] - j["swing_lower"])
                            tl = _ratio_to_joint(j["lift_lower"], j["lift_upper"], lr)
                            td = _ratio_to_joint(j["drop_lower"], j["drop_upper"], dr)
                            ts = _ratio_to_joint(j["swing_lower"], j["swing_upper"], sr)
                            targets[j["lift_idx"]]  = def_l + rp * (tl - def_l)
                            targets[j["drop_idx"]]  = def_d + rp * (td - def_d)
                            targets[j["swing_idx"]] = def_s + rp * (ts - def_s)
                            continue
                else:
                    touchdown_ramp[lid] = 0

                targets[j["lift_idx"]]  = _ratio_to_joint(j["lift_lower"], j["lift_upper"], lr)
                targets[j["drop_idx"]]  = _ratio_to_joint(j["drop_lower"], j["drop_upper"], dr)
                targets[j["swing_idx"]] = _ratio_to_joint(j["swing_lower"], j["swing_upper"], sr)

            gym.set_actor_dof_position_targets(env, actor, targets)
            gym.simulate(sim); gym.fetch_results(sim, True)

            _state_flag = gymapi.STATE_ALL if return_yaw_stats else gymapi.STATE_POS
            states = gym.get_actor_rigid_body_states(env, actor, _state_flag)
            if states is not None and len(states) > 0:
                p = states["pose"]["p"][0]
                com_trail.append([float(p["x"]), float(p["y"])])
                if return_yaw_stats:
                    try:
                        yaw_acc.append(float(states["vel"]["angular"][0]["z"]))
                    except Exception:
                        # Compute yaw-rate numerically from CoM heading change as fallback
                        if len(com_trail) >= 3:
                            _d1 = np.array(com_trail[-1]) - np.array(com_trail[-2])
                            _d0 = np.array(com_trail[-2]) - np.array(com_trail[-3])
                            _a1 = math.atan2(_d1[1], _d1[0])
                            _a0 = math.atan2(_d0[1], _d0[0])
                            _da = ((_a1 - _a0 + math.pi) % (2 * math.pi)) - math.pi
                            yaw_acc.append(_da / max(dt, 1e-9))

            sim_time += dt

            # Early exit: stop once min_travel is covered AND we're past SIM_STEPS
            if step >= SIM_STEPS and min_travel > 0.0 and len(com_trail) > 1:
                trail_arr = np.array(com_trail, dtype=float)
                disp = trail_arr[-1] - trail_arr[0]
                covered = abs(float(np.dot(disp, fwd)))
                if covered >= min_travel:
                    break

        if return_yaw_stats:
            tail = yaw_acc[-60:] if len(yaw_acc) >= 60 else yaw_acc
            yaw_stats = {
                "yaw_rate_mean":     float(np.mean(tail))         if tail else 0.0,
                "yaw_rate_abs_mean": float(np.mean(np.abs(tail))) if tail else 0.0,
            }
            return com_trail, forward_axis, yaw_stats
        return com_trail, forward_axis

    finally:
        gym.destroy_sim(sim)


# ---------------------------------------------------------------------------
# Per-robot iterative gait optimizer
# ---------------------------------------------------------------------------

def _apply_amp_correction(plan: dict, measured_yaw: float, strength: float = 0.55) -> dict:
    """Return a copy of *plan* with per_leg_stride_amplitudes corrected for yaw.

    Physics model
    -------------
    Yaw torque from leg i ≈ base_amp_i × |yaw_lever_i|, where yaw_lever_i is
    the signed perpendicular distance from the CoM to the forward-force line
    (positive = left of forward axis → CCW yaw; negative = right → CW yaw).

    Correction rule (proportional to lateral lever):
        s_i = 1 − strength × sign(measured_yaw) × yaw_lever_i / max_lever
    Legs on the same side as the yaw direction get reduced amplitude;
    legs on the opposite side get increased amplitude.

    This is physically motivated and dimensionally consistent: both
    ``measured_yaw`` sign and ``yaw_lever_i`` sign encode the same geometry.
    """
    yaw_levers = {int(k): float(v)
                  for k, v in plan.get("yaw_balance", {}).get("yaw_levers", {}).items()}
    base_amps  = plan.get("topology", {}).get("per_leg_stride_amplitudes", {})
    plan_copy  = dict(plan)

    if not yaw_levers or not base_amps or abs(measured_yaw) < 1e-6:
        return plan_copy

    max_lever = max(abs(v) for v in yaw_levers.values())
    if max_lever < 1e-9:
        return plan_copy

    yaw_sign = math.copysign(1.0, measured_yaw)
    corrected: Dict[str, float] = {}
    for lid_str, amp in base_amps.items():
        lever = yaw_levers.get(int(lid_str), 0.0)
        # s_i < 1 for legs that feed the current yaw; s_i > 1 for counter-legs
        s = 1.0 - strength * yaw_sign * lever / max_lever
        s = float(np.clip(s, 0.25, 1.75))
        corrected[lid_str] = float(np.clip(float(amp) * s, 0.08, 0.85))

    plan_copy["_per_amp_override"] = corrected
    plan_copy["_yaw_correction_strength"] = strength
    return plan_copy




# ---------------------------------------------------------------------------
# Reusable per-robot sim context — avoids repeated create_sim/destroy_sim
# ---------------------------------------------------------------------------

class _RobotSimCtx:
    """One Isaac Gym simulator per robot, reused across all probe episodes.

    Root cause of segfault (Exit 139):
        Each run_gait_sim call previously did create_sim / destroy_sim.
        After ~100 such cycles in one process, PhysX GPU memory gets
        corrupted → segfault.

    Fix:
        _RobotSimCtx creates the simulator ONCE per robot, runs the sag-
        controller warm-up ONCE, then resets DOF positions + all velocities
        between probe episodes without touching create_sim / destroy_sim.
    """

    def __init__(self, description: dict, urdf_path: Path, use_gpu: bool = True):
        from isaacgym import gymapi as _ga
        self._ga = _ga

        gym = _ga.acquire_gym()
        sp = _ga.SimParams()
        sp.up_axis   = _ga.UP_AXIS_Z
        sp.gravity   = _ga.Vec3(0.0, 0.0, -9.81)
        sp.dt        = 1.0 / 60.0
        sp.substeps  = 2
        sp.physx.use_gpu = use_gpu
        sp.physx.num_position_iterations = 8
        sp.physx.num_velocity_iterations = 2
        gpu_dev = 0 if use_gpu else -1
        sim = gym.create_sim(0, gpu_dev, _ga.SIM_PHYSX, sp)
        if sim is None:
            raise RuntimeError("_RobotSimCtx: create_sim failed")

        pp = _ga.PlaneParams()
        pp.normal = _ga.Vec3(0.0, 0.0, 1.0)
        pp.static_friction = 1.8; pp.dynamic_friction = 1.6; pp.restitution = 0.0
        gym.add_ground(sim, pp)

        ao = _ga.AssetOptions()
        ao.default_dof_drive_mode = int(_ga.DOF_MODE_POS)
        ao.fix_base_link = False
        ao.collapse_fixed_joints = True
        urdf_path = urdf_path.resolve()
        asset = gym.load_asset(sim, str(urdf_path.parent), urdf_path.name, ao)
        if asset is None:
            gym.destroy_sim(sim)
            raise RuntimeError("_RobotSimCtx: load_asset failed")

        env   = gym.create_env(sim, _ga.Vec3(-3, -3, 0), _ga.Vec3(3, 3, 2), 1)
        pose  = _ga.Transform()
        pose.p = _ga.Vec3(0.0, 0.0, BODY_HEIGHT)
        actor = gym.create_actor(env, asset, pose, "r", 0, 1)

        dof_props = gym.get_actor_dof_properties(env, actor)
        dof_props["driveMode"].fill(_ga.DOF_MODE_POS)
        dof_props["stiffness"].fill(200.0)
        dof_props["damping"].fill(20.0)
        if "effort"   in dof_props.dtype.names: dof_props["effort"].fill(1000.0)
        if "armature" in dof_props.dtype.names: dof_props["armature"].fill(0.01)
        dof_names = gym.get_asset_dof_names(asset)
        for idx, name in enumerate(dof_names):
            if "_swing" in name:
                dof_props["stiffness"][idx] = 100.0; dof_props["damping"][idx] = 10.0
            elif "_drop" in name:
                dof_props["stiffness"][idx] = 250.0; dof_props["damping"][idx] = 25.0
        gym.set_actor_dof_properties(env, actor, dof_props)

        lower    = np.where(np.isfinite(dof_props["lower"]), dof_props["lower"], -0.5).astype(np.float32)
        upper    = np.where(np.isfinite(dof_props["upper"]), dof_props["upper"],  0.5).astype(np.float32)
        finite_lo = np.where(np.isfinite(dof_props["lower"]), dof_props["lower"], -1e9)
        finite_hi = np.where(np.isfinite(dof_props["upper"]), dof_props["upper"],  1e9)

        name2idx = {n: i for i, n in enumerate(dof_names)}
        triplets: Dict[int, dict] = {}
        for lid in range(int(description.get("num_legs", 0))):
            ln, sn, dn = f"leg_{lid}_lift", f"leg_{lid}_swing", f"leg_{lid}_drop"
            if ln not in name2idx:
                continue
            triplets[lid] = {
                "lift_idx":    name2idx[ln], "swing_idx": name2idx[sn], "drop_idx":  name2idx[dn],
                "lift_lower":  float(lower[name2idx[ln]]), "lift_upper":  float(upper[name2idx[ln]]),
                "swing_lower": float(lower[name2idx[sn]]), "swing_upper": float(upper[name2idx[sn]]),
                "drop_lower":  float(lower[name2idx[dn]]), "drop_upper":  float(upper[name2idx[dn]]),
            }

        foot_z_vals = [float(lk["default_world_origin"][2])
                       for lk in description.get("links", [])
                       if lk.get("role") == "foot" and lk.get("leg_id") is not None]
        mean_foot_z = float(np.mean(foot_z_vals)) if foot_z_vals else -0.35
        feet_at_ground = abs(mean_foot_z + BODY_HEIGHT) < 0.12

        def _rtj(lo, hi, r):
            return float(lo + max(0.0, min(1.0, r)) * (hi - lo))

        stand = 0.5 * (lower + upper).astype(np.float32)
        for lid, j in triplets.items():
            if feet_at_ground:
                _lr = max(0.0, min(1.0, (0.0 - j["lift_lower"]) / max(j["lift_upper"] - j["lift_lower"], 1e-9)))
                _dr = max(0.0, min(1.0, (0.0 - j["drop_lower"]) / max(j["drop_upper"] - j["drop_lower"], 1e-9)))
            else:
                _lr, _dr = 0.05, 0.995
            stand[j["lift_idx"]]  = _rtj(j["lift_lower"],  j["lift_upper"],  _lr)
            stand[j["drop_idx"]]  = _rtj(j["drop_lower"],  j["drop_upper"],  _dr)
            stand[j["swing_idx"]] = _rtj(j["swing_lower"], j["swing_upper"], 0.5)
        stand = np.clip(stand, finite_lo, finite_hi)

        ds = gym.get_actor_dof_states(env, actor, _ga.STATE_ALL)
        ds["pos"] = stand; ds["vel"].fill(0.0)
        gym.set_actor_dof_states(env, actor, ds, _ga.STATE_ALL)

        if feet_at_ground and triplets:
            fj   = next(iter(triplets.values()))
            _lrr = max(fj["lift_upper"] - fj["lift_lower"], 1e-9)
            _drr = max(fj["drop_upper"] - fj["drop_lower"], 1e-9)
            _sl  = max(0.0, min(1.0, (0.0  - fj["lift_lower"]) / _lrr))
            _swl = max(0.0, min(1.0, (+0.25 - fj["lift_lower"]) / _lrr))
            _sd  = max(0.0, min(1.0, (0.0  - fj["drop_lower"]) / _drr))
            _swd = _sd
        else:
            _sl, _swl, _sd, _swd = STANCE_LIFT_RATIO, SWING_LIFT_RATIO, STANCE_DROP_RATIO, SWING_DROP_RATIO

        # ── Sag-controller warm-up (run ONCE, result cached) ─────────────────
        stand_ev = stand.copy()
        for _ in range(HOLD_STEPS):
            ds_pos = gym.get_actor_dof_states(env, actor, _ga.STATE_POS)
            jpos = np.asarray(ds_pos["pos"], dtype=np.float32)
            se   = stand_ev.copy()
            for idx, name in enumerate(dof_names):
                sag = se[idx] - jpos[idx]
                if "_drop" in name and sag > 0.004:
                    se[idx] = min(finite_hi[idx], se[idx] + min(0.012, 0.22 * sag))
                elif "_lift" in name and sag > 0.004:
                    se[idx] = max(finite_lo[idx], se[idx] - min(0.008, 0.16 * sag))
            stand_ev = np.clip(se, finite_lo, finite_hi)
            gym.set_actor_dof_position_targets(env, actor, stand_ev)
            gym.simulate(sim); gym.fetch_results(sim, True)

        # ── Foot position map ─────────────────────────────────────────────────
        fmap: Dict[int, np.ndarray] = {}
        for lk in description.get("links", []):
            if lk.get("role") == "foot" and lk.get("leg_id") is not None:
                fmap[int(lk["leg_id"])] = np.asarray(lk["default_world_origin"], dtype=float)[:2]
        if fmap:
            foot_pts = np.array(list(fmap.values()), dtype=float)
            body_length = float(np.max(foot_pts.max(axis=0) - foot_pts.min(axis=0)))
        else:
            body_length = 0.5

        self.gym = gym; self.sim = sim; self.env = env; self.actor = actor
        self.dof_names = list(dof_names)
        self.triplets  = triplets
        self.lower = lower; self.upper = upper
        self.finite_lo = finite_lo; self.finite_hi = finite_hi
        self.stand = stand; self.stand_ev = stand_ev.copy()
        self._sl = _sl; self._swl = _swl; self._sd = _sd; self._swd = _swd
        self.fmap = fmap; self.body_length = body_length
        self._closed = False

    # ------------------------------------------------------------------
    def _reset(self):
        """Teleport robot back to origin and zero all velocities."""
        ga = self._ga
        gym, sim, env, actor = self.gym, self.sim, self.env, self.actor

        # Reset DOF positions and velocities
        ds = gym.get_actor_dof_states(env, actor, ga.STATE_ALL)
        ds["pos"] = self.stand_ev
        ds["vel"].fill(0.0)
        gym.set_actor_dof_states(env, actor, ds, ga.STATE_ALL)

        # Reset root body pose + velocity
        try:
            rb = gym.get_actor_rigid_body_states(env, actor, ga.STATE_ALL)
            rb["pose"]["p"][0]["x"] = 0.0
            rb["pose"]["p"][0]["y"] = 0.0
            rb["pose"]["p"][0]["z"] = BODY_HEIGHT
            rb["pose"]["r"][0]["x"] = 0.0
            rb["pose"]["r"][0]["y"] = 0.0
            rb["pose"]["r"][0]["z"] = 0.0
            rb["pose"]["r"][0]["w"] = 1.0
            rb["vel"]["linear"][0]["x"]  = 0.0
            rb["vel"]["linear"][0]["y"]  = 0.0
            rb["vel"]["linear"][0]["z"]  = 0.0
            rb["vel"]["angular"][0]["x"] = 0.0
            rb["vel"]["angular"][0]["y"] = 0.0
            rb["vel"]["angular"][0]["z"] = 0.0
            gym.set_actor_rigid_body_states(env, actor, rb, ga.STATE_ALL)
        except Exception:
            pass   # If root-state setter is unavailable, DOF reset is sufficient

        # Let physics settle for a few steps
        for _ in range(5):
            gym.set_actor_dof_position_targets(env, actor, self.stand_ev)
            gym.simulate(sim); gym.fetch_results(sim, True)

    # ------------------------------------------------------------------
    def run_episode(self, plan: dict, n_steps: int,
                    return_yaw_stats: bool = False,
                    min_travel: float = 0.0,
                    min_steps: int = 0) -> tuple:
        """Reset robot and run one gait episode.

        Returns (com_trail, forward_axis, yaw_stats_or_None).
        min_travel / min_steps: early-exit when both conditions met (for full sims).
        """
        ga = self._ga
        gym, sim, env, actor = self.gym, self.sim, self.env, self.actor

        self._reset()

        forward_axis = list(plan.get("final_forward_axis", [1.0, 0.0]))
        topo    = plan["topology"]
        group_a = topo["groups"]["group_a"]
        group_b = topo["groups"]["group_b"]
        group_c = topo["groups"].get("group_c", [])

        fwd = np.asarray(forward_axis, dtype=float)
        fwd = fwd / max(float(np.linalg.norm(fwd)), 1e-9)
        lat = np.array([-fwd[1], fwd[0]], dtype=float)

        per_amp = {str(k): float(v) for k, v in topo.get("per_leg_stride_amplitudes", {}).items()}
        if "_per_amp_override" in plan:
            per_amp = {str(k): float(v) for k, v in plan["_per_amp_override"].items()}

        def _rtj(lo, hi, r):
            return float(lo + max(0.0, min(1.0, r)) * (hi - lo))
        def _ss(e0, e1, x):
            t = max(0.0, min(1.0, (x - e0) / max(e1 - e0, 1e-9)))
            return t * t * (3.0 - 2.0 * t)

        _sl, _swl, _sd, _swd = self._sl, self._swl, self._sd, self._swd
        com_trail:      List[List[float]] = []
        yaw_acc:        List[float]       = []
        touchdown_ramp: Dict[int, int]    = {}
        sim_time = 0.0
        dt = 1.0 / 60.0

        for step in range(n_steps):
            phase_now = 2.0 * math.pi * GAIT_FREQUENCY * sim_time
            targets   = self.stand_ev.copy()

            for lid, j in self.triplets.items():
                if lid in group_c:
                    targets[j["lift_idx"]]  = _rtj(j["lift_lower"], j["lift_upper"], _sl)
                    targets[j["drop_idx"]]  = _rtj(j["drop_lower"], j["drop_upper"], _sd)
                    targets[j["swing_idx"]] = _rtj(j["swing_lower"], j["swing_upper"], 0.5)
                    continue

                if   lid in group_b: lg_ph = phase_now + math.pi
                elif lid in group_a: lg_ph = phase_now
                else:                lg_ph = 0.0

                sw    = float(math.sin(lg_ph))
                alpha = _ss(-0.30, 0.30, sw)
                fv    = self.fmap.get(lid, np.zeros(2))
                dsign = -1.0 if float(np.dot(fv, lat)) > 0.0 else 1.0
                eff_amp = SWING_AMP * float(per_amp.get(str(lid), 1.0))

                lr = _sl + (_swl - _sl) * alpha
                dr = _sd + (_swd - _sd) * alpha
                sr = 0.5 + eff_amp * dsign * sw

                is_sw = sw > 0.0
                if not is_sw:
                    if lid in touchdown_ramp:
                        touchdown_ramp[lid] += 1
                        rp = min(touchdown_ramp[lid] / 8, 1.0)
                        if rp >= 1.0:
                            del touchdown_ramp[lid]
                        else:
                            dl  = _rtj(j["lift_lower"],  j["lift_upper"],  0.5)
                            dd  = _rtj(j["drop_lower"],  j["drop_upper"],  0.5)
                            ds_ = j["swing_lower"] + 0.5 * (j["swing_upper"] - j["swing_lower"])
                            targets[j["lift_idx"]]  = dl  + rp * (_rtj(j["lift_lower"],  j["lift_upper"],  lr) - dl)
                            targets[j["drop_idx"]]  = dd  + rp * (_rtj(j["drop_lower"],  j["drop_upper"],  dr) - dd)
                            targets[j["swing_idx"]] = ds_ + rp * (_rtj(j["swing_lower"], j["swing_upper"], sr) - ds_)
                            continue
                else:
                    touchdown_ramp[lid] = 0

                targets[j["lift_idx"]]  = _rtj(j["lift_lower"], j["lift_upper"], lr)
                targets[j["drop_idx"]]  = _rtj(j["drop_lower"], j["drop_upper"], dr)
                targets[j["swing_idx"]] = _rtj(j["swing_lower"], j["swing_upper"], sr)

            gym.set_actor_dof_position_targets(env, actor, targets)
            gym.simulate(sim); gym.fetch_results(sim, True)

            _sflag = ga.STATE_ALL if return_yaw_stats else ga.STATE_POS
            states = gym.get_actor_rigid_body_states(env, actor, _sflag)
            if states is not None and len(states) > 0:
                p = states["pose"]["p"][0]
                com_trail.append([float(p["x"]), float(p["y"])])
                if return_yaw_stats:
                    try:
                        yaw_acc.append(float(states["vel"]["angular"][0]["z"]))
                    except Exception:
                        if len(com_trail) >= 3:
                            _d1 = np.array(com_trail[-1]) - np.array(com_trail[-2])
                            _d0 = np.array(com_trail[-2]) - np.array(com_trail[-3])
                            _da = ((math.atan2(_d1[1], _d1[0]) - math.atan2(_d0[1], _d0[0]) + math.pi) % (2 * math.pi)) - math.pi
                            yaw_acc.append(_da / max(dt, 1e-9))

            sim_time += dt

            # Early exit for full-sim mode
            if min_travel > 0.0 and step >= min_steps and len(com_trail) > 1:
                arr = np.array(com_trail, dtype=float)
                covered = abs(float(np.dot(arr[-1] - arr[0], fwd)))
                if covered >= min_travel:
                    break

        yaw_stats = None
        if return_yaw_stats:
            valid  = [v for v in yaw_acc if abs(v) < 20.0]
            mean_y = float(np.mean(valid)) if valid else 0.0
            yaw_stats = {"yaw_rate_mean": mean_y,
                         "yaw_rate_std":  float(np.std(valid)) if valid else 0.0}
        return com_trail, forward_axis, yaw_stats

    # ------------------------------------------------------------------
    def close(self):
        if not self._closed:
            self.gym.destroy_sim(self.sim)
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# Per-robot iterative gait optimizer (reference robot — runs full sim)
# ---------------------------------------------------------------------------

def optimize_and_simulate(
    description: dict, urdf_path: Path,
) -> Tuple[List[List[float]], List[float]]:
    """Probe → correct → full-sim, all inside ONE sim context (no segfault).

    Thin wrapper around probe_and_correct_plan that also runs the full simulation.
    """
    plan, fwd_axis = probe_and_correct_plan(description, urdf_path)
    fwd = np.asarray(fwd_axis, dtype=float)
    fwd = fwd / max(float(np.linalg.norm(fwd)), 1e-9)

    _full_steps = max(MAX_SIM_STEPS, SIM_STEPS)

    with _RobotSimCtx(description, urdf_path, use_gpu=USE_GPU) as ctx:
        _min_travel = ctx.body_length * max(MIN_TRAVEL_BODY_LENGTHS, 0.0)
        trail_f, ax_f, _ = ctx.run_episode(plan, _full_steps,
                                            min_travel=_min_travel, min_steps=SIM_STEPS)
        return trail_f, list(ax_f)


def probe_and_correct_plan(
    description: dict, urdf_path: Path,
) -> Tuple[dict, List[float]]:
    """Probe → correct plan using a single reusable sim (ONE create_sim per robot).

    All probe episodes share one _RobotSimCtx instance.  Between episodes the
    robot is teleported back to origin and all velocities are zeroed — NO
    additional create_sim / destroy_sim calls are made.
    """
    from adaptive_gait import compute_adaptive_plan

    PROBE_STEPS      = 360
    YAW_BAD_THRESH   = 0.18
    FWD_STUCK_THRESH = 0.004
    MAX_YAW_ITERS    = 3
    _dt              = 1.0 / 60.0

    def _metrics(trail, fwd_ax, n_steps):
        if len(trail) < 2 or n_steps == 0:
            return 0.0, 0.0
        arr = np.array(trail, dtype=float)
        disp = arr[-1] - arr[0]
        elapsed = max(n_steps * _dt, _dt)
        return float(np.dot(disp, fwd_ax)) / elapsed, float(np.linalg.norm(disp)) / elapsed

    plan0 = compute_adaptive_plan(description, {})
    fwd0  = np.asarray(plan0["final_forward_axis"], dtype=float)
    fwd0  = fwd0 / max(float(np.linalg.norm(fwd0)), 1e-9)

    with _RobotSimCtx(description, urdf_path, use_gpu=USE_GPU) as ctx:
        # ── Probe 1 ────────────────────────────────────────────────────────
        trail1, _, stats1 = ctx.run_episode(plan0, PROBE_STEPS, return_yaw_stats=True)
        yaw1           = stats1["yaw_rate_mean"]
        fwd_vel1, spd1 = _metrics(trail1, fwd0, len(trail1))
        print(f"  [Probe1] yaw={yaw1:+.3f} rad/s  fwd_vel={fwd_vel1:+.4f} m/s  speed={spd1:.4f} m/s")

        # ── Case A ─────────────────────────────────────────────────────────
        if abs(yaw1) < YAW_BAD_THRESH and fwd_vel1 > FWD_STUCK_THRESH:
            print("  [Opt] Plan OK → queued for batch sim")
            return plan0, fwd0.tolist()

        # ── Case B: stuck ──────────────────────────────────────────────────
        if fwd_vel1 < FWD_STUCK_THRESH:
            major = np.asarray(plan0["initial_virtual_forward_axis"], dtype=float)
            major = major / max(float(np.linalg.norm(major)), 1e-9)
            minor = np.array([-major[1], major[0]], dtype=float)
            best_score = fwd_vel1 - 0.3 * abs(yaw1)
            best_plan  = plan0; best_fwd = fwd0.copy()
            for cand in [-major, minor, -minor]:
                cand_n = cand / max(float(np.linalg.norm(cand)), 1e-9)
                if float(np.dot(cand_n, fwd0)) > 0.90:
                    continue
                plan_c = compute_adaptive_plan(description, {}, forced_axis=cand_n.tolist())
                trail_c, _, stats_c = ctx.run_episode(plan_c, PROBE_STEPS, return_yaw_stats=True)
                yaw_c    = stats_c["yaw_rate_mean"]
                fwd_c, _ = _metrics(trail_c, cand_n, len(trail_c))
                score_c  = fwd_c - 0.3 * abs(yaw_c)
                print(f"  [AltAxis] {[round(v,3) for v in cand_n.tolist()]}  "
                      f"yaw={yaw_c:+.3f}  fwd={fwd_c:+.4f}  score={score_c:.4f}")
                if score_c > best_score:
                    best_score = score_c; best_plan = plan_c; best_fwd = cand_n.copy()
            if abs(yaw1) > YAW_BAD_THRESH:
                best_plan = _apply_amp_correction(best_plan, yaw1)
            print(f"  [Opt] Axis chosen: {[round(v,3) for v in best_fwd.tolist()]} → queued")
            return best_plan, list(best_fwd)

        # ── Case C: spinning ───────────────────────────────────────────────
        best_plan  = plan0; best_score = fwd_vel1 - 0.3 * abs(yaw1)
        cur_yaw = yaw1; cur_plan = plan0
        for iter_i in range(MAX_YAW_ITERS):
            if abs(cur_yaw) < YAW_BAD_THRESH:
                break
            strength  = 0.55 + 0.18 * iter_i
            plan_corr = _apply_amp_correction(cur_plan, cur_yaw, strength=strength)
            trail_c, _, stats_c = ctx.run_episode(plan_corr, PROBE_STEPS, return_yaw_stats=True)
            yaw_c    = stats_c["yaw_rate_mean"]
            fwd_c, _ = _metrics(trail_c, fwd0, len(trail_c))
            score_c  = fwd_c - 0.3 * abs(yaw_c)
            print(f"  [YawIter{iter_i+1}] strength={strength:.2f}  "
                  f"yaw={yaw_c:+.3f} rad/s  fwd={fwd_c:+.4f} m/s  score={score_c:.4f}")
            if score_c > best_score:
                best_score = score_c; best_plan = plan_corr
            cur_yaw = yaw_c; cur_plan = plan_corr
        print(f"  [Opt] Best score={best_score:.4f} → queued for batch sim")
        return best_plan, fwd0.tolist()


# ---------------------------------------------------------------------------
# Subprocess worker — called in a fresh process to isolate CUDA state
# ---------------------------------------------------------------------------

def _subbatch_worker(robot_infos: List[dict], use_gpu: bool, result_file: str) -> None:
    """Run one sub-batch inside an isolated subprocess.

    Results are written to ``result_file`` (pickle) BEFORE the process exits.
    This is critical: the segfault from PhysX CUDA cleanup happens AFTER
    ``run_subbatch_full_pipeline`` returns, so we write results first.
    The main process can read results regardless of the worker's exit code.
    """
    import pickle as _pkl
    try:
        probe_results, sim_results = run_subbatch_full_pipeline(robot_infos, use_gpu=use_gpu)
    except Exception as _e:
        probe_results = [
            (info["initial_plan"], info["initial_plan"]["final_forward_axis"])
            for info in robot_infos
        ]
        sim_results = [([], [1.0, 0.0])] * len(robot_infos)
    # Write results BEFORE process exits (segfault during CUDA cleanup is OK after this)
    with open(result_file, "wb") as _f:
        _pkl.dump((probe_results, sim_results), _f)


def run_subbatch_isolated(
    robot_infos: List[dict],
    use_gpu: bool = True,
) -> Tuple[List[Tuple[dict, List[float]]], List[Tuple[list, list]]]:
    """Run sub-batch in a fresh subprocess to prevent CUDA memory corruption.

    Isaac Gym's PhysX backend has a bug where CUDA cleanup after destroy_sim()
    can corrupt memory and eventually cause a SIGABRT/SIGSEGV in the main process
    (typically after ~12 destroy_sim cycles with 8-robot sims).

    By running each sub-batch in its own subprocess:
    - CUDA state is isolated per sub-batch: no cross-contamination.
    - If the subprocess segfaults during CUDA cleanup, only the child dies.
    - The main process reads results from a file written before the segfault.

    Uses ``multiprocessing.get_context('spawn')`` to start a clean process
    (avoids inheriting CUDA fd state from fork).
    """
    import multiprocessing as _mp
    import pickle as _pkl
    import tempfile as _tmp
    import os as _os

    with _tmp.NamedTemporaryFile(suffix="_subbatch.pkl", delete=False) as tf:
        result_file = tf.name

    fallback = (
        [(info["initial_plan"], info["initial_plan"]["final_forward_axis"])
         for info in robot_infos],
        [([], [1.0, 0.0])] * len(robot_infos),
    )

    try:
        ctx = _mp.get_context("spawn")
        p   = ctx.Process(
            target=_subbatch_worker,
            args=(robot_infos, use_gpu, result_file),
        )
        p.start()
        p.join(timeout=7200)       # 2-hour safety timeout
        if p.is_alive():
            p.terminate(); p.join(30)
            print(f"  [IsolatedBatch] WARNING: worker timed out, using fallback")
            return fallback
        # exit code 0 = clean exit, 139 = SIGABRT/SIGSEGV (CUDA cleanup), both OK
        if _os.path.exists(result_file) and _os.path.getsize(result_file) > 0:
            with open(result_file, "rb") as f:
                return _pkl.load(f)
        print(f"  [IsolatedBatch] WARNING: worker exited {p.exitcode}, no results file; "
              "using fallback")
        return fallback
    finally:
        try:
            _os.unlink(result_file)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Unified sub-batch pipeline: probes + correction + full-sim in ONE sim
# ---------------------------------------------------------------------------

def run_subbatch_full_pipeline(
    robot_infos: List[dict],
    use_gpu: bool = True,
) -> Tuple[List[Tuple[dict, List[float]]], List[Tuple[list, list]]]:
    """Run probe-correction-full-sim for a sub-batch of robots in ONE shared sim.

    This is the core function that eliminates repeated create_sim / destroy_sim
    cycles.  For a sub-batch of N ≤ GPU_BATCH_SIZE robots:

    1.  create_sim called ONCE.
    2.  Warmup (sag controller, HOLD_STEPS steps) run ONCE for all robots.
    3.  Probe round 1 — all robots with initial plans.
    4.  Classification → compute corrected plans in Python (no sim needed).
    5.  Probe rounds 2-4 — only robots still needing correction.
        Between rounds: robot states are reset in-place (set_actor_rigid_body_states).
    6.  Full final simulation — all robots reset to stand_ev, run MAX_SIM_STEPS.
    7.  destroy_sim called ONCE.

    Total create_sim calls per call of this function: EXACTLY 1.
    Total calls from main() for N valid robots: ceil(N / GPU_BATCH_SIZE) ≈ 17.
    Grand total including reference robot: ≈ 18 — well within safe limits.

    Parameters
    ----------
    robot_infos : list of dicts (each has ``description``, ``urdf_path``,
                  ``initial_plan``, ``robot_name``)
    use_gpu     : enable PhysX GPU.

    Returns
    -------
    probe_results : list of ``(corrected_plan, forward_axis)`` per robot.
    sim_results   : list of ``(com_trail, forward_axis)`` per robot.
    """
    from adaptive_gait import compute_adaptive_plan
    from isaacgym import gymapi

    PROBE_STEPS      = 360
    YAW_BAD_THRESH   = 0.18
    FWD_STUCK_THRESH = 0.004
    _dt              = 1.0 / 60.0
    SPACING          = 14.0

    def _rtj(lo, hi, r):
        return float(lo + max(0.0, min(1.0, r)) * (hi - lo))
    def _ss(e0, e1, x):
        t = max(0.0, min(1.0, (x - e0) / max(e1 - e0, 1e-9)))
        return t * t * (3.0 - 2.0 * t)
    def _fwd_vel(trail, fwd_ax):
        if len(trail) < 2:
            return 0.0
        arr = np.array(trail, dtype=float)
        return float(np.dot(arr[-1] - arr[0], fwd_ax)) / max(len(trail) * _dt, _dt)

    N = len(robot_infos)
    if N == 0:
        return [], []

    # ── Create sim ───────────────────────────────────────────────────────────
    gym = gymapi.acquire_gym()
    sp = gymapi.SimParams()
    sp.up_axis   = gymapi.UP_AXIS_Z
    sp.gravity   = gymapi.Vec3(0.0, 0.0, -9.81)
    sp.dt        = 1.0 / 60.0
    sp.substeps  = 2
    sp.physx.use_gpu = use_gpu
    sp.physx.num_position_iterations = 8
    sp.physx.num_velocity_iterations = 2
    gpu_dev = 0 if use_gpu else -1
    sim = gym.create_sim(0, gpu_dev, gymapi.SIM_PHYSX, sp)
    if sim is None:
        # Fallback: per-robot sequential
        probe_results = []
        sim_results   = []
        for info in robot_infos:
            try:
                plan, fwd = probe_and_correct_plan(info["description"], info["urdf_path"])
                trail, ax = run_gait_sim(info["description"], info["urdf_path"],
                                         gait_plan_override=plan, use_gpu=False)
                probe_results.append((plan, fwd))
                sim_results.append((trail, ax))
            except Exception:
                probe_results.append((info["initial_plan"],
                                      info["initial_plan"]["final_forward_axis"]))
                sim_results.append(([], info["initial_plan"]["final_forward_axis"]))
        return probe_results, sim_results

    try:
        pp = gymapi.PlaneParams()
        pp.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        pp.static_friction = 1.8; pp.dynamic_friction = 1.6; pp.restitution = 0.0
        gym.add_ground(sim, pp)

        ao = gymapi.AssetOptions()
        ao.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
        ao.fix_base_link = False
        ao.collapse_fixed_joints = True

        cols = max(int(math.ceil(math.sqrt(N))), 1)
        robot_states: List[Optional[dict]] = []

        for i, info in enumerate(robot_infos):
            urdf_path = info["urdf_path"].resolve()
            asset = gym.load_asset(sim, str(urdf_path.parent), urdf_path.name, ao)
            if asset is None:
                robot_states.append(None)
                continue

            ox = (i % cols) * SPACING
            oy = (i // cols) * SPACING
            env = gym.create_env(sim,
                                 gymapi.Vec3(ox - SPACING / 2, oy - SPACING / 2, 0),
                                 gymapi.Vec3(ox + SPACING / 2, oy + SPACING / 2, 2), N)
            pose      = gymapi.Transform()
            pose.p    = gymapi.Vec3(ox, oy, BODY_HEIGHT)
            actor     = gym.create_actor(env, asset, pose, f"r{i}", i, 1)

            dof_props = gym.get_actor_dof_properties(env, actor)
            dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
            dof_props["stiffness"].fill(200.0); dof_props["damping"].fill(20.0)
            if "effort"   in dof_props.dtype.names: dof_props["effort"].fill(1000.0)
            if "armature" in dof_props.dtype.names: dof_props["armature"].fill(0.01)
            dof_names = gym.get_asset_dof_names(asset)
            for idx, name in enumerate(dof_names):
                if "_swing" in name:
                    dof_props["stiffness"][idx] = 100.0; dof_props["damping"][idx] = 10.0
                elif "_drop" in name:
                    dof_props["stiffness"][idx] = 250.0; dof_props["damping"][idx] = 25.0
            gym.set_actor_dof_properties(env, actor, dof_props)

            lower     = np.where(np.isfinite(dof_props["lower"]), dof_props["lower"], -0.5).astype(np.float32)
            upper     = np.where(np.isfinite(dof_props["upper"]), dof_props["upper"],  0.5).astype(np.float32)
            finite_lo = np.where(np.isfinite(dof_props["lower"]), dof_props["lower"], -1e9)
            finite_hi = np.where(np.isfinite(dof_props["upper"]), dof_props["upper"],  1e9)
            name2idx  = {n: k for k, n in enumerate(dof_names)}

            triplets: Dict[int, dict] = {}
            description = info["description"]
            for lid in range(int(description.get("num_legs", 0))):
                ln, sn, dn = f"leg_{lid}_lift", f"leg_{lid}_swing", f"leg_{lid}_drop"
                if ln not in name2idx:
                    continue
                triplets[lid] = {
                    "lift_idx":    name2idx[ln], "swing_idx": name2idx[sn], "drop_idx":  name2idx[dn],
                    "lift_lower":  float(lower[name2idx[ln]]), "lift_upper":  float(upper[name2idx[ln]]),
                    "swing_lower": float(lower[name2idx[sn]]), "swing_upper": float(upper[name2idx[sn]]),
                    "drop_lower":  float(lower[name2idx[dn]]), "drop_upper":  float(upper[name2idx[dn]]),
                }

            foot_z_vals     = [float(lk["default_world_origin"][2])
                               for lk in description.get("links", [])
                               if lk.get("role") == "foot" and lk.get("leg_id") is not None]
            mean_foot_z     = float(np.mean(foot_z_vals)) if foot_z_vals else -0.35
            feet_at_ground  = abs(mean_foot_z + BODY_HEIGHT) < 0.12

            stand = 0.5 * (lower + upper).astype(np.float32)
            for lid, j in triplets.items():
                if feet_at_ground:
                    _lr = max(0.0, min(1.0, (0.0 - j["lift_lower"]) / max(j["lift_upper"] - j["lift_lower"], 1e-9)))
                    _dr = max(0.0, min(1.0, (0.0 - j["drop_lower"]) / max(j["drop_upper"] - j["drop_lower"], 1e-9)))
                else:
                    _lr, _dr = 0.05, 0.995
                stand[j["lift_idx"]]  = _rtj(j["lift_lower"],  j["lift_upper"],  _lr)
                stand[j["drop_idx"]]  = _rtj(j["drop_lower"],  j["drop_upper"],  _dr)
                stand[j["swing_idx"]] = _rtj(j["swing_lower"], j["swing_upper"], 0.5)
            stand = np.clip(stand, finite_lo, finite_hi)

            ds = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
            ds["pos"] = stand; ds["vel"].fill(0.0)
            gym.set_actor_dof_states(env, actor, ds, gymapi.STATE_ALL)

            if feet_at_ground and triplets:
                fj   = next(iter(triplets.values()))
                _lrr = max(fj["lift_upper"] - fj["lift_lower"], 1e-9)
                _drr = max(fj["drop_upper"] - fj["drop_lower"], 1e-9)
                _sl  = max(0.0, min(1.0, (0.0   - fj["lift_lower"]) / _lrr))
                _swl = max(0.0, min(1.0, (+0.25  - fj["lift_lower"]) / _lrr))
                _sd  = max(0.0, min(1.0, (0.0   - fj["drop_lower"]) / _drr))
                _swd = _sd
            else:
                _sl, _swl, _sd, _swd = STANCE_LIFT_RATIO, SWING_LIFT_RATIO, STANCE_DROP_RATIO, SWING_DROP_RATIO

            fmap: Dict[int, np.ndarray] = {}
            for lk in description.get("links", []):
                if lk.get("role") == "foot" and lk.get("leg_id") is not None:
                    fmap[int(lk["leg_id"])] = np.asarray(lk["default_world_origin"], dtype=float)[:2]
            if fmap:
                fp = np.array(list(fmap.values()), dtype=float)
                body_length = float(np.max(fp.max(axis=0) - fp.min(axis=0)))
            else:
                body_length = 0.5

            robot_states.append({
                "env": env, "actor": actor, "asset": asset,
                "triplets": triplets, "stand": stand.copy(), "stand_ev": stand.copy(),
                "lower": lower, "upper": upper, "finite_lo": finite_lo, "finite_hi": finite_hi,
                "dof_names": list(dof_names),
                "_sl": _sl, "_swl": _swl, "_sd": _sd, "_swd": _swd,
                "fmap": fmap, "body_length": body_length,
                "offset": np.array([ox, oy], dtype=float),
            })

        valid_idx = [i for i, rs in enumerate(robot_states) if rs is not None]
        if not valid_idx:
            gym.destroy_sim(sim)
            fallback_plan  = [(info["initial_plan"], info["initial_plan"]["final_forward_axis"])
                              for info in robot_infos]
            return fallback_plan, [([], [1.0, 0.0])] * N

        # ── Warmup (sag-controller, run ONCE for all robots) ─────────────────
        for _ in range(HOLD_STEPS):
            for i in valid_idx:
                rs   = robot_states[i]
                dp   = gym.get_actor_dof_states(rs["env"], rs["actor"], gymapi.STATE_POS)
                jpos = np.asarray(dp["pos"], dtype=np.float32)
                se   = rs["stand_ev"]
                for idx, name in enumerate(rs["dof_names"]):
                    sag = se[idx] - jpos[idx]
                    if "_drop" in name and sag > 0.004:
                        se[idx] = min(rs["finite_hi"][idx], se[idx] + min(0.012, 0.22 * sag))
                    elif "_lift" in name and sag > 0.004:
                        se[idx] = max(rs["finite_lo"][idx], se[idx] - min(0.008, 0.16 * sag))
                gym.set_actor_dof_position_targets(rs["env"], rs["actor"], se)
            gym.simulate(sim); gym.fetch_results(sim, True)

        # ──────────────────────────────────────────────────────────────────────
        # Inner helpers (run inside the sim lifetime)
        # ──────────────────────────────────────────────────────────────────────

        def _reset_robot(rs):
            """Teleport robot back to its grid offset, zero velocities, reset DOF."""
            ds = gym.get_actor_dof_states(rs["env"], rs["actor"], gymapi.STATE_ALL)
            ds["pos"] = rs["stand_ev"]
            ds["vel"].fill(0.0)
            gym.set_actor_dof_states(rs["env"], rs["actor"], ds, gymapi.STATE_ALL)
            try:
                ox, oy = rs["offset"]
                rb = gym.get_actor_rigid_body_states(rs["env"], rs["actor"], gymapi.STATE_ALL)
                rb["pose"]["p"][0]["x"] = float(ox)
                rb["pose"]["p"][0]["y"] = float(oy)
                rb["pose"]["p"][0]["z"] = BODY_HEIGHT
                rb["pose"]["r"][0]["x"] = 0.0
                rb["pose"]["r"][0]["y"] = 0.0
                rb["pose"]["r"][0]["z"] = 0.0
                rb["pose"]["r"][0]["w"] = 1.0
                rb["vel"]["linear"][0]["x"]  = 0.0
                rb["vel"]["linear"][0]["y"]  = 0.0
                rb["vel"]["linear"][0]["z"]  = 0.0
                rb["vel"]["angular"][0]["x"] = 0.0
                rb["vel"]["angular"][0]["y"] = 0.0
                rb["vel"]["angular"][0]["z"] = 0.0
                gym.set_actor_rigid_body_states(rs["env"], rs["actor"], rb, gymapi.STATE_ALL)
            except Exception:
                pass
            for _ in range(5):
                gym.set_actor_dof_position_targets(rs["env"], rs["actor"], rs["stand_ev"])
                gym.simulate(sim); gym.fetch_results(sim, True)

        def _apply_plan_to_rs(rs, plan):
            """Update per-robot state dict with a new plan's axes / per_amp."""
            topo     = plan["topology"]
            fwd_arr  = np.asarray(plan.get("final_forward_axis", [1.0, 0.0]), dtype=float)
            fwd_arr  = fwd_arr / max(float(np.linalg.norm(fwd_arr)), 1e-9)
            lat_arr  = np.array([-fwd_arr[1], fwd_arr[0]], dtype=float)
            per_amp  = {str(k): float(v) for k, v in topo.get("per_leg_stride_amplitudes", {}).items()}
            if "_per_amp_override" in plan:
                per_amp = {str(k): float(v) for k, v in plan["_per_amp_override"].items()}
            rs["fwd"]       = fwd_arr
            rs["lat"]       = lat_arr
            rs["per_amp"]   = per_amp
            rs["group_a"]   = topo["groups"]["group_a"]
            rs["group_b"]   = topo["groups"]["group_b"]
            rs["group_c"]   = topo["groups"].get("group_c", [])
            rs["fwd_list"]  = fwd_arr.tolist()

        def _run_episode(active_indices, plan_map, n_steps, track_yaw=True):
            """Run one episode for active robots.  plan_map[i] = gait_plan for robot i."""
            for i in active_indices:
                _apply_plan_to_rs(robot_states[i], plan_map[i])
            for i in active_indices:
                robot_states[i]["com_trail"]  = []
                robot_states[i]["yaw_acc"]    = []
                robot_states[i]["td_ramp"]    = {}
            sim_t = 0.0
            for _step in range(n_steps):
                ph_now = 2.0 * math.pi * GAIT_FREQUENCY * sim_t
                for i in active_indices:
                    rs  = robot_states[i]
                    tgt = rs["stand_ev"].copy()
                    for lid, j in rs["triplets"].items():
                        if lid in rs["group_c"]:
                            tgt[j["lift_idx"]]  = _rtj(j["lift_lower"], j["lift_upper"], rs["_sl"])
                            tgt[j["drop_idx"]]  = _rtj(j["drop_lower"], j["drop_upper"], rs["_sd"])
                            tgt[j["swing_idx"]] = _rtj(j["swing_lower"], j["swing_upper"], 0.5)
                            continue
                        if   lid in rs["group_b"]: lg_ph = ph_now + math.pi
                        elif lid in rs["group_a"]: lg_ph = ph_now
                        else:                      lg_ph = 0.0
                        sw    = float(math.sin(lg_ph))
                        alpha = _ss(-0.30, 0.30, sw)
                        fv    = rs["fmap"].get(lid, np.zeros(2))
                        dsign = -1.0 if float(np.dot(fv, rs["lat"])) > 0.0 else 1.0
                        ea    = SWING_AMP * float(rs["per_amp"].get(str(lid), 1.0))
                        lr    = rs["_sl"] + (rs["_swl"] - rs["_sl"]) * alpha
                        dr    = rs["_sd"] + (rs["_swd"] - rs["_sd"]) * alpha
                        sr    = 0.5 + ea * dsign * sw
                        if sw <= 0.0:
                            if lid in rs["td_ramp"]:
                                rs["td_ramp"][lid] += 1
                                rp = min(rs["td_ramp"][lid] / 8, 1.0)
                                if rp < 1.0:
                                    dl  = _rtj(j["lift_lower"],  j["lift_upper"],  0.5)
                                    dd  = _rtj(j["drop_lower"],  j["drop_upper"],  0.5)
                                    ds_ = j["swing_lower"] + 0.5*(j["swing_upper"]-j["swing_lower"])
                                    tgt[j["lift_idx"]]  = dl  + rp*(_rtj(j["lift_lower"],  j["lift_upper"],  lr)-dl)
                                    tgt[j["drop_idx"]]  = dd  + rp*(_rtj(j["drop_lower"],  j["drop_upper"],  dr)-dd)
                                    tgt[j["swing_idx"]] = ds_ + rp*(_rtj(j["swing_lower"], j["swing_upper"], sr)-ds_)
                                    continue
                                else:
                                    del rs["td_ramp"][lid]
                        else:
                            rs["td_ramp"][lid] = 0
                        tgt[j["lift_idx"]]  = _rtj(j["lift_lower"], j["lift_upper"], lr)
                        tgt[j["drop_idx"]]  = _rtj(j["drop_lower"], j["drop_upper"], dr)
                        tgt[j["swing_idx"]] = _rtj(j["swing_lower"], j["swing_upper"], sr)
                    gym.set_actor_dof_position_targets(rs["env"], rs["actor"], tgt)
                gym.simulate(sim); gym.fetch_results(sim, True)
                _sflag = gymapi.STATE_ALL if track_yaw else gymapi.STATE_POS
                for i in active_indices:
                    rs  = robot_states[i]
                    sts = gym.get_actor_rigid_body_states(rs["env"], rs["actor"], _sflag)
                    if sts is not None and len(sts) > 0:
                        p   = sts["pose"]["p"][0]
                        ox, oy = rs["offset"]
                        rs["com_trail"].append([float(p["x"]) - ox, float(p["y"]) - oy])
                        if track_yaw:
                            try:
                                rs["yaw_acc"].append(float(sts["vel"]["angular"][0]["z"]))
                            except Exception:
                                ct = rs["com_trail"]
                                if len(ct) >= 3:
                                    _d1 = np.array(ct[-1]) - np.array(ct[-2])
                                    _d0 = np.array(ct[-2]) - np.array(ct[-3])
                                    _da = ((math.atan2(_d1[1],_d1[0])-math.atan2(_d0[1],_d0[0])+math.pi)%(2*math.pi))-math.pi
                                    rs["yaw_acc"].append(_da / max(_dt, 1e-9))
                sim_t += _dt
            # Compute yaw_stats
            results = {}
            for i in active_indices:
                rs  = robot_states[i]
                vy  = [v for v in rs.get("yaw_acc", []) if abs(v) < 20.0]
                results[i] = {
                    "trail":    rs["com_trail"],
                    "fwd_list": rs.get("fwd_list", [1.0, 0.0]),
                    "yaw_stats": {
                        "yaw_rate_mean": float(np.mean(vy)) if vy else 0.0,
                        "yaw_rate_std":  float(np.std(vy))  if vy else 0.0,
                    } if track_yaw else None,
                }
            return results

        # ── Phase 1: Probe rounds ─────────────────────────────────────────────
        best_plan   = [info["initial_plan"] for info in robot_infos]
        best_fwd    = [np.asarray(info["initial_plan"]["final_forward_axis"], dtype=float)
                       for info in robot_infos]
        for i in range(N):
            best_fwd[i] /= max(float(np.linalg.norm(best_fwd[i])), 1e-9)
        best_score  = [-1e9] * N
        alt_state   = [None] * N
        yaw_state   = [None] * N
        needs_next  = list(valid_idx)          # start: all valid robots need probing

        # Round 1 — initial plans
        plan_map_r1 = {i: best_plan[i] for i in valid_idx}
        for i in valid_idx:
            _reset_robot(robot_states[i])
        r1 = _run_episode(valid_idx, plan_map_r1, PROBE_STEPS, track_yaw=True)

        needs_round2 = []
        for i in valid_idx:
            trail = r1[i]["trail"]
            stats = r1[i]["yaw_stats"]
            yaw   = stats["yaw_rate_mean"]
            fv    = _fwd_vel(trail, best_fwd[i])
            score = fv - 0.3 * abs(yaw)
            best_score[i] = score
            info  = robot_infos[i]
            print(f"    [{info['robot_name']}] P1 yaw={yaw:+.3f}  fwd={fv:+.4f}  score={score:.4f}", end="")
            if abs(yaw) < YAW_BAD_THRESH and fv > FWD_STUCK_THRESH:
                print("  → OK")
            elif fv < FWD_STUCK_THRESH:
                major = np.asarray(best_plan[i]["initial_virtual_forward_axis"], dtype=float)
                major /= max(float(np.linalg.norm(major)), 1e-9)
                minor  = np.array([-major[1], major[0]], dtype=float)
                alt_state[i] = {"candidates": [-major, minor, -minor], "cidx": 0}
                cand = -major
                if float(np.dot(cand, best_fwd[i])) > 0.90:
                    alt_state[i]["cidx"] = 1; cand = minor
                plan_c = compute_adaptive_plan(info["description"], {}, forced_axis=cand.tolist())
                needs_round2.append((i, plan_c, cand, "alt_axis"))
                print(f"  → stuck, trying {[round(v,3) for v in cand.tolist()]}")
            else:
                yaw_state[i] = {"cur_yaw": yaw, "cur_plan": best_plan[i], "iter": 0}
                plan_c = _apply_amp_correction(best_plan[i], yaw, strength=0.55)
                yaw_state[i]["cur_plan"] = plan_c; yaw_state[i]["iter"] = 1
                needs_round2.append((i, plan_c, best_fwd[i], "yaw_corr"))
                print(f"  → spinning s=0.55")

        # Rounds 2-4
        for round_num in range(2, 5):
            if not needs_round2:
                break
            active_r     = [tup[0] for tup in needs_round2]
            plan_map_rn  = {tup[0]: tup[1] for tup in needs_round2}
            fwd_map_rn   = {tup[0]: tup[2] for tup in needs_round2}
            for i in active_r:
                _reset_robot(robot_states[i])
            rn  = _run_episode(active_r, plan_map_rn, PROBE_STEPS, track_yaw=True)
            needs_round2 = []
            for (i, plan_used, fwd_used, kind) in [(t[0],t[1],t[2],t[3]) for t in
                                                    [(tup[0],tup[1],tup[2],tup[3])
                                                     for tup in [(a, plan_map_rn[a], fwd_map_rn[a],
                                                                  "alt_axis" if alt_state[a] else "yaw_corr")
                                                                 for a in active_r]]]:
                trail = rn[i]["trail"]
                stats = rn[i]["yaw_stats"]
                yaw   = stats["yaw_rate_mean"]
                fv    = _fwd_vel(trail, fwd_used)
                score = fv - 0.3 * abs(yaw)
                info  = robot_infos[i]
                print(f"    [{info['robot_name']}] R{round_num} yaw={yaw:+.3f}  fwd={fv:+.4f}  score={score:.4f}", end="")
                if score > best_score[i]:
                    best_score[i] = score
                    best_plan[i]  = plan_used
                    best_fwd[i]   = np.asarray(fwd_used, dtype=float)
                if kind == "alt_axis":
                    st  = alt_state[i]
                    st["cidx"] += 1
                    if st["cidx"] >= len(st["candidates"]):
                        print(f"  → axis {[round(v,3) for v in best_fwd[i].tolist()]}")
                    else:
                        cand = st["candidates"][st["cidx"]]
                        fwd0 = np.asarray(robot_infos[i]["initial_plan"]["final_forward_axis"], dtype=float)
                        fwd0 /= max(float(np.linalg.norm(fwd0)), 1e-9)
                        if float(np.dot(cand, fwd0)) > 0.90:
                            st["cidx"] += 1
                            if st["cidx"] < len(st["candidates"]):
                                cand = st["candidates"][st["cidx"]]
                            else:
                                print(f"  → axis {[round(v,3) for v in best_fwd[i].tolist()]}")
                                continue
                        plan_c = compute_adaptive_plan(info["description"], {}, forced_axis=cand.tolist())
                        needs_round2.append((i, plan_c, cand, "alt_axis"))
                        print(f"  → try {[round(v,3) for v in cand.tolist()]}")
                else:
                    st = yaw_state[i]
                    st["cur_yaw"] = yaw; st["cur_plan"] = plan_used
                    strengths = [0.55, 0.73, 0.90]
                    if abs(yaw) < YAW_BAD_THRESH or st["iter"] >= len(strengths):
                        print(f"  → yaw done score={best_score[i]:.4f}")
                    else:
                        s = strengths[st["iter"]]; st["iter"] += 1
                        pc = _apply_amp_correction(st["cur_plan"], yaw, strength=s)
                        st["cur_plan"] = pc
                        fwd0 = np.asarray(robot_infos[i]["initial_plan"]["final_forward_axis"], dtype=float)
                        fwd0 /= max(float(np.linalg.norm(fwd0)), 1e-9)
                        needs_round2.append((i, pc, fwd0, "yaw_corr"))
                        print(f"  → yaw still {yaw:+.3f} s={s:.2f}")

        probe_results = [(best_plan[i], best_fwd[i].tolist()) for i in range(N)]

        # ── Phase 2: Full final simulation (all robots, same sim) ─────────────
        full_steps = max(MAX_SIM_STEPS, SIM_STEPS)
        plan_map_f = {i: best_plan[i] for i in valid_idx}
        for i in valid_idx:
            rs = robot_states[i]
            rs["min_travel"] = rs["body_length"] * max(MIN_TRAVEL_BODY_LENGTHS, 0.0)
            rs["done"]       = False
            _reset_robot(rs)

        sim_t = 0.0
        for _step in range(full_steps):
            ph_now = 2.0 * math.pi * GAIT_FREQUENCY * sim_t
            still_going = [i for i in valid_idx if not robot_states[i]["done"]]
            if not still_going:
                break
            for i in still_going:
                rs  = robot_states[i]
                tgt = rs["stand_ev"].copy()
                for lid, j in rs["triplets"].items():
                    if lid in rs["group_c"]:
                        tgt[j["lift_idx"]]  = _rtj(j["lift_lower"], j["lift_upper"], rs["_sl"])
                        tgt[j["drop_idx"]]  = _rtj(j["drop_lower"], j["drop_upper"], rs["_sd"])
                        tgt[j["swing_idx"]] = _rtj(j["swing_lower"], j["swing_upper"], 0.5)
                        continue
                    if   lid in rs["group_b"]: lg_ph = ph_now + math.pi
                    elif lid in rs["group_a"]: lg_ph = ph_now
                    else:                      lg_ph = 0.0
                    sw    = float(math.sin(lg_ph))
                    alpha = _ss(-0.30, 0.30, sw)
                    fv    = rs["fmap"].get(lid, np.zeros(2))
                    dsign = -1.0 if float(np.dot(fv, rs["lat"])) > 0.0 else 1.0
                    ea    = SWING_AMP * float(rs["per_amp"].get(str(lid), 1.0))
                    lr    = rs["_sl"] + (rs["_swl"] - rs["_sl"]) * alpha
                    dr    = rs["_sd"] + (rs["_swd"] - rs["_sd"]) * alpha
                    sr    = 0.5 + ea * dsign * sw
                    if sw <= 0.0:
                        if lid in rs["td_ramp"]:
                            rs["td_ramp"][lid] += 1
                            rp = min(rs["td_ramp"][lid] / 8, 1.0)
                            if rp < 1.0:
                                dl  = _rtj(j["lift_lower"],  j["lift_upper"],  0.5)
                                dd  = _rtj(j["drop_lower"],  j["drop_upper"],  0.5)
                                ds_ = j["swing_lower"]+0.5*(j["swing_upper"]-j["swing_lower"])
                                tgt[j["lift_idx"]]  = dl +rp*(_rtj(j["lift_lower"], j["lift_upper"], lr)-dl)
                                tgt[j["drop_idx"]]  = dd +rp*(_rtj(j["drop_lower"], j["drop_upper"], dr)-dd)
                                tgt[j["swing_idx"]] = ds_+rp*(_rtj(j["swing_lower"],j["swing_upper"],sr)-ds_)
                                continue
                            else:
                                del rs["td_ramp"][lid]
                    else:
                        rs["td_ramp"][lid] = 0
                    tgt[j["lift_idx"]]  = _rtj(j["lift_lower"], j["lift_upper"], lr)
                    tgt[j["drop_idx"]]  = _rtj(j["drop_lower"], j["drop_upper"], dr)
                    tgt[j["swing_idx"]] = _rtj(j["swing_lower"], j["swing_upper"], sr)
                gym.set_actor_dof_position_targets(rs["env"], rs["actor"], tgt)
            gym.simulate(sim); gym.fetch_results(sim, True)
            for i in still_going:
                rs  = robot_states[i]
                sts = gym.get_actor_rigid_body_states(rs["env"], rs["actor"], gymapi.STATE_POS)
                if sts is not None and len(sts) > 0:
                    p   = sts["pose"]["p"][0]
                    ox, oy = rs["offset"]
                    rs["com_trail"].append([float(p["x"]) - ox, float(p["y"]) - oy])
                if _step >= SIM_STEPS and len(rs["com_trail"]) > 1 and rs["min_travel"] > 0:
                    arr     = np.array(rs["com_trail"], dtype=float)
                    covered = abs(float(np.dot(arr[-1] - arr[0], rs["fwd"])))
                    if covered >= rs["min_travel"]:
                        rs["done"] = True
            sim_t += _dt

        sim_results = []
        for i in range(N):
            rs = robot_states[i] if robot_states[i] is not None else None
            if rs is None:
                sim_results.append(([], [1.0, 0.0]))
            else:
                sim_results.append((rs["com_trail"], rs.get("fwd_list", [1.0, 0.0])))

    finally:
        gym.destroy_sim(sim)

    return probe_results, sim_results



# ---------------------------------------------------------------------------
# GPU Parallel Final Simulation — runs N robots in one Isaac Gym sim
# ---------------------------------------------------------------------------

def run_gait_sim_parallel_final(
    configs: List[Tuple[dict, Path, dict]],
    use_gpu: bool = True,
    n_max_steps: Optional[int] = None,
) -> List[Tuple[List[List[float]], List[float]]]:
    """Run the final (full-length) gait simulation for multiple robots in parallel.

    All robots share a single Isaac Gym simulation instance.  The physics step
    is executed once per simulation step for all robots simultaneously (GPU),
    while per-robot joint targets are computed sequentially in Python (CPU).

    Parameters
    ----------
    configs    : list of (description, urdf_path, gait_plan) tuples.
    use_gpu    : enable GPU physics (requires NVIDIA GPU).

    n_max_steps: override the simulation step limit.

    Returns
    -------
    List of (com_trail, forward_axis) — one entry per input config.
    """
    from isaacgym import gymapi

    N = len(configs)
    if N == 0:
        return []
    if N == 1:
        desc, urdf, plan = configs[0]
        t, a = run_gait_sim(desc, urdf, gait_plan_override=plan,
                            n_max_steps=n_max_steps, use_gpu=use_gpu)
        return [(t, a)]

    def _rtj(lo, hi, r):
        return float(lo + max(0.0, min(1.0, r)) * (hi - lo))

    def _ss(e0, e1, x):
        t = max(0.0, min(1.0, (x - e0) / max(e1 - e0, 1e-9)))
        return t * t * (3.0 - 2.0 * t)

    gym = gymapi.acquire_gym()
    sp = gymapi.SimParams()
    sp.up_axis   = gymapi.UP_AXIS_Z
    sp.gravity   = gymapi.Vec3(0.0, 0.0, -9.81)
    sp.dt        = 1.0 / 60.0
    sp.substeps  = 2
    sp.physx.use_gpu = use_gpu
    sp.physx.num_position_iterations = 8
    sp.physx.num_velocity_iterations = 2

    gpu_dev = 0 if use_gpu else -1
    sim = gym.create_sim(0, gpu_dev, gymapi.SIM_PHYSX, sp)
    if sim is None:
        print(f"  [ParallelSim] create_sim failed, falling back to sequential (GPU={use_gpu})")
        results = []
        for desc, urdf, plan in configs:
            t, a = run_gait_sim(desc, urdf, gait_plan_override=plan,
                                n_max_steps=n_max_steps, use_gpu=False)
            results.append((t, a))
        return results

    SPACING = 14.0  # metres between robots to prevent interaction
    _dt     = 1.0 / 60.0

    try:
        pp = gymapi.PlaneParams()
        pp.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        pp.static_friction  = 1.8
        pp.dynamic_friction = 1.6
        pp.restitution = 0.0
        gym.add_ground(sim, pp)

        ao = gymapi.AssetOptions()
        ao.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
        ao.fix_base_link = False
        ao.collapse_fixed_joints = True

        robot_states: List[Optional[dict]] = []

        for i, (description, urdf_path, gait_plan) in enumerate(configs):
            urdf_path = urdf_path.resolve()
            asset = gym.load_asset(sim, str(urdf_path.parent), urdf_path.name, ao)
            if asset is None:
                robot_states.append(None)
                continue

            cols = max(int(math.ceil(math.sqrt(N))), 1)
            ox = (i % cols) * SPACING
            oy = (i // cols) * SPACING

            env = gym.create_env(
                sim,
                gymapi.Vec3(ox - SPACING / 2, oy - SPACING / 2, 0),
                gymapi.Vec3(ox + SPACING / 2, oy + SPACING / 2, 2),
                N,
            )
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(ox, oy, BODY_HEIGHT)
            actor = gym.create_actor(env, asset, pose, f"r{i}", i, 1)

            dof_props = gym.get_actor_dof_properties(env, actor)
            dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
            dof_props["stiffness"].fill(200.0)
            dof_props["damping"].fill(20.0)
            if "effort"   in dof_props.dtype.names: dof_props["effort"].fill(1000.0)
            if "armature" in dof_props.dtype.names: dof_props["armature"].fill(0.01)
            dof_names = gym.get_asset_dof_names(asset)
            for idx, name in enumerate(dof_names):
                if "_swing" in name:
                    dof_props["stiffness"][idx] = 100.0; dof_props["damping"][idx] = 10.0
                elif "_drop" in name:
                    dof_props["stiffness"][idx] = 250.0; dof_props["damping"][idx] = 25.0
            gym.set_actor_dof_properties(env, actor, dof_props)

            lower    = np.where(np.isfinite(dof_props["lower"]), dof_props["lower"], -0.5).astype(np.float32)
            upper    = np.where(np.isfinite(dof_props["upper"]), dof_props["upper"],  0.5).astype(np.float32)
            finite_lo = np.where(np.isfinite(dof_props["lower"]), dof_props["lower"], -1e9)
            finite_hi = np.where(np.isfinite(dof_props["upper"]), dof_props["upper"],  1e9)
            name2idx = {n: k for k, n in enumerate(dof_names)}

            triplets: Dict[int, dict] = {}
            for lid in range(int(description.get("num_legs", 0))):
                ln, sn, dn = f"leg_{lid}_lift", f"leg_{lid}_swing", f"leg_{lid}_drop"
                if ln not in name2idx:
                    continue
                triplets[lid] = {
                    "lift_idx":    name2idx[ln], "swing_idx": name2idx[sn], "drop_idx":  name2idx[dn],
                    "lift_lower":  float(lower[name2idx[ln]]), "lift_upper":  float(upper[name2idx[ln]]),
                    "swing_lower": float(lower[name2idx[sn]]), "swing_upper": float(upper[name2idx[sn]]),
                    "drop_lower":  float(lower[name2idx[dn]]), "drop_upper":  float(upper[name2idx[dn]]),
                }

            foot_z_vals = [float(lk["default_world_origin"][2])
                           for lk in description.get("links", [])
                           if lk.get("role") == "foot" and lk.get("leg_id") is not None]
            mean_foot_z = float(np.mean(foot_z_vals)) if foot_z_vals else -0.35
            feet_at_ground = abs(mean_foot_z + BODY_HEIGHT) < 0.12

            stand = 0.5 * (lower + upper).astype(np.float32)
            for lid, j in triplets.items():
                if feet_at_ground:
                    _lr = max(0.0, min(1.0, (0.0 - j["lift_lower"]) / max(j["lift_upper"] - j["lift_lower"], 1e-9)))
                    _dr = max(0.0, min(1.0, (0.0 - j["drop_lower"]) / max(j["drop_upper"] - j["drop_lower"], 1e-9)))
                else:
                    _lr, _dr = 0.05, 0.995
                stand[j["lift_idx"]]  = _rtj(j["lift_lower"],  j["lift_upper"],  _lr)
                stand[j["drop_idx"]]  = _rtj(j["drop_lower"],  j["drop_upper"],  _dr)
                stand[j["swing_idx"]] = _rtj(j["swing_lower"], j["swing_upper"], 0.5)
            stand = np.clip(stand, finite_lo, finite_hi)

            ds = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
            ds["pos"] = stand; ds["vel"].fill(0.0)
            gym.set_actor_dof_states(env, actor, ds, gymapi.STATE_ALL)

            if feet_at_ground and triplets:
                fj = next(iter(triplets.values()))
                _lr_r = max(fj["lift_upper"] - fj["lift_lower"], 1e-9)
                _dr_r = max(fj["drop_upper"] - fj["drop_lower"], 1e-9)
                _sl  = max(0.0, min(1.0, (0.0   - fj["lift_lower"]) / _lr_r))
                _swl = max(0.0, min(1.0, (+0.25  - fj["lift_lower"]) / _lr_r))
                _sd  = max(0.0, min(1.0, (0.0   - fj["drop_lower"]) / _dr_r))
                _swd = _sd
            else:
                _sl, _swl, _sd, _swd = STANCE_LIFT_RATIO, SWING_LIFT_RATIO, STANCE_DROP_RATIO, SWING_DROP_RATIO

            topo    = gait_plan["topology"]
            fwd_arr = np.asarray(gait_plan.get("final_forward_axis", [1.0, 0.0]), dtype=float)
            fwd_arr = fwd_arr / max(float(np.linalg.norm(fwd_arr)), 1e-9)
            lat_arr = np.array([-fwd_arr[1], fwd_arr[0]], dtype=float)

            fmap: Dict[int, np.ndarray] = {}
            for lk in description.get("links", []):
                if lk.get("role") == "foot" and lk.get("leg_id") is not None:
                    fmap[int(lk["leg_id"])] = np.asarray(lk["default_world_origin"], dtype=float)[:2]

            per_amp = {str(k): float(v) for k, v in topo.get("per_leg_stride_amplitudes", {}).items()}
            if "_per_amp_override" in gait_plan:
                per_amp = {str(k): float(v) for k, v in gait_plan["_per_amp_override"].items()}

            if fmap:
                foot_pts = np.array(list(fmap.values()), dtype=float)
                body_length = float(np.max(foot_pts.max(axis=0) - foot_pts.min(axis=0)))
            else:
                body_length = 0.5

            robot_states.append({
                "env": env, "actor": actor,
                "triplets": triplets,
                "stand":    stand.copy(),
                "stand_ev": stand.copy(),
                "lower": lower, "upper": upper, "finite_lo": finite_lo, "finite_hi": finite_hi,
                "dof_names": list(dof_names),
                "feet_at_ground": feet_at_ground,
                "_sl": _sl, "_swl": _swl, "_sd": _sd, "_swd": _swd,
                "fwd": fwd_arr, "lat": lat_arr, "fmap": fmap, "per_amp": per_amp,
                "group_a":  topo["groups"]["group_a"],
                "group_b":  topo["groups"]["group_b"],
                "group_c":  topo["groups"].get("group_c", []),
                "touchdown_ramp": {},
                "com_trail": [],
                "forward_axis": fwd_arr.tolist(),
                "min_travel": body_length * max(MIN_TRAVEL_BODY_LENGTHS, 0.0),
                "offset": np.array([ox, oy], dtype=float),
                "done": False,
            })

        valid_idx = [i for i, rs in enumerate(robot_states) if rs is not None]
        if not valid_idx:
            gym.destroy_sim(sim)
            return [([], list(c[2].get("final_forward_axis", [1.0, 0.0]))) for c in configs]

        print(f"  [ParallelSim] Running {len(valid_idx)}/{N} robots on GPU={use_gpu}")

        # ── Warmup ──────────────────────────────────────────────────────────
        for _ in range(HOLD_STEPS):
            for i in valid_idx:
                rs = robot_states[i]
                ds_pos = gym.get_actor_dof_states(rs["env"], rs["actor"], gymapi.STATE_POS)
                jpos = np.asarray(ds_pos["pos"], dtype=np.float32)
                se = rs["stand_ev"]
                for idx, name in enumerate(rs["dof_names"]):
                    sag = se[idx] - jpos[idx]
                    if "_drop" in name and sag > 0.004:
                        se[idx] = min(rs["finite_hi"][idx], se[idx] + min(0.012, 0.22 * sag))
                    elif "_lift" in name and sag > 0.004:
                        se[idx] = max(rs["finite_lo"][idx], se[idx] - min(0.008, 0.16 * sag))
                se = np.clip(se, rs["finite_lo"], rs["finite_hi"])
                rs["stand_ev"] = se
                gym.set_actor_dof_position_targets(rs["env"], rs["actor"], se)
            gym.simulate(sim)
            gym.fetch_results(sim, True)

        # Record initial absolute CoM (post-warmup)
        for i in valid_idx:
            rs = robot_states[i]
            st = gym.get_actor_rigid_body_states(rs["env"], rs["actor"], gymapi.STATE_POS)
            if st is not None and len(st) > 0:
                p = st["pose"]["p"][0]
                rs["init_com"] = np.array([float(p["x"]), float(p["y"])], dtype=float)
            else:
                rs["init_com"] = rs["offset"].copy()

        # ── Gait loop ────────────────────────────────────────────────────────
        _max_steps = n_max_steps if n_max_steps is not None else max(MAX_SIM_STEPS, SIM_STEPS)
        sim_time = 0.0

        for step in range(_max_steps):
            phase_now = 2.0 * math.pi * GAIT_FREQUENCY * sim_time

            for i in valid_idx:
                rs = robot_states[i]
                if rs["done"]:
                    continue
                targets  = rs["stand_ev"].copy()
                tri      = rs["triplets"]
                group_a  = rs["group_a"]
                group_b  = rs["group_b"]
                group_c  = rs["group_c"]
                lat      = rs["lat"]
                per_amp  = rs["per_amp"]
                _sl, _swl, _sd, _swd = rs["_sl"], rs["_swl"], rs["_sd"], rs["_swd"]
                tdr      = rs["touchdown_ramp"]

                for lid, j in tri.items():
                    if lid in group_c:
                        targets[j["lift_idx"]]  = _rtj(j["lift_lower"], j["lift_upper"], _sl)
                        targets[j["drop_idx"]]  = _rtj(j["drop_lower"], j["drop_upper"], _sd)
                        targets[j["swing_idx"]] = _rtj(j["swing_lower"], j["swing_upper"], 0.5)
                        continue

                    if   lid in group_b: lg_ph = phase_now + math.pi
                    elif lid in group_a: lg_ph = phase_now
                    else:               lg_ph = 0.0

                    sw    = float(math.sin(lg_ph))
                    alpha = _ss(-0.30, 0.30, sw)
                    fv    = rs["fmap"].get(lid, np.zeros(2))
                    dsign = -1.0 if float(np.dot(fv, lat)) > 0.0 else 1.0
                    eff_amp = SWING_AMP * float(per_amp.get(str(lid), 1.0))

                    lr = _sl  + (_swl - _sl) * alpha
                    dr = _sd  + (_swd - _sd) * alpha
                    sr = 0.5  + eff_amp * dsign * sw

                    is_sw = sw > 0.0
                    if not is_sw:
                        if lid in tdr:
                            tdr[lid] += 1
                            rp = min(tdr[lid] / 8, 1.0)
                            if rp >= 1.0:
                                del tdr[lid]
                            else:
                                dl = _rtj(j["lift_lower"], j["lift_upper"], 0.5)
                                dd = _rtj(j["drop_lower"], j["drop_upper"], 0.5)
                                ds = j["swing_lower"] + 0.5 * (j["swing_upper"] - j["swing_lower"])
                                targets[j["lift_idx"]]  = dl + rp * (_rtj(j["lift_lower"],  j["lift_upper"],  lr) - dl)
                                targets[j["drop_idx"]]  = dd + rp * (_rtj(j["drop_lower"],  j["drop_upper"],  dr) - dd)
                                targets[j["swing_idx"]] = ds + rp * (_rtj(j["swing_lower"], j["swing_upper"], sr) - ds)
                                continue
                    else:
                        tdr[lid] = 0

                    targets[j["lift_idx"]]  = _rtj(j["lift_lower"], j["lift_upper"], lr)
                    targets[j["drop_idx"]]  = _rtj(j["drop_lower"], j["drop_upper"], dr)
                    targets[j["swing_idx"]] = _rtj(j["swing_lower"], j["swing_upper"], sr)

                gym.set_actor_dof_position_targets(rs["env"], rs["actor"], targets)

            # ── GPU physics step (all robots simultaneously) ──────────────
            gym.simulate(sim)
            gym.fetch_results(sim, True)

            # Collect CoM positions (relative to post-warmup start)
            for i in valid_idx:
                rs = robot_states[i]
                if rs["done"]:
                    continue
                st = gym.get_actor_rigid_body_states(rs["env"], rs["actor"], gymapi.STATE_POS)
                if st is not None and len(st) > 0:
                    p = st["pose"]["p"][0]
                    abs_com = np.array([float(p["x"]), float(p["y"])], dtype=float)
                    rs["com_trail"].append((abs_com - rs["init_com"]).tolist())

            sim_time += _dt

            # Early exit: mark done when each robot covers MIN_TRAVEL
            if step >= SIM_STEPS and MIN_TRAVEL_BODY_LENGTHS > 0.0:
                for i in valid_idx:
                    rs = robot_states[i]
                    if rs["done"] or len(rs["com_trail"]) < 2:
                        continue
                    trail_arr = np.array(rs["com_trail"], dtype=float)
                    covered = abs(float(np.dot(trail_arr[-1] - trail_arr[0], rs["fwd"])))
                    if covered >= rs["min_travel"]:
                        rs["done"] = True

        results = []
        for i in range(N):
            if robot_states[i] is None:
                fwd_fallback = list(configs[i][2].get("final_forward_axis", [1.0, 0.0]))
                results.append(([], fwd_fallback))
            else:
                rs = robot_states[i]
                # Convert relative trail back to absolute for compatibility with plot_demo
                init = rs["init_com"]
                abs_trail = [[p[0] + float(init[0]), p[1] + float(init[1])]
                             for p in rs["com_trail"]]
                results.append((abs_trail, rs["forward_axis"]))
        return results

    finally:
        gym.destroy_sim(sim)



_PLOT_SCRIPT = REPO_ROOT / "_batch_plot.py"


def plot_demo(
    robot_name: str,
    description: dict,
    ssm_result: dict,
    com_trail: List[List[float]],
    forward_axis: List[float],
    fwd_dist: float,
    lat_dist: float,
    out_path: Path,
) -> None:
    import tempfile
    data = {
        "robot_name":  robot_name,
        "description": description,
        "ssm_result":  ssm_result,
        "com_trail":   com_trail,
        "forward_axis": forward_axis,
        "fwd_dist":    fwd_dist,
        "lat_dist":    lat_dist,
        "out_path":    str(out_path),
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [GEN_PYTHON, str(_PLOT_SCRIPT), tmp_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  [Plot] FAILED:\n{result.stderr[:300]}")
        else:
            print(f"  [Plot] Saved → {out_path}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Batch Analysis Report
# ---------------------------------------------------------------------------

def generate_batch_report(out_root: Path, summary_rows: List[dict]) -> Path:
    """Analyse summary_rows and write a Markdown report to out_root/report.md."""

    ok_rows = [r for r in summary_rows if r.get("status") == "ok"]
    fail_rows = [r for r in summary_rows if r.get("status") != "ok"]

    # ── Categorise ok robots ─────────────────────────────────────────────────
    good, stuck, circular, backward = [], [], [], []
    for r in ok_rows:
        fwd = r.get("fwd_dist", 0.0)
        lat = r.get("lat_dist", 0.0)
        abs_fwd = abs(fwd)
        ratio = abs(lat) / max(abs_fwd, 0.01)
        if fwd < -0.10:
            backward.append(r)
        elif abs_fwd < 0.15:
            stuck.append(r)
        elif ratio > 1.8:
            circular.append(r)
        else:
            good.append(r)

    total = len(summary_rows)
    n_ok  = len(ok_rows)

    fwd_vals  = [r["fwd_dist"] for r in ok_rows] if ok_rows else [0.0]
    lat_vals  = [abs(r["lat_dist"]) for r in ok_rows] if ok_rows else [0.0]
    ssm_vals  = [r["ssm"] for r in summary_rows if "ssm" in r]
    leg_vals  = [r["num_legs"] for r in ok_rows if "num_legs" in r]

    def _fmt_list(lst):
        return ", ".join(r["robot"] for r in lst) if lst else "—"

    lines = [
        f"# Batch Test Report",
        f"",
        f"**Run directory**: `{out_root}`  ",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Config**: {NUM_ROBOTS} robots, seeds={SEEDS[:5]}{'…' if len(SEEDS)>5 else ''}  ",
        f"**GPU**: {USE_GPU}, batch_size={GPU_BATCH_SIZE}",
        f"**EKF Online**: {'enabled' if USE_ONLINE_EKF else 'disabled'} (drift threshold={EKF_DRIFT_THRESHOLD})",
        f"",
        f"---",
        f"",
        f"## 1. Summary",
        f"",
        f"| Category | Count | % |",
        f"|---|---:|---:|",
        f"| Total robots | {total} | 100% |",
        f"| Simulation OK | {n_ok} | {100*n_ok//max(total,1)}% |",
        f"| **Good (straight forward)** | {len(good)} | {100*len(good)//max(total,1)}% |",
        f"| Circular motion | {len(circular)} | {100*len(circular)//max(total,1)}% |",
        f"| Stuck / barely moving | {len(stuck)} | {100*len(stuck)//max(total,1)}% |",
        f"| Backward motion | {len(backward)} | {100*len(backward)//max(total,1)}% |",
        f"| Failed (gen/sim error) | {len(fail_rows)} | {100*len(fail_rows)//max(total,1)}% |",
        f"",
        f"## 2. Motion Statistics (OK robots only)",
        f"",
        f"| Metric | Min | Mean | Max |",
        f"|---|---:|---:|---:|",
        f"| Forward dist (m) | {min(fwd_vals):+.3f} | {sum(fwd_vals)/len(fwd_vals):+.3f} | {max(fwd_vals):+.3f} |",
        f"| |Lateral| dist (m) | {min(lat_vals):.3f} | {sum(lat_vals)/len(lat_vals):.3f} | {max(lat_vals):.3f} |",
    ]
    if ssm_vals:
        lines += [
            f"| SSM (m) | {min(ssm_vals):.4f} | {sum(ssm_vals)/len(ssm_vals):.4f} | {max(ssm_vals):.4f} |",
        ]
    if leg_vals:
        from collections import Counter
        leg_dist = Counter(leg_vals)
        lines += [
            f"",
            f"**Leg count distribution**: " +
            ", ".join(f"{k}-legged: {v}" for k, v in sorted(leg_dist.items())),
        ]

    lines += [
        f"",
        f"## 3. Robot Categories",
        f"",
        f"**✓ Good (forward, low yaw)**:  ",
        f"{_fmt_list(good)}",
        f"",
        f"**↺ Circular motion** (|lat|/|fwd| > 1.8):  ",
        f"{_fmt_list(circular)}",
        f"",
        f"**✗ Stuck** (|fwd| < 0.15 m):  ",
        f"{_fmt_list(stuck)}",
        f"",
        f"**← Backward** (fwd < −0.10 m):  ",
        f"{_fmt_list(backward)}",
        f"",
        f"**⚠ Failed**:  ",
        f"{_fmt_list(fail_rows)}",
        f"",
        f"## 4. Per-Robot Details",
        f"",
        f"| Robot | Status | Legs | SSM | Fwd (m) | Lat (m) | Category |",
        f"|---|---|---:|---:|---:|---:|---|",
    ]
    for r in summary_rows:
        if r.get("status") == "ok":
            fwd = r.get("fwd_dist", 0.0)
            lat = r.get("lat_dist", 0.0)
            ratio = abs(lat) / max(abs(fwd), 0.01)
            if fwd < -0.10:                cat = "← backward"
            elif abs(fwd) < 0.15:          cat = "✗ stuck"
            elif ratio > 1.8:              cat = "↺ circular"
            else:                          cat = "✓ good"
        else:
            fwd = lat = float("nan")
            cat = f"⚠ {r.get('status', '?')}"
        ssm_str = f"{r['ssm']:.3f}" if "ssm" in r else "—"
        legs_str = str(r.get("num_legs", "—"))
        fwd_str = f"{fwd:+.3f}" if not math.isnan(fwd) else "—"
        lat_str = f"{lat:+.3f}" if not math.isnan(lat) else "—"
        lines.append(
            f"| {r['robot']} | {r.get('status','')} | {legs_str} | {ssm_str}"
            f" | {fwd_str} | {lat_str} | {cat} |"
        )

    lines += [
        f"",
        f"## 5. Diagnosis & Recommendations",
        f"",
    ]

    # Auto-generated diagnosis
    if len(circular) > len(good):
        lines += [
            f"**[高偏航率]** {len(circular)}/{n_ok} 个机器人发生圆弧运动，",
            f"说明步态偏航补偿仍不充分。建议：",
            f"- 增大 `YAW_COMP_GAIN`（adaptive_gait.py）至 0.70+",
            f"- 检查 `yaw_levers` 是否过小（对称机器人应接近 0）",
            f"- 考虑增加偏航修正迭代次数 `MAX_YAW_ITERS`",
            f"",
        ]
    if len(stuck) > 0.2 * n_ok:
        lines += [
            f"**[卡死/无法移动]** {len(stuck)}/{n_ok} 个机器人无明显前进，",
            f"可能原因：",
            f"- 形态极端不对称导致方向选择失败",
            f"- SSM 过低导致站立不稳",
            f"- 步态频率/幅度不匹配该形态关节范围",
            f"建议：降低 `SWING_AMP`，增加 `PROBE_STEPS`",
            f"",
        ]
    if len(backward) > 0:
        lines += [
            f"**[后退]** {len(backward)} 个机器人向后运动，",
            f"说明前进方向判断出错（swing vector 方向与实际相反）。",
            f"建议：检查 `_build_default_swing_vector` 中的 outward 方向约定",
            f"",
        ]
    if not (len(circular) > len(good) or len(stuck) > 0.2 * n_ok or backward):
        lines += [
            f"整体表现良好。好的机器人比例：{len(good)}/{n_ok}。",
            f"",
        ]

    fwd_mean = sum(fwd_vals) / len(fwd_vals) if fwd_vals else 0
    lat_mean = sum(lat_vals) / len(lat_vals) if lat_vals else 0
    drift_mean = sum(drift_vals) / len(drift_vals) if drift_vals else 0
    lines += [
        f"**平均前进距离**: {fwd_mean:+.3f} m，**平均偏移**: {lat_mean:.3f} m",
        f"**平均漂移比**: {drift_mean:.3f}（越小越直；< 0.3 视为良好）",
        f"**EKF在线修正**: {ekf_count}/{len(ok_rows)} 机器人触发 ({ekf_count/max(len(ok_rows),1)*100:.1f}%)",
        f"",
        f"---",
        f"*Auto-generated by batch_test.py*",
    ]

    report_path = out_root / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------------

def generate_html_report(out_root: Path, summary_rows: List[dict]) -> Path:
    """Generate a self-contained HTML report with sortable table and trajectory thumbnails.

    Images are referenced as relative paths so the HTML works as long as the
    out_root directory structure is intact.  No external CDN dependencies.
    """
    import base64, math as _math

    ok_rows = [r for r in summary_rows if r.get("status") == "ok"]
    total   = len(summary_rows)
    n_ok    = len(ok_rows)

    # ── Categorise ────────────────────────────────────────────────────────────
    def _cat(r: dict) -> str:
        if r.get("status") != "ok":
            return "failed"
        fwd = r.get("fwd_dist", 0.0)
        lat = r.get("lat_dist", 0.0)
        ratio = abs(lat) / max(abs(fwd), 0.01)
        if fwd < -0.10:         return "backward"
        if abs(fwd) < 0.15:     return "stuck"
        if ratio > 1.8:         return "circular"
        return "good"

    cat_label  = {"good": "✓ Good", "circular": "↺ Circular",
                  "stuck": "✗ Stuck", "backward": "← Backward", "failed": "⚠ Failed"}
    cat_color  = {"good": "#d4edda", "circular": "#fff3cd",
                  "stuck": "#f8d7da", "backward": "#f8d7da", "failed": "#e2e3e5"}
    cat_counts = {k: 0 for k in cat_label}
    for r in summary_rows:
        cat_counts[_cat(r)] += 1

    fwd_vals = [r["fwd_dist"] for r in ok_rows] if ok_rows else [0.0]
    lat_vals = [abs(r["lat_dist"]) for r in ok_rows] if ok_rows else [0.0]
    drift_vals = [r.get("drift_ratio", abs(r["lat_dist"])/max(abs(r["fwd_dist"]),0.01))
                  for r in ok_rows] if ok_rows else [0.0]
    ekf_count = sum(1 for r in ok_rows if r.get("strategy") == "ekf_online")
    ssm_vals = [r["ssm"] for r in summary_rows if "ssm" in r]

    def _mean(lst): return sum(lst) / len(lst) if lst else 0.0

    # ── Build table rows (HTML) ───────────────────────────────────────────────
    row_html_parts: list[str] = []
    for r in summary_rows:
        name   = r["robot"]
        status = r.get("status", "?")
        cat    = _cat(r)
        bg     = cat_color[cat]
        label  = cat_label[cat]

        ssm_str  = f"{r['ssm']:.3f}"      if "ssm"      in r else "—"
        legs_str = str(r.get("num_legs", "—"))
        fwd      = r.get("fwd_dist", _math.nan)
        lat      = r.get("lat_dist", _math.nan)
        drift    = r.get("drift_ratio", _math.nan)
        strategy = r.get("strategy", "baseline")
        fwd_str  = f"{fwd:+.3f}" if not _math.isnan(fwd) else "—"
        lat_str  = f"{lat:+.3f}" if not _math.isnan(lat) else "—"
        drift_str = f"{drift:.3f}" if not _math.isnan(drift) else "—"
        fwd_data = f"{fwd:.6f}"  if not _math.isnan(fwd) else "-9999"
        lat_data = f"{lat:.6f}"  if not _math.isnan(lat) else "-9999"
        drift_data = f"{drift:.6f}" if not _math.isnan(drift) else "-9999"
        ssm_data = f"{r['ssm']:.6f}" if "ssm" in r else "-9999"
        legs_data = str(r.get("num_legs", 0))

        # Thumbnail: relative path from report.html sibling location
        img_rel  = f"{name}/trajectory.png"
        img_path = out_root / name / "trajectory.png"
        if img_path.exists():
            thumb_td = (
                f'<td class="thumb-cell">'
                f'<img class="thumb" src="{img_rel}" alt="{name}" '
                f'onclick="openModal(\'{img_rel}\', \'{name}\')" loading="lazy"/>'
                f'</td>'
            )
        else:
            thumb_td = '<td class="thumb-cell"><span style="color:#aaa">—</span></td>'

        row_html_parts.append(
            f'<tr style="background:{bg}" data-cat="{cat}">'
            f'<td>{name}</td>'
            f'<td>{status}</td>'
            f'<td>{legs_str}</td>'
            f'<td data-val="{ssm_data}">{ssm_str}</td>'
            f'<td data-val="{fwd_data}">{fwd_str}</td>'
            f'<td data-val="{lat_data}">{lat_str}</td>'
            f'<td data-val="{drift_data}">{drift_str}</td>'
            f'<td>{strategy}</td>'
            f'<td>{label}</td>'
            f'{thumb_td}'
            f'</tr>'
        )

    rows_html = "\n".join(row_html_parts)

    # ── Summary section ────────────────────────────────────────────────────────
    def _pct(n): return f"{100*n//max(total,1)}%"
    summary_rows_html = "".join(
        f"<tr><td>{name}</td><td>{cat_counts[k]}</td><td>{_pct(cat_counts[k])}</td></tr>"
        for k, name in cat_label.items()
    )

    # ── Assemble full HTML ─────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Batch Test Report — {out_root.name}</title>
<style>
  body  {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px;
           background: #f5f5f5; color: #333; }}
  h1   {{ font-size: 1.5rem; margin-bottom: 4px; }}
  h2   {{ font-size: 1.1rem; border-bottom: 2px solid #ccc; padding-bottom: 4px;
           margin-top: 28px; }}
  .meta {{ color: #666; font-size: 0.85rem; margin-bottom: 20px; }}
  .cards {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 12px 0 20px; }}
  .card {{ border-radius: 8px; padding: 12px 20px; min-width: 120px;
           box-shadow: 0 1px 3px rgba(0,0,0,.15); text-align: center; }}
  .card .num {{ font-size: 2rem; font-weight: 700; line-height: 1; }}
  .card .lbl {{ font-size: 0.78rem; color: #555; margin-top: 4px; }}
  .c-good     {{ background:#d4edda; }}
  .c-circular {{ background:#fff3cd; }}
  .c-stuck    {{ background:#f8d7da; }}
  .c-backward {{ background:#f8d7da; }}
  .c-failed   {{ background:#e2e3e5; }}
  .c-ok       {{ background:#cce5ff; }}
  table    {{ border-collapse: collapse; width: 100%; background: #fff;
              box-shadow: 0 1px 3px rgba(0,0,0,.1); font-size: 0.82rem; }}
  th       {{ background: #343a40; color: #fff; padding: 8px 10px;
              cursor: pointer; user-select: none; white-space: nowrap; }}
  th:hover {{ background: #495057; }}
  th .sort-icon {{ margin-left: 4px; opacity: .5; }}
  td       {{ padding: 5px 10px; border-bottom: 1px solid #e0e0e0;
              vertical-align: middle; }}
  tr:hover {{ filter: brightness(0.95); }}
  .thumb-cell {{ width: 90px; text-align: center; padding: 3px; }}
  .thumb   {{ width: 80px; height: auto; cursor: zoom-in; border-radius: 4px;
              transition: transform .15s; }}
  .thumb:hover {{ transform: scale(1.12); }}
  .filter-bar {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 12px 0; }}
  .filter-btn {{ border: none; border-radius: 16px; padding: 5px 14px;
                 cursor: pointer; font-size: 0.82rem; opacity: 0.55;
                 transition: opacity .15s; }}
  .filter-btn.active {{ opacity: 1; font-weight: 600; box-shadow: 0 0 0 2px #333; }}
  .stats-table {{ width: auto; min-width: 320px; }}
  .stats-table td, .stats-table th {{ padding: 6px 14px; }}
  /* Modal lightbox */
  #modal {{ display:none; position:fixed; inset:0; background:rgba(0,0,0,.75);
            z-index:999; align-items:center; justify-content:center; flex-direction:column; }}
  #modal.open {{ display:flex; }}
  #modal img  {{ max-width:90vw; max-height:85vh; border-radius:6px;
                 box-shadow:0 4px 20px rgba(0,0,0,.5); }}
  #modal-caption {{ color:#fff; margin-top:10px; font-size:0.9rem; }}
  #modal-close {{ position:fixed; top:14px; right:20px; color:#fff; font-size:2rem;
                  cursor:pointer; line-height:1; }}
</style>
</head>
<body>

<h1>Batch Test Report</h1>
<div class="meta">
  Run: <code>{out_root}</code> &nbsp;|&nbsp;
  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp;
  {NUM_ROBOTS} robots
</div>

<h2>1. Overview</h2>
<div class="cards">
  <div class="card c-ok">
    <div class="num">{total}</div><div class="lbl">Total</div>
  </div>
  <div class="card c-ok">
    <div class="num">{n_ok}</div><div class="lbl">Sim OK ({_pct(n_ok)})</div>
  </div>
  <div class="card c-good">
    <div class="num">{cat_counts['good']}</div><div class="lbl">Good ({_pct(cat_counts['good'])})</div>
  </div>
  <div class="card c-circular">
    <div class="num">{cat_counts['circular']}</div><div class="lbl">Circular</div>
  </div>
  <div class="card c-stuck">
    <div class="num">{cat_counts['stuck']}</div><div class="lbl">Stuck</div>
  </div>
  <div class="card c-backward">
    <div class="num">{cat_counts['backward']}</div><div class="lbl">Backward</div>
  </div>
  <div class="card c-failed">
    <div class="num">{cat_counts['failed']}</div><div class="lbl">Failed</div>
  </div>
</div>

<h2>2. Motion Statistics (OK robots only)</h2>
<table class="stats-table">
  <thead><tr><th>Metric</th><th>Min</th><th>Mean</th><th>Max</th></tr></thead>
  <tbody>
    <tr><td>Forward dist (m)</td>
        <td>{min(fwd_vals):+.3f}</td>
        <td>{_mean(fwd_vals):+.3f}</td>
        <td>{max(fwd_vals):+.3f}</td></tr>
    <tr><td>|Lateral| dist (m)</td>
        <td>{min(lat_vals):.3f}</td>
        <td>{_mean(lat_vals):.3f}</td>
        <td>{max(lat_vals):.3f}</td></tr>
    {"<tr><td>SSM (m)</td><td>" + f"{min(ssm_vals):.4f}</td><td>{_mean(ssm_vals):.4f}</td><td>{max(ssm_vals):.4f}</td></tr>" if ssm_vals else ""}
    <tr><td>Drift ratio</td>
        <td>{min(drift_vals):.3f}</td>
        <td>{_mean(drift_vals):.3f}</td>
        <td>{max(drift_vals):.3f}</td></tr>
    <tr><td colspan="4">EKF online triggered: {ekf_count}/{total} robots</td></tr>
  </tbody>
</table>

<h2>3. Per-Robot Details</h2>
<div class="filter-bar" id="filter-bar">
  <button class="filter-btn active c-ok" data-cat="all" onclick="filterCat(this,'all')">All ({total})</button>
  <button class="filter-btn c-good" data-cat="good" onclick="filterCat(this,'good')">✓ Good ({cat_counts['good']})</button>
  <button class="filter-btn c-circular" data-cat="circular" onclick="filterCat(this,'circular')">↺ Circular ({cat_counts['circular']})</button>
  <button class="filter-btn c-stuck" data-cat="stuck" onclick="filterCat(this,'stuck')">✗ Stuck ({cat_counts['stuck']})</button>
  <button class="filter-btn c-backward" data-cat="backward" onclick="filterCat(this,'backward')">← Backward ({cat_counts['backward']})</button>
  <button class="filter-btn c-failed" data-cat="failed" onclick="filterCat(this,'failed')">⚠ Failed ({cat_counts['failed']})</button>
</div>
<table id="robot-table">
  <thead>
    <tr>
      <th onclick="sortTable(0)">Robot<span class="sort-icon">⇅</span></th>
      <th onclick="sortTable(1)">Status<span class="sort-icon">⇅</span></th>
      <th onclick="sortTable(2)">Legs<span class="sort-icon">⇅</span></th>
      <th onclick="sortTable(3)">SSM<span class="sort-icon">⇅</span></th>
      <th onclick="sortTable(4)">Fwd (m)<span class="sort-icon">⇅</span></th>
      <th onclick="sortTable(5)">Lat (m)<span class="sort-icon">⇅</span></th>
      <th onclick="sortTable(6)">Drift<span class="sort-icon">⇅</span></th>
      <th onclick="sortTable(7)">Strategy<span class="sort-icon">⇅</span></th>
      <th onclick="sortTable(8)">Category<span class="sort-icon">⇅</span></th>
      <th>Trajectory</th>
    </tr>
  </thead>
  <tbody id="robot-tbody">
{rows_html}
  </tbody>
</table>

<!-- Lightbox modal -->
<div id="modal" onclick="closeModal()">
  <span id="modal-close" onclick="closeModal()">&#x2715;</span>
  <img id="modal-img" src="" alt=""/>
  <div id="modal-caption"></div>
</div>

<script>
// ── Sort ──────────────────────────────────────────────────────────────────
let _sortCol = -1, _sortAsc = true;
function sortTable(col) {{
  const tbody = document.getElementById('robot-tbody');
  const rows  = Array.from(tbody.querySelectorAll('tr'));
  _sortAsc = (_sortCol === col) ? !_sortAsc : true;
  _sortCol = col;
  rows.sort((a, b) => {{
    const ta = a.querySelectorAll('td')[col];
    const tb = b.querySelectorAll('td')[col];
    // Use data-val for numeric cols (3,4,5)
    let va = (col >= 3 && col <= 5)
              ? parseFloat(ta.dataset.val ?? ta.textContent)
              : ta.textContent.trim();
    let vb = (col >= 3 && col <= 5)
              ? parseFloat(tb.dataset.val ?? tb.textContent)
              : tb.textContent.trim();
    if (typeof va === 'string') return _sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
    return _sortAsc ? va - vb : vb - va;
  }});
  rows.forEach(r => tbody.appendChild(r));
  // Update header icons
  document.querySelectorAll('th .sort-icon').forEach((ic, i) => {{
    ic.textContent = (i === col) ? (_sortAsc ? ' ↑' : ' ↓') : ' ⇅';
    ic.style.opacity = (i === col) ? '1' : '0.5';
  }});
}}

// ── Filter ────────────────────────────────────────────────────────────────
let _activeCat = 'all';
function filterCat(btn, cat) {{
  _activeCat = cat;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('#robot-tbody tr').forEach(row => {{
    row.style.display = (cat === 'all' || row.dataset.cat === cat) ? '' : 'none';
  }});
}}

// ── Lightbox ──────────────────────────────────────────────────────────────
function openModal(src, caption) {{
  document.getElementById('modal-img').src = src;
  document.getElementById('modal-caption').textContent = caption;
  document.getElementById('modal').classList.add('open');
}}
function closeModal() {{
  document.getElementById('modal').classList.remove('open');
}}
document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closeModal(); }});
</script>

<p style="color:#aaa;font-size:0.75rem;margin-top:30px">
  Auto-generated by batch_test.py &mdash; {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
</p>
</body>
</html>
"""

    html_path = out_root / "report.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path


# ---------------------------------------------------------------------------
# EKF Online Yaw Correction (方向C 集成)
# ---------------------------------------------------------------------------

def run_ekf_online_simulation(
    description: dict,
    urdf_path: Path,
    plan: dict,
    n_steps: int = 480,
) -> Tuple[List[List[float]], List[float], Optional[Dict]]:
    """Run a single-robot simulation with EKF online yaw correction.

    Returns (com_trail, forward_axis, yaw_stats).
    """
    from online_state_estimator import OnlineStateEstimator, make_observation

    fwd = np.asarray(plan.get("final_forward_axis", [1.0, 0.0]), dtype=float)
    fwd = fwd / max(float(np.linalg.norm(fwd)), 1e-9)
    lat = np.array([-fwd[1], fwd[0]], dtype=float)

    topo = plan["topology"]
    group_a = topo["groups"]["group_a"]
    group_b = topo["groups"]["group_b"]
    group_c = topo["groups"].get("group_c", [])
    base_amps = {str(k): float(v) for k, v in topo.get("per_leg_stride_amplitudes", {}).items()}

    total_mass = sum(
        float(lk.get("mass_properties", {}).get("mass", 0.0))
        for lk in description.get("links", [])
    )
    estimator = OnlineStateEstimator(description, plan, total_mass=max(total_mass, 1.0))

    with _RobotSimCtx(description, urdf_path, use_gpu=USE_GPU) as ctx:
        ga = ctx._ga
        gym, sim, env, actor = ctx.gym, ctx.sim, ctx.env, ctx.actor
        ctx._reset()

        def _rtj(lo, hi, r):
            return float(lo + max(0.0, min(1.0, r)) * (hi - lo))
        def _ss(e0, e1, x):
            t = max(0.0, min(1.0, (x - e0) / max(e1 - e0, 1e-9)))
            return t * t * (3.0 - 2.0 * t)

        _sl, _swl, _sd, _swd = ctx._sl, ctx._swl, ctx._sd, ctx._swd
        com_trail: List[List[float]] = []
        yaw_acc:   List[float] = []
        touchdown_ramp: Dict[int, int] = {}
        sim_time = 0.0
        _dt = 1.0 / 60.0

        for step in range(n_steps):
            phase_now = 2.0 * math.pi * GAIT_FREQUENCY * sim_time
            targets = ctx.stand_ev.copy()

            adapt = estimator.get_state()
            online_scales = {str(k): float(v) for k, v in adapt.adaptive_stride_scales.items()}

            for lid, j in ctx.triplets.items():
                if lid in group_c:
                    targets[j["lift_idx"]]  = _rtj(j["lift_lower"], j["lift_upper"], _sl)
                    targets[j["drop_idx"]]  = _rtj(j["drop_lower"], j["drop_upper"], _sd)
                    targets[j["swing_idx"]] = _rtj(j["swing_lower"], j["swing_upper"], 0.5)
                    continue

                if lid in group_b: lg_ph = phase_now + math.pi
                elif lid in group_a: lg_ph = phase_now
                else: lg_ph = 0.0

                sw_val = float(math.sin(lg_ph))
                alpha = _ss(-0.30, 0.30, sw_val)
                fv = ctx.fmap.get(lid, np.zeros(2))
                dsign = -1.0 if float(np.dot(fv, lat)) > 0.0 else 1.0

                base_scale = float(base_amps.get(str(lid), 1.0))
                online_scale = float(online_scales.get(str(lid), 1.0))
                eff_amp = SWING_AMP * base_scale * online_scale

                lr = _sl + (_swl - _sl) * alpha
                dr = _sd + (_swd - _sd) * alpha
                sr = 0.5 + eff_amp * dsign * sw_val

                is_sw = sw_val > 0.0
                if not is_sw:
                    if lid in touchdown_ramp:
                        touchdown_ramp[lid] += 1
                        rp = min(touchdown_ramp[lid] / 8, 1.0)
                        if rp >= 1.0:
                            del touchdown_ramp[lid]
                        else:
                            dl  = _rtj(j["lift_lower"],  j["lift_upper"],  0.5)
                            dd  = _rtj(j["drop_lower"],  j["drop_upper"],  0.5)
                            ds_ = j["swing_lower"] + 0.5 * (j["swing_upper"] - j["swing_lower"])
                            targets[j["lift_idx"]]  = dl  + rp * (_rtj(j["lift_lower"],  j["lift_upper"],  lr) - dl)
                            targets[j["drop_idx"]]  = dd  + rp * (_rtj(j["drop_lower"],  j["drop_upper"],  dr) - dd)
                            targets[j["swing_idx"]] = ds_ + rp * (_rtj(j["swing_lower"], j["swing_upper"], sr) - ds_)
                            continue
                else:
                    touchdown_ramp[lid] = 0

                targets[j["lift_idx"]]  = _rtj(j["lift_lower"], j["lift_upper"], lr)
                targets[j["drop_idx"]]  = _rtj(j["drop_lower"], j["drop_upper"], dr)
                targets[j["swing_idx"]] = _rtj(j["swing_lower"], j["swing_upper"], sr)

            gym.set_actor_dof_position_targets(env, actor, targets)
            gym.simulate(sim)
            gym.fetch_results(sim, True)

            states = gym.get_actor_rigid_body_states(env, actor, ga.STATE_ALL)
            if states is not None and len(states) > 0:
                qw = float(states["pose"]["r"][0]["w"])
                qx = float(states["pose"]["r"][0]["x"])
                qy = float(states["pose"]["r"][0]["y"])
                qz = float(states["pose"]["r"][0]["z"])
                siny = 2.0 * (qw * qz + qx * qy)
                cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
                yaw = math.atan2(siny, cosy)

                p = states["pose"]["p"][0]
                com_trail.append([float(p["x"]), float(p["y"])])
                yaw_acc.append(float(states["vel"]["angular"][0]["z"]))

                obs = make_observation(
                    body_xy=[float(p["x"]), float(p["y"])],
                    body_attitude=(0.0, 0.0, yaw),
                    sim_time=sim_time, dt=_dt,
                )
                estimator.step(obs)

            sim_time += _dt

        valid = [v for v in yaw_acc if abs(v) < 20.0]
        yaw_stats = {
            "yaw_rate_mean": float(np.mean(valid)) if valid else 0.0,
            "yaw_rate_std": float(np.std(valid)) if valid else 0.0,
        }
        return com_trail, fwd.tolist(), yaw_stats


# ---------------------------------------------------------------------------
# Main batch loop
# ---------------------------------------------------------------------------

def main() -> None:
    if not _GYM_AVAILABLE:
        print("[ERROR] Isaac Gym not available. Check environment.")
        sys.exit(1)

    # Ensure seed list is long enough
    seeds = list(SEEDS)
    while len(seeds) < NUM_ROBOTS:
        seeds.append(random.randint(1, 9999))
    seeds = seeds[:NUM_ROBOTS]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = REPO_ROOT / OUTPUT_DIR / timestamp
    out_root.mkdir(parents=True, exist_ok=True)
    # Create per-batch png/ folder immediately — plots are synced in real-time
    png_dir = out_root / "png"
    png_dir.mkdir(exist_ok=True)
    print(f"[Batch] Output directory: {out_root}")
    print(f"[Batch] Testing {NUM_ROBOTS} robots — seeds: {seeds}\n")

    summary_rows = []
    all_png_paths: List[Path] = []  # collect for flat compare dir

    # ── Optional: standard hexapod as reference entry (index -1) ──────────────
    standard_desc_path = ASSET_ROOT / "standard_hexapod" / "robot_description.json"
    standard_urdf_path = ASSET_ROOT / "standard_hexapod" / "generated_robot.urdf"
    if INCLUDE_STANDARD_HEXAPOD and standard_desc_path.exists() and standard_urdf_path.exists():
        robot_name = "robot_ref_standard"
        print(f"{'='*60}")
        print(f"[Ref] {robot_name}")
        robot_dir = out_root / robot_name
        robot_dir.mkdir(parents=True, exist_ok=True)
        try:
            description = json.loads(standard_desc_path.read_text(encoding="utf-8"))
            # copy assets so URDF mesh paths work
            import shutil as _shutil
            _shutil.copy2(standard_desc_path, robot_dir / "robot_description.json")
            _shutil.copy2(standard_urdf_path, robot_dir / "robot.urdf")
            std_meshes = ASSET_ROOT / "standard_hexapod" / "meshes"
            if std_meshes.exists():
                dst_meshes = robot_dir / "meshes"
                if dst_meshes.exists():
                    _shutil.rmtree(dst_meshes)
                _shutil.copytree(std_meshes, dst_meshes)
            urdf_path_std = robot_dir / "robot.urdf"
            print(f"  [Gen]  legs={description.get('num_legs')}  (standard hexapod)")
            ssm_result = check_stability(description)
            print(f"  [SSM]  ssm={ssm_result['ssm']:.4f}m  passed={ssm_result['passed']}")
            com_trail, forward_axis = optimize_and_simulate(description, urdf_path_std)
            trail = np.array(com_trail, dtype=float) if len(com_trail) > 1 else np.zeros((2, 2))
            fwd   = np.asarray(forward_axis, dtype=float)
            fwd   = fwd / max(float(np.linalg.norm(fwd)), 1e-9)
            lat   = np.array([-fwd[1], fwd[0]], dtype=float)
            disp  = trail[-1] - trail[0] if len(trail) > 1 else np.zeros(2)
            fwd_dist = float(np.dot(disp, fwd))
            lat_dist = float(np.dot(disp, lat))
            print(f"  [Sim]  steps={len(com_trail)}  fwd={fwd_dist:+.3f}m  lat={lat_dist:+.3f}m")
            out_img = robot_dir / "trajectory.png"
            plot_demo(robot_name, description, ssm_result, com_trail, forward_axis,
                      fwd_dist, lat_dist, out_img)
            if out_img.exists():
                all_png_paths.append(out_img)
                shutil.copy2(out_img, png_dir / (robot_name + ".png"))
            summary_rows.append({
                "robot": robot_name, "status": "ok",
                "ssm": ssm_result["ssm"], "num_legs": description.get("num_legs"),
                "fwd_dist": fwd_dist, "lat_dist": lat_dist,
            })
        except Exception as e:
            print(f"  [Ref]  FAILED: {e}")
            summary_rows.append({"robot": robot_name, "status": "ref_failed"})

    # ── Phase 1: generate robots + stability check ─────────────────────────
    pending_probe: List[dict] = []   # valid robots ready for batch probing

    for idx, seed in enumerate(seeds):
        robot_name = f"robot_{idx:02d}_seed{seed}"
        print(f"{'='*60}")
        print(f"[{idx+1}/{NUM_ROBOTS}] {robot_name}")
        robot_dir = out_root / robot_name

        # ── Step 1: generate ──
        try:
            desc_path, urdf_path = generate_robot(seed, robot_dir)
            description = json.loads(desc_path.read_text(encoding="utf-8"))
            print(f"  [Gen]  legs={description.get('num_legs')}  "
                  f"desc={desc_path.name}  urdf={urdf_path.name}")
        except Exception as e:
            print(f"  [Gen]  FAILED: {e}")
            summary_rows.append({"robot": robot_name, "status": "gen_failed"})
            continue

        # ── Step 2: static stability ──
        ssm_result = check_stability(description)
        print(f"  [SSM]  ssm={ssm_result['ssm']:.4f}m  passed={ssm_result['passed']}")

        if SKIP_UNSTABLE and not ssm_result["passed"]:
            print(f"  [Skip] SSM below threshold, skipping gait test.")
            summary_rows.append({"robot": robot_name, "status": "unstable", "ssm": ssm_result["ssm"]})
            out_img = robot_dir / "trajectory.png"
            plot_demo(robot_name, description, ssm_result, [], [1.0, 0.0], 0.0, 0.0, out_img)
            if out_img.exists():
                all_png_paths.append(out_img)
                shutil.copy2(out_img, png_dir / (robot_name + ".png"))
            continue

        # ── Step 3: compute initial gait plan (no simulation) ──
        from adaptive_gait import compute_adaptive_plan
        try:
            initial_plan = compute_adaptive_plan(description, {})
        except Exception as e:
            print(f"  [Plan] FAILED: {e}")
            summary_rows.append({"robot": robot_name, "status": "plan_failed", "ssm": ssm_result["ssm"]})
            continue

        pending_probe.append({
            "robot_name":   robot_name,
            "description":  description,
            "urdf_path":    urdf_path,
            "ssm_result":   ssm_result,
            "initial_plan": initial_plan,
            "robot_dir":    robot_dir,
        })

    # ── Unified pipeline: probe + correction + full-sim, ONE sim per sub-batch ─
    # run_subbatch_full_pipeline creates exactly 1 sim for up to GPU_BATCH_SIZE
    # robots, runs all probe rounds + the full final simulation inside that ONE
    # sim, then destroys it.  Total create_sim calls = ceil(N / GPU_BATCH_SIZE).
    if pending_probe:
        print(f"\n{'='*60}")
        total_batches = math.ceil(len(pending_probe) / GPU_BATCH_SIZE)
        print(f"[Batch] Unified pipeline: {len(pending_probe)} robots, "
              f"{total_batches} sub-batch(es) of ≤{GPU_BATCH_SIZE}, GPU={USE_GPU}")
        for batch_start in range(0, len(pending_probe), GPU_BATCH_SIZE):
            batch = pending_probe[batch_start:batch_start + GPU_BATCH_SIZE]
            batch_num = batch_start // GPU_BATCH_SIZE + 1
            print(f"\n  [Sub-batch {batch_num}/{total_batches}] "
                  f"robots {batch_start}–{batch_start+len(batch)-1}")
            try:
                probe_results, sim_results = run_subbatch_isolated(
                    batch, use_gpu=USE_GPU
                )
            except Exception as e:
                print(f"  [Sub-batch] FAILED: {e} — falling back to sequential")
                probe_results = [
                    (info["initial_plan"], info["initial_plan"]["final_forward_axis"])
                    for info in batch
                ]
                sim_results = [([], [1.0, 0.0])] * len(batch)

            for info, (corrected_plan, forward_axis), (com_trail, _) in zip(
                batch, probe_results, sim_results
            ):
                robot_name  = info["robot_name"]
                description = info["description"]
                ssm_result  = info["ssm_result"]
                robot_dir   = info["robot_dir"]

                fwd  = np.asarray(forward_axis, dtype=float)
                fwd  = fwd / max(float(np.linalg.norm(fwd)), 1e-9)
                lat  = np.array([-fwd[1], fwd[0]], dtype=float)
                trail = np.array(com_trail, dtype=float) if len(com_trail) > 1 else np.zeros((2, 2))
                disp  = trail[-1] - trail[0] if len(trail) > 1 else np.zeros(2)
                fwd_dist = float(np.dot(disp, fwd))
                lat_dist = float(np.dot(disp, lat))
                drift_ratio = abs(lat_dist) / max(abs(fwd_dist), 0.01)
                strategy = "baseline"
                print(f"  [Sim]  {robot_name}  steps={len(com_trail)}"
                      f"  fwd={fwd_dist:+.3f}m  lat={lat_dist:+.3f}m  "
                      f"drift={drift_ratio:.2f}")

                # ── EKF online fallback for high-drift robots ──────────────
                if USE_ONLINE_EKF and drift_ratio > EKF_DRIFT_THRESHOLD and len(com_trail) > 1:
                    try:
                        print(f"    [EKF] drift={drift_ratio:.2f} > {EKF_DRIFT_THRESHOLD}, "
                              f"trying online correction …", end=" ", flush=True)
                        ekf_trail, ekf_fwd, ekf_stats = run_ekf_online_simulation(
                            description, urdf_path, corrected_plan, EKF_PROBE_STEPS
                        )
                        if len(ekf_trail) > 1:
                            ekf_arr = np.array(ekf_trail, dtype=float)
                            ekf_fwd_v = np.asarray(ekf_fwd, dtype=float)
                            ekf_fwd_v = ekf_fwd_v / max(float(np.linalg.norm(ekf_fwd_v)), 1e-9)
                            ekf_disp = ekf_arr[-1] - ekf_arr[0]
                            ekf_lat = np.array([-ekf_fwd_v[1], ekf_fwd_v[0]], dtype=float)
                            ekf_fwd_d = float(np.dot(ekf_disp, ekf_fwd_v))
                            ekf_lat_d = float(np.dot(ekf_disp, ekf_lat))
                            ekf_drift = abs(ekf_lat_d) / max(abs(ekf_fwd_d), 0.01)

                            if ekf_drift < drift_ratio:
                                com_trail = ekf_trail
                                forward_axis = ekf_fwd
                                fwd_dist = ekf_fwd_d
                                lat_dist = ekf_lat_d
                                drift_ratio = ekf_drift
                                strategy = "ekf_online"
                                print(f"✓ improved: drift={ekf_drift:.2f}")
                            else:
                                print(f"✗ not better: drift={ekf_drift:.2f}")
                        else:
                            print("✗ failed")
                    except Exception as e:
                        print(f"✗ error: {e}")

                out_img = robot_dir / "trajectory.png"
                plot_demo(robot_name, description, ssm_result, com_trail, forward_axis,
                          fwd_dist, lat_dist, out_img)
                if out_img.exists():
                    all_png_paths.append(out_img)
                    shutil.copy2(out_img, png_dir / (robot_name + ".png"))

                summary_rows.append({
                    "robot":    robot_name,
                    "status":   "ok",
                    "ssm":      ssm_result["ssm"],
                    "num_legs": description.get("num_legs"),
                    "fwd_dist": fwd_dist,
                    "lat_dist": lat_dist,
                    "drift_ratio": round(drift_ratio, 3),
                    "strategy": strategy,
                })


    # ── Summary table ──
    print(f"\n{'='*60}")
    print(f"[Batch] Summary  ({out_root})")
    print(f"{'Robot':<28} {'Status':<12} {'SSM':>7} {'Legs':>5} {'Fwd':>8} {'Lat':>8} {'Drift':>7} {'Strategy':<14}")
    print("-" * 95)
    for r in summary_rows:
        print(f"{r['robot']:<28} {r.get('status',''):<12} "
              f"{r.get('ssm', float('nan')):>7.3f} "
              f"{r.get('num_legs', '-'):>5} "
              f"{r.get('fwd_dist', float('nan')):>8.3f} "
              f"{r.get('lat_dist', float('nan')):>8.3f} "
              f"{r.get('drift_ratio', float('nan')):>7.3f} "
              f"{r.get('strategy', 'baseline'):<14}")

    # Save summary JSON
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    # ── Copy all PNGs to png/ (real-time copies are already done; this is a dedup pass) ──
    if all_png_paths:
        for png in all_png_paths:
            dst = png_dir / (png.parent.name + ".png")
            if not dst.exists():
                shutil.copy2(png, dst)
        print(f"[Batch] PNG folder: {png_dir}  ({len(all_png_paths)} images)")

    # ── Generate batch analysis report ──
    report_path = generate_batch_report(out_root, summary_rows)
    print(f"[Batch] Report  → {report_path}")

    html_path = generate_html_report(out_root, summary_rows)
    print(f"[Batch] HTML    → {html_path}")

    print(f"\n[Batch] Done. Results in {out_root}")


if __name__ == "__main__":
    main()
