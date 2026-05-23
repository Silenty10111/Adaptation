#!/usr/bin/env python3
"""Batch robot generation + stability check + gait test with trajectory plots.

每次运行会在 OUTPUT_DIR 下新建一个带时间戳的子目录，
对每个机器人生成一张 demo 图（机器人轮廓 + 前进方向直线 + 实际质心轨迹曲线）。

=== 在此处修改批量参数 ====================================================="""

from __future__ import annotations

# ─────────────────────────── 配置区（直接在这里改参数）──────────────────────

# 要生成的机器人数量
NUM_ROBOTS = 50

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
SSM_THRESHOLD        = 0.0

# 是否跳过静态稳定性不通过的机器人（False = 仍然进行步态测试）
SKIP_UNSTABLE        = False

# 是否将标准六足（robot_assets/standard_hexapod）作为额外参照机器人加入测试
INCLUDE_STANDARD_HEXAPOD = True

# 输出根目录
OUTPUT_DIR           = "batch_results"

# ─────────────────────────── 以下无需修改 ────────────────────────────────────

import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

TARGET_PYTHON  = "/data/conda/envs/unitree-rl/bin/python"
TARGET_LD_PATH = "/data/conda/envs/unitree-rl/lib"
REPO_ROOT      = Path(__file__).resolve().parent
GEN_PYTHON     = "/data/conda/envs/Adaptation/bin/python"
GEN_SCRIPT     = REPO_ROOT / "generate_geometry.py"
GEN_URDF       = REPO_ROOT / "generate_urdf.py"
ASSET_ROOT     = REPO_ROOT / "robot_assets"


# ---------------------------------------------------------------------------
# Auto re-exec into unitree-rl environment (required for Isaac Gym)
# ---------------------------------------------------------------------------

def _maybe_reexec():
    if os.environ.get("BATCH_REEXEC") == "1":
        return
    if sys.executable == TARGET_PYTHON:
        return
    if not Path(TARGET_PYTHON).exists():
        return
    env = dict(os.environ)
    ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{TARGET_LD_PATH}:{ld}" if ld else TARGET_LD_PATH
    env["BATCH_REEXEC"] = "1"
    print("[INFO] 切换到 unitree-rl 环境执行...")
    os.execvpe(TARGET_PYTHON, [TARGET_PYTHON, *sys.argv], env)


_maybe_reexec()

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

def run_gait_sim(description: dict, urdf_path: Path) -> Tuple[List[List[float]], List[float]]:
    """Run headless gait simulation, return (com_trail, forward_axis)."""
    from isaacgym import gymapi

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

    gait_plan = _load_plan(description)
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
    sp.physx.use_gpu = False
    sp.physx.num_position_iterations = 8
    sp.physx.num_velocity_iterations = 2

    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, sp)
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
        touchdown_ramp: Dict[int, int] = {}
        com_trail: List[List[float]] = []
        sim_time = 0.0
        dt = sp.dt
        phase_now = 0.0

        # Estimate body length from foot bounding box (max axis extent)
        if fmap:
            foot_pts = np.array(list(fmap.values()), dtype=float)
            body_length = float(np.max(foot_pts.max(axis=0) - foot_pts.min(axis=0)))
        else:
            body_length = 0.5  # fallback
        min_travel = body_length * max(MIN_TRAVEL_BODY_LENGTHS, 0.0)
        max_steps  = max(MAX_SIM_STEPS, SIM_STEPS)

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

            states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)
            if states is not None and len(states) > 0:
                p = states["pose"]["p"][0]
                com_trail.append([float(p["x"]), float(p["y"])])

            sim_time += dt

            # Early exit: stop once min_travel is covered AND we're past SIM_STEPS
            if step >= SIM_STEPS and min_travel > 0.0 and len(com_trail) > 1:
                trail_arr = np.array(com_trail, dtype=float)
                disp = trail_arr[-1] - trail_arr[0]
                covered = abs(float(np.dot(disp, fwd)))
                if covered >= min_travel:
                    break

        return com_trail, forward_axis

    finally:
        gym.destroy_sim(sim)


# ---------------------------------------------------------------------------
# Demo plot (delegated to Adaptation env to avoid numpy/PIL incompatibility)
# ---------------------------------------------------------------------------

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
            com_trail, forward_axis = run_gait_sim(description, urdf_path_std)
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
            summary_rows.append({
                "robot": robot_name, "status": "ok",
                "ssm": ssm_result["ssm"], "num_legs": description.get("num_legs"),
                "fwd_dist": fwd_dist, "lat_dist": lat_dist,
            })
        except Exception as e:
            print(f"  [Ref]  FAILED: {e}")
            summary_rows.append({"robot": robot_name, "status": "ref_failed"})

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
            # Still generate a plot with empty trajectory
            out_img = robot_dir / "trajectory.png"
            plot_demo(robot_name, description, ssm_result, [], [1.0, 0.0], 0.0, 0.0, out_img)
            continue

        # ── Step 3: gait simulation ──
        try:
            com_trail, forward_axis = run_gait_sim(description, urdf_path)
        except Exception as e:
            print(f"  [Sim]  FAILED: {e}")
            summary_rows.append({"robot": robot_name, "status": "sim_failed", "ssm": ssm_result["ssm"]})
            continue

        # Compute forward / lateral distances
        trail = np.array(com_trail, dtype=float) if len(com_trail) > 1 else np.zeros((2, 2))
        fwd   = np.asarray(forward_axis, dtype=float)
        fwd   = fwd / max(float(np.linalg.norm(fwd)), 1e-9)
        lat   = np.array([-fwd[1], fwd[0]], dtype=float)
        disp  = trail[-1] - trail[0] if len(trail) > 1 else np.zeros(2)
        fwd_dist = float(np.dot(disp, fwd))
        lat_dist = float(np.dot(disp, lat))
        print(f"  [Sim]  steps={len(com_trail)}  fwd={fwd_dist:+.3f}m  lat={lat_dist:+.3f}m")

        # ── Step 4: plot ──
        out_img = robot_dir / "trajectory.png"
        plot_demo(robot_name, description, ssm_result, com_trail, forward_axis,
                  fwd_dist, lat_dist, out_img)
        if out_img.exists():
            all_png_paths.append(out_img)

        summary_rows.append({
            "robot":    robot_name,
            "status":   "ok",
            "ssm":      ssm_result["ssm"],
            "num_legs": description.get("num_legs"),
            "fwd_dist": fwd_dist,
            "lat_dist": lat_dist,
        })

    # ── Summary table ──
    print(f"\n{'='*60}")
    print(f"[Batch] Summary  ({out_root})")
    print(f"{'Robot':<28} {'Status':<12} {'SSM':>7} {'Legs':>5} {'Fwd':>8} {'Lat':>8}")
    print("-" * 72)
    for r in summary_rows:
        print(f"{r['robot']:<28} {r.get('status',''):<12} "
              f"{r.get('ssm', float('nan')):>7.3f} "
              f"{r.get('num_legs', '-'):>5} "
              f"{r.get('fwd_dist', float('nan')):>8.3f} "
              f"{r.get('lat_dist', float('nan')):>8.3f}")

    # Save summary JSON
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    # ── Copy all PNGs to a flat png/ directory for side-by-side viewing ──
    if all_png_paths:
        png_dir = out_root / "png"
        png_dir.mkdir(exist_ok=True)
        for png in all_png_paths:
            shutil.copy2(png, png_dir / (png.parent.name + ".png"))
        print(f"[Batch] PNG folder: {png_dir}  ({len(all_png_paths)} images)")

    print(f"\n[Batch] Done. Results in {out_root}")


if __name__ == "__main__":
    main()
