#!/usr/bin/env python3
"""Test standard hexapod with amputated legs — missing 1 or 2 legs.

从标准六足机器人 (robot_assets/standard_hexapod) 出发，生成缺 1 条腿（5 腿）
和缺 2 条腿（4 腿）的变体，依次进行静态稳定性检验和步态仿真，并与完整六足
基线对比。

输出：每变体一张 trajectory.png + 一份 JSON 汇总。
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════════

TARGET_PYTHON  = os.environ.get("ISAAC_PYTHON", "/data/conda/envs/unitree-rl/bin/python")
TARGET_LD_PATH = os.environ.get("ISAAC_LD_LIBRARY_PATH", "/data/conda/envs/unitree-rl/lib")
GEN_PYTHON     = os.environ.get("ADAPTATION_PYTHON", "/data/conda/envs/Adaptation/bin/python")
REPO_ROOT      = Path(__file__).resolve().parent.parent
GEN_URDF       = REPO_ROOT / "scripts" / "generate_urdf.py"
ASSET_ROOT     = REPO_ROOT / "robot_assets"
STANDARD_DESC  = ASSET_ROOT / "standard_hexapod" / "robot_description.json"
STANDARD_URDF  = ASSET_ROOT / "standard_hexapod" / "generated_robot.urdf"
STANDARD_MESHES = ASSET_ROOT / "standard_hexapod" / "meshes"

SIM_STEPS             = 1200
MAX_SIM_STEPS         = 6000
MIN_TRAVEL_BODY_LENGTHS = 10.0
SSM_THRESHOLD         = 0.05
SKIP_UNSTABLE         = False   # 缺腿变体即使 SSM 不过也尝试仿真
USE_GPU               = True
OUTPUT_DIR            = "amputation_results"

# ── 缺 1 条腿的测试用例 ──────────────────────────────────────────────────────
# 每条腿单独缺失一次
MISSING_ONE_LEG_CASES = [
    {"name": "missing_leg_0",  "remove_leg_ids": [0]},
    {"name": "missing_leg_1",  "remove_leg_ids": [1]},
    {"name": "missing_leg_2",  "remove_leg_ids": [2]},
    {"name": "missing_leg_3",  "remove_leg_ids": [3]},
    {"name": "missing_leg_4",  "remove_leg_ids": [4]},
    {"name": "missing_leg_5",  "remove_leg_ids": [5]},
]

# ── 缺 2 条腿的测试用例 ──────────────────────────────────────────────────────
# 从 15 种组合 (C(6,2)) 中选取代表性案例
MISSING_TWO_LEGS_CASES = [
    # 同侧相邻 (same side adjacent)
    {"name": "missing_legs_01", "remove_leg_ids": [0, 1]},   # 右侧前+中
    {"name": "missing_legs_12", "remove_leg_ids": [1, 2]},   # 右侧中+后
    {"name": "missing_legs_34", "remove_leg_ids": [3, 4]},   # 左侧后+中
    {"name": "missing_legs_45", "remove_leg_ids": [4, 5]},   # 左侧中+前
    # 对角线 (diagonal, 1 per side)
    {"name": "missing_legs_03", "remove_leg_ids": [0, 3]},   # 右前+左后
    {"name": "missing_legs_14", "remove_leg_ids": [1, 4]},   # 右中+左中
    {"name": "missing_legs_25", "remove_leg_ids": [2, 5]},   # 右后+左前
    # 前后两端 (both ends)
    {"name": "missing_legs_05", "remove_leg_ids": [0, 5]},   # 右前+左前
    {"name": "missing_legs_23", "remove_leg_ids": [2, 3]},   # 右后+左后
    # 跳隔 (skip-one)
    {"name": "missing_legs_02", "remove_leg_ids": [0, 2]},   # 右侧前+后
    {"name": "missing_legs_35", "remove_leg_ids": [3, 5]},   # 左侧后+前
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Auto re-exec into unitree-rl environment
# ═══════════════════════════════════════════════════════════════════════════════

sys.path.insert(0, str(REPO_ROOT))
from reexec import maybe_reexec
maybe_reexec(TARGET_PYTHON, TARGET_LD_PATH)

# ─── After re-exec (unitree-rl environment) ───────────────────────────────────

try:
    from isaacgym import gymapi
    _GYM_AVAILABLE = True
except Exception:
    _GYM_AVAILABLE = False

from adaptation.sim import (
    GAIT_FREQUENCY, SWING_AMP, SWING_LIFT_RATIO, STANCE_LIFT_RATIO,
    SWING_DROP_RATIO, STANCE_DROP_RATIO, BODY_HEIGHT, HOLD_STEPS, _RobotSimCtx,
)
from adaptation.stability import evaluate_ssm
from adaptation.gait import compute_adaptive_plan


# ═══════════════════════════════════════════════════════════════════════════════
#  Leg amputation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def amputate_legs(description: dict, remove_leg_ids: List[int]) -> dict:
    """Return a deep copy of *description* with the given leg IDs removed.

    Links belonging to removed legs are deleted.  Joints that reference any
    removed link are deleted.  Remaining legs are renumbered so leg_ids stay
    contiguous (0 .. N-1).  Link names and joint names are rewritten accordingly.
    **mesh_path references keep their original filenames** — we only copy the
    original STL files, not rename them.

    Parameters
    ----------
    description : dict
        Loaded robot_description.json.
    remove_leg_ids : list[int]
        Leg IDs to remove (0-based).

    Returns
    -------
    dict  new description with fewer legs.
    """
    desc = copy.deepcopy(description)
    remove_set = set(remove_leg_ids)

    # ── Build old→new leg ID map ──────────────────────────────────────────
    all_leg_ids = sorted({
        lk["leg_id"] for lk in desc["links"] if lk.get("leg_id") is not None
    })
    remaining = [lid for lid in all_leg_ids if lid not in remove_set]
    old_to_new: Dict[int, int] = {old: new for new, old in enumerate(remaining)}

    def _rename_leg_ref(s: str) -> str:
        """Replace leg_<old>_ with leg_<new>_ for all renumbered legs."""
        for old_lid in all_leg_ids:
            if old_lid in remove_set:
                continue
            new_lid = old_to_new[old_lid]
            if old_lid == new_lid:
                continue
            s = s.replace(f"leg_{old_lid}_", f"leg_{new_lid}_")
        return s

    # ── Filter and rename links ───────────────────────────────────────────
    new_links = []
    for link in desc["links"]:
        lid = link.get("leg_id")
        if lid is not None and lid in remove_set:
            continue
        new_link = copy.deepcopy(link)
        if lid is not None:
            new_link["leg_id"] = old_to_new[lid]
            # Rename link name: leg_<old>_* → leg_<new>_*
            new_link["name"] = _rename_leg_ref(new_link["name"])
            # IMPORTANT: do NOT rename mesh_path — original STL filenames are kept
        new_links.append(new_link)
    desc["links"] = new_links

    # ── Build link name set for joint filtering ────────────────────────────
    link_names = {lk["name"] for lk in new_links}

    # ── Filter and rename joints ──────────────────────────────────────────
    new_joints = []
    for joint in desc["joints"]:
        parent = joint.get("parent", "")
        child  = joint.get("child", "")
        if parent not in link_names or child not in link_names:
            continue
        new_joint = copy.deepcopy(joint)
        new_joint["parent"] = _rename_leg_ref(parent)
        new_joint["child"]  = _rename_leg_ref(child)
        new_joint["name"]   = _rename_leg_ref(new_joint.get("name", ""))
        new_joints.append(new_joint)
    desc["joints"] = new_joints

    # ── Update metadata ───────────────────────────────────────────────────
    desc["num_legs"] = len(remaining)
    desc["robot_name"] = f"hexapod_{len(remaining)}legs"

    return desc


def copy_meshes_for_variant(desc: dict, dst_meshes_dir: Path) -> None:
    """Copy STL meshes referenced by *desc* from standard hexapod meshes.

    Mesh paths in the (possibly renumbered) description still point to the
    original filenames (e.g. ``meshes/leg_1_hip.stl``), so they exist in
    the standard hexapod mesh directory.
    """
    src_meshes = STANDARD_MESHES
    dst_meshes_dir.mkdir(parents=True, exist_ok=True)
    needed = set()
    for link in desc["links"]:
        mp = link.get("mesh_path", "")
        if mp:
            needed.add(Path(mp).name)
    for mesh_name in sorted(needed):
        src = src_meshes / mesh_name
        if src.exists():
            shutil.copy2(src, dst_meshes_dir / mesh_name)
        else:
            print(f"    [WARN] mesh not found: {mesh_name}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Gait simulation — probe → correct → full-sim (matches batch_test.py logic)
# ═══════════════════════════════════════════════════════════════════════════════

PROBE_STEPS  = 1800
YAW_BAD      = 0.18     # rad/s — above this → apply yaw correction
FWD_STUCK    = 0.004    # m/s   — below this → stuck
BACKWARD_M   = -0.10    # m — backward displacement threshold for flip
TREND_WINDOW = 240      # steps for steady-state check
_dt          = 1.0 / 60.0


def _disp_score(trail, fwd_ax):
    """Return (forward_displacement, lateral_displacement) in metres."""
    if len(trail) < 2:
        return 0.0, 0.0
    arr = np.array(trail, dtype=float)
    d = arr[-1, :2] - arr[0, :2] if arr.ndim == 2 else np.array(arr[-1][:2]) - np.array(arr[0][:2])
    lat = np.array([-fwd_ax[1], fwd_ax[0]], dtype=float)
    return float(np.dot(d, fwd_ax)), float(np.dot(d, lat))


def _trend_disp(trail, fwd_ax):
    """Forward displacement over the last TREND_WINDOW steps, or None."""
    if len(trail) < TREND_WINDOW + 1:
        return None, None
    arr = np.array(trail, dtype=float)
    d = arr[-1, :2] - arr[-TREND_WINDOW - 1, :2]
    lat = np.array([-fwd_ax[1], fwd_ax[0]], dtype=float)
    return float(np.dot(d, fwd_ax)), float(np.dot(d, lat))


def optimize_and_simulate(
    description: dict, urdf_path: Path,
) -> Tuple[List[List[float]], List[float]]:
    """Probe → correct → full-sim inside ONE _RobotSimCtx.

    Matches batch_test.py's optimize_and_simulate:
    1. Run probe (1800 steps) to measure yaw rate and forward displacement.
    2. If backward (>0.1m in wrong direction), flip forward axis 180°.
    3. If yaw rate is high (>0.18 rad/s), apply stride-amplitude correction.
    4. If stuck (fwd < 0.05m), boost stride amplitudes.
    5. Run full simulation with the corrected plan.
    """
    plan0 = compute_adaptive_plan(description, {})
    fwd0  = np.asarray(plan0["final_forward_axis"], dtype=float)
    fwd0  = fwd0 / max(float(np.linalg.norm(fwd0)), 1e-9)

    with _RobotSimCtx(description, urdf_path, use_gpu=USE_GPU) as ctx:
        # ── Probe 1 ──────────────────────────────────────────────────────
        t1, _, s1 = ctx.run_episode(plan0, PROBE_STEPS, return_yaw_stats=True)
        yaw1 = s1["yaw_rate_mean"]
        fd1, ld1 = _disp_score(t1, fwd0)
        fv1 = fd1 / max(len(t1) * _dt, _dt)
        ft1, lt1 = _trend_disp(t1, fwd0)
        ft1_str = f"{ft1:+.3f}m" if ft1 is not None else "N/A"
        print(f"  [Probe] yaw={yaw1:+.3f}  fwd_disp={fd1:+.3f}m  "
              f"lat={ld1:+.3f}m  fwd_vel={fv1:+.4f}  trend_fwd={ft1_str}", flush=True)

        best_plan = plan0
        best_fwd  = fwd0.copy()
        tf = None

        # Use trend if overall is ambiguous
        _dir_fwd = fd1
        if ft1 is not None:
            _trend_vel = ft1 / max(TREND_WINDOW * _dt, _dt)
            if (_trend_vel > 0.0) != (fv1 > 0.0):
                _dir_fwd = ft1
                print(f"  [Trend] overall={fd1:+.3f}m vs trend={ft1:+.3f}m "
                      f"— trusting trend", flush=True)

        # Backward-flip check
        if abs(yaw1) >= YAW_BAD or fv1 <= FWD_STUCK:
            if fv1 <= FWD_STUCK and _dir_fwd < BACKWARD_M:
                flip = -fwd0.copy()
                print(f"  [Backward] fwd_disp={_dir_fwd:+.3f}m < {BACKWARD_M:.1f}m, "
                      f"trying flip...", end=" ", flush=True)
                plan_f = compute_adaptive_plan(description, {}, forced_axis=flip.tolist())
                tf, _, sf = ctx.run_episode(plan_f, PROBE_STEPS, return_yaw_stats=True)
                fdf, ldf = _disp_score(tf, flip)
                ftf, _ = _trend_disp(tf, flip)
                ftf_str = f"{ftf:+.3f}m" if ftf is not None else "N/A"
                print(f"fwd_disp={fdf:+.3f}m  trend={ftf_str}", flush=True)
                if fdf > 0.0 and fdf > fd1:
                    best_plan = plan_f; best_fwd = flip.copy()
                    print(f"  [Backward] flip accepted (fwd={fdf:+.3f}m > 0)", flush=True)
                else:
                    print(f"  [Backward] flip REJECTED "
                          f"(fdf={fdf:+.3f}m, need >0 and >{fd1:+.3f}m)", flush=True)

            # Yaw correction
            if abs(yaw1) > YAW_BAD:
                best_plan = _apply_amp_correction(best_plan, yaw1)
                print(f"  [Corr] yaw={yaw1:+.3f}", flush=True)

            # Stuck recovery
            _ref_trail = tf if (tf is not None and best_plan is not plan0) else t1
            fwd_chk, _ = _disp_score(_ref_trail, best_fwd)
            if fwd_chk < 0.05:
                print(f"  [Stuck] fwd_disp={fwd_chk:.3f}m, boost...", flush=True)
                bp = dict(best_plan)
                ba = best_plan.get("topology", {}).get("per_leg_stride_amplitudes", {})
                bo = {k: float(np.clip(float(v)*1.6, 0.25, 0.85)) for k, v in ba.items()}
                bp["_per_amp_override"] = bo
                tb, _, _ = ctx.run_episode(bp, PROBE_STEPS, return_yaw_stats=True)
                fdb, _ = _disp_score(tb, best_fwd)
                if fdb > fwd_chk:
                    best_plan = bp
                    print(f"  [Stuck] boost accepted (fwd_disp={fdb:+.3f}m)", flush=True)

        print(f"  [Opt] axis={[round(v,3) for v in best_fwd.tolist()]}", flush=True)

        # ── Full simulation ───────────────────────────────────────────────
        _full_steps = max(MAX_SIM_STEPS, SIM_STEPS)
        _min_travel = ctx.body_length * max(MIN_TRAVEL_BODY_LENGTHS, 0.0)
        trail_f, ax_f, _ = ctx.run_episode(best_plan, _full_steps,
                                            min_travel=_min_travel,
                                            min_steps=SIM_STEPS)
        return trail_f, list(ax_f)


def _apply_amp_correction(plan: dict, yaw_rate: float) -> dict:
    """Adjust per-leg stride amplitudes to counter yaw."""
    topo = plan.get("topology", {})
    amps = {str(k): float(v) for k, v in topo.get("per_leg_stride_amplitudes", {}).items()}
    correction = float(np.clip(abs(yaw_rate) * 0.15, 0.02, 0.20))
    groups = topo.get("groups", {})
    group_a = [int(x) for x in groups.get("group_a", [])]
    group_b = [int(x) for x in groups.get("group_b", [])]

    if yaw_rate > 0:
        # Reduce group that drives positive yaw
        for lid in group_a:
            amps[str(lid)] = max(0.10, float(amps.get(str(lid), 0.5)) - correction)
        for lid in group_b:
            amps[str(lid)] = min(0.85, float(amps.get(str(lid), 0.5)) + correction)
    else:
        for lid in group_b:
            amps[str(lid)] = max(0.10, float(amps.get(str(lid), 0.5)) - correction)
        for lid in group_a:
            amps[str(lid)] = min(0.85, float(amps.get(str(lid), 0.5)) + correction)

    new_plan = dict(plan)
    new_plan["topology"] = dict(topo)
    new_plan["topology"]["per_leg_stride_amplitudes"] = amps
    return new_plan


# ═══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_demo(robot_name: str, description: dict, ssm_result: dict,
              com_trail: List, forward_axis: List[float],
              fwd_dist: float, lat_dist: float, out_path: Path) -> None:
    """Generate a trajectory overview plot via a subprocess that uses the
    Adaptation conda environment (which has a working matplotlib)."""
    import tempfile
    data = {
        "robot_name": robot_name,
        "description": description,
        "ssm_result": ssm_result,
        "com_trail": com_trail,
        "forward_axis": forward_axis,
        "fwd_dist": fwd_dist,
        "lat_dist": lat_dist,
        "out_path": str(out_path),
    }
    # Write data to temp file, run plotter in Adaptation env
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(data, tf)
        tmp_path = tf.name

    plotter_script = REPO_ROOT / "scripts" / "_plot_trajectory.py"
    env = dict(os.environ)
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT}:{pp}" if pp else str(REPO_ROOT)
    try:
        result = subprocess.run(
            [GEN_PYTHON, str(plotter_script), tmp_path],
            capture_output=True, text=True, timeout=60,
            cwd=str(REPO_ROOT), env=env,
        )
        if result.returncode != 0:
            print(f"    [plot] FAILED: {result.stderr[:200]}")
        else:
            print(f"    [plot] saved → {out_path}")
    except Exception as e:
        print(f"    [plot] FAILED: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def run_one(robot_name: str, description: dict, robot_dir: Path) -> dict:
    """Run SSM check + gait sim for one robot variant.  Returns summary row."""
    print(f"\n{'─'*60}")
    print(f"[Test] {robot_name}  (legs={description.get('num_legs')})")

    # ── Step 1: static stability ──
    ssm_result = evaluate_ssm(description, threshold=SSM_THRESHOLD)
    passed_str = "PASS" if ssm_result["passed"] else "FAIL"
    print(f"  [SSM]  ssm={ssm_result['ssm']:.4f}m  {passed_str}")

    if SKIP_UNSTABLE and not ssm_result["passed"]:
        print(f"  [Skip] SSM below threshold, still attempting gait (SKIP_UNSTABLE=False)")
        # We continue anyway for amputation tests

    # ── Step 2: compute gait plan ──
    try:
        gait_plan = compute_adaptive_plan(description, {})
        groups = gait_plan.get("topology", {}).get("groups", {})
        print(f"  [Plan] group_a={groups.get('group_a', [])}  "
              f"group_b={groups.get('group_b', [])}  "
              f"group_c={groups.get('group_c', [])}")
    except Exception as e:
        print(f"  [Plan] FAILED: {e}")
        return {"robot": robot_name, "status": "plan_failed",
                "ssm": ssm_result["ssm"], "num_legs": description.get("num_legs")}

    # ── Step 3: generate URDF ──
    desc_path = robot_dir / "robot_description.json"
    urdf_path = robot_dir / "robot.urdf"
    desc_path.write_text(json.dumps(description, indent=2), encoding="utf-8")

    # Copy meshes
    copy_meshes_for_variant(description, robot_dir / "meshes")

    # Generate URDF (ensure PYTHONPATH includes repo root for adaptation imports)
    env = dict(os.environ)
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT}:{pp}" if pp else str(REPO_ROOT)
    cmd_urdf = [
        GEN_PYTHON, str(GEN_URDF),
        "--description", str(desc_path),
        "--output", str(urdf_path),
    ]
    result = subprocess.run(cmd_urdf, capture_output=True, text=True,
                            cwd=str(REPO_ROOT), env=env)
    if result.returncode != 0:
        print(f"  [URDF] FAILED: {result.stderr[:400]}")
        return {"robot": robot_name, "status": "urdf_failed",
                "ssm": ssm_result["ssm"], "num_legs": description.get("num_legs")}
    print(f"  [URDF] OK → {urdf_path}")

    # ── Step 4: gait simulation (probe → correct → full-sim) ──
    try:
        com_trail, forward_axis = optimize_and_simulate(description, urdf_path)
    except Exception as e:
        print(f"  [Sim] FAILED: {e}")
        import traceback; traceback.print_exc()
        return {"robot": robot_name, "status": "sim_failed",
                "ssm": ssm_result["ssm"], "num_legs": description.get("num_legs")}

    # ── Step 5: compute metrics ──
    fwd = np.asarray(forward_axis, dtype=float)
    fwd = fwd / max(float(np.linalg.norm(fwd)), 1e-9)
    lat = np.array([-fwd[1], fwd[0]], dtype=float)
    trail = np.array(com_trail, dtype=float) if len(com_trail) > 1 else np.zeros((2, 3))
    disp = trail[-1, :2] - trail[0, :2] if len(trail) > 1 else np.zeros(2)
    fwd_dist = float(np.dot(disp, fwd))
    lat_dist = float(np.dot(disp, lat))
    drift_ratio = abs(lat_dist) / max(abs(fwd_dist), 0.01)
    print(f"  [Sim]  steps={len(com_trail)}  fwd={fwd_dist:+.3f}m  lat={lat_dist:+.3f}m  "
          f"drift={drift_ratio:.2f}")

    # ── Step 6: plot ──
    out_img = robot_dir / "trajectory.png"
    plot_demo(robot_name, description, ssm_result, com_trail, forward_axis,
              fwd_dist, lat_dist, out_img)

    return {
        "robot": robot_name, "status": "ok",
        "ssm": ssm_result["ssm"], "num_legs": description.get("num_legs"),
        "fwd_dist": fwd_dist, "lat_dist": lat_dist,
        "drift_ratio": round(drift_ratio, 3), "steps": len(com_trail),
    }


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = REPO_ROOT / OUTPUT_DIR / timestamp
    out_root.mkdir(parents=True, exist_ok=True)
    png_dir = out_root / "png"
    png_dir.mkdir(exist_ok=True)
    print(f"[Amputation Test] Output: {out_root}\n")

    # Load standard hexapod description
    base_desc = json.loads(STANDARD_DESC.read_text(encoding="utf-8"))
    print(f"[Base] Standard hexapod: {base_desc['num_legs']} legs")

    summary_rows: List[dict] = []

    # ── Baseline: full 6-leg ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Baseline] 标准六足 (完整 6 腿)")
    base_dir = out_root / "baseline_6legs"
    base_dir.mkdir(parents=True, exist_ok=True)
    base_desc_copy = copy.deepcopy(base_desc)
    base_desc_copy["robot_name"] = "standard_hexapod_baseline"
    try:
        # Just copy the standard URDF + meshes directly (no re-generation needed)
        desc_path = base_dir / "robot_description.json"
        urdf_path = base_dir / "robot.urdf"
        desc_path.write_text(json.dumps(base_desc_copy, indent=2), encoding="utf-8")
        shutil.copy2(STANDARD_URDF, urdf_path)
        meshes_dst = base_dir / "meshes"
        if meshes_dst.exists():
            shutil.rmtree(meshes_dst)
        shutil.copytree(STANDARD_MESHES, meshes_dst)
        row = run_one("baseline_6legs", base_desc_copy, base_dir)
        summary_rows.append(row)
        img = base_dir / "trajectory.png"
        if img.exists():
            shutil.copy2(img, png_dir / "baseline_6legs.png")
    except Exception as e:
        print(f"  [Baseline] FAILED: {e}")
        import traceback; traceback.print_exc()
        summary_rows.append({"robot": "baseline_6legs", "status": "failed"})

    # ── Missing 1 leg cases ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Phase 1] 缺 1 条腿（5 腿）测试 — {len(MISSING_ONE_LEG_CASES)} 个案例")
    for case in MISSING_ONE_LEG_CASES:
        robot_name = case["name"]
        remove_ids = case["remove_leg_ids"]
        robot_dir = out_root / robot_name
        robot_dir.mkdir(parents=True, exist_ok=True)

        try:
            variant_desc = amputate_legs(base_desc, remove_ids)
            print(f"\n  → 移除 leg(s) {remove_ids}, 剩余 {variant_desc['num_legs']} 条腿")
            row = run_one(robot_name, variant_desc, robot_dir)
            summary_rows.append(row)
            img = robot_dir / "trajectory.png"
            if img.exists():
                shutil.copy2(img, png_dir / f"{robot_name}.png")
        except Exception as e:
            print(f"  [{robot_name}] FAILED: {e}")
            import traceback; traceback.print_exc()
            summary_rows.append({"robot": robot_name, "status": "failed"})

        # Allow GPU cooldown between tests
        time.sleep(2.0)

    # ── Missing 2 legs cases ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Phase 2] 缺 2 条腿（4 腿）测试 — {len(MISSING_TWO_LEGS_CASES)} 个案例")
    for case in MISSING_TWO_LEGS_CASES:
        robot_name = case["name"]
        remove_ids = case["remove_leg_ids"]
        robot_dir = out_root / robot_name
        robot_dir.mkdir(parents=True, exist_ok=True)

        try:
            variant_desc = amputate_legs(base_desc, remove_ids)
            print(f"\n  → 移除 leg(s) {remove_ids}, 剩余 {variant_desc['num_legs']} 条腿")
            row = run_one(robot_name, variant_desc, robot_dir)
            summary_rows.append(row)
            img = robot_dir / "trajectory.png"
            if img.exists():
                shutil.copy2(img, png_dir / f"{robot_name}.png")
        except Exception as e:
            print(f"  [{robot_name}] FAILED: {e}")
            import traceback; traceback.print_exc()
            summary_rows.append({"robot": robot_name, "status": "failed"})

        time.sleep(2.0)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Summary]")
    print(f"{'Robot':<28} {'Legs':>5} {'Status':>12} {'SSM':>8} {'Fwd(m)':>10} {'Lat(m)':>10} {'Drift':>8}")
    print(f"{'─'*28} {'─'*5} {'─'*12} {'─'*8} {'─'*10} {'─'*10} {'─'*8}")
    for r in summary_rows:
        legs = r.get("num_legs", "—")
        status = r.get("status", "—")
        ssm = r.get("ssm", 0) or 0
        fwd = r.get("fwd_dist", 0) or 0
        lat = r.get("lat_dist", 0) or 0
        drift = r.get("drift_ratio", 0) or 0
        print(f"{r['robot']:<28} {str(legs):>5} {status:>12} {ssm:>8.4f} {fwd:>10.3f} {lat:>10.3f} {drift:>8.3f}")

    # Save summary JSON
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[Summary] saved → {summary_path}")
    print(f"[PNGs]    saved → {png_dir}/")

    # Quick stats
    ok_count = sum(1 for r in summary_rows if r.get("status") == "ok")
    print(f"[Stats]   {ok_count}/{len(summary_rows)} tests passed")

    return summary_rows


if __name__ == "__main__":
    main()
