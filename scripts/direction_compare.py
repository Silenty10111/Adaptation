#!/usr/bin/env python3
from __future__ import annotations

"""
方向 A vs 方向 C 对比实验
=========================
方向 A (强化解析): 仿真引导的 DynSym 优化 + 修正的 WBC QP
方向 C (在线自适应): OnlineStateEstimator 集成到仿真循环

条件对比:
  baseline     : compute_adaptive_plan（现有）
  +online      : baseline + OnlineStateEstimator 实时修正 (方向C)
  +dynsym_sim  : baseline + 仿真引导 DynSym 优化 (方向A)
  +wbc_fixed   : baseline + 带约束 QP 的 WBC (方向A)
"""

import os
import sys
from pathlib import Path

# ─── 必须在导入 numpy 之前完成环境切换 ─────────────────────────────────────

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from reexec import maybe_reexec

TARGET_PYTHON  = os.environ.get("ISAAC_PYTHON", "/data/conda/envs/unitree-rl/bin/python")
TARGET_LD_PATH = os.environ.get("ISAAC_LD_LIBRARY_PATH", "/data/conda/envs/unitree-rl/lib")
maybe_reexec(TARGET_PYTHON, TARGET_LD_PATH)

# ─── 从此处起，已在 unitree-rl 环境 ─────────────────────────────────────────

import copy
import json
import math
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(_REPO))

from adaptation.utils import compute_metrics
from adaptation.gait import compute_adaptive_plan
from adaptation.symmetry import integrate_dynamic_symmetry
from adaptation.wbc import run_centroidal_wbc
from adaptation.estimator import OnlineStateEstimator, make_observation
from adaptation.sim import _RobotSimCtx

_DT = 1.0 / 60.0

# ─── 实验配置 ──────────────────────────────────────────────────────────────

PROBE_STEPS = 480
USE_GPU     = True

BATCH_RUN_DIR = _REPO / "batch_results" / "20260530_221021"
ROBOT_DIRS = [
    "robot_00_seed7",   # 10 legs
    "robot_01_seed42",  # 4 legs
    "robot_03_seed256", # 7 legs
    "robot_04_seed512", # 6 legs
]
OUTPUT_DIR = _REPO / "direction_results"

# ═══════════════════════════════════════════════════════════════════════════════
# 方向 C: 在线估计增强的仿真
# ═══════════════════════════════════════════════════════════════════════════════

def run_episode_with_online(
    ctx: _RobotSimCtx,
    plan: dict,
    n_steps: int,
    description: dict,
) -> Tuple[List[List[float]], List[float], Optional[Dict]]:
    """方向 C: 集成 OnlineStateEstimator 的仿真 episode。

    仿真的完整循环，每步调用在线估计器获取自适应步幅修正。
    """
    ga = ctx._ga
    gym, sim, env, actor = ctx.gym, ctx.sim, ctx.env, ctx.actor
    ctx._reset()

    fwd = np.asarray(plan.get("final_forward_axis", [1.0, 0.0]), dtype=float)
    fwd = fwd / max(float(np.linalg.norm(fwd)), 1e-9)
    lat = np.array([-fwd[1], fwd[0]], dtype=float)

    topo = plan["topology"]
    group_a = topo["groups"]["group_a"]
    group_b = topo["groups"]["group_b"]
    group_c = topo["groups"].get("group_c", [])

    base_per_amp = {str(k): float(v) for k, v in topo.get("per_leg_stride_amplitudes", {}).items()}
    if "_per_amp_override" in plan:
        base_per_amp = {str(k): float(v) for k, v in plan["_per_amp_override"].items()}

    # ── 创建在线估计器 ────────────────────────────────────────────────────
    total_mass = sum(
        float(lk.get("mass_properties", {}).get("mass", 0.0))
        for lk in description.get("links", [])
    )
    estimator = OnlineStateEstimator(description, plan, total_mass=max(total_mass, 1.0))

    def _rtj(lo, hi, r):
        return float(lo + max(0.0, min(1.0, r)) * (hi - lo))
    def _ss(e0, e1, x):
        t = max(0.0, min(1.0, (x - e0) / max(e1 - e0, 1e-9)))
        return t * t * (3.0 - 2.0 * t)

    _sl, _swl, _sd, _swd = ctx._sl, ctx._swl, ctx._sd, ctx._swd
    from adaptation.sim import GAIT_FREQUENCY, SWING_AMP, BODY_HEIGHT
    com_trail: List[List[float]] = []
    yaw_acc:   List[float] = []
    touchdown_ramp: Dict[int, int] = {}
    sim_time = 0.0

    for step in range(n_steps):
        phase_now = 2.0 * math.pi * GAIT_FREQUENCY * sim_time
        targets = ctx.stand_ev.copy()

        # ── 从在线估计器获取当前自适应修正 ──────────────────────────────
        adapt = estimator.get_state()
        online_scales = {
            str(k): float(v) for k, v in adapt.adaptive_stride_scales.items()
        }
        freq_scale = adapt.frequency_scale

        for lid, j in ctx.triplets.items():
            if lid in group_c:
                targets[j["lift_idx"]]  = _rtj(j["lift_lower"], j["lift_upper"], _sl)
                targets[j["drop_idx"]]  = _rtj(j["drop_lower"], j["drop_upper"], _sd)
                targets[j["swing_idx"]] = _rtj(j["swing_lower"], j["swing_upper"], 0.5)
                continue

            if lid in group_b:
                lg_ph = phase_now + math.pi
            elif lid in group_a:
                lg_ph = phase_now
            else:
                lg_ph = 0.0

            sw = float(math.sin(lg_ph * freq_scale))
            alpha = _ss(-0.30, 0.30, sw)
            fv = ctx.fmap.get(lid, np.zeros(2))
            dsign = -1.0 if float(np.dot(fv, lat)) > 0.0 else 1.0

            # 融合在线修正
            base_scale = float(base_per_amp.get(str(lid), 1.0))
            online_scale = float(online_scales.get(str(lid), 1.0))
            eff_amp = SWING_AMP * base_scale * online_scale

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
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # ── 读取状态 → 在线估计器 ─────────────────────────────────────────
        _sflag = ga.STATE_ALL
        states = gym.get_actor_rigid_body_states(env, actor, _sflag)
        if states is not None and len(states) > 0:
            p = states["pose"]["p"][0]
            body_xy = [float(p["x"]), float(p["y"])]

            # 姿态
            qw = float(states["pose"]["r"][0]["w"])
            qx = float(states["pose"]["r"][0]["x"])
            qy = float(states["pose"]["r"][0]["y"])
            qz = float(states["pose"]["r"][0]["z"])
            # 简化: 从 quaternion 提取 yaw
            siny = 2.0 * (qw * qz + qx * qy)
            cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
            yaw = math.atan2(siny, cosy)

            com_trail.append(body_xy)
            yaw_acc.append(float(states["vel"]["angular"][0]["z"]))

            # 构建观测 → 喂给在线估计器
            obs = make_observation(
                body_xy=body_xy,
                body_attitude=(0.0, 0.0, yaw),
                sim_time=sim_time,
                dt=_DT,
            )
            estimator.step(obs)

        sim_time += _DT

    # 统计
    valid = [v for v in yaw_acc if abs(v) < 20.0]
    yaw_stats = {
        "yaw_rate_mean": float(np.mean(valid)) if valid else 0.0,
        "yaw_rate_std": float(np.std(valid)) if valid else 0.0,
    }
    return com_trail, fwd.tolist(), yaw_stats


# ═══════════════════════════════════════════════════════════════════════════════
# 方向 A: 仿真引导的 DynSym 优化
# ═══════════════════════════════════════════════════════════════════════════════

def dynsym_with_sim_guidance(
    description: dict,
    plan_base: dict,
    ctx: _RobotSimCtx,
) -> Dict[str, float]:
    """仿真引导的 DynSym 优化 (Simulation-Guided Dynamic Symmetry)。

    流程:
    1. 运行 DynSym 解析优化作为初始猜测
    2. 在仿真中探测初始解的实际偏航
    3. 若偏航超标，在最优解附近扰动搜索更优的 stride_scale 组合
    4. 返回仿真验证最佳的 per-leg stride amplitudes
    """
    from adaptation.symmetry import DynamicSymmetryGaitPlan

    PROBE = 360
    YAW_THRESH = 0.15

    # ── Step 1: 解析 DynSym 优化 ──────────────────────────────────────────
    try:
        ds_plan: DynamicSymmetryGaitPlan = integrate_dynamic_symmetry(
            description, plan_base,
            frequency_hz=0.85, optimize=True, max_iterations=20
        )
    except Exception:
        # Fallback: baseline amplitudes
        topo = plan_base["topology"]
        return {str(k): float(v) for k, v in topo.get("per_leg_stride_amplitudes", {}).items()}

    # 构建初始 amplitudes
    topo_base = plan_base["topology"]
    base_amp = {str(k): float(v) for k, v in topo_base.get("per_leg_stride_amplitudes", {}).items()}

    init_amps: Dict[str, float] = {}
    for lid, params in ds_plan.leg_params.items():
        base = float(base_amp.get(str(lid), 1.0))
        init_amps[str(lid)] = float(np.clip(base * params.stride_scale, 0.05, 2.0))

    # 填充未出现在 DynSym 结果中的腿
    for k, v in base_amp.items():
        if k not in init_amps:
            init_amps[k] = v

    # ── Step 2: 探头仿真 ──────────────────────────────────────────────────
    def _probe(amps: Dict[str, float]) -> float:
        probe_plan = dict(plan_base)
        probe_plan["_per_amp_override"] = amps
        trail, _, stats = ctx.run_episode(probe_plan, PROBE, return_yaw_stats=True)
        return abs(stats["yaw_rate_mean"]) if stats else 999.0

    yaw0 = _probe(init_amps)
    print(f"    [DynSym+Sim] 初始 yaw_abs={yaw0:.4f} rad/s")

    if yaw0 < YAW_THRESH:
        print(f"    [DynSym+Sim] 已达标，无需额外搜索")
        return init_amps

    # ── Step 3: 局部扰动搜索 ──────────────────────────────────────────────
    best_amps = dict(init_amps)
    best_yaw = yaw0
    n_trials = 0
    max_trials = 15

    # 对有最大 stride_scale 偏差的腿做扰动
    candidates = sorted(init_amps.keys(),
                        key=lambda k: abs(init_amps[k] - base_amp.get(k, 1.0)),
                        reverse=True)[:4]

    rng = random.Random(42)

    for _ in range(max_trials):
        trial = dict(best_amps)
        # 随机选择 1-2 条腿做小扰动
        n_perturb = rng.randint(1, min(2, len(candidates)))
        for lid in rng.sample(candidates, n_perturb):
            delta = rng.uniform(-0.20, 0.20)
            trial[lid] = float(np.clip(trial[lid] + delta, 0.05, 2.0))

        yaw_t = _probe(trial)
        n_trials += 1

        if yaw_t < best_yaw:
            best_yaw = yaw_t
            best_amps = dict(trial)
            print(f"    [DynSym+Sim] 改善: yaw={best_yaw:.4f} (trial {n_trials})")
            if best_yaw < YAW_THRESH:
                break

    print(f"    [DynSym+Sim] 最终 yaw={best_yaw:.4f} (搜索{n_trials}次)")
    return best_amps


# ═══════════════════════════════════════════════════════════════════════════════
# 主实验
# ═══════════════════════════════════════════════════════════════════════════════

CONDITIONS = ["baseline", "+online", "+dynsym_sim", "+wbc_fixed"]


def run_experiment_for_robot(
    robot_name: str,
    description: dict,
    urdf_path: Path,
) -> Dict[str, Dict[str, float]]:
    """对单个机器人跑所有条件。"""
    print(f"\n{'='*60}")
    print(f"  Robot: {robot_name}  ({description.get('num_legs', '?')} legs)")
    print(f"{'='*60}")

    results: Dict[str, Dict[str, float]] = {}

    # ── 离线规划 ──────────────────────────────────────────────────────────
    plan_base = compute_adaptive_plan(description, {})
    base_amps = {
        str(k): float(v)
        for k, v in plan_base["topology"].get("per_leg_stride_amplitudes", {}).items()
    }
    base_fwd = plan_base["final_forward_axis"]

    # +wbc_fixed
    try:
        wbc_amps = run_centroidal_wbc(description, plan_base)
    except Exception:
        wbc_amps = dict(base_amps)

    with _RobotSimCtx(description, urdf_path, use_gpu=USE_GPU) as ctx:
        # ── Condition 1: baseline ──────────────────────────────────────────
        print("  [baseline] running …", end=" ", flush=True)
        t0 = time.perf_counter()
        trail, ax, stats = ctx.run_episode(plan_base, PROBE_STEPS, return_yaw_stats=True)
        results["baseline"] = compute_metrics(trail, base_fwd, stats, len(trail))
        print(f"yaw={results['baseline']['yaw_abs']:.4f}  "
              f"lat={results['baseline']['lat_drift']:+.3f}  "
              f"fwd={results['baseline']['fwd_vel']:+.4f}  "
              f"({time.perf_counter()-t0:.1f}s)")

        # ── Condition 2: +online (方向C) ──────────────────────────────────
        print("  [+online] running …", end=" ", flush=True)
        t0 = time.perf_counter()
        trail, ax, stats = run_episode_with_online(
            ctx, plan_base, PROBE_STEPS, description
        )
        results["+online"] = compute_metrics(trail, base_fwd, stats, len(trail))
        print(f"yaw={results['+online']['yaw_abs']:.4f}  "
              f"lat={results['+online']['lat_drift']:+.3f}  "
              f"fwd={results['+online']['fwd_vel']:+.4f}  "
              f"({time.perf_counter()-t0:.1f}s)")

        # ── Condition 3: +dynsym_sim (方向A) ───────────────────────────────
        print("  [+dynsym_sim] optimizing …", end=" ", flush=True)
        t0 = time.perf_counter()
        ds_amps = dynsym_with_sim_guidance(description, plan_base, ctx)
        ds_plan = dict(plan_base)
        ds_plan["_per_amp_override"] = ds_amps
        trail, ax, stats = ctx.run_episode(ds_plan, PROBE_STEPS, return_yaw_stats=True)
        results["+dynsym_sim"] = compute_metrics(trail, base_fwd, stats, len(trail))
        print(f"    yaw={results['+dynsym_sim']['yaw_abs']:.4f}  "
              f"lat={results['+dynsym_sim']['lat_drift']:+.3f}  "
              f"fwd={results['+dynsym_sim']['fwd_vel']:+.4f}  "
              f"({time.perf_counter()-t0:.1f}s)")

        # ── Condition 4: +wbc_fixed (方向A) ────────────────────────────────
        print("  [+wbc_fixed] running …", end=" ", flush=True)
        t0 = time.perf_counter()
        wbc_plan = dict(plan_base)
        wbc_plan["_per_amp_override"] = wbc_amps
        trail, ax, stats = ctx.run_episode(wbc_plan, PROBE_STEPS, return_yaw_stats=True)
        results["+wbc_fixed"] = compute_metrics(trail, base_fwd, stats, len(trail))
        print(f"yaw={results['+wbc_fixed']['yaw_abs']:.4f}  "
              f"lat={results['+wbc_fixed']['lat_drift']:+.3f}  "
              f"fwd={results['+wbc_fixed']['fwd_vel']:+.4f}  "
              f"({time.perf_counter()-t0:.1f}s)")

    return results


def print_summary(all_results: Dict) -> None:
    """打印汇总对比表。"""
    print(f"\n{'='*90}")
    print("  方向 A vs 方向 C — 汇总对比")
    print(f"{'='*90}")

    robots = list(all_results.keys())
    cond_means = {c: {"yaw": [], "lat": [], "fwd": []} for c in CONDITIONS}
    for rn, rr in all_results.items():
        for cond, m in rr.items():
            for k, store in [("yaw_abs", "yaw"), ("lat_drift", "lat"), ("fwd_vel", "fwd")]:
                v = m.get(k, float("nan"))
                if not math.isnan(v):
                    cond_means[cond][store].append(v)

    # 表头
    hdr = f"{'条件':<16} {'mean|yaw|':>10} {'±std':>8} {'mean|lat|':>10} {'mean fwd':>10} {'vs base':>10}"
    print(hdr)
    print("-" * len(hdr))

    base_ya = float(np.mean([abs(v) for v in cond_means["baseline"]["yaw"]])) if cond_means["baseline"]["yaw"] else 1e-9
    for cond in CONDITIONS:
        ya = cond_means[cond]["yaw"]
        la = cond_means[cond]["lat"]
        fv = cond_means[cond]["fwd"]
        if not ya:
            continue
        ya_m = float(np.mean([abs(v) for v in ya]))
        ya_s = float(np.std([abs(v) for v in ya])) if len(ya) > 1 else 0.0
        la_m = float(np.mean([abs(v) for v in la]))
        fv_m = float(np.mean(fv))
        imp = (base_ya - ya_m) / max(base_ya, 1e-9) * 100
        print(f"{cond:<16} {ya_m:>10.4f} {ya_s:>8.3f} {la_m:>10.4f} {fv_m:>10.4f} {imp:>+9.1f}%")

    # 逐机器人对比
    print(f"\n{'机器人':<20}", end="")
    for c in CONDITIONS:
        print(f"{c:>14}", end="")
    print(f"\n{'-'*76}")
    for rn in robots:
        print(f"{rn:<20}", end="")
        for c in CONDITIONS:
            ya = all_results[rn].get(c, {}).get("yaw_abs", float("nan"))
            print(f"{ya:>14.4f}", end="")
        print()


def main():
    print("=" * 70)
    print("  方向 A vs 方向 C 对比实验")
    print("  Python:", sys.executable)
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载机器人
    robot_data = []
    for rdir_name in ROBOT_DIRS:
        rdir = BATCH_RUN_DIR / rdir_name
        desc_path = rdir / "robot_description.json"
        urdf_path = rdir / "robot.urdf"
        if not desc_path.exists() or not urdf_path.exists():
            print(f"[Load] SKIP {rdir_name}")
            continue
        with open(desc_path, "r", encoding="utf-8") as f:
            desc = json.load(f)
        n_legs = desc.get("num_legs", "?")
        print(f"[Load] {rdir_name}: {n_legs} legs")
        robot_data.append((rdir_name, desc, urdf_path))

    if not robot_data:
        print("[ERROR] No robots found")
        sys.exit(1)

    # 运行实验
    all_results = {}
    for robot_name, desc, urdf_path in robot_data:
        try:
            res = run_experiment_for_robot(robot_name, desc, urdf_path)
            all_results[robot_name] = res
        except Exception as e:
            print(f"[ERROR] {robot_name}: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("[ERROR] All robots failed")
        sys.exit(1)

    print_summary(all_results)

    # 保存结果
    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = OUTPUT_DIR / f"direction_compare_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[Save] → {json_path}")
    print("[Done]")


if __name__ == "__main__":
    main()
