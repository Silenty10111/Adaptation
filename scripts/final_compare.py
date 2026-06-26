from __future__ import annotations

"""
最终综合对比 — 跨所有实验方法挑选每个机器人的最优策略
=========================================================
核心问题: 偏航力矩未能完全抵消 → 走歪
发现: 不同机器人需要不同策略, 没有 universal solution

策略池:
  baseline     — compute_adaptive_plan (含 YAW_COMP_GAIN=0.50)
  +dynsym      — Dynamic Symmetry 解析优化 (原消融)
  +topo        — Topology Invariant Mapper (原消融, seed256 上 -94%)
  +online      — EKF 在线估计 (v1, seed42 -90%, seed512 -98%)

新思路: 不是找一个万能策略, 而是对每个机器人自动选择最优策略.
"""

import os, sys, json, math, time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from reexec import maybe_reexec
TARGET_PYTHON  = os.environ.get("ISAAC_PYTHON", "/data/conda/envs/unitree-rl/bin/python")
TARGET_LD_PATH = os.environ.get("ISAAC_LD_LIBRARY_PATH", "/data/conda/envs/unitree-rl/lib")
maybe_reexec(TARGET_PYTHON, TARGET_LD_PATH)

from typing import Dict, List, Optional, Tuple
import numpy as np

sys.path.insert(0, str(_REPO))

from adaptation.utils import compute_metrics
from adaptation.gait import compute_adaptive_plan
from adaptation.symmetry import integrate_dynamic_symmetry
from adaptation.topology import zero_shot_gait_plan
from adaptation.sim import _RobotSimCtx

_DT = 1.0 / 60.0
PROBE_STEPS = 480
USE_GPU     = True
BATCH_RUN_DIR = _REPO / "batch_results" / "20260530_221021"
ROBOT_DIRS = [
    "robot_00_seed7", "robot_01_seed42",
    "robot_03_seed256", "robot_04_seed512",
]
OUTPUT_DIR = _REPO / "final_results"


def run_online_ekf_episode(ctx, plan, n_steps, description):
    """v1 的 EKF 在线估计 (已证明在 seed42/seed512 上极有效)."""
    from adaptation.estimator import OnlineStateEstimator, make_observation

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
    base_amps = {str(k): float(v) for k, v in topo.get("per_leg_stride_amplitudes", {}).items()}

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

    from adaptation.sim import GAIT_FREQUENCY, SWING_AMP
    _sl, _swl, _sd, _swd = ctx._sl, ctx._swl, ctx._sd, ctx._swd

    com_trail: List[List[float]] = []
    yaw_acc:   List[float] = []
    touchdown_ramp: Dict[int, int] = {}
    sim_time = 0.0

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
                sim_time=sim_time, dt=_DT,
            )
            estimator.step(obs)

        sim_time += _DT

    valid = [v for v in yaw_acc if abs(v) < 20.0]
    yaw_stats = {
        "yaw_rate_mean": float(np.mean(valid)) if valid else 0.0,
        "yaw_rate_std": float(np.std(valid)) if valid else 0.0,
    }
    return com_trail, fwd.tolist(), yaw_stats


def build_topo_plan(description, plan_base):
    try:
        plan_topo = zero_shot_gait_plan(description)
        return plan_topo
    except Exception:
        return plan_base


def build_dynsym_amps(description, plan_base):
    try:
        ds_plan = integrate_dynamic_symmetry(
            description, plan_base, frequency_hz=0.85,
            optimize=True, max_iterations=30
        )
        topo_base = plan_base["topology"]
        base_amp = {str(k): float(v) for k, v in topo_base.get("per_leg_stride_amplitudes", {}).items()}
        amps = {}
        for lid, params in ds_plan.leg_params.items():
            base = float(base_amp.get(str(lid), 1.0))
            amps[str(lid)] = float(np.clip(base * params.stride_scale, 0.05, 2.0))
        for k, v in base_amp.items():
            if k not in amps:
                amps[k] = v
        return amps
    except Exception:
        topo_base = plan_base["topology"]
        return {str(k): float(v) for k, v in topo_base.get("per_leg_stride_amplitudes", {}).items()}


# ═══════════════════════════════════════════════════════════════════════════════

STRATEGIES = ["baseline", "+dynsym", "+topo", "+online"]


def run_all_strategies(robot_name, description, urdf_path):
    print(f"\n{'='*60}")
    print(f"  {robot_name} ({description.get('num_legs','?')} legs)")
    print(f"{'='*60}")

    results = {}
    plan_base = compute_adaptive_plan(description, {})
    base_fwd = plan_base["final_forward_axis"]
    base_amps = {str(k): float(v) for k, v in
                 plan_base["topology"].get("per_leg_stride_amplitudes", {}).items()}

    # 离线准备各策略
    dynsym_amps = build_dynsym_amps(description, plan_base)
    plan_topo = build_topo_plan(description, plan_base)

    with _RobotSimCtx(description, urdf_path, use_gpu=USE_GPU) as ctx:
        for strat in STRATEGIES:
            print(f"  [{strat}] ", end="", flush=True)
            t0 = time.perf_counter()

            if strat == "baseline":
                run_plan = plan_base
                trail, ax, stats = ctx.run_episode(run_plan, PROBE_STEPS, return_yaw_stats=True)

            elif strat == "+dynsym":
                run_plan = dict(plan_base)
                run_plan["_per_amp_override"] = dynsym_amps
                trail, ax, stats = ctx.run_episode(run_plan, PROBE_STEPS, return_yaw_stats=True)

            elif strat == "+topo":
                trail, ax, stats = ctx.run_episode(plan_topo, PROBE_STEPS, return_yaw_stats=True)

            elif strat == "+online":
                trail, ax, stats = run_online_ekf_episode(
                    ctx, plan_base, PROBE_STEPS, description
                )

            m = compute_metrics(trail, base_fwd, stats, len(trail))
            results[strat] = m
            dt = time.perf_counter() - t0
            print(f"yaw={m['yaw_abs']:.4f}  lat={m['lat_drift']:+.3f}  ({dt:.1f}s)")

    return results


def print_summary(all_results):
    print(f"\n{'='*100}")
    print("  最终综合对比 — 跨所有策略的偏航控制效果")
    print(f"{'='*100}")

    # 逐机器人找最优策略
    print(f"\n{'机器人':<20} {'腿数':<6}", end="")
    for s in STRATEGIES:
        print(f"{s:>12}", end="")
    print(f"{'最优':>12} {'改善':>10}")
    print("-" * 94)

    winners = {}
    total_best = {"yaw": [], "lat": [], "fwd": []}

    for rn in sorted(all_results.keys()):
        # 从文件名中提取腿数
        rr = all_results[rn]
        b_yaw = rr["baseline"]["yaw_abs"]
        best_strat = "baseline"
        best_yaw = b_yaw

        for s in STRATEGIES[1:]:
            ya = rr[s]["yaw_abs"]
            if ya < best_yaw:
                best_yaw = ya
                best_strat = s

        winners[rn] = best_strat
        total_best["yaw"].append(best_yaw)
        total_best["lat"].append(abs(rr[best_strat]["lat_drift"]))
        total_best["fwd"].append(rr[best_strat]["fwd_vel"])

        # 提取腿数
        n_legs = rn.split("_seed")[0].replace("robot_0","").replace("robot_","")
        if n_legs == "0": n_legs = "?"

        print(f"{rn:<20} {description_legs.get(rn, '?'):<6}", end="")
        for s in STRATEGIES:
            ya = rr[s]["yaw_abs"]
            marker = " ★" if s == best_strat else ""
            print(f"{ya:>11.4f}{marker}", end="")
        imp = (b_yaw - best_yaw) / max(b_yaw, 1e-9) * 100
        arrow = "↓" if imp > 0 else "↑"
        print(f"{best_strat:>12} {imp:>+9.1f}% {arrow}")

    # 最优组合汇总
    best_mean_yaw = float(np.mean(total_best["yaw"]))
    base_mean_yaw = float(np.mean([all_results[rn]["baseline"]["yaw_abs"] for rn in all_results]))
    print(f"\n{'─'*94}")
    print(f"  最优策略组合 mean|yaw| = {best_mean_yaw:.4f}  (baseline: {base_mean_yaw:.4f})  "
          f"改善: {(base_mean_yaw-best_mean_yaw)/max(base_mean_yaw,1e-9)*100:+.1f}%")
    print(f"  各机器人最优策略: {winners}")


# ═══════════════════════════════════════════════════════════════════════════════

description_legs = {}

def main():
    print("=" * 70)
    print("  最终综合对比 — 自适应策略选择")
    print("=" * 70)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    robot_data = []
    for rdir_name in ROBOT_DIRS:
        rdir = BATCH_RUN_DIR / rdir_name
        desc_path = rdir / "robot_description.json"
        urdf_path = rdir / "robot.urdf"
        if not desc_path.exists() or not urdf_path.exists():
            continue
        with open(desc_path, "r", encoding="utf-8") as f:
            desc = json.load(f)
        n = desc.get("num_legs", "?")
        description_legs[rdir_name] = n
        print(f"[Load] {rdir_name}: {n} legs")
        robot_data.append((rdir_name, desc, urdf_path))

    all_results = {}
    for robot_name, desc, urdf_path in robot_data:
        try:
            res = run_all_strategies(robot_name, desc, urdf_path)
            all_results[robot_name] = res
        except Exception as e:
            print(f"[ERROR] {robot_name}: {e}")
            import traceback; traceback.print_exc()

    print_summary(all_results)

    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = OUTPUT_DIR / f"final_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[Save] → {json_path}")
    print("[Done]")


if __name__ == "__main__":
    main()
