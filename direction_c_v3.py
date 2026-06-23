from __future__ import annotations

"""
方向C v3: 积分式在线偏航修正 (Slow Integral Yaw Correction)
============================================================
核心思想:
  - 不用 PD (噪声放大)，改用纯 I 控制器 (自然滤波)
  - 累积每步的 yaw_rate → 得到累积偏航角误差
  - 缓慢施加修正力抵消累积误差
  - 极低增益 (k_i ≈ 0.05)，不对步态产生冲击

与 v1 的对比:
  v1 EKF:  慢速 CoM 偏移估计 → 全局步幅修正 → seed42(-90%) seed512(-98%)
  v2 PD:   逐帧 yaw_rate 响应 → 噪声放大 → 全部退化
  v3 I:    累积偏航角 → 缓慢力矩补偿 → ???

直觉: 如果你发现自己正在缓慢右转，你不会每一步都猛踩，
      而是轻轻地、持续地增加左侧推力直到方向回正。
"""

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))

from reexec import maybe_reexec
TARGET_PYTHON  = os.environ.get("ISAAC_PYTHON", "/data/conda/envs/unitree-rl/bin/python")
TARGET_LD_PATH = os.environ.get("ISAAC_LD_LIBRARY_PATH", "/data/conda/envs/unitree-rl/lib")
maybe_reexec(TARGET_PYTHON, TARGET_LD_PATH)

import json
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(_REPO))

from utils import compute_metrics
from adaptive_gait import compute_adaptive_plan
from batch_test import _RobotSimCtx

_DT = 1.0 / 60.0
PROBE_STEPS = 480
USE_GPU     = True
BATCH_RUN_DIR = _REPO / "batch_results" / "20260530_221021"
ROBOT_DIRS = [
    "robot_00_seed7", "robot_01_seed42",
    "robot_03_seed256", "robot_04_seed512",
]
OUTPUT_DIR = _REPO / "direction_c_v3_results"


class IntegralYawController:
    """积分式偏航修正器 — 纯 I 控制, 极低增益, 自然抗噪。

    原理:
      θ_err(t) = ∫₀ᵗ yaw_rate(τ) dτ     (累积偏航角, 弧度)
      τ_cancel = Izz * k_i * θ_err       (持续抵消力矩)
      分配 τ_cancel 到支撑腿 → 修正步幅
    """

    def __init__(
        self,
        foot_positions: Dict[int, np.ndarray],
        com_xy: np.ndarray,
        total_mass: float = 5.0,
        body_diag: float = 0.5,          # 体对角线 (m)
        k_i: float = 0.06,               # 积分增益 (1/s²)
        max_amp_adj: float = 0.20,       # 单腿最大步幅修正
        warmup_steps: int = 60,          # 预热 (让步态稳定)
    ):
        self.k_i = k_i
        self.max_amp_adj = max_amp_adj
        self.warmup_steps = warmup_steps

        # 偏航惯性估算
        self.Izz = total_mass * body_diag * body_diag / 6.0

        # 每条腿的偏航力臂 (从 CoM 到足端的距离)
        self.yaw_levers: Dict[int, float] = {}
        for lid, fp in foot_positions.items():
            r = np.linalg.norm(fp - com_xy)
            self.yaw_levers[lid] = max(r, 0.05)

        # 确定"左侧"和"右侧"腿 (基于横向坐标)
        lat_axis = np.array([0.0, 1.0])  # 假设前进方向为 X
        self.leg_side: Dict[int, int] = {}  # +1=左侧, -1=右侧
        for lid, fp in foot_positions.items():
            lat = float(np.dot(fp - com_xy, lat_axis))
            self.leg_side[lid] = 1 if lat > 0 else -1

        # 累积状态
        self._accum_yaw = 0.0           # 累积偏航角 (rad)
        self._step_count = 0
        self._applied_adj: Dict[int, float] = {}

    def step(
        self,
        yaw_rate: float,
        stance_legs: List[int],
        base_amps: Dict[str, float],
    ) -> Dict[str, float]:
        """每步调用, 返回修正后的 amplitudes。"""
        self._step_count += 1

        # 累积偏航角 (梯形积分)
        self._accum_yaw += yaw_rate * _DT

        # 预热期不修正
        if self._step_count < self.warmup_steps:
            return base_amps

        # 计算需要的抵消力矩
        # τ_cancel = Izz * k_i * accum_yaw
        # 目标: 产生反向力矩逐步消除累积偏航
        tau_cancel = self.Izz * self.k_i * self._accum_yaw
        tau_cancel = float(np.clip(tau_cancel, -5.0, 5.0))

        # 分配到支撑腿
        raw_adj: Dict[int, float] = {}
        if stance_legs:
            total_lever = sum(self.yaw_levers.get(l, 0.05) for l in stance_legs)
            for lid in stance_legs:
                lever = self.yaw_levers.get(lid, 0.05)
                weight = lever / max(total_lever, 0.01)
                # 左侧腿 vs 右侧腿: 产生反向力矩 (简化为力的符号)
                side_sign = -self.leg_side.get(lid, 0)
                raw = tau_cancel * weight * side_sign * 0.015
                raw_adj[lid] = float(np.clip(raw, -self.max_amp_adj, self.max_amp_adj))

        # 应用到输出
        result: Dict[str, float] = {}
        for k, v in base_amps.items():
            lid = int(k)
            adj = raw_adj.get(lid, 0.0)
            self._applied_adj[lid] = adj
            result[k] = float(np.clip(v + adj, 0.05, 2.0))

        return result

    def diagnostics(self) -> Dict:
        return {
            "accum_yaw_deg": round(math.degrees(self._accum_yaw), 2),
            "applied_adj": {str(k): round(v, 4) for k, v in self._applied_adj.items()},
        }


# ═══════════════════════════════════════════════════════════════════════════════

def run_episode_integral_yaw(
    ctx: _RobotSimCtx, plan: dict, n_steps: int, description: dict,
) -> Tuple[List[List[float]], List[float], Optional[Dict], Dict]:
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

    # ── 控制器设置 ──────────────────────────────────────────────────────
    foot_positions: Dict[int, np.ndarray] = {}
    for lk in description.get("links", []):
        if lk.get("role") == "foot" and lk.get("leg_id") is not None:
            lid = int(lk["leg_id"])
            foot_positions[lid] = np.asarray(
                lk.get("default_world_origin", [0, 0, 0]), dtype=float
            )[:2]

    com_xy = np.zeros(2, dtype=float)
    total_mass = 0.0
    for lk in description.get("links", []):
        mp = lk.get("mass_properties", {})
        m = float(mp.get("mass", 0))
        if m <= 0:
            continue
        o = np.asarray(lk.get("default_world_origin", [0, 0, 0]), dtype=float)[:2]
        cm = np.asarray(mp.get("center_mass", [0, 0, 0]), dtype=float)[:2]
        com_xy += (o + cm) * m
        total_mass += m
    if total_mass > 1e-9:
        com_xy /= total_mass

    if foot_positions:
        pts = np.array(list(foot_positions.values()))
        body_diag = float(np.max(np.linalg.norm(pts - com_xy, axis=1)) * 2)
    else:
        body_diag = 0.5

    yaw_ctrl = IntegralYawController(
        foot_positions=foot_positions, com_xy=com_xy,
        total_mass=total_mass, body_diag=body_diag,
        warmup_steps=60,
    )

    # ── 仿真 ────────────────────────────────────────────────────────────
    def _rtj(lo, hi, r):
        return float(lo + max(0.0, min(1.0, r)) * (hi - lo))
    def _ss(e0, e1, x):
        t = max(0.0, min(1.0, (x - e0) / max(e1 - e0, 1e-9)))
        return t * t * (3.0 - 2.0 * t)

    from batch_test import GAIT_FREQUENCY, SWING_AMP
    _sl, _swl, _sd, _swd = ctx._sl, ctx._swl, ctx._sd, ctx._swd

    com_trail: List[List[float]] = []
    yaw_acc:   List[float] = []
    touchdown_ramp: Dict[int, int] = {}
    sim_time = 0.0
    last_diag = {}

    for step in range(n_steps):
        phase_now = 2.0 * math.pi * GAIT_FREQUENCY * sim_time
        targets = ctx.stand_ev.copy()

        sw = math.sin(phase_now)
        stance_group = group_b if sw > 0 else group_a
        stance_legs = [lid for lid in stance_group if lid not in group_c]

        yaw_now = yaw_acc[-1] if yaw_acc else 0.0
        corrected_amps = yaw_ctrl.step(yaw_now, stance_legs, base_amps)

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

            sw_val = float(math.sin(lg_ph))
            alpha = _ss(-0.30, 0.30, sw_val)
            fv = ctx.fmap.get(lid, np.zeros(2))
            dsign = -1.0 if float(np.dot(fv, lat)) > 0.0 else 1.0
            eff_amp = SWING_AMP * float(corrected_amps.get(str(lid), base_amps.get(str(lid), 1.0)))

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
            p = states["pose"]["p"][0]
            com_trail.append([float(p["x"]), float(p["y"])])
            yaw_acc.append(float(states["vel"]["angular"][0]["z"]))

        sim_time += _DT

        if step % 120 == 0 and step > 0:
            last_diag = yaw_ctrl.diagnostics()

    valid = [v for v in yaw_acc if abs(v) < 20.0]
    yaw_stats = {
        "yaw_rate_mean": float(np.mean(valid)) if valid else 0.0,
        "yaw_rate_std": float(np.std(valid)) if valid else 0.0,
    }
    return com_trail, fwd.tolist(), yaw_stats, last_diag


# ═══════════════════════════════════════════════════════════════════════════════

CONDITIONS = ["baseline", "+integral_yaw"]


def run_experiment(robot_name, description, urdf_path):
    print(f"\n{'='*60}")
    print(f"  Robot: {robot_name}  ({description.get('num_legs', '?')} legs)")
    print(f"{'='*60}")

    results = {}
    plan_base = compute_adaptive_plan(description, {})
    base_fwd = plan_base["final_forward_axis"]

    with _RobotSimCtx(description, urdf_path, use_gpu=USE_GPU) as ctx:
        print("  [baseline] running …", end=" ", flush=True)
        t0 = time.perf_counter()
        trail, ax, stats = ctx.run_episode(plan_base, PROBE_STEPS, return_yaw_stats=True)
        results["baseline"] = compute_metrics(trail, base_fwd, stats, len(trail))
        print(f"yaw={results['baseline']['yaw_abs']:.4f}  "
              f"lat={results['baseline']['lat_drift']:+.3f}  "
              f"({time.perf_counter()-t0:.1f}s)")

        print("  [+integral_yaw] running …", end=" ", flush=True)
        t0 = time.perf_counter()
        trail, ax, stats, diag = run_episode_integral_yaw(
            ctx, plan_base, PROBE_STEPS, description
        )
        m = compute_metrics(trail, base_fwd, stats, len(trail))
        acc_yaw = diag.get("accum_yaw_deg", float("nan"))
        results["+integral_yaw"] = m
        print(f"yaw={m['yaw_abs']:.4f}  "
              f"lat={m['lat_drift']:+.3f}  "
              f"accum_yaw={acc_yaw:.1f}°  "
              f"({time.perf_counter()-t0:.1f}s)")

    return results


def print_summary(all_results):
    print(f"\n{'='*85}")
    print("  方向C v3: 积分式在线偏航修正 — 汇总")
    print(f"{'='*85}")

    cond_means = {c: {"yaw": [], "lat": [], "fwd": []} for c in CONDITIONS}
    for rn, rr in all_results.items():
        for cond, m in rr.items():
            for k, store in [("yaw_abs", "yaw"), ("lat_drift", "lat"), ("fwd_vel", "fwd")]:
                v = m.get(k, float("nan"))
                if not math.isnan(v):
                    cond_means[cond][store].append(v)

    hdr = f"{'条件':<20} {'mean|yaw|':>10} {'±std':>8} {'mean|lat|':>10} {'vs base':>10}"
    print(hdr)
    print("-" * len(hdr))

    base_ya = float(np.mean([abs(v) for v in cond_means["baseline"]["yaw"]]))
    for cond in CONDITIONS:
        ya = cond_means[cond]["yaw"]
        la = cond_means[cond]["lat"]
        if not ya:
            continue
        ya_m = float(np.mean([abs(v) for v in ya]))
        ya_s = float(np.std([abs(v) for v in ya])) if len(ya) > 1 else 0.0
        la_m = float(np.mean([abs(v) for v in la]))
        imp = (base_ya - ya_m) / max(base_ya, 1e-9) * 100
        print(f"{cond:<20} {ya_m:>10.4f} {ya_s:>8.3f} {la_m:>10.4f} {imp:>+9.1f}%")

    print(f"\n{'机器人':<20}", end="")
    for c in CONDITIONS:
        print(f"{c:>16}", end="")
    print(f"{'改善':>10}")
    print(f"{'-'*62}")
    for rn in sorted(all_results.keys()):
        b_yaw = all_results[rn]["baseline"]["yaw_abs"]
        print(f"{rn:<20}", end="")
        for c in CONDITIONS:
            ya = all_results[rn].get(c, {}).get("yaw_abs", float("nan"))
            print(f"{ya:>16.4f}", end="")
        ya2 = all_results[rn].get("+integral_yaw", {}).get("yaw_abs", float("nan"))
        imp = (b_yaw - ya2) / max(b_yaw, 1e-9) * 100
        print(f"{imp:>+9.1f}%")


def main():
    print("=" * 70)
    print("  方向C v3: 积分式在线偏航修正")
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
        print(f"[Load] {rdir_name}: {desc.get('num_legs', '?')} legs")
        robot_data.append((rdir_name, desc, urdf_path))

    all_results = {}
    for robot_name, desc, urdf_path in robot_data:
        try:
            res = run_experiment(robot_name, desc, urdf_path)
            all_results[robot_name] = res
        except Exception as e:
            print(f"[ERROR] {robot_name}: {e}")
            import traceback; traceback.print_exc()

    print_summary(all_results)

    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = OUTPUT_DIR / f"integral_yaw_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[Save] → {json_path}")
    print("[Done]")


if __name__ == "__main__":
    main()
