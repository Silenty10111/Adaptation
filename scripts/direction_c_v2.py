from __future__ import annotations

"""
方向C 改进版: 在线状态估计 + 实时偏航力矩闭环抵消
==================================================
修正内容:
  1. EKF 收敛预热期 (前60步不修正)
  2. EMA 平滑修正量 (防过冲)
  3. 实时偏航力矩闭环: 每步测量 yaw rate → 计算抵消力矩 → 分配至支撑腿
  4. 修正健康监测: yaw 恶化时自动降低修正强度
"""

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
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

from adaptation.utils import compute_metrics
from adaptation.gait import compute_adaptive_plan
from adaptation.sim import _RobotSimCtx
from adaptation.phase import (
    leg_phase_state,
    resolve_duty_factors,
    resolve_phase_offsets,
    yaw_rate_error,
)

_DT = 1.0 / 60.0

PROBE_STEPS = 480
USE_GPU     = True
BATCH_RUN_DIR = _REPO / "batch_results" / "20260530_221021"
ROBOT_DIRS = [
    "robot_00_seed7",   # 10 legs
    "robot_01_seed42",  # 4 legs
    "robot_03_seed256", # 7 legs
    "robot_04_seed512", # 6 legs
]
OUTPUT_DIR = _REPO / "direction_c_v2_results"


# ═══════════════════════════════════════════════════════════════════════════════
# 改进的在线偏航力矩闭环修正器
# ═══════════════════════════════════════════════════════════════════════════════

class YawCancelController:
    """实时偏航力矩闭环抵消器。

    每步测量实际偏航率 → 计算抵消力矩 → 按力臂分配至支撑腿。
    """

    def __init__(
        self,
        foot_positions: Dict[int, np.ndarray],  # {leg_id: [x, y]}
        com_xy: np.ndarray,
        total_mass: float = 5.0,
        yaw_inertia: float = 0.5,
        # 修正参数
        kp_yaw: float = 3.0,       # 偏航比例增益
        kd_yaw: float = 0.5,       # 偏航微分增益
        max_amp_adj: float = 0.25,  # 单腿最大步幅修正比例
        ema_alpha: float = 0.12,    # 修正量 EMA 系数
        warmup_steps: int = 60,     # 预热步数（不修正）
    ):
        self.foot_positions = foot_positions
        self.com_xy = com_xy.copy()
        self.total_mass = total_mass
        self.Izz = yaw_inertia
        self.kp = kp_yaw
        self.kd = kd_yaw
        self.max_amp_adj = max_amp_adj
        self.alpha = ema_alpha
        self.warmup_steps = warmup_steps

        # 计算每条腿的偏航力臂
        self.yaw_levers: Dict[int, float] = {}
        for lid, fp in foot_positions.items():
            rx = fp[0] - com_xy[0]
            ry = fp[1] - com_xy[1]
            # 力臂 = 足端到 CoM 的切向分量（垂直于径向）
            self.yaw_levers[lid] = math.hypot(rx, ry)

        # 状态
        self._prev_yaw_error = 0.0
        self._ema_amps: Dict[int, float] = {}  # EMA 平滑后的修正量
        self._correction_strength = 1.0        # 自适应修正强度
        self._yaw_history: List[float] = []     # 最近的偏航率历史
        self._step_count = 0

    def step(
        self,
        yaw_rate: float,
        stance_legs: List[int],
        base_amps: Dict[str, float],
        yaw_rate_ref: float = 0.0,
    ) -> Dict[str, float]:
        """每步调用，返回修正后的 per-leg stride amplitudes。

        Args:
            yaw_rate: 当前测量的偏航率 (rad/s)
            stance_legs: 当前支撑腿 ID 列表
            base_amps: 基础 per-leg amplitudes {"0": 1.0, ...}

        Returns:
            {"0": 1.05, ...}  修正后的 amplitudes
        """
        self._step_count += 1
        error = yaw_rate_error(yaw_rate, yaw_rate_ref)
        self._yaw_history.append(error)
        if len(self._yaw_history) > 120:
            self._yaw_history.pop(0)

        # ── 1. 计算偏航加速度（微分项）────────────────────────────────
        yaw_accel = (error - self._prev_yaw_error) / _DT
        self._prev_yaw_error = error

        # ── 2. 计算抵消力矩 ──────────────────────────────────────────
        # τ_cancel = Izz * (kp * yaw_rate + kd * yaw_accel)
        tau_cancel = self.Izz * (self.kp * error + self.kd * yaw_accel)
        # 钳制避免极端值
        tau_cancel = float(np.clip(tau_cancel, -50.0, 50.0))

        # ── 3. 将抵消力矩分配至支撑腿 ────────────────────────────────
        # 分配原则: 力臂越大，分配越多（力矩 = 力 × 力臂）
        raw_adjustments: Dict[int, float] = {}

        if stance_legs and abs(tau_cancel) > 1e-6:
            # 计算每条支撑腿的力矩贡献权重
            total_lever = 0.0
            for lid in stance_legs:
                total_lever += self.yaw_levers.get(lid, 0.1)

            for lid in stance_legs:
                lever = self.yaw_levers.get(lid, 0.1)
                # 该腿需产生的切向力: F_i = τ_cancel * lever_i / (n * Σ lever²)
                # 步幅修正: Δamp = F_i / k_approx
                # 简化: Δamp ∝ τ_cancel * lever_i / total_lever
                weight = lever / max(total_lever, 0.01)
                # 符号: 正 τ → 逆时针 → 右侧腿需增加步幅，左侧需减少
                sign = 1.0 if self.yaw_levers.get(lid, 0) > 0 else -1.0
                raw = -tau_cancel * weight * sign * 0.02  # 缩放因子
                raw_adjustments[lid] = float(np.clip(raw, -self.max_amp_adj, self.max_amp_adj))

        # ── 4. EMA 平滑 ──────────────────────────────────────────────
        for lid in raw_adjustments:
            prev = self._ema_amps.get(lid, 0.0)
            self._ema_amps[lid] = self.alpha * raw_adjustments[lid] + (1 - self.alpha) * prev

        # ── 5. 修正健康监测 ──────────────────────────────────────────
        if self._step_count > self.warmup_steps + 30 and len(self._yaw_history) >= 60:
            recent_yaw = self._yaw_history[-30:]
            older_yaw  = self._yaw_history[-60:-30]
            recent_mean = float(np.mean([abs(y) for y in recent_yaw]))
            older_mean  = float(np.mean([abs(y) for y in older_yaw]))
            if recent_mean > older_mean * 1.5 and recent_mean > 0.02:
                # yaw 在恶化 → 降低修正强度
                self._correction_strength = max(0.1, self._correction_strength - 0.15)
            elif recent_mean < older_mean * 0.7:
                # yaw 在改善 → 缓慢恢复
                self._correction_strength = min(1.0, self._correction_strength + 0.03)

        # ── 6. 组装输出 ──────────────────────────────────────────────
        result: Dict[str, float] = {}
        for k, v in base_amps.items():
            lid = int(k)
            adj = self._ema_amps.get(lid, 0.0) * self._correction_strength
            # 预热期不修正
            if self._step_count < self.warmup_steps:
                adj = 0.0
            result[k] = float(np.clip(v + adj, 0.05, 2.0))

        return result

    def get_diagnostics(self) -> Dict:
        return {
            "correction_strength": round(self._correction_strength, 3),
            "ema_amps": {str(k): round(v, 4) for k, v in self._ema_amps.items()},
            "mean_yaw_recent": round(float(np.mean([abs(y) for y in self._yaw_history[-30:]])), 5)
                if len(self._yaw_history) >= 30 else None,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 改进的仿真 (在线偏航闭环)
# ═══════════════════════════════════════════════════════════════════════════════

def run_episode_yaw_closed_loop(
    ctx: _RobotSimCtx,
    plan: dict,
    n_steps: int,
    description: dict,
) -> Tuple[List[List[float]], List[float], Optional[Dict], Dict]:
    """仿真 episode with 在线偏航力矩闭环修正。"""
    ga = ctx._ga
    gym, sim, env, actor = ctx.gym, ctx.sim, ctx.env, ctx.actor
    ctx._reset()

    fwd = np.asarray(plan.get("final_forward_axis", [1.0, 0.0]), dtype=float)
    fwd = fwd / max(float(np.linalg.norm(fwd)), 1e-9)
    lat = np.array([-fwd[1], fwd[0]], dtype=float)

    topo = plan["topology"]
    group_c = topo["groups"].get("group_c", [])

    base_amps = {str(k): float(v) for k, v in topo.get("per_leg_stride_amplitudes", {}).items()}

    # ── 创建偏航闭环控制器 ─────────────────────────────────────────────
    # 提取足端位置和 CoM
    foot_positions: Dict[int, np.ndarray] = {}
    for lk in description.get("links", []):
        if lk.get("role") == "foot" and lk.get("leg_id") is not None:
            lid = int(lk["leg_id"])
            origin = np.asarray(lk.get("default_world_origin", [0, 0, 0]), dtype=float)
            foot_positions[lid] = origin[:2]

    # CoM 估算
    com_xy = np.zeros(2, dtype=float)
    total_mass = 0.0
    for lk in description.get("links", []):
        mp = lk.get("mass_properties", {})
        m = float(mp.get("mass", 0))
        if m <= 0:
            continue
        o = np.asarray(lk.get("default_world_origin", [0, 0, 0]), dtype=float)
        cm = np.asarray(mp.get("center_mass", [0, 0, 0]), dtype=float)
        com_xy += (o[:2] + cm[:2]) * m
        total_mass += m
    if total_mass > 1e-9:
        com_xy /= total_mass

    # 估算偏航惯性 (近似: 体对角线² × mass / 12)
    if foot_positions:
        all_pts = np.array(list(foot_positions.values()))
        diag = float(np.max(np.linalg.norm(all_pts - com_xy, axis=1)))
    else:
        diag = 0.4
    Izz_est = total_mass * diag * diag / 6.0

    yaw_ctrl = YawCancelController(
        foot_positions=foot_positions,
        com_xy=com_xy,
        total_mass=total_mass,
        yaw_inertia=max(Izz_est, 0.1),
        warmup_steps=60,
    )

    # ── 仿真函数 ──────────────────────────────────────────────────────
    def _rtj(lo, hi, r):
        return float(lo + max(0.0, min(1.0, r)) * (hi - lo))
    def _ss(e0, e1, x):
        t = max(0.0, min(1.0, (x - e0) / max(e1 - e0, 1e-9)))
        return t * t * (3.0 - 2.0 * t)

    from adaptation.sim import GAIT_FREQUENCY, SWING_AMP, BODY_HEIGHT
    phase_offsets = resolve_phase_offsets(plan, ctx.triplets)
    duty_factors, duty_diagnostics = resolve_duty_factors(plan, ctx.triplets)
    for diagnostic in duty_diagnostics:
        print(f"[YawClosedLoop] {diagnostic}")
    gait_frequency = float(plan.get("cpg", {}).get("frequency_hz", GAIT_FREQUENCY))
    _sl, _swl, _sd, _swd = ctx._sl, ctx._swl, ctx._sd, ctx._swd

    com_trail: List[List[float]] = []
    yaw_acc:   List[float] = []
    touchdown_ramp: Dict[int, int] = {}
    sim_time = 0.0
    all_diag: List[Dict] = []

    for step in range(n_steps):
        phase_now = 2.0 * math.pi * gait_frequency * sim_time
        targets = ctx.stand_ev.copy()

        # ── 确定当前支撑腿 ──────────────────────────────────────────
        stance_legs = [
            lid for lid in ctx.triplets
            if lid not in group_c and leg_phase_state(
                phase_now, lid, phase_offsets, duty_factors
            ).is_stance
        ]

        # ── 从偏航闭环控制器获取修正 ─────────────────────────────────
        # 使用上一步测量的 yaw_rate（本步读取后再用于下一步）
        current_yaw = yaw_acc[-1] if yaw_acc else 0.0
        corrected_amps = yaw_ctrl.step(current_yaw, stance_legs, base_amps)

        for lid, j in ctx.triplets.items():
            if lid in group_c:
                targets[j["lift_idx"]]  = _rtj(j["lift_lower"], j["lift_upper"], _sl)
                targets[j["drop_idx"]]  = _rtj(j["drop_lower"], j["drop_upper"], _sd)
                targets[j["swing_idx"]] = _rtj(j["swing_lower"], j["swing_upper"], 0.5)
                continue

            phase_state = leg_phase_state(
                phase_now, lid, phase_offsets, duty_factors
            )
            sw_val = phase_state.fore_aft
            alpha = phase_state.lift
            fv = ctx.fmap.get(lid, np.zeros(2))
            dsign = -1.0 if float(np.dot(fv, lat)) > 0.0 else 1.0

            # 使用闭环修正后的 amplitude
            eff_amp = SWING_AMP * float(corrected_amps.get(str(lid), base_amps.get(str(lid), 1.0)))

            lr = _sl + (_swl - _sl) * alpha
            dr = _sd + (_swd - _sd) * alpha
            sr = 0.5 + eff_amp * dsign * sw_val

            is_sw = not phase_state.is_stance
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

        # ── 读取状态 ──────────────────────────────────────────────────
        states = gym.get_actor_rigid_body_states(env, actor, ga.STATE_ALL)
        if states is not None and len(states) > 0:
            p = states["pose"]["p"][0]
            com_trail.append([float(p["x"]), float(p["y"])])
            yaw_acc.append(float(states["vel"]["angular"][0]["z"]))

        sim_time += _DT

        # 保存诊断（每60步）
        if step % 60 == 0 and step > 0:
            all_diag.append({
                "step": step,
                **yaw_ctrl.get_diagnostics(),
            })

    # ── 统计 ──────────────────────────────────────────────────────────
    valid = [v for v in yaw_acc if abs(v) < 20.0]
    yaw_stats = {
        "yaw_rate_mean": float(np.mean(valid)) if valid else 0.0,
        "yaw_rate_std": float(np.std(valid)) if valid else 0.0,
    }
    return com_trail, fwd.tolist(), yaw_stats, all_diag


# ═══════════════════════════════════════════════════════════════════════════════
# 主实验
# ═══════════════════════════════════════════════════════════════════════════════

CONDITIONS = ["baseline", "+yaw_closed_loop"]


def run_experiment(robot_name, description, urdf_path):
    print(f"\n{'='*60}")
    print(f"  Robot: {robot_name}  ({description.get('num_legs', '?')} legs)")
    print(f"{'='*60}")

    results = {}
    plan_base = compute_adaptive_plan(description, {})
    base_fwd = plan_base["final_forward_axis"]

    with _RobotSimCtx(description, urdf_path, use_gpu=USE_GPU) as ctx:
        # ── baseline ──────────────────────────────────────────────────
        print("  [baseline] running …", end=" ", flush=True)
        t0 = time.perf_counter()
        trail, ax, stats = ctx.run_episode(plan_base, PROBE_STEPS, return_yaw_stats=True)
        results["baseline"] = compute_metrics(trail, base_fwd, stats, len(trail))
        print(f"yaw={results['baseline']['yaw_abs']:.4f}  "
              f"lat={results['baseline']['lat_drift']:+.3f}  "
              f"fwd={results['baseline']['fwd_vel']:+.4f}  "
              f"({time.perf_counter()-t0:.1f}s)")

        # ── +yaw_closed_loop ───────────────────────────────────────────
        print("  [+yaw_closed_loop] running …", end=" ", flush=True)
        t0 = time.perf_counter()
        trail, ax, stats, diag = run_episode_yaw_closed_loop(
            ctx, plan_base, PROBE_STEPS, description
        )
        m = compute_metrics(trail, base_fwd, stats, len(trail))
        results["+yaw_closed_loop"] = m
        # 打印诊断
        cs = diag[-1]["correction_strength"] if diag else 0.0
        print(f"yaw={m['yaw_abs']:.4f}  "
              f"lat={m['lat_drift']:+.3f}  "
              f"fwd={m['fwd_vel']:+.4f}  "
              f"corr_strength={cs:.2f}  "
              f"({time.perf_counter()-t0:.1f}s)")

    return results


def print_summary(all_results):
    print(f"\n{'='*90}")
    print("  方向C v2: 在线偏航力矩闭环修正 — 汇总对比")
    print(f"{'='*90}")

    cond_means = {c: {"yaw": [], "lat": [], "fwd": []} for c in CONDITIONS}
    for rn, rr in all_results.items():
        for cond, m in rr.items():
            for k, store in [("yaw_abs", "yaw"), ("lat_drift", "lat"), ("fwd_vel", "fwd")]:
                v = m.get(k, float("nan"))
                if not math.isnan(v):
                    cond_means[cond][store].append(v)

    hdr = f"{'条件':<20} {'mean|yaw|':>10} {'±std':>8} {'mean|lat|':>10} {'mean fwd':>10} {'vs base':>10}"
    print(hdr)
    print("-" * len(hdr))

    base_ya = float(np.mean([abs(v) for v in cond_means["baseline"]["yaw"]]))
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
        print(f"{cond:<20} {ya_m:>10.4f} {ya_s:>8.3f} {la_m:>10.4f} {fv_m:>10.4f} {imp:>+9.1f}%")

    print(f"\n{'机器人':<20}", end="")
    for c in CONDITIONS:
        print(f"{c:>18}", end="")
    print(f"{'改善':>10}")
    print(f"{'-'*68}")
    for rn in sorted(all_results.keys()):
        b_yaw = all_results[rn]["baseline"]["yaw_abs"]
        print(f"{rn:<20}", end="")
        for c in CONDITIONS:
            ya = all_results[rn].get(c, {}).get("yaw_abs", float("nan"))
            print(f"{ya:>18.4f}", end="")
        ya2 = all_results[rn].get("+yaw_closed_loop", {}).get("yaw_abs", float("nan"))
        imp = (b_yaw - ya2) / max(b_yaw, 1e-9) * 100
        print(f"{imp:>+9.1f}%")


def main():
    print("=" * 70)
    print("  方向C v2: 在线偏航力矩闭环修正")
    print("  Python:", sys.executable)
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
        print(f"[Load] {rdir_name}: {desc.get('num_legs', '?')} legs")
        robot_data.append((rdir_name, desc, urdf_path))

    if not robot_data:
        print("[ERROR] No robots found")
        sys.exit(1)

    all_results = {}
    for robot_name, desc, urdf_path in robot_data:
        try:
            res = run_experiment(robot_name, desc, urdf_path)
            all_results[robot_name] = res
        except Exception as e:
            print(f"[ERROR] {robot_name}: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("[ERROR] All robots failed")
        sys.exit(1)

    print_summary(all_results)

    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = OUTPUT_DIR / f"yaw_closed_loop_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[Save] → {json_path}")
    print("[Done]")


if __name__ == "__main__":
    main()
