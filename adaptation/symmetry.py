#!/usr/bin/env python3
"""
动态对称性步态规划器 (Dynamic Symmetry Gait Planner)
=====================================================
关联论文: Extreme Dynamic Symmetry Enables Omnidirectional
         and Multifunctional Robots

核心思想
--------
"静态不对称"并不意味着无法直线行走。
即使机器人左边 3 条腿、右边 1 条腿，只要在一个完整步态周期内：

    ∫₀ᵀ τ_yaw(t) dt = 0   (偏航力矩时间积分为零)
    ∫₀ᵀ F_lat(t) dt = 0   (横向力时间积分为零)

机器人就能完美走直线 —— 这称为"动态对称性 (Dynamic Symmetry)"。

实现策略
--------
1. DynamicSymmetryAnalyzer  — 分析当前步态的动态对称性残差
2. AsymmetricGaitOptimizer  — 通过调整各腿组的相位偏移、占空比和步幅，
   在保持稳定性的同时使动态对称性条件趋于满足。
3. DynamicSymmetryGaitPlan  — 存储非对称步态参数的数据结构
4. integrate_dynamic_symmetry() — 一键接口，与 adaptation.gait 输出对接

关键创新点
----------
- 允许 group_a 和 group_b 拥有不同的：
    · 相位偏移 (phase_offset_a ≠ π - phase_offset_b)
    · 占空比 (duty_factor_a ≠ duty_factor_b)
    · 步幅缩放 (stride_scale_a ≠ stride_scale_b)
- 通过离散傅里叶分析量化动态对称性质量
- 提供"对称性得分 (Symmetry Score)"用于诊断
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from adaptation.phase import (
    circular_distance,
    resolve_duty_factors,
    resolve_phase_offsets,
    wrap_2pi,
)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class LegGaitParams:
    """单腿步态参数。"""
    leg_id: int
    phase_offset: float    # 相位偏移 [0, 2π]
    duty_factor: float     # 占空比（支撑相比例）[0.3, 0.8]
    stride_scale: float    # 步幅缩放因子 [0.1, 1.0]
    group: str             # 所属组 'a', 'b', 'c'


@dataclass
class DynamicSymmetryGaitPlan:
    """
    动态对称性步态计划。
    扩展了 adaptation.gait 的基础计划，添加非对称参数。
    """
    base_plan: Dict                          # adaptation.gait 原始输出
    leg_params: Dict[int, LegGaitParams]     # 每腿精细参数
    symmetry_score: float                    # 动态对称性得分 [0,1]，1=完美对称
    net_yaw_integral: float                  # 偏航力矩积分（理想=0）
    net_lat_integral: float                  # 横向力积分（理想=0）
    optimization_converged: bool             # 优化是否收敛
    diagnostic: Dict                         # 诊断信息


# ---------------------------------------------------------------------------
# 1. 动态对称性分析器
# ---------------------------------------------------------------------------

class DynamicSymmetryAnalyzer:
    """
    分析步态序列的动态对称性。

    方法
    ----
    给定各腿的相位、占空比和步幅，在时间域上积分偏航力矩和横向力，
    评估动态对称性残差。
    """

    def __init__(
        self,
        foot_positions_xy: Dict[int, np.ndarray],  # 足端 XY 坐标
        com_xy: np.ndarray,                         # 质心 XY
        forward_axis: np.ndarray,                   # 前进方向（单位向量）
        n_samples: int = 200,                       # 时域采样点数
    ):
        self._feet = foot_positions_xy
        self._com = com_xy[:2].copy()
        fwd = np.asarray(forward_axis[:2], dtype=float)
        fwd_norm = float(np.linalg.norm(fwd))
        self._fwd = fwd / fwd_norm if fwd_norm > 1e-9 else np.array([1., 0.])
        self._lat = np.array([-self._fwd[1], self._fwd[0]], dtype=float)
        self._n = n_samples

    def compute_symmetry_integrals(
        self,
        leg_params: Dict[int, LegGaitParams],
        frequency_hz: float = 1.0,
    ) -> Tuple[float, float]:
        """
        计算一个步态周期内的偏航力矩积分和横向力积分。

        步态模型
        --------
        第 i 腿的推进力方向 = forward_axis（简化：匀速行走，忽略重力分量）
        推进力大小 ∝ stride_scale_i × duty_envelope(φ_i(t))

        占空比包络：
            φ_i(t) = 2π * freq * t + phase_offset_i
            q_i(t) = mod(φ_i(t), 2π) / 2π
            contact_i(t) = 1 if q_i(t) < duty_factor_i else 0
            （duty_factor=0.6 意味着 60% 时间在支撑相）

        Returns
        -------
        (yaw_integral, lat_integral)
        """
        T = 1.0 / max(frequency_hz, 0.01)  # 步态周期
        dt = T / self._n
        t_arr = np.linspace(0, T - dt, self._n)

        yaw_sum = 0.0
        lat_sum = 0.0

        for leg_id, params in leg_params.items():
            if leg_id not in self._feet:
                continue
            if params.group == 'c':
                continue  # 被动腿不产生推进力

            foot = self._feet[leg_id]
            r = foot - self._com

            # 偏航杠杆（与 adaptation.gait 一致）
            yaw_lever = float(r[0] * self._fwd[1] - r[1] * self._fwd[0])
            lat_lever = float(np.dot(r, self._lat))

            # 时间域接触函数
            phi = 2.0 * np.pi * frequency_hz * t_arr + params.phase_offset
            normalized_phase = np.mod(phi, 2.0 * np.pi) / (2.0 * np.pi)
            in_contact = (normalized_phase < params.duty_factor).astype(float)

            # 推进力包络（支撑相时产生 forward 推力）
            # 简化：矩形包络，实际可用 smoothstep
            force_fwd = params.stride_scale * in_contact

            # 偏航力矩贡献
            yaw_sum += float(np.sum(force_fwd * yaw_lever)) * dt
            # 横向力贡献（对于纯前进步态，理想横向力=0）
            lat_sum += float(np.sum(force_fwd * lat_lever)) * dt

        return yaw_sum, lat_sum

    def compute_symmetry_score(
        self,
        leg_params: Dict[int, LegGaitParams],
        frequency_hz: float = 1.0,
    ) -> float:
        """
        动态对称性得分 ∈ [0, 1]。

        score = exp(-α * (|yaw_integral| + |lat_integral|))
        α 由各腿步幅的典型值归一化。
        """
        yaw_int, lat_int = self.compute_symmetry_integrals(leg_params, frequency_hz)
        # 归一化：用总推进力积分（理想情况下偏航=0）
        total_thrust = sum(
            p.stride_scale * max(p.duty_factor, 0.1)
            for p in leg_params.values()
            if p.group != 'c'
        )
        T = 1.0 / max(frequency_hz, 0.01)
        norm = max(total_thrust * T * 0.5, 1e-6)

        residual = (abs(yaw_int) + 0.5 * abs(lat_int)) / norm
        score = math.exp(-3.0 * residual)
        return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 2. 非对称步态优化器
# ---------------------------------------------------------------------------

class AsymmetricGaitOptimizer:
    """
    通过迭代优化调整腿组参数，使动态对称性条件满足。

    优化变量
    --------
    对于每个腿组 (group_a, group_b)：
    - 相位偏移调整量 Δφ_a, Δφ_b  ∈ [-π/4, π/4]
    - 占空比调整量 Δd_a, Δd_b     ∈ [-0.15, +0.15]
    - 步幅缩放调整量 Δs_a, Δs_b    ∈ [-0.25, +0.25]

    目标函数
    --------
    min  w₁|Yaw_integral|² + w₂|Lat_integral|²
         + w₃ Σ_i (Δ_i)²  (正则化，避免过大调整)
    s.t.
         duty_factor_i ∈ [0.35, 0.75]
         stride_scale_i ∈ [0.10, 0.90]
         phase_offset_i ∈ [0, 2π]

    求解方法：坐标下降法（轻量，无需 scipy）
    """

    def __init__(
        self,
        analyzer: DynamicSymmetryAnalyzer,
        w_yaw: float = 3.0,
        w_lat: float = 1.0,
        w_reg: float = 0.5,
        max_iterations: int = 50,
        step_size: float = 0.08,
        tol: float = 1e-4,
    ):
        self._analyzer = analyzer
        self._w_yaw = w_yaw
        self._w_lat = w_lat
        self._w_reg = w_reg
        self._max_iter = max_iterations
        self._step = step_size
        self._tol = tol

    def _objective(
        self,
        leg_params: Dict[int, LegGaitParams],
        base_params: Dict[int, LegGaitParams],
        frequency_hz: float,
    ) -> float:
        yaw_int, lat_int = self._analyzer.compute_symmetry_integrals(
            leg_params, frequency_hz
        )
        reg = sum(
            circular_distance(p.phase_offset, base_params[lid].phase_offset) ** 2 +
            (p.duty_factor - base_params[lid].duty_factor) ** 2 * 4.0 +
            (p.stride_scale - base_params[lid].stride_scale) ** 2 * 2.0
            for lid, p in leg_params.items()
            if lid in base_params
        )
        return (
            self._w_yaw * yaw_int ** 2 +
            self._w_lat * lat_int ** 2 +
            self._w_reg * reg
        )

    def optimize(
        self,
        initial_params: Dict[int, LegGaitParams],
        frequency_hz: float = 1.0,
    ) -> Tuple[Dict[int, LegGaitParams], bool, float]:
        """
        坐标下降法优化步态参数。

        Returns
        -------
        (optimized_params, converged, final_score)
        """
        import copy
        params = {lid: copy.copy(p) for lid, p in initial_params.items()}
        base_params = {lid: copy.copy(p) for lid, p in initial_params.items()}

        # 只优化 group_a 和 group_b 中的腿
        active_ids = [
            lid for lid, p in params.items() if p.group in ('a', 'b')
        ]
        if not active_ids:
            score = self._analyzer.compute_symmetry_score(params, frequency_hz)
            return params, True, score

        prev_obj = self._objective(params, base_params, frequency_hz)
        converged = False
        step = self._step

        for iteration in range(self._max_iter):
            improved = False

            for lid in active_ids:
                p = params[lid]

                # --- 优化相位偏移 ---
                orig_phase = p.phase_offset
                p.phase_offset = wrap_2pi(orig_phase + step)
                obj_plus = self._objective(params, base_params, frequency_hz)
                p.phase_offset = wrap_2pi(orig_phase - step)
                obj_minus = self._objective(params, base_params, frequency_hz)
                if obj_plus < prev_obj and obj_plus <= obj_minus:
                    p.phase_offset = wrap_2pi(orig_phase + step)
                    prev_obj = obj_plus
                    improved = True
                elif obj_minus < prev_obj:
                    p.phase_offset = wrap_2pi(orig_phase - step)
                    prev_obj = obj_minus
                    improved = True
                else:
                    p.phase_offset = orig_phase

                for attr, lo, hi, delta in [
                    ("duty_factor", 0.35, 0.75, step * 0.3),
                    ("stride_scale", 0.10, 0.90, step * 0.5),
                ]:
                    orig = getattr(p, attr)

                    # 尝试 +delta
                    setattr(p, attr, float(np.clip(orig + delta, lo, hi)))
                    obj_plus = self._objective(params, base_params, frequency_hz)

                    # 尝试 -delta
                    setattr(p, attr, float(np.clip(orig - delta, lo, hi)))
                    obj_minus = self._objective(params, base_params, frequency_hz)

                    # 选择最优
                    if obj_plus < prev_obj and obj_plus <= obj_minus:
                        setattr(p, attr, float(np.clip(orig + delta, lo, hi)))
                        prev_obj = obj_plus
                        improved = True
                    elif obj_minus < prev_obj:
                        setattr(p, attr, float(np.clip(orig - delta, lo, hi)))
                        prev_obj = obj_minus
                        improved = True
                    else:
                        setattr(p, attr, orig)  # 回退

            # 自适应步长
            if not improved:
                step *= 0.6
                if step < self._tol:
                    converged = True
                    break

        final_score = self._analyzer.compute_symmetry_score(params, frequency_hz)
        return params, converged, final_score


# ---------------------------------------------------------------------------
# 3. 动态对称性步态规划入口
# ---------------------------------------------------------------------------

def build_initial_leg_params(
    gait_plan: Dict,
    foot_positions_xy: Dict[int, np.ndarray],
) -> Dict[int, LegGaitParams]:
    """
    从 adaptation.gait 输出构造初始 LegGaitParams。

    Phase 约定（与 build_gait_targets 一致）：
        group_a → phase_offset = 0
        group_b → phase_offset = π
        group_c → 被动，不优化
    """
    topo = gait_plan.get("topology", {})
    groups = topo.get("groups", {})
    group_a = set(groups.get("group_a", []))
    group_b = set(groups.get("group_b", []))
    group_c = set(groups.get("group_c", []))

    base_amps = {
        int(k): float(v)
        for k, v in topo.get("per_leg_stride_amplitudes", {}).items()
    }

    all_active = group_a | group_b | group_c
    phase_offsets = resolve_phase_offsets(gait_plan, all_active)
    duty_factors, _ = resolve_duty_factors(gait_plan, all_active)
    leg_params: Dict[int, LegGaitParams] = {}

    for lid in all_active:
        if lid in group_a:
            grp = 'a'
        elif lid in group_b:
            grp = 'b'
        else:
            grp = 'c'

        leg_params[lid] = LegGaitParams(
            leg_id=lid,
            phase_offset=phase_offsets[lid],
            duty_factor=duty_factors[lid],
            stride_scale=float(base_amps.get(lid, 0.5)),
            group=grp,
        )

    return leg_params


def integrate_dynamic_symmetry(
    description: Dict,
    gait_plan: Dict,
    frequency_hz: float = 0.85,
    optimize: bool = True,
    w_yaw: float = 3.0,
    max_iterations: int = 40,
) -> DynamicSymmetryGaitPlan:
    """
    分析并优化步态的动态对称性。

    Parameters
    ----------
    description   : robot_description.json
    gait_plan     : compute_adaptive_plan() 输出
    frequency_hz  : 步态频率 (Hz)
    optimize      : 是否运行优化（若 False 仅分析）
    w_yaw         : 偏航力矩权重
    max_iterations: 优化最大迭代次数

    Returns
    -------
    DynamicSymmetryGaitPlan
    """
    # 提取足端位置
    foot_positions_xy: Dict[int, np.ndarray] = {}
    for link in description.get("links", []):
        if link.get("role") != "foot" or link.get("leg_id") is None:
            continue
        origin = np.asarray(link.get("default_world_origin", [0., 0., 0.]), dtype=float)
        foot_positions_xy[int(link["leg_id"])] = origin[:2]

    com_xy = np.asarray(gait_plan.get("projected_com_xy", [0., 0.]), dtype=float)
    forward_axis = np.asarray(gait_plan.get("final_forward_axis", [1., 0.]), dtype=float)

    # 初始步态参数
    leg_params = build_initial_leg_params(gait_plan, foot_positions_xy)

    analyzer = DynamicSymmetryAnalyzer(
        foot_positions_xy, com_xy, forward_axis
    )

    # 分析初始对称性
    yaw_int_init, lat_int_init = analyzer.compute_symmetry_integrals(
        leg_params, frequency_hz
    )
    score_init = analyzer.compute_symmetry_score(leg_params, frequency_hz)
    print(
        f"[DynSymmetry] 初始: yaw_integral={yaw_int_init:.4f}  "
        f"lat_integral={lat_int_init:.4f}  score={score_init:.4f}"
    )

    converged = not optimize
    if optimize and score_init < 0.90:
        optimizer = AsymmetricGaitOptimizer(
            analyzer,
            w_yaw=w_yaw,
            max_iterations=max_iterations,
        )
        leg_params, converged, final_score = optimizer.optimize(
            leg_params, frequency_hz=frequency_hz
        )

        yaw_int_final, lat_int_final = analyzer.compute_symmetry_integrals(
            leg_params, frequency_hz
        )
        print(
            f"[DynSymmetry] 优化后: yaw_integral={yaw_int_final:.4f}  "
            f"lat_integral={lat_int_final:.4f}  score={final_score:.4f}  "
            f"converged={converged}"
        )
    else:
        yaw_int_final, lat_int_final = yaw_int_init, lat_int_init
        final_score = score_init
        print("[DynSymmetry] 对称性已足够好，跳过优化。")

    yaw_int, lat_int = analyzer.compute_symmetry_integrals(leg_params, frequency_hz)

    # 将优化后的 leg_params 反映到 gait_plan 的 per_leg_stride_amplitudes
    topo = gait_plan.get("topology", {})
    per_leg_amps = dict(topo.get("per_leg_stride_amplitudes", {}))
    for lid, params in leg_params.items():
        per_leg_amps[str(lid)] = params.stride_scale

    # 构造相位偏移字典
    phase_offsets_opt: Dict[str, float] = {
        str(lid): params.phase_offset
        for lid, params in leg_params.items()
    }

    # 诊断信息
    diag = {
        "initial_symmetry_score": score_init,
        "final_symmetry_score": final_score,
        "initial_yaw_integral": yaw_int_init,
        "initial_lat_integral": lat_int_init,
        "final_yaw_integral": yaw_int,
        "final_lat_integral": lat_int,
        "optimization_converged": converged,
        "frequency_hz": frequency_hz,
        "per_group_duty_factor": {
            "group_a": float(
                np.mean([leg_params[lid].duty_factor
                         for lid in topo.get("groups", {}).get("group_a", [])
                         if lid in leg_params])
            ) if topo.get("groups", {}).get("group_a") else 0.6,
            "group_b": float(
                np.mean([leg_params[lid].duty_factor
                         for lid in topo.get("groups", {}).get("group_b", [])
                         if lid in leg_params])
            ) if topo.get("groups", {}).get("group_b") else 0.6,
        },
        "per_leg_phase_offsets": phase_offsets_opt,
    }

    # 将优化结果写回 gait_plan (in-place 更新，外部可直接使用)
    topo["per_leg_stride_amplitudes"] = per_leg_amps
    if "cpg" in gait_plan:
        gait_plan["cpg"]["phase_offsets"] = phase_offsets_opt
        gait_plan["cpg"]["per_leg_duty_factors"] = {
            str(lid): float(params.duty_factor)
            for lid, params in leg_params.items()
            if params.group in ('a', 'b')
        }

    return DynamicSymmetryGaitPlan(
        base_plan=gait_plan,
        leg_params=leg_params,
        symmetry_score=final_score,
        net_yaw_integral=yaw_int,
        net_lat_integral=lat_int,
        optimization_converged=converged,
        diagnostic=diag,
    )


# ---------------------------------------------------------------------------
# 扩展 build_gait_targets 支持非对称相位
# ---------------------------------------------------------------------------

def get_asymmetric_phase(
    leg_id: int,
    leg_params: Dict[int, "LegGaitParams"],
    base_phase: float,
) -> float:
    """
    替代 test_gait.py 中的 leg_group_phase()，支持每腿独立相位偏移。

    Parameters
    ----------
    leg_id     : 腿 ID
    leg_params : DynamicSymmetryGaitPlan.leg_params
    base_phase : 当前时间对应的全局相位 = 2π * freq * t

    Returns
    -------
    float  该腿的当前相位
    """
    if leg_id in leg_params:
        return base_phase + leg_params[leg_id].phase_offset
    return base_phase


# ---------------------------------------------------------------------------
# CLI 测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="动态对称性步态规划器测试"
    )
    parser.add_argument("--description", type=Path,
                        default=Path("robot_assets/robot_description.json"))
    parser.add_argument("--gait-plan", type=Path, default=None,
                        help="adaptation.gait 输出 JSON；不传则自动计算")
    parser.add_argument("--frequency", type=float, default=0.85,
                        help="步态频率 (Hz)")
    parser.add_argument("--no-optimize", action="store_true",
                        help="仅分析，不运行优化")
    parser.add_argument("--max-iter", type=int, default=40,
                        help="优化最大迭代次数")
    args = parser.parse_args()

    desc = json.loads(args.description.read_text(encoding="utf-8"))

    if args.gait_plan:
        plan = json.loads(args.gait_plan.read_text(encoding="utf-8"))
    else:
        from adaptation.gait import compute_adaptive_plan
        plan = compute_adaptive_plan(desc, {})

    sym_plan = integrate_dynamic_symmetry(
        desc, plan,
        frequency_hz=args.frequency,
        optimize=not args.no_optimize,
        max_iterations=args.max_iter,
    )

    print(f"\n[结果] 动态对称性得分: {sym_plan.symmetry_score:.4f}")
    print(f"  净偏航积分:   {sym_plan.net_yaw_integral:.6f}")
    print(f"  净横向积分:   {sym_plan.net_lat_integral:.6f}")
    print(f"  优化收敛:     {sym_plan.optimization_converged}")
    print("\n[腿参数]")
    for lid in sorted(sym_plan.leg_params):
        p = sym_plan.leg_params[lid]
        print(f"  leg {lid} [{p.group}]: "
              f"phase={math.degrees(p.phase_offset):.1f}°  "
              f"duty={p.duty_factor:.3f}  "
              f"stride={p.stride_scale:.3f}")
