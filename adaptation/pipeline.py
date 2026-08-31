#!/usr/bin/env python3
"""
自适应步态规划流水线 (Adaptive Gait Pipeline)
==============================================
可配置步态管线，根据机器人形态自动选择最优控制策略。

策略
----
  baseline  – compute_adaptive_plan（原始实现）
  dynsym    – baseline + DynamicSymmetry 时域优化 stride_scale
  topo      – TopologyInvariantMapper 零样本步态

版本管理
--------
iterative_improve.py 在每次迭代前会将本文件快照至
  iterations/<iter_name>/adaptation.pipeline_snapshot.py
可随时复制快照回本目录以回退版本。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, asdict, field
from typing import Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 配置结构
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """步态管线全部可调参数。"""

    name: str = "default"
    description: str = ""

    # ── 对称性门控 ──────────────────────────────────────────────────────────
    # 初始对称性得分 >= skip_threshold 时跳过所有增强，直接返回 baseline
    skip_threshold: float = 0.92

    # ── DynamicSymmetry ─────────────────────────────────────────────────────
    dynsym_enabled: bool = True
    dynsym_max_iter: int = 30          # 迭代数过多会过优化（30是经验值）
    dynsym_w_yaw: float = 3.0          # 偏航力矩惩罚权重（越大越激进）
    dynsym_max_legs: int = 100         # num_legs > 此值时跳过 DynSym（高腿数机器人易退化）
    dynsym_max_stride_dev: float = 0.50  # 限制 stride_scale 与 1.0 的最大偏离量

    # require_convergence: 仅当优化收敛时才应用 DynSym 结果
    # converge_max_legs : num_legs > 此值时才强制要求收敛（腿少的机器人即使未收敛也应用）
    require_convergence: bool = False
    converge_max_legs: int = 100

    # ── TopologyInvariant ───────────────────────────────────────────────────
    topo_enabled: bool = False
    topo_min_legs: int = 6
    topo_max_legs: int = 9             # 仅对腿数在 [min, max] 范围内使用 Topo
    topo_score_threshold: float = 0.80 # 仅对 sym_score < 此值的机器人启用 Topo

    # ── Phase scheduling (binary remains backward-compatible default) ────────
    phase_strategy: str = "binary"
    duty_factor: float = 0.60
    wave_count: float = 1.0
    wave_direction: float = 1.0
    lateral_phase_lag: float = float(np.pi)
    per_leg_duty_factors: Dict[str, float] = field(default_factory=dict)


# 开箱即用的默认配置
DEFAULT_CONFIG = PipelineConfig(
    name="default",
    description="默认配置（DynSym + 收敛门控，不含 Topo）",
)


# ---------------------------------------------------------------------------
# 核心函数
# ---------------------------------------------------------------------------

def build_gait_plan(
    description: Dict,
    config: PipelineConfig = DEFAULT_CONFIG,
    state: Optional[Dict] = None,
) -> Tuple[Dict, str, float, Dict]:
    """
    根据机器人描述和配置构建步态计划。

    Parameters
    ----------
    description : robot_description.json 内容
    config      : 管线配置

    Returns
    -------
    (plan, strategy, sym_score_initial, diagnostics)

    plan
        兼容 batch_test._RobotSimCtx.run_episode() 的步态计划字典。
        如果包含 _per_amp_override 键，run_episode 会自动使用它替换
        topology.per_leg_stride_amplitudes。

    strategy
        实际使用的策略: 'baseline' | 'dynsym' | 'topo'

    sym_score_initial
        DynSym 分析得到的初始对称性得分 [0, 1]

    diagnostics
        调试信息字典（策略选择原因、各步得分等）
    """
    from adaptation.gait import compute_adaptive_plan
    from adaptation.symmetry import integrate_dynamic_symmetry
    from adaptation.topology import zero_shot_gait_plan

    num_legs = int(description.get("num_legs", 6))

    # ── Step 1: 基础 baseline 计划 ─────────────────────────────────────────
    planner_state = copy.deepcopy(state or {})
    cpg_state = planner_state.setdefault("cpg", {})
    cpg_state.setdefault("phase_strategy", config.phase_strategy)
    cpg_state.setdefault("duty_factor", config.duty_factor)
    cpg_state.setdefault("wave_count", config.wave_count)
    cpg_state.setdefault("wave_direction", config.wave_direction)
    cpg_state.setdefault("lateral_phase_lag", config.lateral_phase_lag)
    cpg_state.setdefault("per_leg_duty_factors", dict(config.per_leg_duty_factors))
    plan = compute_adaptive_plan(description, planner_state)
    base_per_amp: Dict[str, float] = {
        str(k): float(v)
        for k, v in plan["topology"].get("per_leg_stride_amplitudes", {}).items()
    }

    # ── Step 2: 对称性诊断（不优化，仅分析） ──────────────────────────────
    sym_score = 1.0
    try:
        dynsym_diag = integrate_dynamic_symmetry(description, plan, optimize=False)
        sym_score = dynsym_diag.symmetry_score
    except Exception as e:
        print(f"[Pipeline] Sym diagnostic failed: {e}")

    diag: Dict = {
        "num_legs": num_legs,
        "sym_score_initial": round(sym_score, 4),
        "config_name": config.name,
        "phase_strategy": plan.get("cpg", {}).get("phase_strategy", "binary"),
    }

    # ── Step 3: 对称性门控 — 得分够高则直接用 baseline ──────────────────
    if sym_score >= config.skip_threshold:
        _tag = f"score={sym_score:.3f} >= skip_threshold={config.skip_threshold}"
        print(f"[Pipeline:{config.name}] {_tag} → baseline (skip)")
        diag.update(strategy="baseline", reason=_tag)
        return plan, "baseline", sym_score, diag

    # ── Step 3b: DynSym 腿数上限门控 ───────────────────────────────────────
    # 腿数过多的机器人 DynSym 的解析模型误差大，强制使用 baseline
    if config.dynsym_enabled and num_legs > config.dynsym_max_legs:
        _tag = f"num_legs={num_legs} > dynsym_max_legs={config.dynsym_max_legs} → baseline"
        print(f"[Pipeline:{config.name}] {_tag}")
        diag.update(strategy="baseline", reason=_tag)
        return plan, "baseline", sym_score, diag

    # ── Step 4: Topo 门控 ─────────────────────────────────────────────────
    if (config.topo_enabled
            and config.topo_min_legs <= num_legs <= config.topo_max_legs
            and sym_score < config.topo_score_threshold):
        try:
            plan_topo = zero_shot_gait_plan(description, state=planner_state)
            _tag = (f"topo: legs={num_legs}∈[{config.topo_min_legs},{config.topo_max_legs}],"
                    f" score={sym_score:.3f}<{config.topo_score_threshold}")
            print(f"[Pipeline:{config.name}] {_tag}")
            diag.update(strategy="topo", reason=_tag)
            return plan_topo, "topo", sym_score, diag
        except Exception as e:
            print(f"[Pipeline:{config.name}] Topo failed ({e}), fallback to DynSym")

    # ── Step 5: DynSym ─────────────────────────────────────────────────────
    if config.dynsym_enabled:
        try:
            dynsym_opt = integrate_dynamic_symmetry(
                description, plan,
                optimize=True,
                w_yaw=config.dynsym_w_yaw,
                max_iterations=config.dynsym_max_iter,
            )
            converged = dynsym_opt.optimization_converged

            # 腿数较多时要求收敛才应用
            if config.require_convergence and num_legs > config.converge_max_legs and not converged:
                _tag = (f"DynSym not converged, legs={num_legs}>{config.converge_max_legs}"
                        f" → baseline")
                print(f"[Pipeline:{config.name}] {_tag}")
                diag.update(strategy="baseline", reason=_tag,
                            dynsym_converged=False)
                return plan, "baseline", sym_score, diag

            override = _build_dynsym_override(
                dynsym_opt, base_per_amp,
                max_stride_dev=config.dynsym_max_stride_dev,
            )
            plan_ds = copy.deepcopy(plan)
            plan_ds["_per_amp_override"] = override
            _tag = (f"dynsym: score {sym_score:.3f}→{dynsym_opt.symmetry_score:.3f},"
                    f" converged={converged}")
            print(f"[Pipeline:{config.name}] {_tag}")
            diag.update(
                strategy="dynsym",
                dynsym_converged=converged,
                sym_score_after=round(dynsym_opt.symmetry_score, 4),
                reason=_tag,
            )
            return plan_ds, "dynsym", sym_score, diag

        except Exception as e:
            print(f"[Pipeline:{config.name}] DynSym failed: {e}")

    # ── Fallback ─────────────────────────────────────────────────────────
    print(f"[Pipeline:{config.name}] fallback to baseline")
    diag.update(strategy="baseline", reason="fallback")
    return plan, "baseline", sym_score, diag


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _build_dynsym_override(
    dynsym_plan,
    base_per_amp: Dict[str, float],
    max_stride_dev: float = 0.50,
) -> Dict[str, float]:
    """
    从 DynamicSymmetryGaitPlan 构建 per_amp 覆盖量。

    max_stride_dev 限制 stride_scale 与 1.0 的最大偏离量，
    防止过优化产生过大的振幅修正导致步态失稳。
    """
    override: Dict[str, float] = {}
    for lid, params in dynsym_plan.leg_params.items():
        base = float(base_per_amp.get(str(lid), 1.0))
        # 将 stride_scale 裁剪到 [1 - max_dev, 1 + max_dev]
        clamped_scale = float(np.clip(
            params.stride_scale,
            1.0 - max_stride_dev,
            1.0 + max_stride_dev,
        ))
        override[str(lid)] = float(np.clip(base * clamped_scale, 0.05, 2.0))
    return override
