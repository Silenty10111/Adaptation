#!/usr/bin/env python3
"""消融实验：对比 5 种步态规划方法的偏航控制效果。

条件
----
  baseline   : compute_adaptive_plan（当前实现，含内置 YAW_COMP_GAIN=0.50）
  +wbc       : baseline + CentroidalWBC 二次规划修正 per_leg_stride_amplitudes
  +dynsym    : baseline + DynamicSymmetry 时域积分优化 stride_scale
  +topo      : TopologyInvariantMapper 零样本拓扑步态（替换 group 分配）
  +topo_wbc  : 拓扑步态 + WBC 二次规划修正（最完整的组合）

指标
----
  yaw_abs   : |yaw_rate_mean| (rad/s)  — 越小越好，衡量偏航漂移
  lat_drift : 横向漂移 (m)             — 越小越好
  fwd_vel   : 前进速度 (m/s)           — 越大越好，确保功能性

机器人来源
---------
  从 batch_results/20260530_221021/ 读取已生成的 5 个机器人
  （seeds: 7, 42, 137, 256, 512）避免重复生成 URDF 的耗时。

运行方式
--------
  python ablation_study.py            # 自动切换到 unitree-rl 环境
  BATCH_REEXEC=1 python ablation_study.py  # 已在目标环境时直接运行
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ─── 配置 ──────────────────────────────────────────────────────────────────

PROBE_STEPS = 480          # 仿真步数（60 Hz × 480 = 8 秒）
USE_GPU     = True

# 机器人来源目录（使用最近一次批量测试的已生成机器人）
_REPO = Path(__file__).resolve().parent.parent
BATCH_RUN_DIR = _REPO / "batch_results" / "20260530_221021"

# 用于消融的机器人子集（相对于 BATCH_RUN_DIR）
ROBOT_DIRS = [
    "robot_00_seed7",
    "robot_01_seed42",
    "robot_02_seed137",
    "robot_03_seed256",
    "robot_04_seed512",
]

OUTPUT_DIR = _REPO / "ablation_results"

# ─── 自动重启到 unitree-rl 环境 ───────────────────────────────

from adaptation.utils import maybe_reexec, compute_metrics

TARGET_PYTHON  = os.environ.get("ISAAC_PYTHON", "/data/conda/envs/unitree-rl/bin/python")
TARGET_LD_PATH = os.environ.get("ISAAC_LD_LIBRARY_PATH", "/data/conda/envs/unitree-rl/lib")

maybe_reexec(TARGET_PYTHON, TARGET_LD_PATH)

# ─── 从此处起，我们已在 unitree-rl Python 环境 ────────────────────────────

sys.path.insert(0, str(_REPO))  # 确保本目录优先导入

from adaptation.gait import compute_adaptive_plan               # noqa: E402
from adaptation.wbc import run_centroidal_wbc                 # noqa: E402
from adaptation.symmetry import integrate_dynamic_symmetry  # noqa: E402
from adaptation.topology import zero_shot_gait_plan     # noqa: E402

# _RobotSimCtx 在 batch_test 里定义，只在实例化时才真正导入 isaacgym
from adaptation.sim import _RobotSimCtx                           # noqa: E402

# ─── 工具函数 ──────────────────────────────────────────────────────────────


def _build_dynsym_override(
    dynsym_plan,
    baseline_per_amp: Dict[str, float],
) -> Dict[str, float]:
    """从 DynamicSymmetryGaitPlan 提取 per_amp override。

    DynSym 的 stride_scale 是乘上基础幅度后的倍率；
    baseline_per_amp 的值已经是最终 per_amp。
    因此组合 = baseline_per_amp * stride_scale（归一化到合理范围）。
    """
    override: Dict[str, float] = {}
    for lid, params in dynsym_plan.leg_params.items():
        base = float(baseline_per_amp.get(str(lid), 1.0))
        # stride_scale 在 [0.1, 1.0]，作用于 SWING_AMP 的倍率
        override[str(lid)] = float(np.clip(base * params.stride_scale, 0.05, 2.0))
    return override


# ─── 主消融流程 ────────────────────────────────────────────────────────────

CONDITIONS = ["baseline", "+wbc", "+dynsym", "+topo", "+topo_wbc"]


def run_ablation_for_robot(
    robot_name: str,
    description: Dict,
    urdf_path: Path,
) -> Dict[str, Dict[str, float]]:
    """对单个机器人跑全部 5 个条件，返回 {condition: metrics} 字典。"""
    print(f"\n{'='*60}")
    print(f"  Robot: {robot_name}")
    print(f"{'='*60}")

    results: Dict[str, Dict[str, float]] = {}

    # ── 1. 离线计算各条件的步态计划（不需要 sim）───────────────────────────
    print("[Plan] Computing baseline plan …")
    plan_base = compute_adaptive_plan(description, {})
    base_per_amp = {
        str(k): float(v)
        for k, v in plan_base["topology"].get("per_leg_stride_amplitudes", {}).items()
    }
    base_fwd = plan_base["final_forward_axis"]

    # +WBC
    print("[Plan] Running CentroidalWBC …")
    try:
        wbc_amps = run_centroidal_wbc(description, plan_base)
    except Exception as e:
        print(f"  [WBC] fallback to baseline amps: {e}")
        wbc_amps = dict(base_per_amp)

    # +DynSym
    print("[Plan] Running DynamicSymmetry …")
    try:
        dynsym_plan = integrate_dynamic_symmetry(
            description, plan_base,
            frequency_hz=0.85, optimize=True, max_iterations=30
        )
        dynsym_amps = _build_dynsym_override(dynsym_plan, base_per_amp)
    except Exception as e:
        print(f"  [DynSym] fallback to baseline amps: {e}")
        dynsym_amps = dict(base_per_amp)

    # +Topo
    print("[Plan] Running TopologyInvariantMapper …")
    try:
        plan_topo = zero_shot_gait_plan(description)
        topo_per_amp = {
            str(k): float(v)
            for k, v in plan_topo["topology"].get("per_leg_stride_amplitudes", {}).items()
        }
    except Exception as e:
        print(f"  [Topo] fallback to baseline plan: {e}")
        plan_topo  = plan_base
        topo_per_amp = dict(base_per_amp)

    # +Topo+WBC
    print("[Plan] Running TopologyInvariantMapper + CentroidalWBC …")
    try:
        topo_wbc_amps = run_centroidal_wbc(description, plan_topo)
    except Exception as e:
        print(f"  [Topo+WBC] fallback to topo amps: {e}")
        topo_wbc_amps = dict(topo_per_amp)

    # 汇总 5 组（plan, _per_amp_override 或直接 plan）
    episodes = [
        ("baseline", plan_base,  None),
        ("+wbc",     plan_base,  wbc_amps),
        ("+dynsym",  plan_base,  dynsym_amps),
        ("+topo",    plan_topo,  None),
        ("+topo_wbc", plan_topo, topo_wbc_amps),
    ]

    # ── 2. 在 ONE sim ctx 里跑 5 个 episode ─────────────────────────────
    print(f"\n[Sim] Opening _RobotSimCtx for {robot_name} …")
    with _RobotSimCtx(description, urdf_path, use_gpu=USE_GPU) as ctx:
        for cond_name, plan, amp_override in episodes:
            # 准备注入 plan
            run_plan = dict(plan)
            if amp_override is not None:
                run_plan["_per_amp_override"] = amp_override

            fwd_axis = run_plan.get("final_forward_axis", [1.0, 0.0])

            print(f"  [{cond_name}] running {PROBE_STEPS} steps …", end=" ", flush=True)
            t0 = time.perf_counter()
            trail, ax, stats = ctx.run_episode(
                run_plan, PROBE_STEPS, return_yaw_stats=True
            )
            dt_wall = time.perf_counter() - t0

            m = compute_metrics(trail, fwd_axis, stats, len(trail))
            results[cond_name] = m
            print(
                f"yaw_abs={m['yaw_abs']:.4f} rad/s  "
                f"lat={m['lat_drift']:+.3f} m  "
                f"fwd={m['fwd_vel']:+.4f} m/s  "
                f"({dt_wall:.1f}s wall)"
            )

    return results


def print_summary_table(all_results: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    """打印多机器人消融汇总表。"""
    print(f"\n{'='*80}")
    print("  消融实验汇总")
    print(f"{'='*80}")

    # 计算各条件均值
    cond_means: Dict[str, Dict[str, List[float]]] = {c: {"yaw_abs": [], "lat_drift": [], "fwd_vel": []} for c in CONDITIONS}
    for robot_name, robot_res in all_results.items():
        for cond, m in robot_res.items():
            for key in ("yaw_abs", "lat_drift", "fwd_vel"):
                v = m.get(key, float("nan"))
                if not math.isnan(v):
                    cond_means[cond][key].append(v)

    header = f"{'条件':<14}  {'yaw_abs(rad/s)':>16}  {'lat_drift(m)':>14}  {'fwd_vel(m/s)':>14}"
    print(header)
    print("-" * len(header))
    for cond in CONDITIONS:
        ya  = cond_means[cond]["yaw_abs"]
        ld  = cond_means[cond]["lat_drift"]
        fv  = cond_means[cond]["fwd_vel"]
        ya_m  = float(np.mean(ya))  if ya  else float("nan")
        ld_m  = float(np.mean([abs(x) for x in ld])) if ld  else float("nan")
        fv_m  = float(np.mean(fv))  if fv  else float("nan")
        ya_s  = float(np.std(ya))   if len(ya)  > 1 else 0.0
        print(f"{cond:<14}  {ya_m:>12.4f}±{ya_s:.3f}  {ld_m:>14.4f}  {fv_m:>14.4f}")

    # 对比改善率（相对于 baseline）
    print(f"\n  相对 baseline 的改善率")
    print("-" * len(header))
    base_ya = float(np.mean(cond_means["baseline"]["yaw_abs"])) if cond_means["baseline"]["yaw_abs"] else 1e-9
    for cond in CONDITIONS[1:]:
        ya = cond_means[cond]["yaw_abs"]
        if not ya:
            continue
        ya_m = float(np.mean(ya))
        improvement = (base_ya - ya_m) / max(base_ya, 1e-9) * 100.0
        arrow = "↑" if improvement > 0 else "↓"
        print(f"{cond:<14}  yaw改善={improvement:+.1f}%  {arrow}")


def save_results(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    out_dir: Path,
) -> None:
    """保存 JSON 结果 + 生成 matplotlib 条形图。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    # JSON
    json_path = out_dir / f"ablation_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[Save] Results → {json_path}")

    # 尝试生成条形图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        robots  = list(all_results.keys())
        n_cond  = len(CONDITIONS)
        x       = np.arange(len(robots))
        width   = 0.14

        # 兼容旧版 matplotlib (< 3.5)
        try:
            cmap = plt.colormaps.get_cmap("tab10")
        except AttributeError:
            cmap = plt.cm.get_cmap("tab10")  # type: ignore[attr-defined]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        metrics_cfg = [
            ("yaw_abs",  "abs(偏航速率) (rad/s)",  "越小越好 ↓"),
            ("lat_drift","横向漂移绝对值 (m)",       "越小越好 ↓"),
            ("fwd_vel",  "前进速度 (m/s)",           "越大越好 ↑"),
        ]

        for ax, (key, ylabel, note) in zip(axes, metrics_cfg):
            for ci, cond in enumerate(CONDITIONS):
                vals = [
                    abs(all_results[r].get(cond, {}).get(key, float("nan")))
                    for r in robots
                ]
                ax.bar(
                    x + (ci - n_cond / 2 + 0.5) * width,
                    vals, width,
                    label=cond,
                    color=cmap(ci),
                    alpha=0.85,
                )
            ax.set_xlabel("机器人")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel}\n({note})")
            ax.set_xticks(x.tolist())
            ax.set_xticklabels([r.replace("robot_", "") for r in robots], rotation=15, fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(axis="y", alpha=0.3)

        plt.suptitle("消融实验：偏航控制方法对比", fontsize=13, fontweight="bold")
        plt.tight_layout()
        png_path = out_dir / f"ablation_{ts}.png"
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Save] Plot → {png_path}")
    except ImportError:
        print("[Save] matplotlib 未安装，跳过绘图")
    except Exception as e:
        import traceback
        print(f"[Save] 绘图失败: {e}")
        traceback.print_exc()


# ─── 入口 ──────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("  消融实验 — 偏航力矩控制模块对比")
    print("  Python:", sys.executable)
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 加载机器人 ──────────────────────────────────────────────────────────
    robot_data: List[Tuple[str, Dict, Path]] = []
    for rdir_name in ROBOT_DIRS:
        rdir = BATCH_RUN_DIR / rdir_name
        desc_path = rdir / "robot_description.json"
        urdf_path = rdir / "robot.urdf"
        if not desc_path.exists():
            print(f"[Load] SKIP {rdir_name}: robot_description.json not found")
            continue
        if not urdf_path.exists():
            print(f"[Load] SKIP {rdir_name}: robot.urdf not found")
            continue
        with open(desc_path, "r", encoding="utf-8") as f:
            desc = json.load(f)
        n_legs = int(desc.get("num_legs", 0))
        print(f"[Load] {rdir_name}: {n_legs} legs  ({desc_path})")
        robot_data.append((rdir_name, desc, urdf_path))

    if not robot_data:
        print("[ERROR] No robots found in", BATCH_RUN_DIR)
        sys.exit(1)

    # ── 逐个机器人跑消融 ────────────────────────────────────────────────────
    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for robot_name, desc, urdf_path in robot_data:
        try:
            res = run_ablation_for_robot(robot_name, desc, urdf_path)
            all_results[robot_name] = res
        except Exception as e:
            print(f"[ERROR] {robot_name} failed: {e}")
            import traceback; traceback.print_exc()

    if not all_results:
        print("[ERROR] All robots failed")
        sys.exit(1)

    # ── 输出结果 ────────────────────────────────────────────────────────────
    print_summary_table(all_results)
    save_results(all_results, OUTPUT_DIR)
    print("\n[Done] 消融实验完成.")


if __name__ == "__main__":
    main()
