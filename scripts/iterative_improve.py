#!/usr/bin/env python3
"""
迭代改进框架 (Iterative Improvement Framework)
===============================================
自动运行"评估 → 分析 → 修正 → 保存"循环，迭代优化步态控制策略。

工作流程
--------
  1. 从初始 PipelineConfig 出发
  2. 运行评估（baseline vs pipeline 双路对比仿真）
  3. 分析结果：找出退化机器人、偏航超标机器人
  4. 自动生成下一轮 PipelineConfig（修正问题）
  5. 将本轮快照（config + 代码 + 结果 + 分析报告）保存到独立文件夹
  6. 重复至 MAX_ITERATIONS 或所有机器人达标

输出目录结构
-----------
iterations/
  iter_01_dynsym_converge_gate/
    config.json                 ← 本轮 PipelineConfig
    adaptation.pipeline_snapshot.py   ← adaptation.pipeline.py 快照（可复制回滚）
    results.json                ← 每个机器人的 baseline/pipeline 指标
    analysis.json               ← 自动分析：退化/超标/改善 分类
    analysis.md                 ← 可读分析报告
    comparison.png              ← 与 baseline 的对比条形图
  iter_02_.../ ...
  SUMMARY.json                  ← 跨轮次汇总
  SUMMARY.png                   ← 跨轮次趋势图

运行方式
--------
  python iterative_improve.py            # 自动切换到 unitree-rl 环境
  BATCH_REEXEC=1 python iterative_improve.py  # 已在目标环境中
"""

from __future__ import annotations

import copy
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ─── 配置区 ────────────────────────────────────────────────────────────────

PROBE_STEPS    = 480          # 每次仿真步数（60 Hz × 480 = 8 s）
USE_GPU        = True
MAX_ITERATIONS = 6            # 最多运行几轮迭代

YAW_TARGET       = 0.035      # 偏航目标（rad/s），低于此值认为合格
REGRESSION_RATIO = 1.15       # pipeline yaw > baseline × 此比例 = 退化

# 机器人数据目录（与 ablation_study.py 相同）
_REPO          = Path(__file__).resolve().parent.parent
BATCH_RUN_DIR  = _REPO / "batch_results" / "20260530_221021"
ROBOT_DIRS     = [
    "robot_00_seed7",
    "robot_01_seed42",
    "robot_02_seed137",
    "robot_03_seed256",
    "robot_04_seed512",
]

ITERATIONS_DIR = _REPO / "iterations"

# ─── 自动重启到 unitree-rl ─────────────────────────────────────────────────

from adaptation.utils import maybe_reexec, compute_metrics

TARGET_PYTHON  = os.environ.get("ISAAC_PYTHON", "/data/conda/envs/unitree-rl/bin/python")
TARGET_LD_PATH = os.environ.get("ISAAC_LD_LIBRARY_PATH", "/data/conda/envs/unitree-rl/lib")

maybe_reexec(TARGET_PYTHON, TARGET_LD_PATH)

# ─── 正式导入（已在 unitree-rl 环境） ─────────────────────────────────────

sys.path.insert(0, str(_REPO))

from adaptation.gait import compute_adaptive_plan          # noqa: E402
from adaptation.pipeline import PipelineConfig, build_gait_plan  # noqa: E402
from adaptation.sim import _RobotSimCtx                      # noqa: E402

# ─── 初始配置（第一轮，基于消融实验 + iter_01 错误分析重新设计） ────────────
#
# 消融实验发现：
#   seed7  (10腿): baseline 0.0108 已够好；DynSym（任何迭代数）均引发退化
#   seed42 ( 4腿): DynSym 30迭代有效 (-49%)，Topo 较差
#   seed256( 7腿): Topo 效果惊人 (-94%)；DynSym 30迭代也好 (-54%)
#   seed512( 6腿): baseline 已够好 (0.0097)，skip_threshold 过滤
#
# iter_01 教训：
#   ❌ 增大 max_iter=50 使 DynSym 过优化，seed256 从改善变退化
#   ❌ converge_max_legs 作为门控无效（DynSym 会收敛但结果差）
#   ✅ 正确门控: dynsym_max_legs=9（排除10腿），max_iter=30（防过优化）
#   ✅ 初始启用 Topo（6-9腿且不对称时自动选择）
#
# 新策略：
#   seed7  (10腿): dynsym_max_legs=9 → baseline ✓
#   seed42 ( 4腿): DynSym(30步), Topo 腿数不满足 → dynsym ✓
#   seed256( 7腿): Topo 门控 (6≤7≤9, score<0.80) → topo ✓
#   seed512( 6腿): skip_threshold=0.92 > score=0.965 → baseline ✓

INITIAL_CONFIG = PipelineConfig(
    name="iter_01_topo_plus_dynsym30",
    description="Topo处理6-9腿不对称机器人，DynSym(30步)处理其余，10+腿用baseline",
    skip_threshold=0.92,
    dynsym_enabled=True,
    dynsym_max_iter=30,
    dynsym_max_legs=9,           # 10腿不用DynSym（模型与仿真不符）
    dynsym_max_stride_dev=0.45,  # 限制步幅修正幅度防过校正
    dynsym_w_yaw=3.0,
    require_convergence=False,
    converge_max_legs=100,       # 关闭收敛门控（由dynsym_max_legs代替）
    topo_enabled=True,
    topo_min_legs=6,
    topo_max_legs=9,
    topo_score_threshold=0.80,
)


# ─── 单轮评估 ──────────────────────────────────────────────────────────────

def run_iteration(
    config: PipelineConfig,
    robots: List[Tuple[str, Dict, Path]],
) -> Dict[str, Dict]:
    """
    对所有机器人跑 baseline 和 pipeline 两个条件，返回结果字典。

    结构: {robot_name: {"baseline": metrics, "pipeline": metrics+diag}}
    """
    results: Dict[str, Dict] = {}

    for robot_name, desc, urdf_path in robots:
        print(f"\n  {'─'*50}")
        print(f"  [{robot_name}]  num_legs={desc.get('num_legs', '?')}")
        print(f"  {'─'*50}")

        try:
            with _RobotSimCtx(desc, urdf_path, use_gpu=USE_GPU) as ctx:
                # ── Baseline ──────────────────────────────────────────────
                plan_base = compute_adaptive_plan(desc, {})
                trail_b, ax_b, stats_b = ctx.run_episode(
                    plan_base, PROBE_STEPS, return_yaw_stats=True
                )
                m_base = compute_metrics(trail_b, ax_b, stats_b, len(trail_b))
                print(f"  [baseline] yaw_abs={m_base['yaw_abs']:.4f}  "
                      f"lat={m_base['lat_drift']:+.3f}  fwd={m_base['fwd_vel']:+.4f}")

                # ── Pipeline ──────────────────────────────────────────────
                plan_p, strategy, sym_score, diag = build_gait_plan(desc, config)
                trail_p, ax_p, stats_p = ctx.run_episode(
                    plan_p, PROBE_STEPS, return_yaw_stats=True
                )
                m_pipe = compute_metrics(trail_p, ax_p, stats_p, len(trail_p))
                m_pipe["strategy"]  = strategy
                m_pipe["sym_score"] = round(sym_score, 4)
                m_pipe.update({k: v for k, v in diag.items() if k not in m_pipe})

                delta = m_base["yaw_abs"] - m_pipe["yaw_abs"]
                sign  = "↓" if delta > 0 else "↑"
                print(f"  [pipeline/{strategy}] yaw_abs={m_pipe['yaw_abs']:.4f}  "
                      f"lat={m_pipe['lat_drift']:+.3f}  fwd={m_pipe['fwd_vel']:+.4f}  "
                      f"Δyaw={delta:+.4f} {sign}")

                results[robot_name] = {"baseline": m_base, "pipeline": m_pipe}

        except Exception as e:
            print(f"  [ERROR] {robot_name}: {e}")
            import traceback; traceback.print_exc()

    return results


# ─── 自动分析 ──────────────────────────────────────────────────────────────

def auto_analyze(results: Dict, config: PipelineConfig) -> Dict:
    """
    分析当前迭代结果，识别三类机器人：
      regressions   : pipeline yaw > baseline × REGRESSION_RATIO
      persistent_bad: yaw > YAW_TARGET（但未退化）
      improvements  : yaw <= YAW_TARGET 且未退化
    """
    regressions   = {}
    persistent_bad = {}
    improvements  = {}

    for robot, res in results.items():
        b_yaw = res["baseline"].get("yaw_abs", 0)
        p_yaw = res["pipeline"].get("yaw_abs", 0)
        strategy = res["pipeline"].get("strategy", "unknown")
        num_legs = res["pipeline"].get("num_legs", 6)

        if math.isnan(p_yaw):
            continue

        if p_yaw > b_yaw * REGRESSION_RATIO and b_yaw > 1e-4:
            regressions[robot] = {
                "baseline_yaw": round(b_yaw, 5),
                "pipeline_yaw": round(p_yaw, 5),
                "degradation_pct": round((p_yaw - b_yaw) / max(b_yaw, 1e-6) * 100, 1),
                "strategy": strategy,
                "num_legs": num_legs,
            }
        elif p_yaw > YAW_TARGET:
            persistent_bad[robot] = {
                "baseline_yaw": round(b_yaw, 5),
                "pipeline_yaw": round(p_yaw, 5),
                "improvement_pct": round((b_yaw - p_yaw) / max(b_yaw, 1e-6) * 100, 1),
                "strategy": strategy,
                "num_legs": num_legs,
            }
        else:
            improvements[robot] = {
                "baseline_yaw": round(b_yaw, 5),
                "pipeline_yaw": round(p_yaw, 5),
                "improvement_pct": round((b_yaw - p_yaw) / max(b_yaw, 1e-6) * 100, 1),
                "strategy": strategy,
                "num_legs": num_legs,
            }

    valid_yaws = [
        res["pipeline"]["yaw_abs"] for res in results.values()
        if not math.isnan(res["pipeline"].get("yaw_abs", float("nan")))
    ]
    mean_yaw = float(np.mean(valid_yaws)) if valid_yaws else float("nan")
    converged = (len(regressions) == 0 and len(persistent_bad) == 0)

    return {
        "regressions": regressions,
        "persistent_bad": persistent_bad,
        "improvements": improvements,
        "summary": {
            "mean_yaw_abs": round(mean_yaw, 5),
            "n_regressions": len(regressions),
            "n_persistent_bad": len(persistent_bad),
            "n_improvements": len(improvements),
            "converged": converged,
            "config_name": config.name,
        },
    }


def format_analysis_report(analysis: Dict, config: PipelineConfig) -> str:
    """生成人可读的分析报告（Markdown 格式）。"""
    s = analysis["summary"]
    lines = [
        f"# 分析报告 — {config.name}",
        f"",
        f"**配置描述**: {config.description}",
        f"",
        f"## 汇总",
        f"- 均值 yaw_abs: **{s['mean_yaw_abs']:.5f} rad/s**",
        f"- 退化机器人: {s['n_regressions']} 个",
        f"- 偏航超标: {s['n_persistent_bad']} 个",
        f"- 达标改善: {s['n_improvements']} 个",
        f"- 全部达标: {'✅ 是' if s['converged'] else '❌ 否'}",
        f"",
    ]
    if analysis["regressions"]:
        lines += ["## ⚠️ 退化（pipeline 比 baseline 更差）", ""]
        for robot, d in analysis["regressions"].items():
            lines.append(
                f"- **{robot}** ({d['num_legs']}腿): "
                f"baseline={d['baseline_yaw']:.4f} → pipeline={d['pipeline_yaw']:.4f} "
                f"({d['degradation_pct']:+.1f}%)  策略={d['strategy']}"
            )
        lines.append("")

    if analysis["persistent_bad"]:
        lines += ["## 🔴 偏航超标（yaw > {:.3f} rad/s）".format(YAW_TARGET), ""]
        for robot, d in analysis["persistent_bad"].items():
            lines.append(
                f"- **{robot}** ({d['num_legs']}腿): "
                f"baseline={d['baseline_yaw']:.4f} → pipeline={d['pipeline_yaw']:.4f} "
                f"({d['improvement_pct']:+.1f}%)  策略={d['strategy']}"
            )
        lines.append("")

    if analysis["improvements"]:
        lines += ["## ✅ 达标改善", ""]
        for robot, d in analysis["improvements"].items():
            lines.append(
                f"- **{robot}** ({d['num_legs']}腿): "
                f"baseline={d['baseline_yaw']:.4f} → pipeline={d['pipeline_yaw']:.4f} "
                f"({d['improvement_pct']:+.1f}%)  策略={d['strategy']}"
            )
        lines.append("")

    return "\n".join(lines)


# ─── 自动调参：生成下一轮配置 ──────────────────────────────────────────────

def auto_tune_next_config(
    current_config: PipelineConfig,
    analysis: Dict,
    next_iter_num: int,
) -> PipelineConfig:
    """
    根据分析结果自动生成下一轮 PipelineConfig。

    决策规则（按优先级）
    --------------------
    1. 退化 → 修补导致退化的策略门控（收紧阈值）
    2. 超标 → 尝试更激进的策略（开启/扩大 Topo，增加 DynSym 迭代）
    3. 全部达标 → 轻微降低 skip_threshold（覆盖更多机器人）
    """
    cfg = copy.deepcopy(current_config)
    cfg.name = f"iter_{next_iter_num:02d}_auto"
    reasons: List[str] = []

    regressions  = analysis["regressions"]
    persist_bad  = analysis["persistent_bad"]

    # ── 1. 修复退化 ─────────────────────────────────────────────────────────
    dynsym_reg_robots = [r for r, d in regressions.items() if d["strategy"] == "dynsym"]
    topo_reg_robots   = [r for r, d in regressions.items() if d["strategy"] == "topo"]

    if dynsym_reg_robots:
        # DynSym 引发退化 → 收紧 dynsym_max_legs 以排除高腿数机器人
        # (此前错误地修改了 converge_max_legs，但 require_convergence=False
        #  时该参数无效；正确做法是直接限制 DynSym 适用的最大腿数)
        bad_legs = [regressions[r]["num_legs"] for r in dynsym_reg_robots]
        max_bad_leg = max(bad_legs)
        if max_bad_leg <= cfg.dynsym_max_legs:
            old_val = cfg.dynsym_max_legs
            # 排除所有退化腿数：dynsym_max_legs = min(退化的腿) - 1
            cfg.dynsym_max_legs = max(4, min(bad_legs) - 1)
            reasons.append(
                f"DynSym 退化于 {dynsym_reg_robots}（腿数 {bad_legs}）"
                f" → dynsym_max_legs: {old_val}→{cfg.dynsym_max_legs}"
            )
        # 同时降低 DynSym 迭代数，防止过优化
        if cfg.dynsym_max_iter > 20:
            old_iter = cfg.dynsym_max_iter
            cfg.dynsym_max_iter = max(20, cfg.dynsym_max_iter - 10)
            if old_iter != cfg.dynsym_max_iter:
                reasons.append(
                    f"降低 DynSym 迭代防过优化: {old_iter}→{cfg.dynsym_max_iter}"
                )

    if topo_reg_robots:
        # Topo 引发退化 → 收紧 topo 腿数范围
        bad_legs = [regressions[r]["num_legs"] for r in topo_reg_robots]
        if any(l > cfg.topo_max_legs - 1 for l in bad_legs):
            old_max = cfg.topo_max_legs
            cfg.topo_max_legs = max(cfg.topo_min_legs, min(bad_legs) - 1)
            reasons.append(
                f"Topo 退化于 {topo_reg_robots} → topo_max_legs: {old_max}→{cfg.topo_max_legs}"
            )
        elif any(l < cfg.topo_min_legs + 1 for l in bad_legs):
            old_min = cfg.topo_min_legs
            cfg.topo_min_legs = min(cfg.topo_max_legs, max(bad_legs) + 1)
            reasons.append(
                f"Topo 退化于 {topo_reg_robots} → topo_min_legs: {old_min}→{cfg.topo_min_legs}"
            )
        else:
            # 在腿数范围内但对称性阈值不够严格
            old_thresh = cfg.topo_score_threshold
            cfg.topo_score_threshold = min(0.90, cfg.topo_score_threshold + 0.05)
            reasons.append(
                f"Topo 退化于 {topo_reg_robots}"
                f" → topo_score_threshold: {old_thresh:.2f}→{cfg.topo_score_threshold:.2f}"
            )

    # ── 2. 处理偏航超标 ──────────────────────────────────────────────────────
    for robot, bad in persist_bad.items():
        strategy  = bad["strategy"]
        num_legs  = bad["num_legs"]

        if strategy == "baseline":
            # 当前选择了 baseline（被门控过滤），需要放开门控
            if num_legs <= 9 and not cfg.topo_enabled and cfg.topo_min_legs <= num_legs <= cfg.topo_max_legs:
                cfg.topo_enabled = True
                reasons.append(f"启用 Topo（{robot} baseline 超标, legs={num_legs}）")
            elif cfg.dynsym_enabled and num_legs <= cfg.dynsym_max_legs:
                # 降低 skip_threshold 以通过门控进 DynSym
                old_st = cfg.skip_threshold
                cfg.skip_threshold = max(0.75, cfg.skip_threshold - 0.05)
                if cfg.skip_threshold != old_st:
                    reasons.append(
                        f"降低 skip_threshold: {old_st:.2f}→{cfg.skip_threshold:.2f}"
                        f"（{robot} baseline 超标, legs={num_legs}）"
                    )

        elif strategy == "dynsym":
            # DynSym 应用了但仍超标 — 先尝试 Topo（如果腿数合适）
            if not cfg.topo_enabled and cfg.topo_min_legs <= num_legs <= cfg.topo_max_legs:
                cfg.topo_enabled = True
                reasons.append(f"启用 Topo（{robot} DynSym 后仍超标, legs={num_legs}）")
            elif cfg.dynsym_max_iter < 80:
                old_iter = cfg.dynsym_max_iter
                cfg.dynsym_max_iter = min(cfg.dynsym_max_iter + 20, 80)
                reasons.append(
                    f"增加 DynSym 迭代: {old_iter}→{cfg.dynsym_max_iter}（{robot} 超标）"
                )

        elif strategy == "topo":
            # Topo 后仍超标 → 尝试组合 Topo+WBC 或 扩大 Topo 范围
            if cfg.dynsym_enabled and num_legs <= cfg.dynsym_max_legs:
                # 降低 skip_threshold 让 DynSym 也能介入
                old_st = cfg.skip_threshold
                cfg.skip_threshold = max(0.75, cfg.skip_threshold - 0.05)
                if cfg.skip_threshold != old_st:
                    reasons.append(
                        f"降低 skip_threshold: {old_st:.2f}→{cfg.skip_threshold:.2f}"
                        f"（{robot} Topo 后仍超标）"
                    )

    # ── 3. 全部达标 → 轻微扩展覆盖 ─────────────────────────────────────────
    if not regressions and not persist_bad:
        if cfg.skip_threshold > 0.78:
            old_st = cfg.skip_threshold
            cfg.skip_threshold = max(0.78, cfg.skip_threshold - 0.04)
            reasons.append(
                f"全部达标，降低 skip_threshold: {old_st:.2f}→{cfg.skip_threshold:.2f}"
            )
        else:
            reasons.append("全部达标，配置已稳定，继续验证")

    if not reasons:
        reasons.append("无需调整（保持当前配置继续验证）")

    cfg.description = "; ".join(reasons)
    return cfg


# ─── 保存与绘图 ────────────────────────────────────────────────────────────

def save_iteration(
    iter_dir: Path,
    config: PipelineConfig,
    results: Dict,
    analysis: Dict,
    analysis_report: str,
) -> None:
    """将本轮所有数据持久化到 iter_dir。"""
    iter_dir.mkdir(parents=True, exist_ok=True)

    # 1. PipelineConfig snapshot
    config_dict = {k: v for k, v in vars(config).items()}
    with open(iter_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    # 2. adaptation.pipeline.py 代码快照（用于版本回退）
    src = _REPO / "gait_pipeline.py"
    if src.exists():
        shutil.copy2(src, iter_dir / "gait_pipeline_snapshot.py")

    # 3. 评估结果
    with open(iter_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 4. 分析 JSON
    with open(iter_dir / "analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    # 5. 分析报告（Markdown）
    with open(iter_dir / "analysis.md", "w", encoding="utf-8") as f:
        f.write(analysis_report)

    print(f"  [Save] 结果写入 {iter_dir.relative_to(_REPO)}/")


def plot_iteration(
    iter_dir: Path,
    results: Dict,
    iter_label: str,
    all_history: Optional[List[Tuple[str, Dict]]] = None,
) -> None:
    """生成对比条形图 (baseline vs pipeline) + 历史趋势图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        robots = list(results.keys())
        x      = np.arange(len(robots))
        width  = 0.35

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        metrics = [
            ("yaw_abs",   "|偏航速率| (rad/s)"),
            ("lat_drift", "横向漂移 (m)"),
            ("fwd_vel",   "前进速度 (m/s)"),
        ]
        try:
            cmap = plt.colormaps.get_cmap("Set2")
        except AttributeError:
            cmap = plt.cm.get_cmap("Set2")  # type: ignore

        for ax, (key, ylabel) in zip(axes, metrics):
            b_vals = [abs(results[r]["baseline"].get(key, float("nan"))) for r in robots]
            p_vals = [abs(results[r]["pipeline"].get(key, float("nan"))) for r in robots]
            ax.bar(x - width / 2, b_vals, width, label="baseline", color=cmap(0), alpha=0.85)
            ax.bar(x + width / 2, p_vals, width, label=f"pipeline\n({iter_label})", color=cmap(1), alpha=0.85)
            if key == "yaw_abs":
                ax.axhline(y=YAW_TARGET, color="red", linestyle="--", alpha=0.6, label=f"target={YAW_TARGET}")
            ax.set_xlabel("机器人")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            ax.set_xticks(x.tolist())
            ax.set_xticklabels([r.replace("robot_", "") for r in robots], rotation=15, fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(axis="y", alpha=0.3)

        plt.suptitle(f"迭代对比: {iter_label}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(iter_dir / "comparison.png", dpi=140, bbox_inches="tight")
        plt.close()

        # 历史趋势图（若有多轮数据）
        if all_history and len(all_history) >= 2:
            labels     = [h[0] for h in all_history]
            mean_yaws  = []
            for _, hist_res in all_history:
                yaws = [hist_res[r]["pipeline"].get("yaw_abs", float("nan"))
                        for r in hist_res if not math.isnan(hist_res[r]["pipeline"].get("yaw_abs", float("nan")))]
                mean_yaws.append(float(np.mean(yaws)) if yaws else float("nan"))

            fig2, ax2 = plt.subplots(figsize=(max(6, len(labels) * 1.5), 4))
            ax2.plot(range(len(labels)), mean_yaws, "o-", color="steelblue", linewidth=2, markersize=8)
            ax2.axhline(y=YAW_TARGET, color="red", linestyle="--", alpha=0.6, label=f"目标 {YAW_TARGET}")
            ax2.set_xticks(range(len(labels)))
            ax2.set_xticklabels(labels, rotation=20, fontsize=8)
            ax2.set_ylabel("均值 |偏航速率| (rad/s)")
            ax2.set_title("跨轮次偏航改善趋势")
            ax2.legend()
            ax2.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(ITERATIONS_DIR / "trend.png", dpi=140, bbox_inches="tight")
            plt.close()

        print(f"  [Plot] 图像已保存")
    except ImportError:
        print("  [Plot] matplotlib 未安装，跳过绘图")
    except Exception as e:
        print(f"  [Plot] 绘图失败: {e}")


# ─── 汇总保存 ──────────────────────────────────────────────────────────────

def save_summary(all_history: List[Tuple[str, Dict, Dict]], out_dir: Path) -> None:
    """将跨轮次汇总写入 SUMMARY.json。"""
    summary = {}
    for iter_name, res, analysis in all_history:
        robots_summary = {}
        for robot, rd in res.items():
            robots_summary[robot] = {
                "baseline_yaw": round(rd["baseline"].get("yaw_abs", float("nan")), 5),
                "pipeline_yaw": round(rd["pipeline"].get("yaw_abs", float("nan")), 5),
                "strategy": rd["pipeline"].get("strategy", "?"),
            }
        summary[iter_name] = {
            "robots": robots_summary,
            "summary": analysis["summary"],
        }

    with open(out_dir / "SUMMARY.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[Summary] → {out_dir / 'SUMMARY.json'}")


# ─── 机器人加载 ────────────────────────────────────────────────────────────

def load_robots() -> List[Tuple[str, Dict, Path]]:
    robots = []
    for rdir_name in ROBOT_DIRS:
        rdir      = BATCH_RUN_DIR / rdir_name
        desc_path = rdir / "robot_description.json"
        urdf_path = rdir / "robot.urdf"
        if not desc_path.exists() or not urdf_path.exists():
            print(f"[Load] SKIP {rdir_name}: 缺少 robot_description.json 或 robot.urdf")
            continue
        with open(desc_path, "r", encoding="utf-8") as f:
            desc = json.load(f)
        print(f"[Load] {rdir_name}: {desc.get('num_legs', '?')} 腿")
        robots.append((rdir_name, desc, urdf_path))
    return robots


# ─── 主循环 ────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("  迭代改进框架 — 自动评估-分析-修正循环")
    print(f"  Python: {sys.executable}")
    print(f"  最大迭代: {MAX_ITERATIONS}  目标 yaw ≤ {YAW_TARGET}")
    print("=" * 70)

    ITERATIONS_DIR.mkdir(parents=True, exist_ok=True)
    robots = load_robots()
    if not robots:
        print("[ERROR] 无可用机器人，退出")
        sys.exit(1)

    config           = INITIAL_CONFIG
    all_history_plot = []   # [(iter_name, results)] 用于趋势图
    all_history_sum  = []   # [(iter_name, results, analysis)] 用于 SUMMARY

    for iter_num in range(1, MAX_ITERATIONS + 1):
        iter_label = config.name if config.name.startswith("iter_") else f"iter_{iter_num:02d}_{config.name}"
        iter_dir   = ITERATIONS_DIR / iter_label

        print(f"\n{'='*70}")
        print(f"  迭代 {iter_num}/{MAX_ITERATIONS}: {iter_label}")
        print(f"  {config.description}")
        print(f"{'='*70}")

        t0 = time.perf_counter()

        # ── 运行评估 ────────────────────────────────────────────────────────
        results = run_iteration(config, robots)
        if not results:
            print("[ERROR] 本轮无结果，跳过")
            continue

        # ── 自动分析 ────────────────────────────────────────────────────────
        analysis = auto_analyze(results, config)
        report   = format_analysis_report(analysis, config)

        s = analysis["summary"]
        print(f"\n  ── 本轮分析 ──────────────────────────────────────────────")
        print(f"  均值 yaw_abs = {s['mean_yaw_abs']:.5f} rad/s")
        print(f"  退化: {s['n_regressions']}  超标: {s['n_persistent_bad']}  改善: {s['n_improvements']}")
        if analysis["regressions"]:
            for r, d in analysis["regressions"].items():
                print(f"  ⚠️  退化 {r}: {d['baseline_yaw']:.4f}→{d['pipeline_yaw']:.4f} "
                      f"({d['degradation_pct']:+.1f}%)  策略={d['strategy']}")
        if analysis["persistent_bad"]:
            for r, d in analysis["persistent_bad"].items():
                print(f"  🔴 超标 {r}: yaw={d['pipeline_yaw']:.4f}  策略={d['strategy']}")

        # ── 持久化 ──────────────────────────────────────────────────────────
        all_history_plot.append((iter_label, results))
        all_history_sum.append((iter_label, results, analysis))

        save_iteration(iter_dir, config, results, analysis, report)
        plot_iteration(iter_dir, results, iter_label, all_history_plot)
        save_summary(all_history_sum, ITERATIONS_DIR)

        elapsed = time.perf_counter() - t0
        print(f"\n  ✓ 迭代 {iter_num} 完成 ({elapsed:.0f}s)")

        # ── 收敛检查 ────────────────────────────────────────────────────────
        if s["converged"]:
            print(f"\n🎉 所有机器人达标！迭代提前结束于第 {iter_num} 轮。")
            break

        # ── 生成下一轮配置 ──────────────────────────────────────────────────
        if iter_num < MAX_ITERATIONS:
            config = auto_tune_next_config(config, analysis, iter_num + 1)
            print(f"\n  → 下一轮: {config.name}")
            print(f"    {config.description}")

    # ── 最终汇总打印 ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  跨轮次结果汇总")
    print(f"{'='*70}")
    print(f"{'迭代':<40}  {'均值yaw':>10}  {'退化':>6}  {'超标':>6}  {'改善':>6}")
    print("-" * 68)
    for iter_name, _, a in all_history_sum:
        s = a["summary"]
        print(f"{iter_name:<40}  {s['mean_yaw_abs']:>10.5f}  "
              f"{s['n_regressions']:>6}  {s['n_persistent_bad']:>6}  {s['n_improvements']:>6}")

    print(f"\n[Done] 所有结果已保存至 iterations/")


if __name__ == "__main__":
    main()
