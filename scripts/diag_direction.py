#!/usr/bin/env python3
"""
前进方向诊断可视化 — 叠加 PCA 初始轴、驱动力评分、最终轴、各腿摆动贡献。

用法:
  python diag_direction.py robot_assets/robot_description.json
  python diag_direction.py batch_results/20260623_125539/robot_00_seed7/robot_description.json
  python diag_direction.py --batch batch_results/20260623_125539  # 批量诊断
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root on path
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from adaptation.gait import compute_adaptive_plan


def plot_direction_diagnostic(
    description: dict,
    out_path: Path,
    robot_name: str = "robot",
    com_trail: List[List[float]] | None = None,
) -> Dict:
    """生成前进方向诊断图。

    Returns 诊断数据字典。
    """
    plan = compute_adaptive_plan(description, {})

    # ── 提取关键数据 ──────────────────────────────────────────────────────
    initial_axis = np.asarray(plan["initial_virtual_forward_axis"], dtype=float)
    final_axis   = np.asarray(plan["final_forward_axis"], dtype=float)
    scores       = plan["direction_scores"]
    pos_score    = scores["positive"]
    neg_score    = scores["negative"]
    drive_res    = np.asarray(plan["drive_resultant_xy"], dtype=float)
    support_poly = np.asarray(plan.get("support_polygon_xy", []), dtype=float)
    safety_corr  = np.asarray(plan.get("safety_corridor_xy", []), dtype=float)
    proj_com     = np.asarray(plan.get("projected_com_xy", [0, 0]), dtype=float)
    trans_comp   = np.asarray(plan.get("translational_compensation_xy", [0, 0]), dtype=float)
    planned_swings = plan.get("planned_swings", {})
    per_leg_amps   = plan["topology"].get("per_leg_stride_amplitudes", {})
    yaw_balance    = plan.get("yaw_balance", {})
    psi_by_leg     = yaw_balance.get("psi_by_leg", {})
    groups         = plan["topology"]["groups"]

    # 归一化
    _norm = lambda v: v / max(float(np.linalg.norm(v)), 1e-9)

    # ── 躯干轮廓 ──────────────────────────────────────────────────────────
    trunk_poly_raw = np.array(description.get("trunk_polygon_xy", []), dtype=float)
    if len(trunk_poly_raw) < 3:
        bl = float(description.get("body_length", 0.3))
        bw = float(description.get("body_width", 0.2))
        trunk_poly_raw = np.array([
            [-bl/2, -bw/2], [bl/2, -bw/2], [bl/2, bw/2], [-bl/2, bw/2]
        ])

    # ── 足端和髋部 ────────────────────────────────────────────────────────
    foot_positions: Dict[int, np.ndarray] = {}
    hip_positions: Dict[int, np.ndarray] = {}
    for lk in description.get("links", []):
        if lk.get("leg_id") is None:
            continue
        origin = np.asarray(lk.get("default_world_origin", [0, 0, 0]), dtype=float)[:2]
        lid = int(lk["leg_id"])
        if lk.get("role") == "foot":
            foot_positions[lid] = origin
        elif str(lk.get("name", "")).endswith("_hip"):
            hip_positions[lid] = origin

    # ── 躯干主轴 (PCA of trunk polygon) ───────────────────────────────────
    trunk_center = trunk_poly_raw.mean(axis=0)
    trunk_centered = trunk_poly_raw - trunk_center
    trunk_cov = trunk_centered.T @ trunk_centered / len(trunk_poly_raw)
    trunk_eigvals, trunk_eigvecs = np.linalg.eigh(trunk_cov)
    trunk_main_axis = trunk_eigvecs[:, -1]  # 最大特征值 = 躯干长轴
    trunk_main_axis /= max(float(np.linalg.norm(trunk_main_axis)), 1e-9)

    # 躯干主轴和 PCA 初始轴的点积 — 方向一致性
    trunk_vs_pca = float(np.dot(trunk_main_axis, _norm(initial_axis)))

    # ── 足端分布 PCA 特征值比 (对称性指标) ───────────────────────────────
    if len(foot_positions) >= 2:
        foot_pts = np.array(list(foot_positions.values()))
        foot_center = foot_pts.mean(axis=0)
        foot_cov = (foot_pts - foot_center).T @ (foot_pts - foot_center) / len(foot_pts)
        foot_eigvals, _ = np.linalg.eigh(foot_cov)
        eig_ratio = min(foot_eigvals) / max(foot_eigvals) if max(foot_eigvals) > 1e-9 else 1.0
    else:
        eig_ratio = 1.0

    # ── 绘图 ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    # ===== 左图: 机器人结构 + 方向分析 =====
    ax = axes[0]

    # 躯干
    poly_c = np.vstack([trunk_poly_raw, trunk_poly_raw[0]])
    ax.fill(poly_c[:, 0], poly_c[:, 1], color="lightgray", alpha=0.6, zorder=2)
    ax.plot(poly_c[:, 0], poly_c[:, 1], "k-", lw=1.5, zorder=3)

    # 髋→足连线
    for lid, fp in foot_positions.items():
        hp = hip_positions.get(lid, fp)
        ax.plot([hp[0], fp[0]], [hp[1], fp[1]], color="dimgray", lw=1.2, zorder=3)
        ax.plot(fp[0], fp[1], "ko", ms=5, zorder=4)
        # 腿 ID 标签
        ax.annotate(str(lid), (fp[0], fp[1]), xytext=(3, 3),
                    textcoords="offset points", fontsize=7, color="darkred")

    # 支撑多边形
    if len(support_poly) >= 3:
        sp_c = np.vstack([support_poly, support_poly[0]])
        ax.plot(sp_c[:, 0], sp_c[:, 1], "g--", lw=1.2, alpha=0.7, label="Support Polygon")

    # 质心
    ax.plot(proj_com[0], proj_com[1], "r*", ms=14, zorder=8, label="Proj. CoM")

    # 躯干主轴 (灰色虚线)
    ta_len = float(np.linalg.norm(trunk_poly_raw.max(axis=0) - trunk_poly_raw.min(axis=0))) * 0.45
    ax.annotate("", xy=(trunk_center[0] + trunk_main_axis[0] * ta_len,
                        trunk_center[1] + trunk_main_axis[1] * ta_len),
                xytext=(trunk_center[0] - trunk_main_axis[0] * ta_len,
                        trunk_center[1] - trunk_main_axis[1] * ta_len),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=1.2, ls="--"),
                zorder=5)
    ax.annotate("Trunk Axis", xy=(trunk_center[0] + trunk_main_axis[0] * ta_len * 0.6,
                                   trunk_center[1] + trunk_main_axis[1] * ta_len * 0.6),
                fontsize=7, color="gray")

    # PCA 初始轴 (蓝色虚线)
    ia_len = ta_len * 0.8
    center = proj_com[:2]
    ax.annotate("", xy=(center[0] + initial_axis[0] * ia_len,
                        center[1] + initial_axis[1] * ia_len),
                xytext=(center[0] - initial_axis[0] * ia_len,
                        center[1] - initial_axis[1] * ia_len),
                arrowprops=dict(arrowstyle="<->", color="blue", lw=1.5, ls="--"),
                zorder=5)
    ax.annotate(f"PCA ({pos_score:.1f})", xy=(center[0] + initial_axis[0] * ia_len * 0.7,
                                               center[1] + initial_axis[1] * ia_len * 0.7),
                fontsize=7, color="blue", fontweight="bold")

    # PCA 反向 (红色虚线 — 低分方向)
    ax.annotate("", xy=(center[0] - initial_axis[0] * ia_len * 0.5,
                        center[1] - initial_axis[1] * ia_len * 0.5),
                xytext=(center[0], center[1]),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.0, ls=":"),
                zorder=5)
    ax.annotate(f"Rev ({neg_score:.1f})", xy=(center[0] - initial_axis[0] * ia_len * 0.3,
                                               center[1] - initial_axis[1] * ia_len * 0.3),
                fontsize=7, color="red")

    # 最终前进方向 (绿色实线)
    ax.annotate("", xy=(center[0] + final_axis[0] * ia_len,
                        center[1] + final_axis[1] * ia_len),
                xytext=(center[0], center[1]),
                arrowprops=dict(arrowstyle="->", color="green", lw=2.5),
                zorder=9)
    ax.annotate("FWD", xy=(center[0] + final_axis[0] * ia_len * 0.5,
                            center[1] + final_axis[1] * ia_len * 0.5),
                fontsize=8, color="green", fontweight="bold")

    # 各腿摆动方向 (小箭头)
    for lid, sinfo in planned_swings.items():
        lid_i = int(lid)
        if lid_i in foot_positions:
            fp = foot_positions[lid_i]
            sv = np.asarray(sinfo.get("swing_vector", [0, 0]), dtype=float)
            sv_n = float(np.linalg.norm(sv))
            if sv_n > 0.01:
                sv_hat = sv / sv_n * 0.08
                proj = sinfo.get("forward_projection", 0)
                # 颜色: 绿=正向贡献, 红=负贡献
                col = "green" if proj > 0 else "red"
                ax.annotate("", xy=(fp[0] + sv_hat[0], fp[1] + sv_hat[1]),
                            xytext=(fp[0], fp[1]),
                            arrowprops=dict(arrowstyle="->", color=col, lw=1.0, alpha=0.7),
                            zorder=6)

    # 图例
    ax.legend(loc="upper right", fontsize=7)
    ax.set_aspect("equal")
    ax.set_title(
        f"Forward Direction Diagnostic\n{robot_name}  "
        f"num_legs={description.get('num_legs','?')}  "
        f"eig_ratio={eig_ratio:.3f}  trunk_vs_pca={trunk_vs_pca:+.2f}",
        fontsize=10,
    )
    ax.grid(True, ls=":", alpha=0.4)

    # ===== 右图: 各腿摆动贡献条形图 =====
    ax2 = axes[1]
    leg_ids = sorted(foot_positions.keys())
    x = np.arange(len(leg_ids))

    # 摆动投影
    projections = [planned_swings.get(str(lid), {}).get("forward_projection", 0) for lid in leg_ids]
    amps = [float(per_leg_amps.get(str(lid), 0)) for lid in leg_ids]
    psis = [float(psi_by_leg.get(str(lid), 0)) for lid in leg_ids]

    colors_proj = ["green" if p > 0 else "red" for p in projections]
    ax2.bar(x - 0.25, projections, 0.25, color=colors_proj, alpha=0.7, label="Swing Projection")
    ax2.bar(x + 0.00, amps, 0.25, color="steelblue", alpha=0.7, label="Stride Amplitude")
    ax2.bar(x + 0.25, psis, 0.25, color="orange", alpha=0.7, label="Yaw Lever (ψ)")

    ax2.axhline(y=0, color="black", lw=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Leg {lid}" for lid in leg_ids], rotation=30, fontsize=8)
    ax2.set_ylabel("Value")
    ax2.set_title(f"Per-Leg Contribution  "
                  f"(score: +{pos_score:.1f} / -{neg_score:.1f} → final=+{max(pos_score,neg_score):.1f})\n"
                  f"group_a={groups['group_a']}  group_b={groups['group_b']}  "
                  f"group_c={groups.get('group_c', [])}",
                  fontsize=9)
    ax2.legend(fontsize=7)
    ax2.grid(axis="y", ls=":", alpha=0.4)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 诊断指标 ──────────────────────────────────────────────────────────
    diag = {
        "robot_name": robot_name,
        "num_legs": description.get("num_legs", 0),
        "pos_score": round(pos_score, 2),
        "neg_score": round(neg_score, 2),
        "score_ratio": round(pos_score / max(neg_score, 0.01), 2),
        "eig_ratio": round(eig_ratio, 3),
        "trunk_vs_pca": round(trunk_vs_pca, 3),
        "trunk_vs_final": round(float(np.dot(trunk_main_axis, final_axis)), 3),
        "pca_vs_final": round(float(np.dot(initial_axis, final_axis)), 3),
        "n_passive_legs": len(groups.get("group_c", [])),
        "mean_swing_proj": round(float(np.mean([abs(p) for p in projections])), 3) if projections else 0,
        "max_yaw_lever": round(max(abs(v) for v in psis), 3) if psis else 0,
        # 警告标记
        "warnings": [],
    }

    # 自动检测问题
    if trunk_vs_pca < 0.3:
        diag["warnings"].append("PCA轴偏离躯干主轴>70° — 腿分布与躯干方向严重不一致")
    if diag["score_ratio"] < 1.5:
        diag["warnings"].append(f"方向评分差距小 ({pos_score:.1f} vs {neg_score:.1f}) — PCA轴可能噪声敏感")
    if max(projections) - min(projections) > 1.5:
        diag["warnings"].append("摆动贡献极不平衡 — 强腿主导方向选择")
    if diag["n_passive_legs"] > len(leg_ids) * 0.25:
        diag["warnings"].append(f"{diag['n_passive_legs']}条腿被标记为被动 — 机器人严重不对称")

    return diag


def batch_diagnose(batch_dir: Path, out_dir: Path):
    """批量诊断 batch_results 目录中的所有机器人。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    all_diags = []

    for robot_dir in sorted(batch_dir.iterdir()):
        if not robot_dir.is_dir() or not robot_dir.name.startswith("robot_"):
            continue
        desc_path = robot_dir / "robot_description.json"
        if not desc_path.exists():
            continue

        try:
            description = json.loads(desc_path.read_text(encoding="utf-8"))
            out_path = out_dir / f"{robot_dir.name}_diag.png"
            diag = plot_direction_diagnostic(description, out_path, robot_dir.name)
            all_diags.append(diag)
            warnings = "; ".join(diag["warnings"]) if diag["warnings"] else "OK"
            print(f"  [{diag['robot_name']}] "
                  f"legs={diag['num_legs']}  "
                  f"score={diag['pos_score']}/{diag['neg_score']}  "
                  f"eig_ratio={diag['eig_ratio']:.3f}  "
                  f"trunk_vs_pca={diag['trunk_vs_pca']:+.2f}  "
                  f"WARN: {warnings}")
        except Exception as e:
            print(f"  [{robot_dir.name}] ERROR: {e}")

    # 保存诊断摘要
    summary_path = out_dir / "direction_diag_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_diags, f, indent=2, ensure_ascii=False)

    # 打印问题机器人
    problem = [d for d in all_diags if d["warnings"]]
    print(f"\n{'='*60}")
    print(f"  问题机器人: {len(problem)}/{len(all_diags)} ({len(problem)/max(len(all_diags),1)*100:.0f}%)")
    print(f"{'='*60}")
    for d in sorted(problem, key=lambda x: len(x["warnings"]), reverse=True):
        print(f"  {d['robot_name']} (legs={d['num_legs']}):")
        for w in d["warnings"]:
            print(f"    ⚠ {w}")


# ─── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="前进方向诊断可视化")
    parser.add_argument("description", nargs="?", help="robot_description.json 路径")
    parser.add_argument("--batch", type=Path, help="batch_results 目录路径")
    parser.add_argument("--output", type=Path, default=Path("png/direction_diag"),
                        help="输出目录 (default: png/direction_diag)")
    parser.add_argument("--trail", type=Path, help="batch 结果中的 trajectory 数据 (summary.json)")
    args = parser.parse_args()

    if args.batch:
        batch_diagnose(args.batch, args.output)
        return

    if not args.description:
        # 默认：用 robot_assets 中的
        desc_path = _REPO / "robot_assets" / "robot_description.json"
        if desc_path.exists():
            args.description = str(desc_path)
        else:
            print("No description file specified. Use --batch or provide path.")
            sys.exit(1)

    desc_path = Path(args.description)
    description = json.loads(desc_path.read_text(encoding="utf-8"))
    out_path = args.output / f"{desc_path.parent.name}_diag.png"
    diag = plot_direction_diagnostic(description, out_path,
                                      robot_name=desc_path.parent.name)
    for w in diag["warnings"]:
        print(f"⚠ {w}")
    print(f"[Save] → {out_path}")


if __name__ == "__main__":
    main()
