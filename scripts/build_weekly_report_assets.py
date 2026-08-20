#!/usr/bin/env python3
"""Build reproducible figures used by the 2026-08-18 weekly report."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Polygon


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "weekly_report_20260818" / "assets"

REPORT_FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
font_manager.fontManager.addfont(str(REPORT_FONT_PATH))
REPORT_FONT = font_manager.FontProperties(fname=REPORT_FONT_PATH).get_name()

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [REPORT_FONT, "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 180,
})


def save(fig, name: str) -> None:
    fig.savefig(OUTPUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def convex_hull(points: np.ndarray) -> np.ndarray:
    pts = sorted(set(map(tuple, points.tolist())))
    if len(pts) <= 2:
        return np.asarray(pts, dtype=float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def load_description(relative: str) -> dict:
    return json.loads((REPO_ROOT / relative).read_text(encoding="utf-8"))


def foot_positions(description: dict) -> np.ndarray:
    rows = [
        link["default_world_origin"][:2]
        for link in description.get("links", [])
        if link.get("role") == "foot"
    ]
    return np.asarray(rows, dtype=float)


def morphology_figure() -> None:
    cases = [
        ("标准六足", "robot_assets/standard_hexapod/robot_description.json"),
        ("缺失腿 0（5 腿）", "validation_results/dev_amputation_fixed/missing_leg_0/robot_description.json"),
        ("缺失腿 2、3（4 腿）", "validation_results/dev_amputation_fixed/missing_legs_23/robot_description.json"),
        ("缺失腿 3、5（4 腿）", "validation_results/dev_amputation_fixed/missing_legs_35/robot_description.json"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharex=True, sharey=True)
    for ax, (title, path) in zip(axes, cases):
        desc = load_description(path)
        trunk = np.asarray(desc["trunk_polygon_xy"], dtype=float)
        feet = foot_positions(desc)
        hull = convex_hull(feet)
        ax.add_patch(Polygon(trunk, closed=True, facecolor="#d1d5db", edgecolor="#374151", lw=1.4))
        if len(hull) >= 3:
            ax.add_patch(Polygon(hull, closed=True, facecolor="#86efac", edgecolor="#16a34a", alpha=0.42, lw=1.4))
        ax.scatter(feet[:, 0], feet[:, 1], c="#dc2626", s=35, marker="s", zorder=3)
        ax.scatter([0], [0], c="#0891b2", s=55, marker="x", linewidths=2.2, zorder=4)
        for idx, point in enumerate(feet):
            ax.text(point[0] + 0.018, point[1] + 0.018, str(idx), fontsize=8, color="#991b1b")
        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_aspect("equal")
        ax.grid(alpha=0.22)
        ax.set_xlim(-0.75, 0.75)
        ax.set_ylim(-0.68, 0.68)
        ax.set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    fig.suptitle("标准六足及缺腿构型的足端支撑多边形", fontsize=15, weight="bold")
    fig.tight_layout()
    save(fig, "amputation_morphologies.png")


def validation_overview() -> None:
    labels = ["标准六足", "单腿缺失", "双腿缺失\n代表集", "定向生成\n构型", "500 台随机\n批测"]
    passed = np.asarray([1, 6, 11, 5, 223])
    total = np.asarray([1, 6, 11, 5, 500])
    rates = passed / total * 100.0
    colors = ["#2563eb", "#0891b2", "#0d9488", "#16a34a", "#f59e0b"]
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    bars = ax.bar(labels, rates, color=colors, width=0.62)
    ax.set_ylim(0, 112)
    ax.set_ylabel("严格标准通过率 (%)")
    ax.set_title("定向回归验证与大规模泛化验证结果", fontsize=15, weight="bold")
    ax.grid(axis="y", alpha=0.24)
    for bar, p, t, rate in zip(bars, passed, total, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 2.2,
                f"{p}/{t}\n{rate:.1f}%", ha="center", va="bottom", fontsize=10, weight="bold")
    ax.text(4, 12, "随机批测更能反映当前泛化水平", ha="center", color="#92400e", fontsize=10)
    fig.tight_layout()
    save(fig, "validation_overview.png")


def batch_by_leg() -> None:
    legs = np.asarray([4, 5, 6, 7, 8, 9, 10])
    totals = np.asarray([26, 54, 101, 125, 101, 63, 30])
    dynamic = np.asarray([12, 45, 94, 121, 94, 59, 29])
    passed = np.asarray([5, 9, 40, 63, 48, 38, 20])
    rates = passed / totals * 100.0
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    width = 0.26
    ax.bar(legs - width, totals, width, label="样本总数", color="#cbd5e1")
    ax.bar(legs, dynamic, width, label="进入动态仿真", color="#60a5fa")
    ax.bar(legs + width, passed, width, label="严格 PASS", color="#22c55e")
    ax.set_xlabel("腿数")
    ax.set_ylabel("机器人数量")
    ax.set_xticks(legs)
    ax.grid(axis="y", alpha=0.22)
    ax.set_title("500 台随机机器人按腿数统计", fontsize=15, weight="bold")
    ax.legend(loc="upper left")
    rate_ax = ax.twinx()
    rate_ax.plot(legs, rates, color="#dc2626", marker="o", linewidth=2.2, label="总体通过率")
    rate_ax.set_ylabel("总体通过率 (%)", color="#b91c1c")
    rate_ax.set_ylim(0, 80)
    rate_ax.tick_params(axis="y", labelcolor="#b91c1c")
    for x, rate in zip(legs, rates):
        rate_ax.text(x, rate + 3, f"{rate:.1f}%", ha="center", color="#991b1b", fontsize=9)
    fig.tight_layout()
    save(fig, "batch_by_leg.png")


def failure_reasons() -> None:
    labels = ["行进方向误差", "漂移比", "前向速度", "侧向速度", "平均机身高度", "周期速度 CV", "高度波动"]
    counts = np.asarray([200, 179, 127, 50, 18, 12, 1])
    order = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    bars = ax.barh(order, counts[::-1], color=["#94a3b8", "#94a3b8", "#60a5fa", "#60a5fa", "#f59e0b", "#f97316", "#ef4444"])
    ax.set_yticks(order, labels[::-1])
    ax.set_xlabel("未通过次数（同一机器人可违反多项指标）")
    ax.set_title("231 台动态未通过机器人的指标频次", fontsize=15, weight="bold")
    ax.grid(axis="x", alpha=0.22)
    for bar, value in zip(bars, counts[::-1]):
        ax.text(value + 3, bar.get_y() + bar.get_height() / 2, str(value), va="center", fontsize=10)
    ax.set_xlim(0, 225)
    fig.tight_layout()
    save(fig, "failure_reasons.png")


def copy_trajectories() -> None:
    source = REPO_ROOT / "batch_results" / "20260818_131804" / "png"
    for source_name, target_name in [
        ("robot_ref_standard.png", "standard_trajectory.png"),
        ("robot_00_seed7.png", "random_pass_trajectory.png"),
        ("robot_01_seed42.png", "random_fail_trajectory.png"),
    ]:
        shutil.copy2(source / source_name, OUTPUT / target_name)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    morphology_figure()
    validation_overview()
    batch_by_leg()
    failure_reasons()
    copy_trajectories()
    print(f"Generated report assets in {OUTPUT}")


if __name__ == "__main__":
    main()
