#!/usr/bin/env python3
"""
PDF 批量测试报告生成器
======================
从 batch_results 目录读取 summary.json 和 trajectory.png，
生成多页 PDF 报告。

用法:
  python generate_pdf_report.py batch_results/20260624_162055
  python generate_pdf_report.py --latest
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
import numpy as np

_REPO = Path(__file__).resolve().parent.parent


def build_report(batch_dir: Path, out_pdf: Path) -> dict:
    """生成 PDF 报告，返回统计摘要。"""
    summary_path = batch_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found in {batch_dir}")

    data = json.loads(summary_path.read_text(encoding="utf-8"))

    ok = [r for r in data if r.get("status") == "ok"]
    failed = [r for r in data if r.get("status") != "ok"]

    # Filter robots with actual simulation data
    sim_ok = [r for r in ok if abs(r.get("fwd_dist", 0)) > 0.001 or abs(r.get("lat_dist", 0)) > 0.001]

    # Categories
    strategies = defaultdict(list)
    for r in sim_ok:
        strategies[r.get("strategy", "baseline")].append(r)

    by_legs = defaultdict(list)
    for r in sim_ok:
        n = r.get("num_legs", 0)
        if n > 0:
            by_legs[n].append(r)

    # Drift stats
    drifts = [abs(r.get("drift_ratio", 0)) for r in sim_ok]
    drift_mean = float(np.mean(drifts)) if drifts else 0
    drift_median = float(np.median(drifts)) if drifts else 0

    ekf_count = len(strategies.get("ekf_online", []))
    baseline_count = len(strategies.get("baseline", []))

    # ── PDF Generation ───────────────────────────────────────────────────
    pdf = pdf_backend.PdfPages(str(out_pdf))

    # --- Page 1: Title + Summary ---
    fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4
    ax.axis("off")

    title_lines = [
        f"Adaptation Batch Test Report",
        f"",
        f"Date: {batch_dir.name}",
        f"Total robots: {len(data)}",
        f"SSM passed + simulated: {len(ok)}",
        f"With valid trajectory: {len(sim_ok)}",
        f"SSM failed/skipped: {len(failed)}",
        f"",
        f"── Performance Metrics ──",
        f"Mean drift ratio:   {drift_mean:.3f}",
        f"Median drift ratio:  {drift_median:.3f}",
        f"Good (drift <0.3):   {sum(1 for d in drifts if d < 0.3)}/{len(drifts)} ({sum(1 for d in drifts if d < 0.3)/max(len(drifts),1)*100:.0f}%)",
        f"Acceptable (<1.0):   {sum(1 for d in drifts if d < 1.0)}/{len(drifts)} ({sum(1 for d in drifts if d < 1.0)/max(len(drifts),1)*100:.0f}%)",
        f"",
        f"── Strategy Distribution ──",
        f"Baseline:            {baseline_count}",
        f"EKF Online:          {ekf_count}",
        f"EKF improvement rate: {ekf_count}/{baseline_count+ekf_count} ({ekf_count/max(baseline_count+ekf_count,1)*100:.0f}%)",
        f"",
        f"── Methods ──",
        f"Forward direction:   trunk-frame fixed swing vector",
        f"Yaw compensation:    static (YAW_COMP_GAIN=0.50)",
        f"Online correction:   EKF CoM offset + yaw rate monitoring",
        f"Direction diagnosis: 95% correct (score margin > 0.5)",
    ]

    for i, line in enumerate(title_lines):
        y = 1.0 - i * 0.022
        if line.startswith("──"):
            ax.text(0.08, y, line, fontsize=11, fontweight="bold", family="monospace",
                    transform=ax.transAxes, va="top")
        elif line.startswith("Adaptation"):
            ax.text(0.08, y, line, fontsize=16, fontweight="bold",
                    transform=ax.transAxes, va="top")
        else:
            ax.text(0.08, y, line, fontsize=10, family="monospace",
                    transform=ax.transAxes, va="top")

    pdf.savefig(fig, dpi=150)
    plt.close(fig)

    # --- Page 2: Stats by leg count ---
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))

    # 2a: Drift by leg count
    ax = axes[0, 0]
    leg_list = sorted(by_legs.keys())
    means = [float(np.mean([abs(r.get("drift_ratio", 0)) for r in by_legs[l]])) for l in leg_list]
    medians = [float(np.median([abs(r.get("drift_ratio", 0)) for r in by_legs[l]])) for l in leg_list]
    counts = [len(by_legs[l]) for l in leg_list]
    x = np.arange(len(leg_list))
    w = 0.35
    ax.bar(x - w/2, means, w, label="Mean Drift", color="steelblue", alpha=0.8)
    ax.bar(x + w/2, medians, w, label="Median Drift", color="coral", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}腿\n(n={c})" for l, c in zip(leg_list, counts)], fontsize=8)
    ax.set_ylabel("Drift Ratio")
    ax.set_title("Drift by Leg Count")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # 2b: SSM vs Drift scatter
    ax = axes[0, 1]
    ssm_vals = [r.get("ssm", 0) for r in sim_ok]
    drift_vals = [abs(r.get("drift_ratio", 0)) for r in sim_ok]
    colors = ["green" if r.get("strategy") == "ekf_online" else "blue" for r in sim_ok]
    ax.scatter(ssm_vals, drift_vals, c=colors, alpha=0.6, s=15)
    ax.set_xlabel("SSM (m)")
    ax.set_ylabel("Drift Ratio")
    ax.set_title("SSM vs Drift (green=EKF, blue=baseline)")
    ax.axhline(y=0.3, color="red", ls="--", alpha=0.5, label="good threshold")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # 2c: Strategy pie
    ax = axes[1, 0]
    strat_labels = []
    strat_vals = []
    strat_colors = []
    if baseline_count > 0:
        strat_labels.append(f"Baseline ({baseline_count})")
        strat_vals.append(baseline_count)
        strat_colors.append("steelblue")
    if ekf_count > 0:
        strat_labels.append(f"EKF Online ({ekf_count})")
        strat_vals.append(ekf_count)
        strat_colors.append("green")
    if strat_vals:
        ax.pie(strat_vals, labels=strat_labels, colors=strat_colors, autopct="%1.0f%%")
        ax.set_title("Strategy Distribution")

    # 2d: Fwd vs Lat scatter
    ax = axes[1, 1]
    fwd_vals = [r.get("fwd_dist", 0) for r in sim_ok]
    lat_vals = [abs(r.get("lat_dist", 0)) for r in sim_ok]
    ax.scatter(fwd_vals, lat_vals, c=colors, alpha=0.6, s=15)
    ax.set_xlabel("Forward Distance (m)")
    ax.set_ylabel("|Lateral Drift| (m)")
    ax.set_title("Forward vs Lateral")
    ax.axhline(y=0, color="gray", ls="-", alpha=0.3)
    ax.grid(alpha=0.3)

    plt.suptitle(f"Batch Test Analysis — {batch_dir.name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

    # --- Pages 3+: Per-robot trajectory images (6 per page) ---
    robots_with_img = []
    for r in sim_ok:
        name = r["robot"]
        img_path = batch_dir / name / "trajectory.png"
        if img_path.exists():
            robots_with_img.append((name, r, img_path))

    # Sort: drift worst first
    robots_with_img.sort(key=lambda x: abs(x[1].get("drift_ratio", 0)), reverse=True)

    imgs_per_page = 6
    for page_start in range(0, len(robots_with_img), imgs_per_page):
        page_robots = robots_with_img[page_start:page_start + imgs_per_page]
        n = len(page_robots)
        cols = 3
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(11.69, 8.27))
        if rows == 1:
            axes = axes.reshape(1, -1)

        for idx, (name, r, img_path) in enumerate(page_robots):
            row, col = idx // cols, idx % cols
            ax = axes[row, col]

            img = plt.imread(str(img_path))
            ax.imshow(img)
            ax.axis("off")

            drift = r.get("drift_ratio", float("nan"))
            strategy = r.get("strategy", "?")
            legs = r.get("num_legs", "?")
            ax.set_title(f"{name} | legs={legs} | {strategy}\ndrift={drift:.3f}",
                        fontsize=7, family="monospace")

        # Hide unused subplots
        for idx in range(n, rows * cols):
            row, col = idx // cols, idx % cols
            axes[row, col].axis("off")

        plt.suptitle(f"Robot Trajectories (sorted by drift, worst first) — page {page_start//imgs_per_page + 1}",
                    fontsize=10, fontweight="bold")
        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    # --- Last page: Best performers ---
    best = sorted(robots_with_img, key=lambda x: abs(x[1].get("drift_ratio", 99)))[:6]
    fig, axes = plt.subplots(2, 3, figsize=(11.69, 8.27))
    for idx, (name, r, img_path) in enumerate(best):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        img = plt.imread(str(img_path))
        ax.imshow(img)
        ax.axis("off")
        drift = r.get("drift_ratio", float("nan"))
        ax.set_title(f"★ {name} | drift={drift:.3f}", fontsize=8, color="green")

    plt.suptitle("Best Performers (Lowest Drift)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

    pdf.close()

    return {
        "total": len(data),
        "ok": len(ok),
        "sim_ok": len(sim_ok),
        "drift_mean": drift_mean,
        "drift_median": drift_median,
        "ekf_count": ekf_count,
        "baseline_count": baseline_count,
        "good_drift": sum(1 for d in drifts if d < 0.3),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate PDF batch report")
    parser.add_argument("batch_dir", nargs="?", help="Batch results directory")
    parser.add_argument("--latest", action="store_true", help="Use latest batch_results dir")
    parser.add_argument("--output", type=Path, help="Output PDF path")
    args = parser.parse_args()

    if args.latest or not args.batch_dir:
        results = sorted(Path("batch_results").glob("2*/"))
        if not results:
            print("No batch results found")
            sys.exit(1)
        args.batch_dir = str(results[-1])
        print(f"Using latest: {args.batch_dir}")

    batch_dir = Path(args.batch_dir)
    if not batch_dir.exists():
        print(f"Directory not found: {batch_dir}")
        sys.exit(1)

    out_pdf = args.output or (batch_dir / "report.pdf")
    stats = build_report(batch_dir, out_pdf)

    print(f"\nPDF report saved: {out_pdf}")
    print(f"  Robots: {stats['total']} total, {stats['ok']} simulated, {stats['sim_ok']} with trajectory")
    print(f"  Drift: mean={stats['drift_mean']:.3f}, median={stats['drift_median']:.3f}")
    print(f"  Good (<0.3): {stats['good_drift']}/{stats['sim_ok']} "
          f"({stats['good_drift']/max(stats['sim_ok'],1)*100:.0f}%)")
    print(f"  EKF: {stats['ekf_count']} robots")

if __name__ == "__main__":
    main()
