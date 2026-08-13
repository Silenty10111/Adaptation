#!/usr/bin/env python3
"""Helper: generate a trajectory overview plot from JSON data.

Called by test_amputated_hexapod.py as a subprocess in the Adaptation conda env.
Usage: python _plot_trajectory.py <data.json>
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    data_path = Path(sys.argv[1])
    data = json.loads(data_path.read_text(encoding="utf-8"))

    robot_name = data["robot_name"]
    description = data["description"]
    ssm_result = data["ssm_result"]
    com_trail = data["com_trail"]
    forward_axis = data["forward_axis"]
    fwd_dist = data["fwd_dist"]
    lat_dist = data["lat_dist"]
    out_path = Path(data["out_path"])

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: robot footprint + SSM ───────────────────────────────────────
    ax = axes[0]
    ax.set_aspect("equal")
    ax.set_title(f"{robot_name}\nSSM = {ssm_result.get('ssm', 0):.4f} m", fontsize=10)

    # Draw trunk polygon
    trunk = description.get("trunk_polygon_xy", [])
    if len(trunk) >= 3:
        tx = [p[0] for p in trunk] + [trunk[0][0]]
        ty = [p[1] for p in trunk] + [trunk[0][1]]
        ax.fill(tx, ty, alpha=0.2, color="gray", label="trunk")

    # Draw foot positions
    foot_pts = {}
    hip_pts = {}
    for lk in description.get("links", []):
        lid = lk.get("leg_id")
        if lid is None:
            continue
        origin = lk.get("default_world_origin", [0, 0, 0])
        role = lk.get("role", "")
        if role == "foot":
            foot_pts[lid] = np.array(origin[:2])
        elif role == "joint_sphere" and "hip" in lk.get("name", ""):
            hip_pts[lid] = np.array(origin[:2])

    for lid, pt in foot_pts.items():
        ax.scatter(pt[0], pt[1], c="red", s=60, zorder=5, marker="s")
        ax.text(pt[0] + 0.02, pt[1] + 0.02, str(lid), fontsize=8, color="red")
    for lid, pt in hip_pts.items():
        ax.scatter(pt[0], pt[1], c="blue", s=30, zorder=4, marker="o")

    # support polygon
    sp = ssm_result.get("support_polygon_xy", [])
    if len(sp) >= 3:
        sx = [p[0] for p in sp] + [sp[0][0]]
        sy = [p[1] for p in sp] + [sp[0][1]]
        ax.fill(sx, sy, alpha=0.3, color="green", label="support polygon")

    # CoM
    com_xy = ssm_result.get("com_xy", [0.0, 0.0])
    ax.scatter(com_xy[0], com_xy[1], c="cyan", s=100, zorder=6, marker="x", linewidths=2)

    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Right: trajectory ─────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_aspect("equal")
    ax2.set_title(f"Trajectory  fwd={fwd_dist:+.3f}m  lat={lat_dist:+.3f}m", fontsize=10)
    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)")

    if len(com_trail) > 1:
        trail = np.array(com_trail)
        ax2.plot(trail[:, 0], trail[:, 1], "b-", linewidth=0.8, alpha=0.7, label="CoM trail")
        ax2.scatter(trail[0, 0], trail[0, 1], c="green", s=50, marker="o", label="start")
        ax2.scatter(trail[-1, 0], trail[-1, 1], c="red", s=50, marker="s", label="end")
        # forward axis arrow
        fwd_arr = np.array(forward_axis[:2])
        fwd_arr = fwd_arr / max(np.linalg.norm(fwd_arr), 1e-9)
        mid = (trail[0] + trail[-1]) / 2.0
        ax2.arrow(mid[0], mid[1], fwd_arr[0] * 0.3, fwd_arr[1] * 0.3,
                  head_width=0.04, head_length=0.06, fc="orange", ec="orange", label="fwd axis")

    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
