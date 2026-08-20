#!/usr/bin/env python3
"""Internal helper: render one trajectory demo image.
Called by batch_test.py via subprocess in the Adaptation conda env.
Usage: python _batch_plot.py <data_json_path>
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def project_to_path_frame(com_trail, forward_axis):
    """Project world XY samples into (lateral, commanded-forward) coordinates."""
    trail = np.asarray(com_trail, dtype=float)
    if trail.ndim != 2 or len(trail) == 0 or trail.shape[1] < 2:
        trail = np.zeros((2, 2), dtype=float)
    fwd = np.asarray(forward_axis, dtype=float)[:2]
    norm = float(np.linalg.norm(fwd))
    if norm < 1e-9:
        fwd = np.array([1.0, 0.0], dtype=float)
    else:
        fwd = fwd / norm
    lat = np.array([-fwd[1], fwd[0]], dtype=float)
    delta = trail[:, :2] - trail[0, :2]
    return delta @ lat, delta @ fwd, fwd, lat


def main():
    data_path = Path(sys.argv[1])
    d = json.loads(data_path.read_text(encoding="utf-8"))

    robot_name   = d["robot_name"]
    description  = d["description"]
    ssm_result   = d["ssm_result"]
    com_trail    = d["com_trail"]
    forward_axis = d["forward_axis"]
    fwd_dist     = d["fwd_dist"]
    lat_dist     = d["lat_dist"]
    metrics      = d.get("metrics", {})
    out_path     = Path(d["out_path"])

    trail = np.array(com_trail, dtype=float) if len(com_trail) > 1 else np.zeros((2, 2))
    lat_vals, fwd_vals, fwd, lat = project_to_path_frame(trail, forward_axis)

    # Rotation matrix: world → display frame (forward = +Y, lateral = +X)
    # R @ [world_x, world_y] = [lat_component, fwd_component]
    R = np.array([[lat[0], lat[1]], [fwd[0], fwd[1]]], dtype=float)

    def _rot(pts: np.ndarray) -> np.ndarray:
        """World-XY → display-XY (lat, fwd)."""
        pts = np.asarray(pts, dtype=float)
        if pts.ndim == 1:
            return R @ pts
        return (R @ pts.T).T

    total_fwd = float(fwd_vals[-1]) if len(fwd_vals) > 0 else 0.0

    # Trunk polygon — rotate to display frame
    trunk_poly_raw = np.array(description.get("trunk_polygon_xy", []), dtype=float)
    trunk_poly = _rot(trunk_poly_raw) if len(trunk_poly_raw) >= 3 else trunk_poly_raw

    # Leg geometry
    foot_links = {int(lk["leg_id"]): lk for lk in description.get("links", [])
                  if lk.get("role") == "foot" and lk.get("leg_id") is not None}
    hip_links  = {int(lk["leg_id"]): lk for lk in description.get("links", [])
                  if lk.get("role") == "joint_sphere"
                  and str(lk.get("name", "")).endswith("_hip")
                  and lk.get("leg_id") is not None}

    # ── figure ──
    fig, ax = plt.subplots(figsize=(5, 10))

    # 1. Robot outline (at origin, rotated so forward = +Y)
    if len(trunk_poly) >= 3:
        poly_closed = np.vstack([trunk_poly, trunk_poly[0]])
        ax.fill(poly_closed[:, 0], poly_closed[:, 1],
                color="lightgray", alpha=0.8, zorder=2)
        ax.plot(poly_closed[:, 0], poly_closed[:, 1],
                "k-", lw=1.0, zorder=3)

    # 2. Legs (rotated to display frame)
    for lid, foot_lk in foot_links.items():
        foot_pos = _rot(np.asarray(foot_lk["default_world_origin"], dtype=float)[:2])
        hip_lk   = hip_links.get(lid)
        hip_pos  = (_rot(np.asarray(hip_lk["default_world_origin"], dtype=float)[:2])
                    if hip_lk is not None else np.zeros(2))
        ax.plot([hip_pos[0], foot_pos[0]], [hip_pos[1], foot_pos[1]],
                color="dimgray", lw=1.5, zorder=3)
        ax.plot(*foot_pos, "ko", ms=4, zorder=4)

    # 2b. Forward direction arrow on robot body
    arrow_len = 0.18
    ax.annotate("", xy=(0.0, arrow_len), xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle="->", color="blue", lw=1.5), zorder=6)

    # 3. Initial base marker.  Isaac Gym rigid-body state index 0 is the root
    # body, not the mass-weighted whole-robot centre of mass.
    ax.plot(0.0, 0.0, "go", ms=5, zorder=8, label="Start")

    # 4. Commanded path (dashed blue, straight line forward = +Y)
    cmd_len = max(float(np.max(fwd_vals)) * 1.05, total_fwd * 1.05, 0.3)
    guide_y = np.linspace(0.0, cmd_len, 100)
    ax.fill_betweenx(guide_y, -0.25 * guide_y, 0.25 * guide_y,
                     color="royalblue", alpha=0.06, zorder=0,
                     label="Drift-ratio 0.25 guide")
    ax.plot([0.0, 0.0], [0.0, cmd_len], "b--", lw=1.5,
            label="Commanded Path", zorder=6)

    # 5. Actual root-body trajectory
    if len(lat_vals) > 1:
        ax.plot(lat_vals, fwd_vals, color="darkorange", lw=1.8,
                label="Actual Base Trajectory", zorder=7)
        ax.plot(lat_vals[-1], fwd_vals[-1], marker="X", color="crimson",
                ms=7, zorder=9, label="End")

    ssm_str = f"SSM={ssm_result['ssm']:.3f}m ({'✓' if ssm_result['passed'] else '✗'})"
    motion_str = ""
    if metrics:
        motion_str = (
            f" | locomotion={'PASS' if metrics.get('passed') else 'FAIL'}"
            f" | v={float(metrics.get('forward_speed', 0.0)):.3f}m/s"
            f" | CV={float(metrics.get('speed_cv', 0.0)):.3f}"
        )
    ax.set_title(
        f"Experimental Log:\nDynamic Base Trajectory\n"
        f"{robot_name} | {ssm_str}{motion_str}\n"
        f"fwd={fwd_dist:+.3f}m  lat={lat_dist:+.3f}m",
        fontsize=9,
    )
    ax.set_xlabel("Lateral Deviation (m)")
    ax.set_ylabel("Forward Distance (m)")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_aspect("equal")
    ax.grid(True, ls=":", alpha=0.5)
    ax.axvline(0, color="gray", lw=0.5, ls="--")

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {out_path}")


if __name__ == "__main__":
    main()
