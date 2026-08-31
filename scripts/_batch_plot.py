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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from adaptation.gait_visualization import build_planned_gait_raster


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


def _time_extent(time_s):
    if len(time_s) <= 1:
        return 0.0, 1.0
    return float(time_s[0]), float(time_s[-1] + (time_s[1] - time_s[0]))


def render_gait_visuals(data, trajectory_path):
    """Render one planned-target heatmap with separate left/right panels."""
    gait_plan = data.get("gait_plan")
    if not isinstance(gait_plan, dict):
        return []
    trail_count = len(data.get("com_trail", []))
    duration_s = float(data.get("gait_duration_s", max(trail_count / 60.0, 2.0)))
    raster = build_planned_gait_raster(
        data["description"], gait_plan, duration_s=duration_s,
    )
    raster["plan_source"] = data.get("gait_plan_source", "executed_final_plan")
    raster["robot_name"] = data["robot_name"]
    metadata_path = trajectory_path.with_name("gait_visualization.json")
    metadata_path.write_text(json.dumps(raster, ensure_ascii=False), encoding="utf-8")

    leg_ids = list(raster["leg_ids"])
    if not leg_ids:
        return []
    labels = [
        f"Leg {leg_id} ({group})"
        for leg_id, group in zip(leg_ids, raster["group_labels"])
    ]
    time_s = np.asarray(raster["time_s"], dtype=float)
    x0, x1 = _time_extent(time_s)
    strategy = raster["selected_phase_strategy"]
    source = raster["plan_source"]

    heatmap_path = trajectory_path.with_name("gait_heatmap.png")
    values = np.asarray(raster["swing_joint_target_delta_rad"], dtype=float)
    side_labels = list(raster["side_labels"])
    side_rows = {
        side: [index for index, value in enumerate(side_labels) if value == side]
        for side in ("left", "right")
    }
    nonempty_sides = [side for side in ("left", "right") if side_rows[side]]
    fig, axes = plt.subplots(
        max(len(nonempty_sides), 1), 1,
        figsize=(13, max(5.5, 0.62 * len(leg_ids) + 2.5)), sharex=True,
        squeeze=False, constrained_layout=True,
    )
    limit = max(float(np.max(np.abs(values))), 1e-6)
    heatmap_image = None
    for plot_index, side in enumerate(nonempty_sides):
        ax = axes[plot_index, 0]
        rows = side_rows[side]
        side_values = values[rows]
        side_labels_for_ticks = [labels[index] for index in rows]
        side_extent = [x0, x1, len(rows) - 0.5, -0.5]
        heatmap_image = ax.imshow(
            side_values, aspect="auto", interpolation="nearest", origin="upper",
            extent=side_extent, cmap="coolwarm", vmin=-limit, vmax=limit,
        )
        ax.set_yticks(range(len(rows)), labels=side_labels_for_ticks, fontsize=8)
        ax.set_ylabel("Leg")
        ax.set_title(f"{side.capitalize()} legs", fontsize=10)
    if heatmap_image is not None:
        fig.colorbar(
            heatmap_image, ax=axes[:len(nonempty_sides), 0].tolist(), pad=0.012,
            label="Planned swing-joint deviation from neutral (rad)",
        )
    axes[len(nonempty_sides) - 1, 0].set_xlabel("Simulation time (s)")
    fig.suptitle(
        f"Left/right planned leg-angle heatmap — {data['robot_name']}\n"
        f"strategy={strategy}; source={source}; not measured DOF states",
        fontsize=12,
    )
    fig.savefig(heatmap_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return [heatmap_path, metadata_path]


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
    for generated_path in render_gait_visuals(d, out_path):
        print(f"[Plot] {generated_path}")


if __name__ == "__main__":
    main()
