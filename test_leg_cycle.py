#!/usr/bin/env python3
"""Generate a PDF report of one complete gait cycle for every leg.

Runs in the Adaptation environment (Python ≥ 3.10).  Computes joint-position
targets over one full period (1 / gait_frequency) and renders:

  1. Lift joint targets  — all legs
  2. Swing joint targets  — all legs
  3. Drop joint targets   — all legs
  4. Phase-state stacked bars (stance / swing per leg)
  5. Summary statistics

Output:  png/leg_cycle_report.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
ASSET_DIR = REPO_ROOT / "robot_assets"
OUTPUT_DIR = REPO_ROOT / "png"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Math helpers (pure numpy, no gymapi)
# ---------------------------------------------------------------------------

def smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = max(0.0, min(1.0, (x - edge0) / max(edge1 - edge0, 1e-9)))
    return t * t * (3.0 - 2.0 * t)


def ratio_to_joint(lower: float, upper: float, ratio: float) -> float:
    return float(lower + max(0.0, min(1.0, ratio)) * (upper - lower))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_description(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_leg_limits(description: dict) -> Dict[int, dict]:
    """Extract per-leg lift / swing / drop joint limits from the description."""
    joints_by_leg: Dict[int, dict] = {}
    for joint in description.get("joints", []):
        name = str(joint.get("name", ""))
        for prefix in ("lift", "swing", "drop"):
            if name.endswith(f"_{prefix}"):
                parts = name.split("_")
                try:
                    leg_id = int(parts[1])
                except (IndexError, ValueError):
                    continue
                limit = joint.get("limit", {})
                joints_by_leg.setdefault(leg_id, {})[prefix] = {
                    "lower": float(limit.get("lower", 0.0)),
                    "upper": float(limit.get("upper", 0.0)),
                }
    return joints_by_leg


def compute_gait_plan(description: dict) -> dict:
    """Obtain the adaptive gait plan (grouping, forward axis, etc.)."""
    try:
        from adaptive_gait import compute_adaptive_plan
        return compute_adaptive_plan(description, {})
    except (ModuleNotFoundError, ImportError):
        num_legs = int(description.get("num_legs", 0))
        ids = list(range(num_legs))
        return {
            "final_forward_axis": [1.0, 0.0],
            "topology": {
                "groups": {"group_a": ids[::2], "group_b": ids[1::2]},
                "phase_offsets": {"group_a": 0.0, "group_b": float(np.pi)},
            },
        }


# ---------------------------------------------------------------------------
# Target computation (replicates build_gait_targets without Isaac Gym)
# ---------------------------------------------------------------------------

def leg_group_phase(leg_id: int, group_a: List[int], group_b: List[int],
                    base_phase: float) -> float:
    if leg_id in group_b:
        return base_phase + float(np.pi)
    if leg_id in group_a:
        return base_phase
    return base_phase


def compute_cycle_targets(
    description: dict,
    gait_plan: dict,
    leg_limits: Dict[int, dict],
    gait_freq: float = 0.85,
    swing_amp: float = 0.26,
    stance_lift: float = 0.05,
    swing_lift: float = 0.78,
    stance_drop: float = 0.90,
    swing_drop: float = 0.38,
    num_samples: int = 120,
) -> dict:
    """Compute joint targets over one full gait cycle.

    Returns
    -------
    dict with keys:
        t             : (num_samples,)  time in seconds
        lift_targets  : (num_legs, num_samples)
        swing_targets : (num_legs, num_samples)
        drop_targets  : (num_legs, num_samples)
        phase_state   : (num_legs, num_samples)  1.0 = swing, 0.0 = stance
        leg_ids       : sorted leg IDs
        group_a, group_b : list[int]
        forward_axis  : [fx, fy]
    """
    period = 1.0 / max(gait_freq, 0.02)
    t = np.linspace(0.0, period, num_samples, endpoint=True)
    leg_ids = sorted(leg_limits.keys())
    num_legs = len(leg_ids)

    group_a = list(gait_plan.get("topology", {}).get("groups", {}).get("group_a", []))
    group_b = list(gait_plan.get("topology", {}).get("groups", {}).get("group_b", []))
    forward_axis = np.asarray(
        gait_plan.get("final_forward_axis", [1.0, 0.0]), dtype=float,
    )
    forward_axis = forward_axis / max(float(np.linalg.norm(forward_axis)), 1e-9)

    # Foot positions for direction sign
    foot_xy: Dict[int, np.ndarray] = {}
    for link in description.get("links", []):
        if link.get("role") == "foot" and link.get("leg_id") is not None:
            origin = np.asarray(
                link.get("default_world_origin", [0.0, 0.0, 0.0]), dtype=float,
            )
            foot_xy[int(link["leg_id"])] = origin[:2]

    lift_targets = np.zeros((num_legs, num_samples), dtype=float)
    swing_targets = np.zeros((num_legs, num_samples), dtype=float)
    drop_targets = np.zeros((num_legs, num_samples), dtype=float)
    phase_state = np.zeros((num_legs, num_samples), dtype=float)

    for row, leg_id in enumerate(leg_ids):
        limits = leg_limits[leg_id]
        ft = foot_xy.get(leg_id, np.zeros(2, dtype=float))
        dir_sign = 1.0 if float(np.dot(ft, forward_axis)) >= 0.0 else -1.0

        for col, time_val in enumerate(t):
            phase = 2.0 * np.pi * gait_freq * time_val
            lg_ph = leg_group_phase(leg_id, group_a, group_b, phase)
            swing_wave = float(np.sin(lg_ph))
            swing_alpha = smoothstep(-0.05, 0.05, swing_wave)

            lift_r = stance_lift + (swing_lift - stance_lift) * swing_alpha
            drop_r = stance_drop + (swing_drop - stance_drop) * swing_alpha
            swing_r = 0.5 + swing_amp * dir_sign * swing_wave

            lift_targets[row, col] = ratio_to_joint(
                limits["lift"]["lower"], limits["lift"]["upper"], lift_r,
            )
            swing_targets[row, col] = ratio_to_joint(
                limits["swing"]["lower"], limits["swing"]["upper"], swing_r,
            )
            drop_targets[row, col] = ratio_to_joint(
                limits["drop"]["lower"], limits["drop"]["upper"], drop_r,
            )
            phase_state[row, col] = float(swing_wave > 0.0)

    return {
        "t": t,
        "lift_targets": lift_targets,
        "swing_targets": swing_targets,
        "drop_targets": drop_targets,
        "phase_state": phase_state,
        "leg_ids": leg_ids,
        "group_a": group_a,
        "group_b": group_b,
        "forward_axis": forward_axis.tolist(),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_joint_panel(ax, t: np.ndarray, data: np.ndarray, leg_ids: List[int],
                     phase: np.ndarray, ylabel: str, title: str,
                     group_a: List[int], group_b: List[int]) -> None:
    """Draw one subplot: joint targets over one cycle with phase background."""
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(leg_ids), 1)))
    period = t[-1]

    for row, leg_id in enumerate(leg_ids):
        # Phase background (swing = grey)
        for col in range(len(t) - 1):
            if phase[row, col] > 0.5:
                ax.axvspan(t[col], t[col + 1], alpha=0.08, color="grey", lw=0)

        label = f"leg_{leg_id}"
        ax.plot(t, data[row], color=colors[row % len(colors)],
                linewidth=1.5, label=label)

    # Style
    ax.set_xlim(0, period)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.8)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(period / 4))


def plot_phase_gantt(ax, t: np.ndarray, phase: np.ndarray, leg_ids: List[int],
                     group_a: List[int], group_b: List[int]) -> None:
    """Gantt-style stacked bars showing stance (green) / swing (grey) per leg."""
    num_legs = len(leg_ids)
    period = t[-1]
    bar_height = 0.65

    for row, leg_id in enumerate(leg_ids):
        # Find stance ↔ swing transitions
        in_swing = False
        seg_start = 0.0
        for col in range(len(t)):
            currently_swing = phase[row, col] > 0.5
            if currently_swing != in_swing or col == len(t) - 1:
                seg_end = t[col]
                duration = seg_end - seg_start
                if duration > 0:
                    color = "#aaaaaa" if in_swing else "#4caf50"
                    ax.barh(row, duration, bar_height, left=seg_start,
                            color=color, edgecolor="none", alpha=0.85)
                seg_start = t[col]
                in_swing = currently_swing

    ax.set_xlim(0, period)
    ax.set_yticks(range(num_legs))
    ax.set_yticklabels([f"leg_{lid}" for lid in leg_ids])
    ax.set_xlabel("Time (s)")
    ax.set_title("Phase state (green = stance, grey = swing)")
    ax.grid(True, alpha=0.25, axis="x", linestyle="--")
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(mticker.MultipleLocator(period / 4))


def plot_report(cycle: dict, args: argparse.Namespace, robot_name: str) -> Path:
    """Build and save the full PDF report."""
    leg_ids = cycle["leg_ids"]
    group_a = cycle["group_a"]
    group_b = cycle["group_b"]
    t = cycle["t"]
    phase = cycle["phase_state"]
    period = t[-1]

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2],
                          hspace=0.35, wspace=0.30,
                          left=0.06, right=0.96, top=0.93, bottom=0.05)

    # Subplot 1: Lift
    ax1 = fig.add_subplot(gs[0, 0])
    plot_joint_panel(ax1, t, cycle["lift_targets"], leg_ids, phase,
                     ylabel="Lift angle (rad)", title="Lift joint targets",
                     group_a=group_a, group_b=group_b)

    # Subplot 2: Swing
    ax2 = fig.add_subplot(gs[0, 1])
    plot_joint_panel(ax2, t, cycle["swing_targets"], leg_ids, phase,
                     ylabel="Swing angle (rad)", title="Swing joint targets",
                     group_a=group_a, group_b=group_b)

    # Subplot 3: Drop
    ax3 = fig.add_subplot(gs[1, 0])
    plot_joint_panel(ax3, t, cycle["drop_targets"], leg_ids, phase,
                     ylabel="Drop angle (rad)", title="Drop joint targets",
                     group_a=group_a, group_b=group_b)

    # Subplot 4: Phase Gantt
    ax4 = fig.add_subplot(gs[1, 1])
    plot_phase_gantt(ax4, t, phase, leg_ids, group_a, group_b)

    # ---- Summary statistics (full-width bottom strip) -----------------------
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis("off")

    stance_ratio = 1.0 - float(np.mean(phase))
    swing_ratio = float(np.mean(phase))

    lift_range = cycle["lift_targets"].max(axis=1) - cycle["lift_targets"].min(axis=1)
    swing_range = cycle["swing_targets"].max(axis=1) - cycle["swing_targets"].min(axis=1)
    drop_range = cycle["drop_targets"].max(axis=1) - cycle["drop_targets"].min(axis=1)

    lines: List[str] = []
    lines.append(f"Robot: {robot_name}   |   Legs: {len(leg_ids)}"
                 f"   |   Frequency: {args.gait_frequency:.2f} Hz"
                 f"   |   Period: {period:.3f} s"
                 f"   |   Samples: {len(t)}")
    lines.append(f"Group A (base phase): {group_a}   |   Group B (+π): {group_b}")
    lines.append(f"Forward axis: [{cycle['forward_axis'][0]:.4f}, {cycle['forward_axis'][1]:.4f}]")
    lines.append("")
    lines.append(f"Stance ratio: {stance_ratio:.1%}   |   Swing ratio: {swing_ratio:.1%}")
    lines.append("")

    # Per-leg stats table
    header = f"{'Leg':>5}  {'Lift mean':>10}  {'Lift Δ':>8}  {'Swing mean':>10}  {'Swing Δ':>8}  {'Drop mean':>10}  {'Drop Δ':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for idx, lid in enumerate(leg_ids):
        lines.append(
            f"  {lid:>3}  {cycle['lift_targets'][idx].mean():10.4f}  "
            f"{lift_range[idx]:8.4f}  {cycle['swing_targets'][idx].mean():10.4f}  "
            f"{swing_range[idx]:8.4f}  {cycle['drop_targets'][idx].mean():10.4f}  "
            f"{drop_range[idx]:8.4f}"
        )

    lines.append("")
    lines.append(f"Lift parameters:   stance_ratio={args.stance_lift_ratio}, "
                 f"swing_ratio={args.swing_lift_ratio}")
    lines.append(f"Swing parameters:  amplitude={args.swing_ratio_amplitude}")
    lines.append(f"Drop parameters:   stance_ratio={args.stance_drop_ratio}, "
                 f"swing_ratio={args.swing_drop_ratio}")

    ax5.text(0.01, 0.98, "\n".join(lines), transform=ax5.transAxes,
             fontsize=9, verticalalignment="top", family="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    # Title
    fig.suptitle(f"Gait Cycle Analysis — {robot_name}",
                 fontsize=16, fontweight="bold", y=0.98)

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--description", type=Path,
                   default=ASSET_DIR / "robot_description.json")
    p.add_argument("--gait-plan", type=Path, default=None,
                   help="Pre-computed gait plan JSON; if omitted, auto-computed.")
    p.add_argument("--output", type=Path,
                   default=OUTPUT_DIR / "leg_cycle_report.pdf")
    p.add_argument("--gait-frequency", type=float, default=0.85)
    p.add_argument("--swing-ratio-amplitude", type=float, default=0.26)
    p.add_argument("--swing-lift-ratio", type=float, default=0.78)
    p.add_argument("--stance-lift-ratio", type=float, default=0.05)
    p.add_argument("--swing-drop-ratio", type=float, default=0.38)
    p.add_argument("--stance-drop-ratio", type=float, default=0.90)
    p.add_argument("--samples", type=int, default=120,
                   help="Number of time samples per cycle.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.description.exists():
        print(f"[ERROR] Description not found: {args.description}")
        return 1

    description = load_description(args.description)
    robot_name = description.get("robot_name", "unknown")

    # Gait plan
    if args.gait_plan and args.gait_plan.exists():
        gait_plan = json.loads(args.gait_plan.read_text(encoding="utf-8"))
    else:
        gait_plan = compute_gait_plan(description)

    # Leg joint limits
    leg_limits = parse_leg_limits(description)
    if not leg_limits:
        print("[ERROR] No leg joint limits found in description.")
        return 1

    # Compute cycle targets
    cycle = compute_cycle_targets(
        description, gait_plan, leg_limits,
        gait_freq=args.gait_frequency,
        swing_amp=args.swing_ratio_amplitude,
        stance_lift=args.stance_lift_ratio,
        swing_lift=args.swing_lift_ratio,
        stance_drop=args.stance_drop_ratio,
        swing_drop=args.swing_drop_ratio,
        num_samples=args.samples,
    )

    # Plot and save
    output_path = plot_report(cycle, args, robot_name)
    print(f"PDF report written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
