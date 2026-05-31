#!/usr/bin/env python3
"""
机器人几何形态与SSM可视化工具（修复版 - 每次生成全新机器人）
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------- matplotlib (lazy — only required by SSMVisualizer) ------------

_plt = None
_available_font = None

def _ensure_matplotlib():
    global _plt, _available_font
    if _plt is not None:
        return _plt, _available_font

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.font_manager as fm

    chinese_fonts = [
        'WenQuanYi Zen Hei',
        'WenQuanYi Micro Hei',
        'Noto Sans CJK SC',
        'Noto Sans SC',
        'SimHei',
        'DejaVu Sans',
    ]

    for font_name in chinese_fonts:
        try:
            font_path = fm.findfont(fm.FontProperties(family=font_name), fallback_to_default=False)
            if font_path:
                _available_font = font_name
                break
        except:
            continue

    if _available_font:
        print(f"[INFO] 使用字体: {_available_font}")
        matplotlib.rcParams['font.family'] = _available_font
    else:
        print("[WARN] 未找到中文字体，使用英文")

    matplotlib.rcParams['axes.unicode_minus'] = False

    import matplotlib.pyplot as _plt_mod
    _plt = _plt_mod
    return _plt, _available_font

# ---------- 配置 ----------
GEN_PYTHON = "/data/conda/envs/Adaptation/bin/python"
REPO_ROOT = Path(__file__).resolve().parent
GEN_SCRIPT = REPO_ROOT / "generate_geometry.py"
OUTPUT_DIR = REPO_ROOT / "png"
DESC_PATH = REPO_ROOT / "robot_assets" / "robot_description.json"
# ------------------------

try:
    from shapely.geometry import MultiPoint
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False


def _signed_polygon_area(polygon_xy: np.ndarray) -> float:
    pts = np.asarray(polygon_xy, dtype=float)
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _ensure_ccw(polygon_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(polygon_xy, dtype=float)
    if len(pts) < 3:
        return pts
    if _signed_polygon_area(pts) < 0.0:
        return pts[::-1].copy()
    return pts


# ---------------------------------------------------------------------------
# Adaptive Gait Planner — Phase 1 & 2
# ---------------------------------------------------------------------------

def compute_adaptive_plan(
    description: Dict,
    state: Optional[Dict] = None,
    forced_axis: Optional[List[float]] = None,
) -> Dict:
    """Compute virtual forward axis, heading direction, leg groups, and gait topology.

    Phase 1 — Forward axis (Section 4b in README):
      1. Collect foot positions and build weighted covariance matrix.
      2. PCA → initial virtual forward axis (eigenvector of max eigenvalue).
      3. Score ±axis by swing-vector contribution, phase gain, and joint-range gain.
      4. The higher-scoring direction becomes ``final_forward_axis``.

    Phase 2 — Leg grouping (Section 6 in README):
      1. Project feet onto ``final_forward_axis``, sort by longitudinal position,
         distribute legs alternately into ``group_a`` / ``group_b``.
      2. Compute translational compensation from CoM → CoS.
      3. Build safety corridor along the forward axis inside the support polygon.

    Parameters
    ----------
    description : dict  robot_description.json content.
    state       : dict  optional gait-state override (phases, swing vectors, etc.).

    Returns
    -------
    dict  adaptive gait plan (see Section 9 in README for field descriptions).
    """
    if state is None:
        state = {}

    EPS = 1e-9

    # ---- foot world positions & leg metadata ---------------------------------
    num_legs = int(description.get("num_legs", 0))
    foot_xy: Dict[int, np.ndarray] = {}
    hip_xy: Dict[int, np.ndarray] = {}
    joint_links: Dict[int, List[Dict]] = {}

    for link in description.get("links", []):
        leg_id = link.get("leg_id")
        if leg_id is None:
            continue
        origin = np.asarray(link.get("default_world_origin", [0.0, 0.0, 0.0]), dtype=float)
        role = str(link.get("role", ""))
        if role == "foot":
            foot_xy[int(leg_id)] = origin[:2]
        elif role == "joint_sphere":
            name = str(link.get("name", ""))
            if name.endswith("_hip"):
                hip_xy[int(leg_id)] = origin[:2]
        joint_links.setdefault(int(leg_id), []).append(link)

    # ---- extract swing-vector range from joints per leg ----------------------
    swing_ranges: Dict[int, float] = {}
    for joint in description.get("joints", []):
        jname = str(joint.get("name", ""))
        for leg_id_str in joint_links:
            if jname.startswith(f"leg_{leg_id_str}_swing"):
                limit = joint.get("limit", {})
                lo = float(limit.get("lower", -0.55))
                hi = float(limit.get("upper", 0.55))
                swing_ranges[int(leg_id_str)] = max(hi - lo, 0.0)
                break

    # ---- support legs --------------------------------------------------------
    state_phases = state.get("phases", {})
    upcoming_stance = state.get("upcoming_stance_leg_ids", [])
    locked_ids = state.get("locked_leg_ids", [])
    missing_ids = state.get("missing_leg_ids", [])

    support_ids: List[int] = []
    near_stance_ids: List[int] = []

    for leg_id, pos in foot_xy.items():
        if leg_id in missing_ids:
            continue
        phase = state_phases.get(str(leg_id), "")
        if phase == "stance":
            support_ids.append(leg_id)
        elif int(leg_id) in upcoming_stance:
            near_stance_ids.append(leg_id)

    # Fallback: treat all valid feet as support
    if not support_ids:
        support_ids = sorted([lid for lid in foot_xy if lid not in missing_ids])

    # ---- weighted covariance → PCA initial axis -----------------------------
    def _unit(vec: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(vec))
        return vec / n if n > EPS else vec

    weights: Dict[int, float] = {}
    for lid in foot_xy:
        if lid in support_ids:
            weights[lid] = 1.0
        elif lid in near_stance_ids:
            weights[lid] = 0.35
        else:
            weights[lid] = 0.0

    sum_w = sum(weights.get(lid, 0.0) for lid in foot_xy)
    if sum_w < EPS:
        sum_w = 1.0
    cos_xy = np.zeros(2, dtype=float)
    for lid, pos in foot_xy.items():
        w = weights.get(lid, 0.0)
        if w <= 0.0:
            continue
        cos_xy += pos * w
    cos_xy /= sum_w

    # ---- COM projection (used for stability / compensation) ------------------
    projected_com_xy = np.zeros(2, dtype=float)
    total_mass = 0.0
    for link in description.get("links", []):
        mp = link.get("mass_properties", {})
        mass = float(mp.get("mass", 0.0))
        if mass <= 0.0:
            continue
        origin = np.asarray(link.get("default_world_origin", [0.0, 0.0, 0.0]), dtype=float)
        cm_local = np.asarray(mp.get("center_mass", [0.0, 0.0, 0.0]), dtype=float)
        world_cm = origin + cm_local
        projected_com_xy += world_cm[:2] * mass
        total_mass += mass
    if total_mass > EPS:
        projected_com_xy /= total_mass

    # ---- Mass-Weighted PCA (Principal Axis of Inertia) -----------------------
    cov_mass = np.zeros((2, 2), dtype=float)
    for link in description.get("links", []):
        mp = link.get("mass_properties", {})
        mass = float(mp.get("mass", 0.0))
        if mass <= 0.0:
            continue
        origin = np.asarray(link.get("default_world_origin", [0.0, 0.0, 0.0]), dtype=float)
        cm_local = np.asarray(mp.get("center_mass", [0.0, 0.0, 0.0]), dtype=float)
        world_cm = origin + cm_local
        diff = world_cm[:2] - projected_com_xy
        cov_mass += mass * np.outer(diff, diff)
    
    if total_mass > EPS:
        cov_mass /= total_mass

    eigenvals, eigenvecs = np.linalg.eigh(cov_mass)
    # Use MAJOR axis (max eigenvalue) as the principal inertia axis for stable movement.
    idx_max = int(np.argmax(eigenvals))
    major_axis = _unit(eigenvecs[:, idx_max].copy())
    # Also include the perpendicular (minor) axis: for robots whose legs extend along
    # the major body axis (e.g. standard hexapod with legs on both Y-sides), the
    # actual forward direction may be along the MINOR axis (perpendicular to the
    # longest body dimension).
    minor_axis = _unit(np.array([-major_axis[1], major_axis[0]], dtype=float))
    candidate_axes = [major_axis, -major_axis, minor_axis, -minor_axis]

    # ---- Phase-1 torque-aware scoring of ±axis -------------------------------
    def _build_default_swing_vector(leg_id: int) -> np.ndarray:
        """Return the swing vector for this leg for plan output (uses final axis).

        This is kept for backward compatibility (plan["swing_vector"] field).
        Direction scoring uses _gait_swing_dir() instead.
        """
        hip = hip_xy.get(leg_id)
        foot = foot_xy.get(leg_id)
        if hip is None or foot is None:
            return np.array([0.0, 0.0], dtype=float)
        hip_to_foot = foot - hip
        reach = float(np.linalg.norm(hip_to_foot))
        if reach < EPS:
            return np.array([0.0, 0.0], dtype=float)
        outward = foot - cos_xy
        foot_vel_dir = np.array([-hip_to_foot[1], hip_to_foot[0]], dtype=float) / reach
        if float(np.dot(foot_vel_dir, outward)) < 0.0:
            foot_vel_dir = -foot_vel_dir
        return foot_vel_dir * reach

    def _gait_swing_dir(leg_id: int, direction: np.ndarray) -> np.ndarray:
        """Return the foot velocity direction during the forward-swing half-cycle
        (sin > 0) using the **same dsign convention** as the gait execution in
        batch_test.py.

        Why this is correct
        -------------------
        The gait chooses dsign based on whether the foot is to the left or right
        of the candidate forward axis:
            lat = rotate_CCW(direction)
            dsign = -1 if dot(foot, lat) > 0 else +1

        dsign > 0 → CCW rotation: foot_vel = [-h2f_y, +h2f_x]
        dsign < 0 → CW  rotation: foot_vel = [+h2f_y, -h2f_x]

        This ensures ALL correctly-placed legs have a positive projection onto
        the true forward direction (both left and right legs conspire to push the
        body forward together).  The old "outward" heuristic failed for diagonal
        legs because CCW and outward are approximately perpendicular for any
        radial leg, so the flip condition never triggered reliably.
        """
        hip = hip_xy.get(leg_id)
        foot = foot_xy.get(leg_id)
        if hip is None or foot is None:
            return np.array([0.0, 0.0], dtype=float)
        h2f = foot - hip
        reach = float(np.linalg.norm(h2f))
        if reach < EPS:
            return np.array([0.0, 0.0], dtype=float)
        # left direction for this candidate
        lat_dir = np.array([-direction[1], direction[0]], dtype=float)
        lat_p = float(np.dot(foot, lat_dir))
        dsign = -1.0 if lat_p > 0.0 else 1.0
        if dsign > 0:
            return np.array([-h2f[1],  h2f[0]], dtype=float)   # CCW
        else:
            return np.array([ h2f[1], -h2f[0]], dtype=float)   # CW

    user_swings = state.get("swing_vectors", {})
    state_phases_legs = state.get("phases", {})

    def _score_direction(direction: np.ndarray) -> float:
        score = 0.0
        for leg_id, pos in foot_xy.items():
            if leg_id in missing_ids or leg_id in locked_ids:
                continue
            swing_raw = user_swings.get(str(leg_id), None)
            if swing_raw is not None:
                v = np.asarray(swing_raw, dtype=float)[:2].copy()
            else:
                v = _gait_swing_dir(leg_id, direction)
            v_norm = float(np.linalg.norm(v))
            if v_norm < EPS:
                continue
            v_hat = v / v_norm
            proj = float(np.dot(direction, v_hat))
            if proj <= 0.0:
                continue  # only contribute positive projection

            # Phase gain
            phase = state_phases_legs.get(str(leg_id), "")
            if phase in ("swing", "lift", "drop"):
                phase_gain = 1.15
            elif phase == "stance":
                phase_gain = 0.75
            else:
                phase_gain = 1.0

            # Joint-range gain  ──  wider swing range ⇒ more drive capacity
            delta = swing_ranges.get(leg_id, 0.0)
            range_gain = 1.0 + min(delta, 1.2)

            score += proj * v_norm * phase_gain * range_gain
            
        # CoM eccentricity penalty - penalize if CoM is far from lateral corridor center
        lat_axis = np.array([-direction[1], direction[0]], dtype=float)
        lat_positions = [float(np.dot(pos, lat_axis)) for lid, pos in foot_xy.items() if lid not in missing_ids]
        if lat_positions:
            corridor_center = (min(lat_positions) + max(lat_positions)) / 2.0
            com_lat = float(np.dot(projected_com_xy, lat_axis))
            eccentricity = abs(com_lat - corridor_center)
            score -= eccentricity * 0.5

        # Lateral leg distribution balance penalty.
        # When most legs fall on one side of the forward axis the alternating gait
        # cannot achieve zero net yaw regardless of amplitude correction.
        # Penalise the normalised lateral first-moment of foot positions:
        #   moment = sum(dot(foot_i, lat)) / N_active
        # For a symmetric arrangement this is 0; for a 6:2 split it is large.
        # Weight is strong (>> yaw torque weight) so a balanced direction always
        # beats an unbalanced one with higher raw thrust.
        _balance_legs = [lid for lid in foot_xy if lid not in missing_ids and lid not in locked_ids]
        if _balance_legs:
            _lat_moment = sum(float(np.dot(foot_xy[lid], lat_axis)) for lid in _balance_legs)
            _lat_moment_norm = _lat_moment / len(_balance_legs)
            LATERAL_BALANCE_WEIGHT = 3.0
            score -= LATERAL_BALANCE_WEIGHT * abs(_lat_moment_norm)

        # Yaw torque balance penalty — robots move straight only when net yaw is near zero.
        # For each driving leg, the yaw torque about the COM is:
        #   τ_i = thrust_i × yaw_lever_i,  yaw_lever_i = cross2d(r_i, direction)
        # where r_i = foot_i − COM.  Penalise the absolute value of the sum.
        net_yaw_torque = 0.0
        for leg_id, pos in foot_xy.items():
            if leg_id in missing_ids or leg_id in locked_ids:
                continue
            swing_raw = user_swings.get(str(leg_id), None)
            if swing_raw is not None:
                sv = np.asarray(swing_raw, dtype=float)[:2].copy()
            else:
                sv = _gait_swing_dir(leg_id, direction)
            v_proj = float(np.dot(sv, direction))
            if v_proj <= 0.0:
                continue
            phase = state_phases_legs.get(str(leg_id), "")
            if phase in ("swing", "lift", "drop"):
                pg = 1.15
            elif phase == "stance":
                pg = 0.75
            else:
                pg = 1.0
            delta = swing_ranges.get(leg_id, 0.0)
            rg = 1.0 + min(delta, 1.2)
            r = pos - projected_com_xy
            yaw_lever = float(r[0] * direction[1] - r[1] * direction[0])
            net_yaw_torque += v_proj * yaw_lever * pg * rg
        # Weight: light enough to avoid false direction flips (dynamic probe corrects residual yaw).
        # Secondary to forward thrust — only tips close-scoring candidates.
        YAW_BALANCE_WEIGHT = 0.8
        score -= YAW_BALANCE_WEIGHT * abs(net_yaw_torque)

        return score

    pos_score = _score_direction(major_axis)  # for report only
    neg_score = _score_direction(-major_axis)  # for report only

    # Evaluate all 4 candidate directions and pick the best, or use forced axis.
    if forced_axis is not None:
        # External caller (e.g. iterative optimizer) supplies the axis directly.
        final_axis = _unit(np.asarray(forced_axis[:2], dtype=float))
        best_score = _score_direction(final_axis)
    else:
        _X_PRIOR = 0.04
        best_score = -1e18
        final_axis = major_axis.copy()
        for cand in candidate_axes:
            s = _score_direction(cand) + _X_PRIOR * float(np.dot(cand, [1.0, 0.0]))
            if s > best_score:
                best_score = s
                final_axis = cand.copy()

    drive_resultant = final_axis * max(pos_score, neg_score)

    # ---- support polygon (CCW hull of all foot positions) --------------------
    all_foot_pts = np.array([p for lid, p in foot_xy.items() if lid not in missing_ids], dtype=float)
    if len(all_foot_pts) >= 3 and SHAPELY_AVAILABLE:
        try:
            hull = MultiPoint(all_foot_pts.tolist()).convex_hull
            if hull.geom_type == "Polygon":
                support_polygon = _ensure_ccw(np.array(hull.exterior.coords[:-1], dtype=float))
            else:
                support_polygon = all_foot_pts
        except Exception:
            support_polygon = all_foot_pts
    else:
        support_polygon = all_foot_pts

    # ---- safety corridor (medial-axis sampling along forward axis) -----------
    safety_corridor_xy: List[List[float]] = []
    if len(support_polygon) >= 3:
        try:
            from shapely.geometry import LineString, Polygon as ShpPoly
            poly = ShpPoly(support_polygon)
            # Walk along forward axis through the polygon
            com_p = projected_com_xy
            step_len = 0.06
            num_steps = 30
            for i in range(-num_steps, num_steps + 1):
                sample_pt = com_p + final_axis * (i * step_len)
                # Build perpendicular scan line
                perp = np.array([-final_axis[1], final_axis[0]], dtype=float)
                scan_half = 1.0
                line = LineString([
                    (sample_pt[0] - perp[0] * scan_half, sample_pt[1] - perp[1] * scan_half),
                    (sample_pt[0] + perp[0] * scan_half, sample_pt[1] + perp[1] * scan_half),
                ])
                inter = line.intersection(poly)
                if inter.is_empty:
                    continue
                if inter.geom_type == "Point":
                    pt = (inter.x, inter.y)
                    safety_corridor_xy.append([float(pt[0]), float(pt[1])])
                elif inter.geom_type == "MultiPoint":
                    coords = [(p.x, p.y) for p in inter.geoms]
                    if len(coords) >= 2:
                        mx = sum(c[0] for c in coords) / len(coords)
                        my = sum(c[1] for c in coords) / len(coords)
                        safety_corridor_xy.append([mx, my])
                elif inter.geom_type == "LineString":
                    mid = inter.interpolate(0.5, normalized=True)
                    safety_corridor_xy.append([float(mid.x), float(mid.y)])
        except Exception:
            pass

    # ---- translational compensation (CoM → CoS pull-back) -------------------
    # Solve min_t ||(CoM + t) - CoS||² + λ||W t||²  →  t ≈ (CoS - CoM) / (1 + λ)
    lambda_reg = 0.25
    torque_weights = state.get("torque_weights", {})
    avg_w = np.mean([float(w) for w in torque_weights.values()]) if torque_weights else 1.0
    effective_lambda = lambda_reg * avg_w
    com_offset = cos_xy - projected_com_xy
    translational_compensation = com_offset / (1.0 + effective_lambda)

    # ---- Phase 2: leg grouping -----------------------------------------------
    # For standard hexapods with symmetric left/right configuration:
    # Group legs diagonally (right-front + left-mid + right-rear vs left-front + right-mid + left-rear)
    # This ensures symmetric and stable tripod gait.
    #
    # For other configurations, sort by polar angle and alternate.
    active_legs = [lid for lid in sorted(foot_xy) if lid not in missing_ids and lid not in locked_ids]

    group_a: List[int] = []
    group_b: List[int] = []

    # Detect standard hexapod: exactly 6 legs, symmetric Y positions (3 left, 3 right)
    if len(active_legs) == 6:
        # Compute Y-coordinate statistics to detect left/right symmetry
        y_coords = [float(foot_xy[lid][1]) for lid in active_legs]
        y_mean = float(np.mean(y_coords))
        y_left = [lid for lid in active_legs if float(foot_xy[lid][1]) > y_mean]
        y_right = [lid for lid in active_legs if float(foot_xy[lid][1]) <= y_mean]
        
        # If we have 3 legs on each side, apply diagonal tripod grouping
        if len(y_left) == 3 and len(y_right) == 3:
            # Sort each side by X coordinate (front → mid → rear)
            y_left.sort(key=lambda lid: float(foot_xy[lid][0]), reverse=True)  # [front, mid, rear]
            y_right.sort(key=lambda lid: float(foot_xy[lid][0]), reverse=True)  # [front, mid, rear]
            # Diagonal tripod: front-right + mid-left + rear-right vs front-left + mid-right + rear-left
            group_a = [y_right[0], y_left[1], y_right[2]]  # [right-front, left-mid, right-rear]
            group_b = [y_left[0], y_right[1], y_left[2]]   # [left-front, right-mid, left-rear]
            print(f"[Grouping] 标准六足对角三脚架: group_a={group_a}  group_b={group_b}")
        else:
            # Fallback to polar angle alternation
            angles = []
            for lid in active_legs:
                dp = foot_xy[lid] - cos_xy
                angles.append((float(np.arctan2(dp[1], dp[0])), lid))
            angles.sort(key=lambda t: t[0])
            for idx, (_, lid) in enumerate(angles):
                if idx % 2 == 0:
                    group_a.append(lid)
                else:
                    group_b.append(lid)
    else:
        # For non-standard configurations, sort by polar angle and alternate
        angles = []
        for lid in active_legs:
            dp = foot_xy[lid] - cos_xy
            angles.append((float(np.arctan2(dp[1], dp[0])), lid))
        angles.sort(key=lambda t: t[0])
        for idx, (_, lid) in enumerate(angles):
            if idx % 2 == 0:
                group_a.append(lid)
            else:
                group_b.append(lid)

    # ---- Per-leg swing projection onto forward axis --------------------------
    # Compute how much each leg's swing joint contributes to forward motion.
    # A leg mounted laterally (foot directly to the side) has projection ≈ 1.0.
    # A leg mounted facing forward/backward has projection ≈ 0.0 (passive).
    PASSIVE_THRESHOLD = 0.30  # legs with |projection| < threshold go into group_c

    fwd_axis = final_axis
    lat_axis = np.array([-fwd_axis[1], fwd_axis[0]], dtype=float)

    swing_projections: Dict[int, float] = {}
    for lid in active_legs:
        h = hip_xy.get(lid)
        f = foot_xy.get(lid)
        if h is not None and f is not None:
            outward = f - h
            outward_n = float(np.linalg.norm(outward))
            if outward_n > EPS:
                tangent = np.array([-outward[1], outward[0]], dtype=float) / outward_n
            else:
                tangent = np.array([0.0, 1.0], dtype=float)
            # Determine swing direction sign (same logic as build_gait_targets)
            lat_pos = float(np.dot(f, lat_axis))
            d_sign = -1.0 if lat_pos > 0.0 else 1.0
            proj = float(np.dot(d_sign * tangent, fwd_axis))
        else:
            proj = 1.0  # no geometry data → assume full contribution
        swing_projections[lid] = proj

    # ---- Per-leg stride amplitude limits ------------------------------------
    # Each leg gets a stride amplitude in [0, 1] (normalised).
    # The limit is derived from the leg's kinematic reach relative to the
    # maximum reach across all legs, scaled further by the forward projection.
    #
    # Rationale: shorter / offset legs should not receive the same full-range
    # joint commands as long legs — doing so causes joint-limit violations and
    # physically mismatched ground clearance across the tripod.
    per_leg_stride_amplitudes: Dict[int, float] = {}

    # Gather outward reach (hip → foot distance) for normalisation.
    reaches: Dict[int, float] = {}
    for lid in active_legs:
        h = hip_xy.get(lid)
        f = foot_xy.get(lid)
        if h is not None and f is not None:
            reaches[lid] = float(np.linalg.norm(f - h))
        else:
            reaches[lid] = 1.0

    max_reach = max(reaches.values()) if reaches else 1.0
    if max_reach < EPS:
        max_reach = 1.0

    # Also normalise by swing joint range (wider-range joints can afford more amp).
    max_swing_range = max(swing_ranges.values()) if swing_ranges else 1.0
    if max_swing_range < EPS:
        max_swing_range = 1.0

    for lid in active_legs:
        reach_ratio = reaches.get(lid, 1.0) / max_reach           # 0..1
        range_ratio = swing_ranges.get(lid, max_swing_range) / max_swing_range  # 0..1
        proj_mag = abs(swing_projections.get(lid, 1.0))            # 0..1
        # Combined: sqrt of geometry factors, clamped to [0.15, 0.75]
        raw = math.sqrt(reach_ratio * range_ratio) * proj_mag
        per_leg_stride_amplitudes[lid] = float(np.clip(raw, 0.15, 0.75))

    # Classify active legs into three groups:
    #   group_a / group_b : alternating active tripods (good or neutral forward projection)
    #   group_c           : passive legs whose swing motion OPPOSES forward motion (proj < 0)
    #
    # Rationale: a leg with proj ≈ 0 (e.g. standard hexapod centerline legs) contributes
    # zero swing thrust but is NOT an obstacle — keeping it in group_a/b retains tripod
    # stability.  Only legs with proj < −PASSIVE_THRESHOLD (swing that actively pushes the
    # body backward) are demoted to group_c.
    PASSIVE_THRESHOLD = 0.10  # legs with proj < −threshold are demoted to group_c
    passive_ids: set = {lid for lid in active_legs
                        if swing_projections.get(lid, 0.0) < -PASSIVE_THRESHOLD}
    group_c: List[int] = sorted(lid for lid in active_legs if lid in passive_ids)
    group_a = [lid for lid in group_a if lid not in passive_ids]
    group_b = [lid for lid in group_b if lid not in passive_ids]

    # Also assign any un-grouped active leg (projection OK but missed by polar alternation)
    # to whichever group it balances — prevents "ghost" legs defaulting to group_a phase.
    assigned = set(group_a) | set(group_b) | set(group_c)
    for lid in active_legs:
        if lid not in assigned:
            if len(group_a) <= len(group_b):
                group_a.append(lid)
            else:
                group_b.append(lid)

    # Cap swing group size to prevent too many legs lifting simultaneously on
    # highly asymmetric robots. The limit scales with the robot: at most
    # ceil(N/2) legs per group (which is exactly what a standard hexapod needs:
    # 3 legs per group). Only demote when a group is genuinely over-sized
    # relative to the balanced split.
    n_active = max(len(group_a) + len(group_b) + len(group_c), 1)
    MAX_SWING_GROUP_SIZE = math.ceil(n_active / 2)
    for grp, other in [(group_a, group_b), (group_b, group_a)]:
        while len(grp) > MAX_SWING_GROUP_SIZE:
            # Demote the leg with lowest swing projection to group_c
            leg_to_demote = min(grp, key=lambda lid: abs(swing_projections.get(lid, 1.0)))
            grp.remove(leg_to_demote)
            group_c.append(leg_to_demote)

    # ---- Yaw torque compensation via per-leg stride amplitude scaling -------
    # Physics: when leg i is in stance the reaction force on the body points along
    # final_axis with magnitude ∝ stride_amplitude_i.  The yaw torque about the
    # COM is  τ_i = stride_i × ψ_i  where the "yaw lever" for leg i is:
    #   ψ_i = swing_proj_i × cross2d(r_i, final_axis)
    #       r_i = foot_i − projected_com_xy
    # Net yaw = Σ ψ_i (with unit amplitudes).
    #
    # Least-squares correction: minimise Σ(s_i − 1)² s.t. Σ s_i ψ_i = 0
    #   → s_i = 1 − λψ_i,   λ = (Σ ψ_i) / (Σ ψ_i²)
    #
    # Then the corrected stride amplitude is clamped to [0.10, 0.85].
    # YAW_COMP_GAIN ∈ [0,1] blends between no correction (0) and full correction (1).
    YAW_COMP_GAIN    = 0.50   # conservative static correction; dynamic probe handles the rest
    YAW_COMP_MAX_ADJ = 0.30   # maximum ±adjustment fraction (±30 % of base amplitude)

    # Per-leg yaw lever: ψ_i = swing_proj_i × cross2d(r_i, final_axis)
    yaw_levers_by_leg: Dict[int, float] = {}
    psi_by_leg: Dict[int, float] = {}
    for lid in active_legs:
        r = foot_xy[lid] - projected_com_xy
        yaw_lever = float(r[0] * final_axis[1] - r[1] * final_axis[0])
        yaw_levers_by_leg[lid] = yaw_lever
        psi_by_leg[lid] = swing_projections.get(lid, 0.0) * yaw_lever

    net_yaw_nominal = sum(psi_by_leg.values())  # with all s_i = 1
    denom_psi = sum(w * w for w in psi_by_leg.values())

    if denom_psi > EPS:
        lam = YAW_COMP_GAIN * net_yaw_nominal / denom_psi
        for lid in active_legs:
            s_raw = 1.0 - lam * psi_by_leg[lid]
            s_clamped = float(np.clip(s_raw,
                                      1.0 - YAW_COMP_MAX_ADJ,
                                      1.0 + YAW_COMP_MAX_ADJ))
            base_amp = per_leg_stride_amplitudes.get(lid, 0.5)
            per_leg_stride_amplitudes[lid] = float(np.clip(base_amp * s_clamped, 0.10, 0.85))
        # net yaw after correction: Σ ψ_i × s_i  ≈ 0 by construction
        net_yaw_corrected = sum(
            psi_by_leg[lid] * per_leg_stride_amplitudes[lid]
            for lid in active_legs
        )
        _initial_amps = {lid: (1.0 - lam * psi_by_leg[lid]) for lid in active_legs}
        _max_adj = max(abs(1.0 - _initial_amps[lid]) for lid in active_legs)
        print(f"[YawComp] net_yaw_nominal={net_yaw_nominal:.4f}  λ={lam:.4f}  "
              f"max_adj={_max_adj:.3f}")
    else:
        net_yaw_corrected = net_yaw_nominal

    # ---- CPG coupling matrix (geometry-aware) --------------------------------
    cpg_cfg = state.get("cpg", {}) if isinstance(state.get("cpg", {}), dict) else {}
    freq_hz = float(cpg_cfg.get("frequency_hz", 0.85))
    duty_factor = float(cpg_cfg.get("duty_factor", 0.60))
    coupling_gain = float(cpg_cfg.get("coupling_gain", 1.0))
    long_scale = float(cpg_cfg.get("longitudinal_scale", 0.35))
    lat_scale = float(cpg_cfg.get("lateral_scale", 0.25))
    long_scale = max(long_scale, 1e-3)
    lat_scale = max(lat_scale, 1e-3)

    # fwd_axis / lat_axis already defined above (swing projection section)

    n_legs = len(active_legs)
    coupling = np.zeros((n_legs, n_legs), dtype=float)
    phase_bias = np.zeros((n_legs, n_legs), dtype=float)

    group_a_set = set(group_a)
    group_b_set = set(group_b)
    for i, lid_i in enumerate(active_legs):
        for j, lid_j in enumerate(active_legs):
            if i == j:
                continue
            delta = foot_xy[lid_j] - foot_xy[lid_i]
            d_long = float(np.dot(delta, fwd_axis))
            d_lat = float(np.dot(delta, lat_axis))
            w = math.exp(-0.5 * ((d_long / long_scale) ** 2 + (d_lat / lat_scale) ** 2))
            coupling[i, j] = coupling_gain * w
            if (lid_i in group_a_set and lid_j in group_b_set) or (lid_i in group_b_set and lid_j in group_a_set):
                phase_bias[i, j] = float(np.pi)

    leg_phase_offsets: Dict[str, float] = {}
    for lid in active_legs:
        if lid in group_b_set:
            leg_phase_offsets[str(lid)] = float(np.pi)
        else:
            leg_phase_offsets[str(lid)] = 0.0

    # ---- inhibition rules ----------------------------------------------------
    inhibition_rules: List[Dict] = []
    for lid in locked_ids:
        inhibition_rules.append({
            "leg_id": int(lid),
            "reason": "locked",
            "in_degree": 0.0,
            "out_degree": 0.0,
        })
    for lid in missing_ids:
        inhibition_rules.append({
            "leg_id": int(lid),
            "reason": "missing",
            "in_degree": 0.0,
            "out_degree": 0.0,
        })
    for lid in range(num_legs):
        if lid not in foot_xy:
            inhibition_rules.append({
                "leg_id": lid,
                "reason": "no_foot_data",
                "in_degree": 0.0,
                "out_degree": 0.0,
            })

    # ---- planned swings ------------------------------------------------------
    planned_swings: Dict[str, Dict] = {}
    for lid in foot_xy:
        v_raw = user_swings.get(str(lid))
        if v_raw is not None:
            v = np.asarray(v_raw, dtype=float)[:2]
        else:
            v = _build_default_swing_vector(lid)
        fwd_comp = float(np.dot(v, final_axis))
        planned_swings[str(lid)] = {
            "swing_vector": v.tolist(),
            "forward_component": fwd_comp,
            "magnitude": float(np.linalg.norm(v)),
        }

    plan = {
        "support_center_xy": cos_xy.tolist(),
        "projected_com_xy": projected_com_xy.tolist(),
        "initial_virtual_forward_axis": major_axis.tolist(),
        "final_forward_axis": final_axis.tolist(),
        "drive_resultant_xy": drive_resultant.tolist(),
        "direction_scores": {"positive": pos_score, "negative": neg_score},
        "support_polygon_xy": support_polygon.tolist() if len(support_polygon) > 0 else [],
        "safety_corridor_xy": safety_corridor_xy,
        "translational_compensation_xy": translational_compensation.tolist(),
        "planned_swings": planned_swings,
        "support_leg_ids": support_ids,
        "near_stance_leg_ids": near_stance_ids,
        "yaw_balance": {
            # net_yaw_nominal: net yaw torque before amplitude correction (unit amplitudes).
            # Values near 0 mean naturally balanced; large values indicate asymmetry.
            "net_yaw_nominal": float(net_yaw_nominal),
            # net_yaw_corrected: estimated residual after amplitude correction.
            "net_yaw_corrected": float(net_yaw_corrected),
            # per-leg yaw lever ψ_i = swing_proj_i × cross2d(r_i, final_axis).
            # Positive = foot is on the left of the forward line (adds CCW yaw when thrust).
            # Negative = foot is on the right (adds CW yaw when thrust).
            "psi_by_leg": {str(lid): float(psi_by_leg.get(lid, 0.0)) for lid in active_legs},
            "yaw_levers": {str(lid): float(yaw_levers_by_leg.get(lid, 0.0)) for lid in active_legs},
        },
        "cpg": {
            "active_leg_ids": active_legs,
            "phase_offsets": leg_phase_offsets,
            "coupling_weights": coupling.tolist(),
            "coupling_phase_bias": phase_bias.tolist(),
            "frequency_hz": freq_hz,
            "omega": float(2.0 * np.pi * freq_hz),
            "duty_factor": duty_factor,
            "longitudinal_scale": long_scale,
            "lateral_scale": lat_scale,
        },
        "impedance": {
            "space": str(state.get("impedance", {}).get("space", "joint")),
            "stance_kp": float(state.get("impedance", {}).get("stance_kp", 180.0)),
            "stance_kd": float(state.get("impedance", {}).get("stance_kd", 12.0)),
            "swing_kp": float(state.get("impedance", {}).get("swing_kp", 80.0)),
            "swing_kd": float(state.get("impedance", {}).get("swing_kd", 6.0)),
            "max_deflection": float(state.get("impedance", {}).get("max_deflection", 0.05)),
        },
        "topology": {
            "groups": {"group_a": group_a, "group_b": group_b, "group_c": group_c},
            "phase_offsets": {"group_a": 0.0, "group_b": float(np.pi)},
            "swing_projections": {str(lid): swing_projections[lid] for lid in active_legs},
            "per_leg_stride_amplitudes": {str(lid): per_leg_stride_amplitudes.get(lid, 1.0)
                                          for lid in active_legs},
            "inhibition_rules": inhibition_rules,
            "coupling_matrix_zeroed_edges": [],
        },
    }
    return plan


# ---------------------------------------------------------------------------
# SSM Visualiser (matplotlib-based, for offline inspection)
# ---------------------------------------------------------------------------

class SSMVisualizer:
    """SSM 可视化器"""
    
    def __init__(self, description: Dict):
        self.desc = description
        self._parse_data()
    
    def _parse_data(self):
        self.body_outline = self._get_body_outline()
        self.legs_info = self._get_legs_info()
        self.foot_positions = self._get_foot_positions()
        self.com_xy = self._compute_com_xy()
        self.support_polygon = self._compute_support_polygon()
        self.ssm_value, self.closest_edge = self._compute_ssm()
    
    def _get_body_outline(self) -> np.ndarray:
        if "trunk_polygon_xy" in self.desc:
            pts = np.array(self.desc["trunk_polygon_xy"], dtype=float)
            if len(pts) > 0 and pts.shape[1] >= 2:
                return pts[:, :2]
        
        if "body_outline" in self.desc:
            pts = np.array(self.desc["body_outline"], dtype=float)
            if len(pts) > 0 and pts.shape[1] >= 2:
                return pts[:, :2]
        
        bl = float(self.desc.get("body_length", 0.3))
        bw = float(self.desc.get("body_width", 0.2))
        hw = bw / 2
        hl = bl / 2
        return np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]], dtype=float)
    
    def _get_legs_info(self) -> List[Dict]:
        legs = []
        for link in self.desc.get("links", []):
            role = link.get("role", "")
            if role in ("hip", "upper_link", "lower_link", "foot"):
                origin = np.array(link.get("default_world_origin", [0, 0, 0]), dtype=float)
                leg_id = link.get("leg_id")
                
                found = False
                for leg in legs:
                    if leg.get("leg_id") == leg_id:
                        leg["segments"].append({"role": role, "origin": origin})
                        if role == "foot":
                            leg["foot_xy"] = origin[:2]
                        found = True
                        break
                
                if not found and leg_id is not None:
                    new_leg = {"leg_id": leg_id, "segments": [{"role": role, "origin": origin}]}
                    if role == "foot":
                        new_leg["foot_xy"] = origin[:2]
                    legs.append(new_leg)
        
        return legs
    
    def _get_foot_positions(self) -> np.ndarray:
        feet = []
        for leg in self.legs_info:
            if "foot_xy" in leg:
                feet.append(leg["foot_xy"])
        
        if not feet:
            for link in self.desc.get("links", []):
                if link.get("role") == "foot":
                    origin = np.array(link.get("default_world_origin", [0, 0, 0]), dtype=float)
                    feet.append(origin[:2])
        
        return np.array(feet) if feet else np.zeros((0, 2))
    
    def _compute_com_xy(self) -> np.ndarray:
        weighted_sum = np.zeros(2, dtype=float)
        total_mass = 0.0
        
        for link in self.desc.get("links", []):
            mass_props = link.get("mass_properties", {})
            mass = float(mass_props.get("mass", 0.0))
            if mass <= 0:
                continue
            
            origin = np.array(link.get("default_world_origin", [0, 0, 0]), dtype=float)
            cm_local = np.array(mass_props.get("center_mass", [0, 0, 0]), dtype=float)
            world_cm = origin + cm_local
            weighted_sum += world_cm[:2] * mass
            total_mass += mass
        
        return weighted_sum / total_mass if total_mass > 1e-9 else np.zeros(2)
    
    def _compute_support_polygon(self) -> np.ndarray:
        if len(self.foot_positions) < 3:
            return self.foot_positions.copy()
        
        if SHAPELY_AVAILABLE:
            try:
                hull = MultiPoint(self.foot_positions.tolist()).convex_hull
                if hull.geom_type == 'Polygon':
                    return _ensure_ccw(np.array(hull.exterior.coords[:-1]))
            except:
                pass
        
        center = self.foot_positions.mean(axis=0)
        angles = np.arctan2(
            self.foot_positions[:, 1] - center[1],
            self.foot_positions[:, 0] - center[0]
        )
        return _ensure_ccw(self.foot_positions[np.argsort(angles)])
    
    def _compute_ssm(self) -> Tuple[float, Optional[Dict]]:
        n = len(self.support_polygon)
        if n < 3:
            return 0.0, None
        
        pts = self.support_polygon
        p = self.com_xy
        
        min_dist = float('inf')
        closest_edge = None
        
        for i in range(n):
            a = pts[i]
            b = pts[(i + 1) % n]
            edge = b - a
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-9:
                continue
            
            signed_dist = (edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0])) / edge_len
            
            if signed_dist < min_dist:
                min_dist = signed_dist
                closest_edge = {
                    'start': a.copy(),
                    'end': b.copy(),
                    'signed_distance': signed_dist,
                    'edge_index': i
                }
        
        return float(min_dist) if np.isfinite(min_dist) else 0.0, closest_edge
    
    def plot(self, save_path: Path, robot_name: str, seed: int):
        """绘制可视化图片"""
        plt, available_font = _ensure_matplotlib()
        fig, ax = plt.subplots(1, 1, figsize=(14, 11))
        
        # 1. 躯干轮廓
        body_closed = np.vstack([self.body_outline, self.body_outline[0]])
        ax.plot(body_closed[:, 0], body_closed[:, 1], 'k-', linewidth=2.5, 
                label='躯干轮廓' if available_font else 'Body Outline')
        ax.fill(body_closed[:, 0], body_closed[:, 1], alpha=0.15, color='gray')
        
        # 2. 腿部链节
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(self.legs_info), 1)))
        for idx, leg in enumerate(self.legs_info):
            color = colors[idx % len(colors)]
            segments = leg.get("segments", [])
            
            for i in range(len(segments) - 1):
                p1 = segments[i]["origin"][:2]
                p2 = segments[i + 1]["origin"][:2]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color=color, linewidth=1.5, alpha=0.7)
            
            for seg in segments:
                pt = seg["origin"][:2]
                ax.plot(pt[0], pt[1], 'o', color=color, markersize=4, alpha=0.6)
        
        # 3. 落足点
        if len(self.foot_positions) > 0:
            ax.scatter(
                self.foot_positions[:, 0], 
                self.foot_positions[:, 1],
                s=120, c='red', marker='s', zorder=5, edgecolors='darkred', linewidths=1.5,
                label='落足点' if available_font else 'Foot Points'
            )
            for i, fp in enumerate(self.foot_positions):
                ax.annotate(
                    f'F{i+1}', (fp[0], fp[1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold', color='darkred',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8)
                )
        
        # 4. 支撑多边形
        if len(self.support_polygon) >= 3:
            poly_closed = np.vstack([self.support_polygon, self.support_polygon[0]])
            
            if self.ssm_value >= 0:
                poly_color = 'green'
                poly_alpha = 0.25
                edge_color = 'darkgreen'
            else:
                poly_color = 'red'
                poly_alpha = 0.15
                edge_color = 'darkred'
            
            poly_label = f'支撑多边形 ({len(self.support_polygon)}边形)' if available_font else f'Support Polygon ({len(self.support_polygon)}-gon)'
            ax.fill(poly_closed[:, 0], poly_closed[:, 1], alpha=poly_alpha, color=poly_color)
            ax.plot(poly_closed[:, 0], poly_closed[:, 1], '--', color=edge_color, linewidth=2, label=poly_label)
            
            for i, v in enumerate(self.support_polygon):
                ax.annotate(
                    f'V{i}', (v[0], v[1]),
                    xytext=(-8, -8), textcoords='offset points',
                    fontsize=8, color=edge_color
                )
        
        # 5. 质心位置
        com_label = '质心投影 CoM' if available_font else 'CoM Projection'
        ax.scatter(
            self.com_xy[0], self.com_xy[1],
            s=200, c='blue', marker='*', zorder=10, edgecolors='darkblue', linewidths=2,
            label=com_label
        )
        
        # 6. SSM标注
        if self.closest_edge is not None:
            edge_data = self.closest_edge
            edge_start = edge_data['start']
            edge_end = edge_data['end']
            
            edge_vec = edge_end - edge_start
            edge_len = np.linalg.norm(edge_vec)
            if edge_len > 1e-9:
                edge_unit = edge_vec / edge_len
                t = np.dot(self.com_xy - edge_start, edge_unit)
                t = np.clip(t, 0, edge_len)
                foot_point = edge_start + t * edge_unit
                
                ax.plot(
                    [self.com_xy[0], foot_point[0]], 
                    [self.com_xy[1], foot_point[1]],
                    'g--', linewidth=2, alpha=0.8
                )
                
                mid_point = (self.com_xy + foot_point) / 2
                ax.annotate(
                    f'SSM = {self.ssm_value:.4f} m',
                    mid_point,
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=11, fontweight='bold',
                    color='green' if self.ssm_value >= 0 else 'red',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6)
                )
        
        # 7. 标题
        status_text = "✓ 通过 (静态稳定)" if self.ssm_value >= 0 else "✗ 不通过 (静态不稳定)"
        status_color = 'green' if self.ssm_value >= 0 else 'red'
        
        title = (
            f"机器人静态稳定性分析 (种子: {seed})\n"
            f"模型: {robot_name} | "
            f"足端数: {len(self.foot_positions)} | "
            f"SSM = {self.ssm_value:.4f} m | "
            f"{status_text}"
        )
        ax.set_title(title, fontsize=14, fontweight='bold', color=status_color)
        
        # 8. 图表设置
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        
        # 9. 信息文本
        info_text = (
            f"SSM (静态稳定裕度): {self.ssm_value:.4f} m\n"
            f"阈值: 0.0 m\n"
            f"足端数量: {len(self.foot_positions)}\n"
            f"支撑多边形顶点: {len(self.support_polygon)}\n"
            f"质心坐标: ({self.com_xy[0]:.4f}, {self.com_xy[1]:.4f})\n"
            f"判定: {'稳定' if self.ssm_value >= 0 else '不稳定'}\n"
            f"种子: {seed}"
        )
        
        props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9)
        ax.text(
            0.02, 0.98, info_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=props,
            family='monospace'
        )
        
        # 10. 保存
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[OK] 图片已保存: {save_path}")


def generate_robot_ignore_ssm(robot_name: str, seed: int, leg_placement: str = "random",
                               body_length=None, body_width=None, body_height=None,
                               upper_length=None, lower_length=None) -> bool:
    """
    生成机器人并忽略SSM检测
    原理：先生成几何，如果SSM不通过，用另一种方式强制生成描述文件
    """
    # 方案：先删除旧的描述文件
    if DESC_PATH.exists():
        DESC_PATH.unlink()
        print(f"[INFO] 已删除旧描述文件")
    
    cmd = [
        GEN_PYTHON, str(GEN_SCRIPT),
        "--robot-name", robot_name,
        "--seed", str(seed),
        "--leg-placement", leg_placement,
    ]
    
    for param_name, flag in [
        ("body_length", "--body-length"),
        ("body_width", "--body-width"),
        ("body_height", "--body-height"),
        ("upper_length", "--upper-length"),
        ("lower_length", "--lower-length"),
    ]:
        val = locals().get(param_name)
        if val is not None:
            cmd.extend([flag, str(val)])
    
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # 检查是否生成了描述文件
    if DESC_PATH.exists():
        # 验证文件是否是刚生成的
        file_time = DESC_PATH.stat().st_mtime
        current_time = time.time()
        if current_time - file_time < 10:  # 10秒内生成的文件
            print(f"[INFO] 描述文件生成成功 (时间戳匹配)")
            return True
    
    # 如果SSM失败导致没有生成描述文件，尝试强制生成
    print(f"[WARN] 标准生成失败，尝试强制生成模式...")
    print(f"[STDERR] {result.stderr[:300]}")
    
    # 这里可以添加备用生成逻辑
    # 例如：使用更宽松的参数重新生成
    # 或者：直接构造一个简化的描述文件
    return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="SSM Visualization Tool")
    parser.add_argument("--single", action="store_true", help="Single generation mode")
    parser.add_argument("--count", type=int, default=10, help="Batch count")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--seed-base", type=int, default=7, help="Batch base seed")
    parser.add_argument("--seed-step", type=int, default=17, help="Batch seed step")
    parser.add_argument("--leg-placement", default="random", choices=["uniform", "random"])
    parser.add_argument("--body-length", type=float, default=None)
    parser.add_argument("--body-width", type=float, default=None)
    parser.add_argument("--body-height", type=float, default=None)
    parser.add_argument("--upper-length", type=float, default=None)
    parser.add_argument("--lower-length", type=float, default=None)
    
    args = parser.parse_args()
    
    if not GEN_SCRIPT.exists():
        print(f"[ERROR] {GEN_SCRIPT} not found")
        sys.exit(1)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.single:
        # 单次模式
        robot_name = f"robot_s{args.seed}"
        print(f"\n{'='*60}")
        print(f"生成机器人: {robot_name}")
        print(f"{'='*60}")
        
        # 生成机器人
        success = generate_robot_ignore_ssm(
            robot_name, args.seed, args.leg_placement,
            args.body_length, args.body_width, args.body_height,
            args.upper_length, args.lower_length
        )
        
        if success and DESC_PATH.exists():
            with open(DESC_PATH, 'r') as f:
                description = json.load(f)
            
            visualizer = SSMVisualizer(description)
            ssm_status = "pass" if visualizer.ssm_value >= 0 else "fail"
            filename = f"{robot_name}_{ssm_status}_ssm{visualizer.ssm_value:.4f}.png"
            save_path = OUTPUT_DIR / filename
            
            visualizer.plot(save_path, robot_name, args.seed)
            
            print(f"\n结果:")
            print(f"  种子: {args.seed}")
            print(f"  足端数: {len(visualizer.foot_positions)}")
            print(f"  SSM: {visualizer.ssm_value:.4f} m")
            print(f"  状态: {'✓ 通过' if visualizer.ssm_value >= 0 else '✗ 不通过'}")
        else:
            print(f"[ERROR] 无法生成机器人")
    else:
        # 批量模式
        print(f"\n批量生成 {args.count} 个机器人...")
        
        for i in range(args.count):
            seed = args.seed_base + i * args.seed_step
            robot_name = f"robot_{i:03d}_s{seed}"
            
            print(f"\n[{i+1}/{args.count}] 种子={seed}")
            
            success = generate_robot_ignore_ssm(
                robot_name, seed, args.leg_placement,
                args.body_length, args.body_width, args.body_height,
                args.upper_length, args.lower_length
            )
            
            if success and DESC_PATH.exists():
                with open(DESC_PATH, 'r') as f:
                    description = json.load(f)
                
                visualizer = SSMVisualizer(description)
                ssm_status = "pass" if visualizer.ssm_value >= 0 else "fail"
                filename = f"{robot_name}_{ssm_status}_ssm{visualizer.ssm_value:.4f}.png"
                save_path = OUTPUT_DIR / filename
                
                visualizer.plot(save_path, robot_name, seed)
                print(f"  SSM: {visualizer.ssm_value:.4f} m")
            
            time.sleep(0.1)


if __name__ == "__main__":
    main()