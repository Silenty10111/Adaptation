#!/usr/bin/env python3
"""
自适应步态规划器 — 基于 PCA 的虚拟体轴估计、支撑安全走廊提取、
头尾方向确定、质心平移补偿、腿部分组与拓扑掩蔽、偏航力矩消除。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

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

    # ---- COM projection (used for adaptation.stability / compensation) ------------------
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

    # ---- Trunk main axis (fixed reference for swing direction) ------------------
    trunk_pts = []
    if "trunk_polygon_xy" in description:
        trunk_pts = np.array(description["trunk_polygon_xy"], dtype=float)
    if len(trunk_pts) < 3:
        bl = float(description.get("body_length", 0.72))
        bw = float(description.get("body_width", 0.44))
        trunk_pts = np.array([
            [-bl/2, -bw/2], [bl/2, -bw/2], [bl/2, bw/2], [-bl/2, bw/2]
        ], dtype=float)
    trunk_center = trunk_pts.mean(axis=0)
    trunk_centered = trunk_pts - trunk_center
    trunk_cov = trunk_centered.T @ trunk_centered / len(trunk_pts)
    trunk_eigvals, trunk_eigvecs = np.linalg.eigh(trunk_cov)
    trunk_main = trunk_eigvecs[:, -1] / max(float(np.linalg.norm(trunk_eigvecs[:, -1])), EPS)
    trunk_lat  = np.array([-trunk_main[1], trunk_main[0]], dtype=float)

    def _fixed_swing_dir(leg_id: int) -> np.ndarray:
        """Fixed-frame swing direction using TRUNK lateral axis for dsign.

        Unlike _gait_swing_dir, does NOT depend on candidate forward axis.
        Breaks the self-referential loop that made +/- axis scores identical.
        """
        hip = hip_xy.get(leg_id)
        foot = foot_xy.get(leg_id)
        if hip is None or foot is None:
            return np.array([0.0, 0.0], dtype=float)
        h2f = foot - hip
        reach = float(np.linalg.norm(h2f))
        if reach < EPS:
            return np.array([0.0, 0.0], dtype=float)
        lat_p = float(np.dot(foot, trunk_lat))
        dsign = -1.0 if lat_p > 0.0 else 1.0
        if dsign > 0:
            return np.array([-h2f[1],  h2f[0]], dtype=float)
        else:
            return np.array([ h2f[1], -h2f[0]], dtype=float)

    # ---- Phase-1 torque-aware scoring of ±axis -------------------------------
    def _build_default_swing_vector(leg_id: int) -> np.ndarray:
        """Return the swing vector for this leg for plan output (uses final axis).

        This is kept for backward compatibility (plan["swing_vector"] field).
        Direction scoring uses _fixed_swing_dir() instead.
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
                v = _fixed_swing_dir(leg_id)
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
            LATERAL_BALANCE_WEIGHT = 1.8
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
                sv = _fixed_swing_dir(leg_id)
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

        # Trunk alignment — preference for body natural forward direction.
        score += 0.40 * float(np.dot(direction, trunk_main))

        # Forward half-space bonus — feet in forward hemisphere.
        # Uses trunk center (~origin) as reference to avoid cos_xy shift bias.
        _fwd_count = 0
        for lid in foot_xy:
            if lid not in missing_ids and lid not in locked_ids:
                if float(np.dot(foot_xy[lid], direction)) > 0.0:
                    _fwd_count += 1
        _fwd_ratio = _fwd_count / max(len(_balance_legs), 1)
        score += 0.25 * _fwd_ratio

        return score

    pos_score = _score_direction(major_axis)  # for report only
    neg_score = _score_direction(-major_axis)  # for report only

    # Evaluate all 4 candidate directions and pick the best, or use forced axis.
    if forced_axis is not None:
        # External caller (e.g. iterative optimizer) supplies the axis directly.
        final_axis = _unit(np.asarray(forced_axis[:2], dtype=float))
        best_score = _score_direction(final_axis)
    else:
        best_score = -1e18
        final_axis = major_axis.copy()
        for cand in candidate_axes:
            s = _score_direction(cand)
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
        # Combined: sqrt of geometry factors, clamped to [0.25, 0.78]
        # ↑ min 0.15→0.25: SWING_AMP=0.32×0.15=0.048 effective, too small to produce motion
        raw = math.sqrt(reach_ratio * range_ratio) * proj_mag
        per_leg_stride_amplitudes[lid] = float(np.clip(raw, 0.25, 0.78))

    # Classify active legs into three groups:
    #   group_a / group_b : alternating active tripods (good or neutral forward projection)
    #   group_c           : passive legs whose swing motion OPPOSES forward motion (proj < 0)
    #
    # Rationale: a leg with proj ≈ 0 (e.g. standard hexapod centerline legs) contributes
    # zero swing thrust but is NOT an obstacle — keeping it in group_a/b retains tripod
    # adaptation.stability.  Only legs with proj < −PASSIVE_THRESHOLD (swing that actively pushes the
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
            per_leg_stride_amplitudes[lid] = float(np.clip(base_amp * s_clamped, 0.20, 0.85))
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