#!/usr/bin/env python3
"""Static Stability Margin (SSM) computation via Centre of Gravity Projection Method (CGPM).

Algorithm reference:
  McGhee & Frank (1968), "On the stability properties of quadruped creeping gaits."

CGPM principle
--------------
1. Compute the robot's composite Centre of Mass (CoM) and project it onto the
   horizontal ground plane → P_xy.
2. Collect all foot contact positions and compute their convex hull → support
   polygon S with CCW-ordered vertices V_0 … V_{n-1}.
3. For each directed edge e_i = (V_i → V_{i+1 mod n}):

       d_i = cross(V_{i+1} - V_i,  P_xy - V_i) / |V_{i+1} - V_i|

   where the 2D scalar cross product is:

       cross(a, b) = a_x · b_y − a_y · b_x

4. The Static Stability Margin is:

       SSM = min_i(d_i)

   - SSM > 0  ⟹  P_xy is strictly inside every edge's half-plane  ⟹  stable.
   - SSM = 0  ⟹  P_xy lies on the boundary.
   - SSM < 0  ⟹  P_xy is outside the support polygon  ⟹  unstable.

All public functions operate on the `description` dict produced by
`generate_geometry.py`/`generate_urdf.py`.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


EPS = 1e-9


# ---------------------------------------------------------------------------
# Sub-function 1: CoM projection
# ---------------------------------------------------------------------------

def compute_projected_com_xy(description: Dict[str, object]) -> np.ndarray:
    """Return the XY projection of the robot's composite Centre of Mass.

    For every link *k* with mass m_k, world-frame CoM origin o_k, and local
    centre-of-mass offset cm_k the contribution is:

        weighted_sum += m_k * (o_k[:2] + cm_k[:2])

    Total CoM:

        P_xy = weighted_sum / Σ m_k

    Returns
    -------
    np.ndarray  shape (2,)  [x, y] in metres.
    """
    weighted_sum = np.zeros(2, dtype=float)
    total_mass = 0.0
    for link in description.get("links", []):
        mass_props = link.get("mass_properties", {})
        mass = float(mass_props.get("mass", 0.0))
        if mass <= 0.0:
            continue
        origin = np.asarray(
            link.get("default_world_origin", [0.0, 0.0, 0.0]), dtype=float
        )
        cm = np.asarray(
            mass_props.get("center_mass", [0.0, 0.0, 0.0]), dtype=float
        )
        world_cm = origin + cm
        weighted_sum += world_cm[:2] * mass
        total_mass += mass
    if total_mass <= EPS:
        return np.zeros(2, dtype=float)
    return weighted_sum / total_mass


# ---------------------------------------------------------------------------
# Sub-function 2: Support polygon (convex hull of feet)
# ---------------------------------------------------------------------------

def compute_support_polygon_xy(description: Dict[str, object]) -> np.ndarray:
    """Compute the CCW convex hull of all foot contact positions.

    Returns
    -------
    np.ndarray  shape (N, 2)  in CCW order.  N ≥ 3 if a proper hull exists;
    otherwise returns whatever points are available (can be < 3).
    """
    foot_pts: List[List[float]] = []
    for link in description.get("links", []):
        if link.get("role") != "foot":
            continue
        origin = np.asarray(
            link.get("default_world_origin", [0.0, 0.0, 0.0]), dtype=float
        )
        foot_pts.append(origin[:2].tolist())

    if len(foot_pts) < 3:
        return np.array(foot_pts, dtype=float) if foot_pts else np.zeros((0, 2), dtype=float)

    try:
        from shapely.geometry import MultiPoint
        from shapely.geometry import Polygon as ShapelyPolygon

        hull_geom = MultiPoint(foot_pts).convex_hull
        if isinstance(hull_geom, ShapelyPolygon):
            return np.array(hull_geom.exterior.coords[:-1], dtype=float)
        # Degenerate line or point — fall through to angle-sort fallback
    except ImportError:
        pass

    # Fallback: angle-sort around centroid (approximate CCW hull)
    pts = np.array(foot_pts, dtype=float)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(angles)]


# ---------------------------------------------------------------------------
# Sub-function 3: SSM calculation
# ---------------------------------------------------------------------------

def compute_ssm(polygon_xy: np.ndarray, com_xy: np.ndarray) -> float:
    """Compute the Static Stability Margin of *com_xy* w.r.t. *polygon_xy*.

    The polygon must be in CCW vertex order for the sign convention to be
    correct (positive inside, negative outside).

    For each directed edge from A_i to B_i:

        d_i = ( (B_i - A_i) × (P - A_i) ) / |B_i - A_i|

    where × is the 2D scalar cross product:

        cross(u, v) = u_x · v_y − u_y · v_x

    Returns
    -------
    float  SSM in metres.  Positive ⟹ stable, negative ⟹ unstable.
    """
    n = len(polygon_xy)
    if n < 3:
        return 0.0
    pts = np.asarray(polygon_xy, dtype=float)
    p = np.asarray(com_xy, dtype=float)

    min_signed = float("inf")
    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]
        edge = b - a
        edge_len = float(np.linalg.norm(edge))
        if edge_len < EPS:
            continue
        diff = p - a
        # 2D scalar cross product: edge × diff
        signed_dist = (edge[0] * diff[1] - edge[1] * diff[0]) / edge_len
        if signed_dist < min_signed:
            min_signed = signed_dist

    return float(min_signed) if np.isfinite(min_signed) else 0.0


# ---------------------------------------------------------------------------
# Sub-function 4: Full evaluation entry point
# ---------------------------------------------------------------------------

def evaluate_ssm(
    description: Dict[str, object],
    threshold: float = 0.0,
) -> Dict[str, object]:
    """Run a complete CGPM static stability check.

    Parameters
    ----------
    description : dict  Robot description as produced by generate_geometry.py.
    threshold   : float Minimum acceptable SSM in metres (default: 0.0 → any
                        positive margin passes).

    Returns
    -------
    dict with keys:
        com_xy            (list[float])  XY CoM projection.
        support_polygon_xy (list[list])  CCW hull vertices.
        ssm               (float)        Static Stability Margin (m).
        threshold         (float)        User-specified threshold.
        passed            (bool)         True iff SSM ≥ threshold.
    """
    com_xy = compute_projected_com_xy(description)
    polygon_xy = compute_support_polygon_xy(description)
    ssm = compute_ssm(polygon_xy, com_xy)
    return {
        "com_xy": com_xy.tolist(),
        "support_polygon_xy": polygon_xy.tolist(),
        "ssm": float(ssm),
        "threshold": float(threshold),
        "passed": bool(ssm >= threshold),
    }
