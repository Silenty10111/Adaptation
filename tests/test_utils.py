#!/usr/bin/env python3
"""Tests for utils.py shared utility functions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from utils import (
    ratio_to_joint,
    smoothstep,
    quat_to_euler,
    foot_xy_map,
    compute_metrics,
)


# ── ratio_to_joint ──────────────────────────────────────────────────────────

class TestRatioToJoint:
    def test_midpoint(self):
        assert ratio_to_joint(0.0, 10.0, 0.5) == 5.0

    def test_endpoints(self):
        assert ratio_to_joint(-1.0, 1.0, 0.0) == -1.0
        assert ratio_to_joint(-1.0, 1.0, 1.0) == 1.0

    def test_clamped(self):
        assert ratio_to_joint(0.0, 100.0, -0.5) == 0.0
        assert ratio_to_joint(0.0, 100.0, 1.5) == 100.0

    def test_float_precision(self):
        assert abs(ratio_to_joint(0.0, 1.0, 0.333333) - 0.333333) < 1e-6


# ── smoothstep ──────────────────────────────────────────────────────────────

class TestSmoothstep:
    def test_boundaries(self):
        assert smoothstep(0.0, 1.0, 0.0) == 0.0
        assert smoothstep(0.0, 1.0, 1.0) == 1.0

    def test_clamped(self):
        assert smoothstep(0.0, 1.0, -1.0) == 0.0
        assert smoothstep(0.0, 1.0, 2.0) == 1.0

    def test_midpoint(self):
        assert smoothstep(0.0, 1.0, 0.5) == 0.5  # Hermite at t=0.5

    def test_monotonic(self):
        vals = [smoothstep(0.0, 1.0, t) for t in np.linspace(0, 1, 20)]
        assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))


# ── quat_to_euler ───────────────────────────────────────────────────────────

class TestQuatToEuler:
    def test_identity(self):
        r, p, y = quat_to_euler(1.0, 0.0, 0.0, 0.0)
        assert r == 0.0 and p == 0.0 and y == 0.0

    def test_yaw_90(self):
        # Rotation of 90° around Z: q = (cos45, 0, 0, sin45)
        import math
        half = math.pi / 4
        r, p, y = quat_to_euler(math.cos(half), 0.0, 0.0, math.sin(half))
        assert abs(p) < 1e-9
        assert abs(r) < 1e-9
        assert abs(y - math.pi / 2) < 1e-9


# ── foot_xy_map ─────────────────────────────────────────────────────────────

class TestFootXYMap:
    def test_extracts_feet(self):
        desc = {
            "links": [
                {"name": "body", "role": "trunk", "leg_id": None,
                 "default_world_origin": [0, 0, 0]},
                {"name": "leg_0_foot", "role": "foot", "leg_id": 0,
                 "default_world_origin": [0.3, 0.1, 0]},
                {"name": "leg_1_foot", "role": "foot", "leg_id": 1,
                 "default_world_origin": [-0.3, 0.1, 0]},
                {"name": "leg_0_swing", "role": "swing", "leg_id": 0,
                 "default_world_origin": [0.1, 0.1, 0]},
            ],
        }
        result = foot_xy_map(desc)
        assert len(result) == 2
        assert 0 in result and 1 in result
        assert np.allclose(result[0], [0.3, 0.1])
        assert np.allclose(result[1], [-0.3, 0.1])

    def test_empty(self):
        assert foot_xy_map({"links": []}) == {}


# ── compute_metrics ─────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_straight_forward(self):
        trail = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
        metrics = compute_metrics(trail, [1.0, 0.0], None, n_steps=120)
        assert metrics["fwd_vel"] > 0.5
        assert abs(metrics["lat_drift"]) < 1e-9

    def test_lateral_drift(self):
        trail = [[0.0, 0.0], [1.0, 0.5]]
        metrics = compute_metrics(trail, [1.0, 0.0], None, n_steps=60)
        assert metrics["lat_drift"] > 0.4

    def test_with_yaw_stats(self):
        trail = [[0.0, 0.0], [0.5, 0.0]]
        metrics = compute_metrics(trail, [1.0, 0.0],
                                  {"yaw_rate_mean": 0.05}, n_steps=60)
        assert metrics["yaw_abs"] == 0.05

    def test_short_trail(self):
        trail = [[0.0, 0.0]]
        metrics = compute_metrics(trail, [1.0, 0.0], None, n_steps=60)
        assert metrics["fwd_vel"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
