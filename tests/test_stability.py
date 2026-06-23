#!/usr/bin/env python3
"""Tests for the stability module (SSM computation)."""

import sys
from pathlib import Path

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from stability import (
    _signed_polygon_area,
    _ensure_ccw,
    compute_projected_com_xy,
    compute_support_polygon_xy,
    compute_ssm,
    evaluate_ssm,
)


# ── _signed_polygon_area ────────────────────────────────────────────────────

class TestSignedPolygonArea:
    def test_square_ccw(self):
        pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        assert _signed_polygon_area(pts) > 0

    def test_square_cw(self):
        pts = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=float)
        assert _signed_polygon_area(pts) < 0

    def test_triangle(self):
        pts = np.array([[0, 0], [2, 0], [0, 2]], dtype=float)
        assert abs(_signed_polygon_area(pts) - 2.0) < 1e-9

    def test_degenerate(self):
        assert _signed_polygon_area(np.array([[0, 0], [1, 1]], dtype=float)) == 0.0
        assert _signed_polygon_area(np.array([], dtype=float).reshape(0, 2)) == 0.0


# ── _ensure_ccw ─────────────────────────────────────────────────────────────

class TestEnsureCCW:
    def test_ccw_unchanged(self):
        ccw = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        result = _ensure_ccw(ccw)
        assert np.allclose(result[0], ccw[0])

    def test_cw_reversed(self):
        cw = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=float)
        result = _ensure_ccw(cw)
        assert _signed_polygon_area(result) > 0

    def test_few_points(self):
        pts = np.array([[0, 0], [1, 1]], dtype=float)
        result = _ensure_ccw(pts)
        assert np.array_equal(result, pts)


# ── compute_ssm ─────────────────────────────────────────────────────────────

class TestComputeSSM:
    def test_inside_square(self):
        poly = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        com = np.array([0.5, 0.5], dtype=float)
        ssm = compute_ssm(poly, com)
        assert ssm > 0.0  # inside → positive
        assert abs(ssm - 0.5) < 1e-9  # distance to each edge = 0.5

    def test_outside(self):
        poly = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        com = np.array([1.5, 0.5], dtype=float)
        ssm = compute_ssm(poly, com)
        assert ssm < 0.0  # outside → negative

    def test_on_edge(self):
        poly = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        com = np.array([0.5, 0.0], dtype=float)
        ssm = compute_ssm(poly, com)
        assert abs(ssm) < 1e-9

    def test_few_vertices(self):
        assert compute_ssm(np.array([[0, 0], [1, 1]], dtype=float),
                          np.array([0.5, 0.5])) == 0.0


# ── compute_projected_com_xy ────────────────────────────────────────────────

class TestComputeProjectedComXY:
    def test_simple_robot(self):
        desc = {
            "links": [
                {
                    "name": "body",
                    "role": "trunk",
                    "default_world_origin": [0, 0, 0],
                    "mass_properties": {
                        "mass": 1.0,
                        "center_mass": [0, 0, 0],
                    },
                },
                {
                    "name": "leg_0_foot",
                    "role": "foot",
                    "default_world_origin": [1, 0, 0],
                    "mass_properties": {
                        "mass": 0.1,
                        "center_mass": [0, 0, 0],
                    },
                },
            ],
        }
        com = compute_projected_com_xy(desc)
        # CoM = (1.0*[0,0] + 0.1*[1,0]) / 1.1 ≈ [0.0909, 0.0]
        assert abs(com[0] - 0.1 / 1.1) < 1e-6
        assert abs(com[1]) < 1e-9

    def test_empty_robot(self):
        desc = {"links": []}
        com = compute_projected_com_xy(desc)
        assert np.allclose(com, [0, 0])


# ── evaluate_ssm ────────────────────────────────────────────────────────────

class TestEvaluateSSM:
    def test_full_evaluation(self):
        desc = {
            "links": [
                {"name": "body", "role": "trunk",
                 "default_world_origin": [0, 0, 0],
                 "mass_properties": {"mass": 2.0, "center_mass": [0, 0, 0]}},
                {"name": "f1", "role": "foot",
                 "default_world_origin": [0.5, 0.5, 0],
                 "mass_properties": {"mass": 0.1, "center_mass": [0, 0, 0]}},
                {"name": "f2", "role": "foot",
                 "default_world_origin": [-0.5, 0.5, 0],
                 "mass_properties": {"mass": 0.1, "center_mass": [0, 0, 0]}},
                {"name": "f3", "role": "foot",
                 "default_world_origin": [-0.5, -0.5, 0],
                 "mass_properties": {"mass": 0.1, "center_mass": [0, 0, 0]}},
                {"name": "f4", "role": "foot",
                 "default_world_origin": [0.5, -0.5, 0],
                 "mass_properties": {"mass": 0.1, "center_mass": [0, 0, 0]}},
            ],
        }
        result = evaluate_ssm(desc)
        assert "ssm" in result
        assert "passed" in result
        assert result["passed"]  # CoM at origin, feet form a square around it
        assert result["ssm"] > 0.0

    def test_unstable(self):
        # All feet on one side → CoM outside support polygon
        desc = {
            "links": [
                {"name": "body", "role": "trunk",
                 "default_world_origin": [0, 0, 0],
                 "mass_properties": {"mass": 5.0, "center_mass": [0, 0, 0]}},
                {"name": "f1", "role": "foot",
                 "default_world_origin": [1.0, 0.0, 0],
                 "mass_properties": {"mass": 0.1, "center_mass": [0, 0, 0]}},
                {"name": "f2", "role": "foot",
                 "default_world_origin": [1.5, 0.1, 0],
                 "mass_properties": {"mass": 0.1, "center_mass": [0, 0, 0]}},
                {"name": "f3", "role": "foot",
                 "default_world_origin": [1.5, 0.3, 0],
                 "mass_properties": {"mass": 0.1, "center_mass": [0, 0, 0]}},
            ],
        }
        result = evaluate_ssm(desc)
        assert not result["passed"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
