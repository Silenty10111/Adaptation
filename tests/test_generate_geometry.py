import json
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Point

from scripts.generate_geometry import (
    DEFAULT_BODY_LENGTH,
    DEFAULT_BODY_WIDTH,
    DEFAULT_LEG_POOL,
    DEFAULT_LEG_PROBABILITIES,
    MAX_EXPLICIT_LEGS,
    coplanarize_lower_leg_vectors,
    compute_mount_points,
    create_irregular_trunk_polygon,
    create_serial_rigid_trunk_polygon,
)
from scripts.generate_urdf import validate_static_stability_before_export


def test_irregular_body_dimensions_are_final_envelope():
    polygon = create_irregular_trunk_polygon(0.72, 0.44, np.random.default_rng(7))
    min_x, min_y, max_x, max_y = polygon.bounds
    assert max_x - min_x == pytest.approx(0.72)
    assert max_y - min_y == pytest.approx(0.44)
    assert 0.5 * (max_x + min_x) == pytest.approx(0.0, abs=1e-12)
    assert 0.5 * (max_y + min_y) == pytest.approx(0.0, abs=1e-12)


def test_default_profile_is_elongated_and_many_legged():
    assert DEFAULT_BODY_LENGTH / DEFAULT_BODY_WIDTH >= 2.5
    assert DEFAULT_LEG_POOL.tolist() == list(range(8, MAX_EXPLICIT_LEGS + 1))
    assert DEFAULT_LEG_PROBABILITIES.sum() == pytest.approx(1.0)
    assert float(np.dot(DEFAULT_LEG_POOL, DEFAULT_LEG_PROBABILITIES)) >= 10.0


@pytest.mark.parametrize("leg_count", [10, 14])
def test_random_mount_spacing_includes_perimeter_wrap_gap(leg_count):
    polygon = create_irregular_trunk_polygon(0.72, 0.44, np.random.default_rng(19))
    mounts = compute_mount_points(
        polygon, leg_count, "random", np.random.default_rng(23), upper_length=0.28
    )
    boundary = polygon.exterior
    distances = sorted(
        float(boundary.project(Point(mount["point_xy"]))) for mount in mounts
    )
    circular_gaps = [
        distances[index + 1] - distances[index]
        for index in range(len(distances) - 1)
    ]
    circular_gaps.append(boundary.length - distances[-1] + distances[0])
    expected_minimum = min(0.28, 0.80 * boundary.length / leg_count)
    assert min(circular_gaps) >= expected_minimum - 1e-9


def test_serial_rigid_outline_is_valid_and_has_requested_envelope():
    polygon = create_serial_rigid_trunk_polygon(0.60, 0.40)
    assert polygon.is_valid
    assert polygon.bounds == pytest.approx((-0.30, -0.20, 0.30, 0.20))


def test_urdf_export_gate_rejects_failed_ssm_by_default():
    description_path = (
        Path(__file__).resolve().parents[1]
        / "robot_assets" / "standard_hexapod" / "robot_description.json"
    )
    description = json.loads(description_path.read_text(encoding="utf-8"))
    with pytest.raises(SystemExit):
        validate_static_stability_before_export(description, threshold=10.0)


def test_coplanar_leg_adjustment_preserves_lengths_and_levels_feet():
    knees = [
        [0.2, 0.1, -0.11],
        [-0.2, 0.1, -0.09],
        [0.0, -0.1, -0.13],
    ]
    lowers = [
        [0.05, 0.01, -0.2956],
        [-0.04, 0.02, -0.2966],
        [0.03, -0.02, -0.2979],
    ]
    original_lengths = [np.linalg.norm(value) for value in lowers]
    adjusted, target_z = coplanarize_lower_leg_vectors(knees, lowers)
    foot_z = [knee[2] + lower[2] for knee, lower in zip(knees, adjusted)]
    assert foot_z == pytest.approx([target_z] * len(knees), abs=1e-12)
    assert [np.linalg.norm(value) for value in adjusted] == pytest.approx(
        original_lengths, abs=1e-12
    )
