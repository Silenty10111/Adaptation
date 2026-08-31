import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adaptation.autotune import (
    generate_candidates,
    refine_axis_from_trajectory,
    score_metrics,
)


STANDARD = Path(__file__).resolve().parents[1] / "robot_assets" / "standard_hexapod" / "robot_description.json"


def test_candidates_include_two_controller_waveforms():
    description = json.loads(STANDARD.read_text())
    candidates = generate_candidates(description, max_candidates=4)
    assert candidates[0].plan["cpg"]["mode"] == "tripod"
    assert candidates[1].plan["cpg"]["mode"] == "legacy_sine"


def test_accepted_probe_always_wins():
    passed = {"passed": True, "forward_speed": 0.03}
    fast_but_failed = {"passed": False, "forward_speed": 10.0}
    assert score_metrics(passed) > score_metrics(fast_but_failed)


def test_candidates_keep_opposite_walking_directions():
    description = json.loads(STANDARD.read_text())
    candidates = generate_candidates(description, max_candidates=18)
    axes = [np.asarray(item.plan["final_forward_axis"]) for item in candidates]
    first = axes[0] / np.linalg.norm(axes[0])
    assert any(float(np.dot(first, axis / np.linalg.norm(axis))) < -0.999
               for axis in axes)


def test_axis_refinement_preserves_motor_plan_and_learns_course():
    plan = {
        "final_forward_axis": [1.0, 0.0],
        "drive_resultant_xy": [0.2, 0.0],
        "topology": {
            "groups": {"group_a": [0, 2], "group_b": [1, 3]},
            "per_leg_stride_amplitudes": {"0": 0.3, "1": 0.4},
        },
        "cpg": {"mode": "tripod", "phase_offsets": {"0": 0.1, "1": 2.2}},
    }
    trajectory = [[0.01 * i, 0.02 * i, 0.0, 0.4] for i in range(100)]

    refined = refine_axis_from_trajectory({}, plan, trajectory)

    np.testing.assert_allclose(refined["actuation_forward_axis"], [1.0, 0.0])
    np.testing.assert_allclose(
        refined["final_forward_axis"],
        np.asarray([1.0, 2.0]) / np.sqrt(5.0),
    )
    assert refined["topology"] == plan["topology"]
    assert refined["cpg"] == plan["cpg"]
    assert "actuation_forward_axis" not in plan


def test_axis_refinement_keeps_existing_actuation_axis():
    plan = {
        "final_forward_axis": [0.0, 1.0],
        "actuation_forward_axis": [1.0, 0.0],
        "drive_resultant_xy": [0.0, 0.2],
    }
    trajectory = [[0.0, -0.01 * i] for i in range(20)]

    refined = refine_axis_from_trajectory({}, plan, trajectory)

    np.testing.assert_allclose(refined["actuation_forward_axis"], [1.0, 0.0])
    np.testing.assert_allclose(refined["final_forward_axis"], [0.0, -1.0])
