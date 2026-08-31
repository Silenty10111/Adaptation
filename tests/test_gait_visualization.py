import json
from pathlib import Path

import numpy as np

from adaptation.gait import compute_adaptive_plan
from adaptation.gait_visualization import build_planned_gait_raster


STANDARD = (
    Path(__file__).resolve().parents[1]
    / "robot_assets" / "standard_hexapod" / "robot_description.json"
)


def _standard():
    return json.loads(STANDARD.read_text(encoding="utf-8"))


def test_planned_raster_is_finite_and_matches_leg_count():
    description = _standard()
    plan = compute_adaptive_plan(description, {"cpg": {"phase_strategy": "binary"}})
    raster = build_planned_gait_raster(description, plan, duration_s=2.0)
    values = np.asarray(raster["swing_joint_target_delta_rad"])
    states = np.asarray(raster["contact_state"])
    assert values.shape == states.shape
    assert values.shape[0] == 6
    assert np.all(np.isfinite(values))
    assert set(np.unique(states)) <= {0, 1, 2}
    assert raster["side_labels"].count("left") == 3
    assert raster["side_labels"].count("right") == 3


def test_binary_tripods_have_half_cycle_shifted_contact_sequences():
    description = _standard()
    plan = compute_adaptive_plan(description, {"cpg": {"phase_strategy": "binary"}})
    raster = build_planned_gait_raster(description, plan, duration_s=1.5)
    states = np.asarray(raster["contact_state"])
    ids = raster["leg_ids"]
    groups = plan["topology"]["groups"]
    row = {leg_id: ids.index(leg_id) for leg_id in ids}
    first_a = states[row[groups["group_a"][0]]]
    first_b = states[row[groups["group_b"][0]]]
    assert all(np.array_equal(states[row[leg_id]], first_a) for leg_id in groups["group_a"])
    assert all(np.array_equal(states[row[leg_id]], first_b) for leg_id in groups["group_b"])
    assert not np.array_equal(first_a, first_b)
    assert set(first_a) == {0, 1}
    assert set(first_b) == {0, 1}


def test_passive_leg_is_all_stance_with_zero_target_delta():
    description = _standard()
    plan = compute_adaptive_plan(
        description,
        {"locked_leg_ids": [1], "cpg": {"phase_strategy": "binary"}},
    )
    raster = build_planned_gait_raster(description, plan, duration_s=1.5)
    row = raster["leg_ids"].index(1)
    assert set(raster["contact_state"][row]) == {0}
    assert np.allclose(raster["swing_joint_target_delta_rad"][row], 0.0)
    assert raster["group_labels"][row] == "C/passive"
