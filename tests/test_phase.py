import json
import math
from pathlib import Path

import numpy as np
import pytest

from adaptation.gait import compute_adaptive_plan
from adaptation.phase import (
    ContactPhaseGateConfig,
    TWO_PI,
    balanced_geometry_wave_phase_offsets,
    blend_phase_offsets,
    circular_distance,
    circular_interpolate,
    circular_mean,
    contact_aware_swing_set,
    geometry_wave_phase_offsets,
    leg_phase_state,
    phase_state,
    resolve_duty_factors,
    resolve_phase_offsets,
    uniform_geometry_wave_phase_offsets,
    yaw_rate_error,
)


STANDARD = (
    Path(__file__).resolve().parents[1]
    / "robot_assets"
    / "standard_hexapod"
    / "robot_description.json"
)


def _standard_description():
    return json.loads(STANDARD.read_text(encoding="utf-8"))


def _foot_positions(description):
    return {
        int(link["leg_id"]): link["default_world_origin"][:2]
        for link in description["links"]
        if link.get("role") == "foot" and link.get("leg_id") is not None
    }


@pytest.mark.parametrize("duty, expected", [(0.60, 0.60), (0.50, 0.50)])
def test_sampled_stance_ratio_matches_duty_factor(duty, expected):
    samples = 20_000
    stance_count = sum(
        phase_state(TWO_PI * index / samples, duty).is_stance
        for index in range(samples)
    )
    assert stance_count / samples == pytest.approx(expected, abs=1.0 / samples)


def test_explicit_per_leg_offsets_win_over_legacy_groups():
    plan = {
        "cpg": {"phase_offsets": {"0": 1.25, "1": 0.25}},
        "topology": {"groups": {"group_a": [0], "group_b": [1]}},
    }
    assert resolve_phase_offsets(plan, [0, 1]) == pytest.approx({0: 1.25, 1: 0.25})


def test_missing_offsets_fall_back_to_binary_groups():
    old_plan = {
        "topology": {
            "groups": {"group_a": [2], "group_b": [5], "group_c": [8]},
        },
    }
    offsets = resolve_phase_offsets(old_plan, [2, 5])
    assert offsets[2] == 0.0
    assert offsets[5] == pytest.approx(math.pi)


def test_standard_hexapod_geometry_wave_is_tripod_equivalent():
    feet = _foot_positions(_standard_description())
    offsets = geometry_wave_phase_offsets(feet, [1.0, 0.0], wave_count=1.0)

    # On the +lateral side, front/middle/rear is 0/pi/0.  The other side
    # receives one global pi lag.  Circular distance accepts 0 == 2*pi.
    for leg_id, expected in {5: 0.0, 4: math.pi, 3: 0.0}.items():
        assert circular_distance(offsets[leg_id], expected) < 1e-10
    for positive, negative in ((5, 0), (4, 1), (3, 2)):
        assert circular_distance(offsets[negative], offsets[positive] + math.pi) < 1e-10


def _synthetic_feet(count):
    # IDs deliberately have no relationship to longitudinal rank.
    angles = np.linspace(0.0, TWO_PI, count, endpoint=False) + 0.17
    ids = [100 + count - index for index in range(count)]
    return {
        leg_id: [float(0.8 * math.cos(angle)), float(0.5 * math.sin(angle))]
        for leg_id, angle in zip(ids, angles)
    }


@pytest.mark.parametrize("count", [4, 6, 7, 10])
def test_arbitrary_leg_counts_have_finite_deterministic_offsets(count):
    feet = _synthetic_feet(count)
    first = geometry_wave_phase_offsets(feet, [0.9, 0.2], wave_count=1.25)
    second = geometry_wave_phase_offsets(feet, [0.9, 0.2], wave_count=1.25)
    assert first == second
    assert len(first) == count
    assert all(math.isfinite(value) and 0.0 <= value < TWO_PI for value in first.values())


def test_leg_id_iteration_order_does_not_change_geometry_offsets():
    feet = _synthetic_feet(7)
    reversed_feet = dict(reversed(list(feet.items())))
    normal = geometry_wave_phase_offsets(feet, [1.0, 0.0], wave_count=1.5)
    reversed_result = geometry_wave_phase_offsets(
        reversed_feet, [1.0, 0.0], wave_count=1.5
    )
    assert reversed_result == pytest.approx(normal)


def test_missing_leg_recomputes_chain_and_preserves_longitudinal_formula():
    feet = {
        10: [3.0, 1.0], 11: [2.0, 1.0], 12: [1.0, 1.0], 13: [0.0, 1.0],
        20: [3.0, -1.0], 21: [2.0, -1.0], 22: [1.0, -1.0], 23: [0.0, -1.0],
    }
    offsets = geometry_wave_phase_offsets(
        feet, [1.0, 0.0], missing_leg_ids=[11], wave_count=1.0
    )
    assert 11 not in offsets
    for leg_id, expected_s in ((10, 0.0), (12, 2.0 / 3.0), (13, 1.0)):
        assert circular_distance(offsets[leg_id], TWO_PI * expected_s) < 1e-10


def test_locked_leg_is_passive_and_its_coupling_edges_are_zero():
    plan = compute_adaptive_plan(
        _standard_description(),
        {"locked_leg_ids": [1], "cpg": {"phase_strategy": "geometry_wave"}},
    )
    groups = plan["topology"]["groups"]
    assert 1 in groups["group_c"]
    assert 1 not in plan["cpg"]["active_leg_ids"]
    assert "1" not in plan["cpg"]["phase_offsets"]

    order = plan["cpg"]["coupling_leg_ids"]
    row = order.index(1)
    matrix = np.asarray(plan["cpg"]["coupling_weights"])
    assert np.all(matrix[row, :] == 0.0)
    assert np.all(matrix[:, row] == 0.0)
    assert any(
        edge["leg_i"] == 1 or edge["leg_j"] == 1
        for edge in plan["topology"]["coupling_matrix_zeroed_edges"]
    )


def test_missing_leg_is_removed_from_phase_targets_and_zeroed_couplings():
    plan = compute_adaptive_plan(
        _standard_description(),
        {"missing_leg_ids": ["1"], "cpg": {"phase_strategy": "geometry_wave"}},
    )
    assert 1 not in plan["cpg"]["active_leg_ids"]
    assert "1" not in plan["cpg"]["phase_offsets"]
    assert "1" not in plan["planned_swings"]
    order = plan["cpg"]["coupling_leg_ids"]
    row = order.index(1)
    matrix = np.asarray(plan["cpg"]["coupling_weights"])
    assert np.all(matrix[row, :] == 0.0)
    assert np.all(matrix[:, row] == 0.0)


def test_circular_mean_interpolation_and_plan_switch_near_wrap():
    near_wrap = circular_mean([TWO_PI - 0.1, 0.1])
    assert circular_distance(near_wrap, 0.0) < 1e-10
    midpoint = circular_interpolate(TWO_PI - 0.2, 0.2, 0.5)
    assert circular_distance(midpoint, 0.0) < 1e-10
    blended = blend_phase_offsets({3: TWO_PI - 0.2}, {3: 0.2}, 0.5)
    assert circular_distance(blended[3], 0.0) < 1e-10


def test_periodic_target_is_continuous_at_liftoff_and_cycle_boundary():
    duty = 0.60
    epsilon = 1e-7
    joins = (
        (phase_state(TWO_PI * (duty - epsilon), duty),
         phase_state(TWO_PI * (duty + epsilon), duty)),
        (phase_state(TWO_PI * (1.0 - epsilon), duty),
         phase_state(TWO_PI * epsilon, duty)),
    )
    for before, after in joins:
        assert before.fore_aft == pytest.approx(after.fore_aft, abs=1e-9)
        assert before.lift == pytest.approx(after.lift, abs=1e-9)


def test_old_plan_and_per_leg_duty_still_execute_with_clipping_diagnostics():
    plan = {
        "cpg": {"duty_factor": 0.60, "per_leg_duty_factors": {"1": 0.95}},
        "topology": {"groups": {"group_a": [0], "group_b": [1]}},
    }
    offsets = resolve_phase_offsets(plan, [0, 1])
    duties, diagnostics = resolve_duty_factors(plan, [0, 1])
    assert duties == pytest.approx({0: 0.60, 1: 0.85})
    assert diagnostics and "leg_duty_factors[1] clipped" in diagnostics[0]
    assert leg_phase_state(0.0, 0, offsets, duties).is_stance
    legacy_b = leg_phase_state(0.0, 1, offsets, duties)
    assert legacy_b.normalized_phase == pytest.approx(0.5)
    assert legacy_b.duty_factor == pytest.approx(0.85)


def test_nonzero_yaw_reference_is_used_in_tracking_error():
    assert yaw_rate_error(0.7, 0.5) == pytest.approx(0.2)
    assert yaw_rate_error(0.5, 0.5) == pytest.approx(0.0)


def test_compute_plan_preserves_explicit_phase_offsets_and_duty_clipping():
    plan = compute_adaptive_plan(
        _standard_description(),
        {"cpg": {
            "phase_strategy": "binary",
            "phase_offsets": {"0": 0.37},
            "duty_factor": 0.2,
        }},
    )
    assert plan["cpg"]["phase_offsets"]["0"] == pytest.approx(0.37)
    assert plan["cpg"]["duty_factor"] == pytest.approx(0.35)
    assert plan["cpg"]["duty_factor_diagnostics"]


def test_contact_gate_delays_liftoff_when_measured_support_is_too_small():
    states = {
        0: phase_state(TWO_PI * 0.62, 0.60),
        1: phase_state(TWO_PI * 0.64, 0.60),
        2: phase_state(TWO_PI * 0.20, 0.60),
        3: phase_state(TWO_PI * 0.30, 0.60),
    }
    admitted = contact_aware_swing_set(
        states,
        actual_contact_leg_ids=[0, 1, 2],
        config=ContactPhaseGateConfig(
            minimum_support_count=3,
            maximum_simultaneous_swing=2,
        ),
    )
    assert admitted == set()


def test_contact_gate_admits_only_available_capacity_and_keeps_active_swing():
    early = phase_state(TWO_PI * 0.64, 0.60)
    late = phase_state(TWO_PI * 0.85, 0.60)
    states = {0: early, 1: early, 2: late, 3: phase_state(0.2, 0.60)}
    config = ContactPhaseGateConfig(
        minimum_support_count=3,
        maximum_simultaneous_swing=2,
        early_touchdown_progress=0.35,
    )
    admitted = contact_aware_swing_set(
        states,
        actual_contact_leg_ids=[0, 1, 2, 3],
        config=config,
    )
    assert len(admitted) == 1
    assert admitted <= {0, 1}

    # Lift-off admission does not mistake residual contact for touchdown; the
    # stateful executor requires a confirmed airborne sample first.
    carried = contact_aware_swing_set(
        states,
        actual_contact_leg_ids=[0, 1, 2, 3],
        previous_swing_leg_ids=[2],
        config=config,
    )
    assert 2 in carried


def test_uniform_geometry_wave_has_unique_evenly_spaced_offsets():
    feet = _synthetic_feet(12)
    offsets = uniform_geometry_wave_phase_offsets(feet, [1.0, 0.0])
    ordered = sorted(offsets.values())
    gaps = [
        (ordered[(index + 1) % len(ordered)] - ordered[index]) % TWO_PI
        for index in range(len(ordered))
    ]
    assert len(offsets) == 12
    assert gaps == pytest.approx([TWO_PI / 12] * 12)

    duty = 0.72
    maximum_swing = max(
        sum(
            not phase_state(base + offset, duty).is_stance
            for offset in offsets.values()
        )
        for base in np.linspace(0.0, TWO_PI, 1440, endpoint=False)
    )
    assert maximum_swing == 4


def test_balanced_geometry_wave_spreads_each_side_without_endpoint_duplicate():
    feet = {
        **{index: [2.0 - index, 1.0] for index in range(6)},
        **{6 + index: [2.0 - index, -1.0] for index in range(6)},
    }
    offsets = balanced_geometry_wave_phase_offsets(feet, [1.0, 0.0])
    for side in (range(6), range(6, 12)):
        values = sorted(offsets[index] for index in side)
        assert len(set(round(value, 10) for value in values)) == 6
    maximum_swing = max(
        sum(
            not phase_state(base + offset, 0.72).is_stance
            for offset in offsets.values()
        )
        for base in np.linspace(0.0, TWO_PI, 1440, endpoint=False)
    )
    assert maximum_swing <= 4
