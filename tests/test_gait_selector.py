import json
import math
from pathlib import Path

import numpy as np
import pytest

from adaptation.gait_selector import (
    CandidateSearchConfig,
    GaitCandidate,
    GaitFeasibilityConfig,
    blend_and_validate_phase_switch,
    candidate_from_plan,
    evaluate_gait_candidate,
    generate_gait_candidates,
    select_gait_candidate,
)
from adaptation.gait import compute_adaptive_plan
from adaptation.phase import (
    binary_phase_offsets,
    circular_distance,
    phase_state,
    yaw_rate_error,
)


STANDARD = (
    Path(__file__).resolve().parents[1]
    / "robot_assets" / "standard_hexapod" / "robot_description.json"
)
GROUPS = {"group_a": [0, 4, 2], "group_b": [5, 1, 3], "group_c": []}


def _standard_feet():
    description = json.loads(STANDARD.read_text(encoding="utf-8"))
    return {
        int(link["leg_id"]): link["default_world_origin"][:2]
        for link in description["links"]
        if link.get("role") == "foot"
    }


def _binary_candidate(duty=0.5):
    offsets = binary_phase_offsets(GROUPS, range(6))
    return GaitCandidate(
        "binary", "binary", offsets, {leg_id: duty for leg_id in range(6)}
    )


def _config(**overrides):
    values = dict(
        samples_per_cycle=360,
        minimum_ssm=0.0,
        minimum_stance_count=3,
        allow_dynamic_support=False,
        maximum_phase_jump=math.pi,
        minimum_adjacent_swing_spacing=0.01,
    )
    values.update(overrides)
    return GaitFeasibilityConfig(**values)


def test_feasibility_uses_same_q_less_than_duty_rule_as_executor():
    candidate = _binary_candidate(0.60)
    result = evaluate_gait_candidate(
        active_leg_ids=range(6), foot_positions=_standard_feet(),
        com_xy=[0.0, 0.0], forward_axis=[1.0, 0.0],
        candidate=candidate, config=_config(samples_per_cycle=100),
    )
    expected = sorted(
        leg_id for leg_id in range(6)
        if phase_state(candidate.phase_offsets[leg_id], 0.60).is_stance
    )
    assert result.initial_stance_leg_ids == expected


def test_standard_tripod_passes_full_cycle_check():
    result = evaluate_gait_candidate(
        active_leg_ids=range(6), foot_positions=_standard_feet(),
        com_xy=[0.0, 0.0], forward_axis=[1.0, 0.0],
        candidate=_binary_candidate(), config=_config(),
    )
    assert result.feasible
    assert result.minimum_stance_count == 3
    assert result.maximum_swing_count == 3
    assert result.ssm_min > 0.0


def test_all_legs_swinging_together_is_rejected():
    candidate = GaitCandidate(
        "synchronized", "test", {leg_id: 0.0 for leg_id in range(6)},
        {leg_id: 0.5 for leg_id in range(6)},
    )
    result = evaluate_gait_candidate(
        active_leg_ids=range(6), foot_positions=_standard_feet(),
        com_xy=[0.0, 0.0], forward_axis=[1.0, 0.0],
        candidate=candidate, config=_config(),
    )
    assert not result.feasible
    assert result.maximum_swing_count == 6
    assert "degenerate_support_polygon" in result.rejection_reasons


def test_ssm_below_configured_threshold_is_rejected():
    result = evaluate_gait_candidate(
        active_leg_ids=range(6), foot_positions=_standard_feet(),
        com_xy=[0.0, 0.0], forward_axis=[1.0, 0.0],
        candidate=_binary_candidate(), config=_config(minimum_ssm=0.5),
    )
    assert not result.feasible
    assert "ssm_below_threshold" in result.rejection_reasons


def test_locked_leg_cannot_be_active():
    result = evaluate_gait_candidate(
        active_leg_ids=range(6), foot_positions=_standard_feet(),
        com_xy=[0.0, 0.0], forward_axis=[1.0, 0.0],
        candidate=_binary_candidate(), locked_leg_ids=[1], config=_config(),
    )
    assert not result.feasible
    assert "locked_leg_active" in result.rejection_reasons


def test_missing_leg_is_not_accessed():
    result = evaluate_gait_candidate(
        active_leg_ids=range(6), foot_positions=_standard_feet(),
        com_xy=[0.0, 0.0], forward_axis=[1.0, 0.0],
        candidate=_binary_candidate(), missing_leg_ids=[1], config=_config(),
    )
    assert not result.feasible
    assert "missing_leg_access" in result.rejection_reasons


def _synthetic_feet(count):
    angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False) + 0.11
    return {
        100 + count - index: [0.8 * math.cos(angle), 0.5 * math.sin(angle)]
        for index, angle in enumerate(angles)
    }


@pytest.mark.parametrize("count", [4, 6, 7, 10])
def test_arbitrary_leg_counts_generate_deterministic_candidates(count):
    feet = _synthetic_feet(count)
    ids = sorted(feet)
    groups = {"group_a": ids[::2], "group_b": ids[1::2], "group_c": []}
    search = CandidateSearchConfig(
        wave_counts=(1.0,), global_phase_origins=(0.0,),
        lateral_phase_offsets=(math.pi,), duty_factors=(0.6,),
    )
    first = generate_gait_candidates(
        active_leg_ids=ids, foot_positions=feet, forward_axis=[1.0, 0.0],
        groups=groups, search=search,
    )
    second = generate_gait_candidates(
        active_leg_ids=reversed(ids), foot_positions=dict(reversed(list(feet.items()))),
        forward_axis=[1.0, 0.0], groups=groups, search=search,
    )
    assert [candidate.name for candidate in first] == [candidate.name for candidate in second]
    for left, right in zip(first, second):
        assert right.phase_offsets == pytest.approx(left.phase_offsets)


def test_global_phase_translation_changes_initial_support_timing():
    feet = _standard_feet()
    search = CandidateSearchConfig(
        wave_counts=(1.0,), global_phase_origins=(0.0, math.pi),
        lateral_phase_offsets=(math.pi,), duty_factors=(0.5,),
    )
    candidates = generate_gait_candidates(
        active_leg_ids=range(6), foot_positions=feet,
        forward_axis=[1.0, 0.0], groups=GROUPS, search=search,
    )
    geometry = [candidate for candidate in candidates if candidate.strategy == "geometry_wave"]
    results = [evaluate_gait_candidate(
        active_leg_ids=range(6), foot_positions=feet, com_xy=[0.0, 0.0],
        forward_axis=[1.0, 0.0], candidate=candidate, config=_config(),
    ) for candidate in geometry]
    assert results[0].initial_stance_leg_ids != results[1].initial_stance_leg_ids


def test_phase_switch_near_wrap_uses_shortest_arc_and_remains_finite():
    previous = GaitCandidate("old", "binary", {0: 2 * math.pi - 0.2}, {0: 0.6})
    target = GaitCandidate("new", "geometry_wave", {0: 0.2}, {0: 0.7})
    candidate, result = blend_and_validate_phase_switch(
        previous=previous, target=target, alpha=0.5, active_leg_ids=[0],
        foot_positions={0: [0.5, 0.3], 1: [-0.5, -0.3], 2: [0.5, -0.3]},
        com_xy=[0.0, 0.0], forward_axis=[1.0, 0.0],
        passive_leg_ids=[1, 2],
        config=_config(minimum_stance_count=1, allow_dynamic_support=True, minimum_ssm=-1.0),
    )
    assert circular_distance(candidate.phase_offsets[0], 0.0) < 1e-10
    assert math.isfinite(result.phase_jump_cost)


def test_all_candidates_failing_returns_checked_all_stance_stop():
    search = CandidateSearchConfig(
        wave_counts=(1.0,), global_phase_origins=(0.0,),
        lateral_phase_offsets=(math.pi,), duty_factors=(0.5,),
    )
    selected = select_gait_candidate(
        active_leg_ids=range(6), foot_positions=_standard_feet(),
        com_xy=[10.0, 10.0], forward_axis=[1.0, 0.0], groups=GROUPS,
        feasibility_config=_config(minimum_ssm=0.1), search_config=search,
    )
    assert selected.selection_mode == "all_stance_stop_fallback"
    assert selected.candidate.strategy == "stop"
    assert selected.candidate.phase_offsets == {}
    assert selected.feasibility.rejection_reasons


def test_old_gait_plan_resolves_to_candidate_and_yaw_reference_still_applies():
    old_plan = {"topology": {"groups": GROUPS}}
    candidate = candidate_from_plan(old_plan, range(6))
    assert candidate.phase_offsets[0] == 0.0
    assert candidate.phase_offsets[1] == pytest.approx(math.pi)
    assert candidate.duty_factors[0] == pytest.approx(0.60)
    assert yaw_rate_error(0.8, 0.5) == pytest.approx(0.3)


def test_adaptive_plan_exposes_configurable_checked_phase_transition():
    description = json.loads(STANDARD.read_text(encoding="utf-8"))
    state = {
        "cpg": {
            "phase_strategy": "adaptive_wave",
            "phase_offsets": {str(leg_id): 0.0 for leg_id in range(6)},
            "selector": {
                "samples_per_cycle": 72,
                "transition_cycles": 4,
                "transition_cycle_index": 1,
            },
            "candidate_search": {
                "wave_counts": [1.0],
                "global_phase_origins": [0.0, math.pi],
                "lateral_phase_offsets": [math.pi],
                "duty_factors": [0.5, 0.6],
            },
        }
    }
    plan = compute_adaptive_plan(description, state)
    transition = plan["cpg"]["selection_diagnostics"]["phase_transition"]
    assert transition is not None
    assert transition["requested_alpha"] == pytest.approx(0.25)
    assert isinstance(transition["feasible"], bool)
