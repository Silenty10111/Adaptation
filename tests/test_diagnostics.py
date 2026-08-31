import math

import pytest

from adaptation.diagnostics import (
    EpisodeDiagnosticConfig,
    EpisodeDiagnosticAccumulator,
    EpisodeStepTelemetry,
    diagnose_trajectory_only,
)


def _accumulator():
    return EpisodeDiagnosticAccumulator(
        [1.0, 0.0], active_leg_count=6,
        config=EpisodeDiagnosticConfig(contact_failure_persistence_s=0.0),
    )


def _step(time_s, x, y=0.0, roll=0.0, pitch=0.0, actual=None):
    return EpisodeStepTelemetry(
        time_s=time_s,
        body_position=[x, y, 0.5],
        body_rpy=[roll, pitch, 0.0],
        commanded_stance_leg_ids=[0, 2, 4],
        actual_contact_leg_ids=actual,
    )


@pytest.mark.parametrize(
    "kind, steps, expected",
    [
        ("roll", [_step(0.0, 0.0), _step(1.0, 0.1, roll=math.radians(50))], "fall_roll"),
        ("pitch", [_step(0.0, 0.0), _step(1.0, 0.1, pitch=math.radians(50))], "fall_pitch"),
        ("reverse", [_step(0.0, 0.0), _step(1.0, -0.1)], "reverse_motion"),
        ("drift", [_step(0.0, 0.0), _step(1.0, 0.1, y=0.1)], "excessive_lateral_drift"),
        ("contact", [_step(0.0, 0.0, actual=[])], "insufficient_contact"),
    ],
)
def test_failure_classifier_distinguishes_root_causes(kind, steps, expected):
    accumulator = _accumulator()
    for telemetry in steps:
        accumulator.update(telemetry)
    diagnostic = accumulator.finalize()
    assert diagnostic.failure_reason == expected
    assert diagnostic.first_failure_time_s is not None


def test_first_failure_is_preserved_over_later_terminal_failure():
    accumulator = _accumulator()
    accumulator.update(_step(0.0, 0.0, actual=[]))
    accumulator.update(_step(1.0, -0.2, actual=[0, 2, 4]))
    diagnostic = accumulator.finalize()
    assert diagnostic.failure_reason == "insufficient_contact"
    assert diagnostic.first_failure_time_s == 0.0
    assert any(event["reason"] == "reverse_motion" for event in diagnostic.failure_events)


def test_unmeasured_values_are_null_with_reasons_not_fabricated_numbers():
    accumulator = _accumulator()
    accumulator.update(_step(0.0, 0.0))
    accumulator.update(_step(1.0, 0.1))
    diagnostic = accumulator.finalize().to_dict()
    for field in (
        "actual_contact_leg_ids", "minimum_actual_contact_count",
        "contact_mismatch_ratio", "foot_slip_ratio",
        "leg_body_collision_count", "leg_leg_collision_count",
        "peak_joint_torque_ratio", "ssm_min", "ssm_p05",
    ):
        assert diagnostic[field] is None
        assert field in diagnostic["measurement_unavailable_reasons"]


def test_single_transient_support_switch_is_not_a_root_failure():
    accumulator = EpisodeDiagnosticAccumulator([1.0, 0.0], active_leg_count=6)
    accumulator.update(_step(0.0, 0.0, actual=[0, 2]))
    accumulator.update(_step(0.05, 0.01, actual=[0, 2, 4]))
    accumulator.update(_step(1.0, 0.10, actual=[0, 2, 4]))
    assert accumulator.finalize().failure_reason is None


def test_persistent_contact_failure_reports_the_onset_time():
    accumulator = EpisodeDiagnosticAccumulator([1.0, 0.0], active_leg_count=6)
    accumulator.update(_step(0.0, 0.0, actual=[]))
    accumulator.update(_step(0.12, 0.01, actual=[]))
    accumulator.update(_step(1.0, 0.10, actual=[0, 2, 4]))
    diagnostic = accumulator.finalize()
    assert diagnostic.failure_reason == "insufficient_contact"
    assert diagnostic.first_failure_time_s == 0.0


def test_trajectory_only_does_not_invent_attitude_height_or_contacts():
    diagnostic = diagnose_trajectory_only([[0.0, 0.0], [0.1, 0.0]], [1.0, 0.0])
    for field in ("yaw_tracking_rmse", "roll_rmse", "pitch_rmse", "min_body_height"):
        assert diagnostic[field] is None
        assert field in diagnostic["measurement_unavailable_reasons"]
