"""Task 9 record, seed and aggregate tests (hardware-free)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from sim2real.evaluation.records import (
    TERMINATION_REASONS,
    EvalEpisodeRecord,
    EvalRecordError,
    EvalStep,
    aggregate_records,
    derive_eval_seeds,
    task_completion,
    wilson_interval,
)


def make_step(index: int = 0, *, collision: bool = False) -> EvalStep:
    return EvalStep(
        stamp_ns=index * 50_000_000,
        step_index=index,
        joint_positions=np.zeros(7),
        joint_velocities=np.zeros(7),
        ee_pose=np.eye(4),
        action=np.zeros(8),
        gripper_position=0.0,
        unintended_collision=collision,
    )


def make_record(*, reason: str = "TIMEOUT", steps: int = 3) -> EvalEpisodeRecord:
    return EvalEpisodeRecord(
        episode_id="run_ep0000",
        eval_run_id="run",
        seed=20000,
        episode_index=0,
        randomization_profile="default",
        randomization={"nominal": False, "marker": 1},
        initial_object_pose=(0.45, -0.12, 0.0),
        friction_sliding=0.9,
        object_mass=0.07,
        light_intensity=1.0,
        color_temperature_k=6500.0,
        camera_jitter_norm=0.01,
        task_instruction="pick up the red cube and place it in the left zone",
        success=reason == "SUCCESS",
        task_completion=0.5,
        collision_count=1,
        termination_reason=reason,
        policy_id="zero",
        checkpoint_hash="none",
        config_hash="abc",
        git_commit="deadbeef",
        git_dirty=False,
        n_action_steps=10,
        renderer="fake-analytic",
        steps=[make_step(index=i) for i in range(steps)],
    )


def test_record_required_fields_and_schema_version():
    record = make_record()
    payload = record.to_dict()
    for field in (
        "episode_id",
        "eval_run_id",
        "seed",
        "randomization",
        "initial_object_pose",
        "friction_sliding",
        "object_mass",
        "light_intensity",
        "color_temperature_k",
        "camera_jitter_norm",
        "task_instruction",
        "success",
        "task_completion",
        "collision_count",
        "termination_reason",
        "policy_id",
        "checkpoint_hash",
        "config_hash",
        "git_commit",
        "git_dirty",
        "n_action_steps",
        "renderer",
        "steps",
    ):
        assert field in payload
    assert payload["schema_version"] == 1


def test_record_json_round_trip_is_identical():
    record = make_record()
    payload = json.loads(record.to_json())
    restored = EvalEpisodeRecord.from_json(record.to_json())
    assert restored.to_dict() == payload
    assert restored.to_json() == record.to_json()


def test_invalid_termination_reason_is_rejected():
    with pytest.raises(EvalRecordError):
        make_record(reason="NOT_A_REASON")
    assert "SUCCESS" in TERMINATION_REASONS


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nan_and_infinite_values_are_rejected(bad: float):
    with pytest.raises(EvalRecordError):
        EvalEpisodeRecord(
            episode_id="run_ep0000",
            eval_run_id="run",
            seed=20000,
            episode_index=0,
            randomization_profile="default",
            randomization={},
            initial_object_pose=(bad, 0.0, 0.0),
            friction_sliding=0.9,
            object_mass=0.07,
            light_intensity=1.0,
            color_temperature_k=6500.0,
            camera_jitter_norm=0.01,
            task_instruction="x",
            success=False,
            task_completion=0.5,
            collision_count=0,
            termination_reason="TIMEOUT",
            policy_id="zero",
            checkpoint_hash="none",
            config_hash="abc",
            git_commit="deadbeef",
            git_dirty=False,
            n_action_steps=10,
            renderer="fake-analytic",
        )


def test_success_flag_must_match_termination_reason():
    record = make_record(reason="TIMEOUT")
    record.success = True
    with pytest.raises(EvalRecordError):
        record.validate()


def test_derived_seed_lists_are_stable_and_distinct():
    first = derive_eval_seeds(20000, 8)
    assert first == derive_eval_seeds(20000, 8)
    assert first != derive_eval_seeds(20001, 8)
    assert len(set(first)) == len(first)
    assert all(0 <= seed < 2**31 for seed in first)


def test_wilson_interval_matches_known_numeric_case():
    low, high = wilson_interval(5, 10)
    assert low == pytest.approx(0.2366, abs=1.0e-3)
    assert high == pytest.approx(0.7634, abs=1.0e-3)
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_aggregates_counts_and_edge_cases():
    records = [make_record(reason="SUCCESS"), make_record(reason="TIMEOUT")]
    aggregate = aggregate_records(records)
    assert aggregate["num_episodes"] == 2
    assert aggregate["num_success"] == 1
    assert aggregate["success_rate"] == 0.5
    assert aggregate["termination_counts"] == {"SUCCESS": 1, "TIMEOUT": 1}
    assert aggregate["wilson_95"][0] < 0.5 < aggregate["wilson_95"][1]

    empty = aggregate_records([])
    assert empty["num_episodes"] == 0
    assert empty["success_rate"] == 0.0
    assert empty["wilson_95"] == [0.0, 0.0]

    all_success = aggregate_records([make_record(reason="SUCCESS") for _ in range(5)])
    assert all_success["success_rate"] == 1.0
    assert all_success["wilson_95"][0] <= all_success["success_rate"] <= all_success["wilson_95"][1]
    all_failure = aggregate_records([make_record(reason="TIMEOUT") for _ in range(5)])
    assert all_failure["success_rate"] == 0.0
    assert all_failure["wilson_95"][0] <= all_failure["success_rate"] <= all_failure["wilson_95"][1]


def test_task_completion_is_bounded_and_handles_zero_start_distance():
    zone = np.zeros(2)
    assert task_completion(np.array([1.0, 0.0]), np.array([0.0, 0.0]), zone) == pytest.approx(1.0)
    assert task_completion(np.array([1.0, 0.0]), np.array([2.0, 0.0]), zone) == pytest.approx(0.0)
    assert task_completion(np.array([1.0, 0.0]), np.array([0.5, 0.0]), zone) == pytest.approx(0.5)
    assert task_completion(zone, zone, zone) == 1.0
