"""Decision-engine tests: thresholds, determinism, malformed input, label leakage."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from sim2real import contracts
from sim2real.filtering.config import HardGates, MetricWeights, SoftThresholds, load_filter_config
from sim2real.filtering.decision import decide
from sim2real.filtering.metrics import compute_metrics
from sim2real.filtering.pipeline import FilterPipeline
from sim2real.filtering.synthetic import MID_RANGE_ARM, make_episode, min_jerk_positions

BASE = load_filter_config("configs/filtering/default.yaml")
PERMISSIVE = replace(
    BASE,
    hard_gates=HardGates(
        require_success=True,
        max_collisions=10,
        min_joint_margin=-1.0,
        min_velocity_margin=-1.0,
    ),
    soft_thresholds=SoftThresholds(
        max_dimensionless_squared_jerk=1.0e15,
        min_sparc=-1.0e6,
        max_path_length_m=1.0e6,
    ),
    weights=MetricWeights(1.0, 1.0, 1.0),
    accept_score_threshold=0.0,
)


def _clean_episode():
    times, positions = min_jerk_positions(2.0, 0.4)
    return make_episode(positions=positions, timestamps_s=times), compute_metrics(
        make_episode(positions=positions, timestamps_s=times), BASE
    )


def test_each_soft_threshold_flips_the_decision_with_its_reason():
    episode, metrics = _clean_episode()
    cases = (
        (
            "JERK_TOO_HIGH",
            replace(
                PERMISSIVE,
                soft_thresholds=replace(
                    PERMISSIVE.soft_thresholds,
                    max_dimensionless_squared_jerk=metrics.dimensionless_squared_jerk * 0.99,
                ),
            ),
            replace(
                PERMISSIVE,
                soft_thresholds=replace(
                    PERMISSIVE.soft_thresholds,
                    max_dimensionless_squared_jerk=metrics.dimensionless_squared_jerk * 1.01,
                ),
            ),
        ),
        (
            "SPARC_TOO_LOW",
            replace(
                PERMISSIVE,
                soft_thresholds=replace(
                    PERMISSIVE.soft_thresholds, min_sparc=metrics.sparc + 0.001
                ),
            ),
            replace(
                PERMISSIVE,
                soft_thresholds=replace(
                    PERMISSIVE.soft_thresholds, min_sparc=metrics.sparc - 0.001
                ),
            ),
        ),
        (
            "PATH_TOO_LONG",
            replace(
                PERMISSIVE,
                soft_thresholds=replace(
                    PERMISSIVE.soft_thresholds, max_path_length_m=metrics.path_length_m * 0.99
                ),
            ),
            replace(
                PERMISSIVE,
                soft_thresholds=replace(
                    PERMISSIVE.soft_thresholds, max_path_length_m=metrics.path_length_m * 1.01
                ),
            ),
        ),
    )
    for code, outside, inside in cases:
        rejected = decide(episode, outside)
        accepted = decide(episode, inside)
        assert rejected.decision == contracts.DECISION_REJECT
        assert code in [reason.code for reason in rejected.reasons]
        assert accepted.decision == contracts.DECISION_ACCEPT, (code, accepted.reasons)


def test_hard_gates_override_a_high_score():
    _episode, metrics = _clean_episode()
    strict = replace(
        PERMISSIVE,
        hard_gates=HardGates(
            require_success=True,
            max_collisions=0,
            min_joint_margin=0.0,
            min_velocity_margin=0.0,
        ),
    )
    failed = make_episode(
        positions=min_jerk_positions(2.0, 0.4)[1], outcome=contracts.OUTCOME_FAILURE
    )
    result = decide(failed, PERMISSIVE)
    assert result.decision == contracts.DECISION_REJECT
    assert "NOT_SUCCESS" in [reason.code for reason in result.reasons]
    assert result.acceptance_score > 0.9

    collision_flags = np.zeros(metrics.num_steps, dtype=bool)
    collision_flags[10:15] = True
    collided = make_episode(positions=min_jerk_positions(2.0, 0.4)[1], collisions=collision_flags)
    collision_result = decide(collided, strict)
    assert "COLLISION" in [reason.code for reason in collision_result.reasons]

    beyond = np.tile(np.asarray(MID_RANGE_ARM), (metrics.num_steps, 1))
    beyond[:, 0] = BASE.limits.joint_limits[0][0] - 0.2
    joint_result = decide(
        make_episode(positions=min_jerk_positions(2.0, 0.4)[1], joint_positions=beyond),
        strict,
    )
    assert "JOINT_LIMIT_MARGIN" in [reason.code for reason in joint_result.reasons]

    too_fast = np.tile(np.asarray(MID_RANGE_ARM), (metrics.num_steps, 1))
    speeds = np.zeros_like(too_fast)
    speeds[:, 0] = BASE.limits.velocity_limits[0] * 2.0
    velocity_result = decide(
        make_episode(
            positions=min_jerk_positions(2.0, 0.4)[1],
            joint_positions=too_fast,
            joint_velocities=speeds,
        ),
        strict,
    )
    assert "VELOCITY_LIMIT_MARGIN" in [reason.code for reason in velocity_result.reasons]


def test_decision_is_deterministic_and_reason_order_is_stable():
    _episode, metrics = _clean_episode()
    config = replace(
        PERMISSIVE,
        soft_thresholds=SoftThresholds(
            max_dimensionless_squared_jerk=metrics.dimensionless_squared_jerk * 0.5,
            min_sparc=metrics.sparc + 0.5,
            max_path_length_m=metrics.path_length_m * 0.5,
        ),
    )
    episode = make_episode(
        positions=min_jerk_positions(2.0, 0.4)[1], outcome=contracts.OUTCOME_FAILURE
    )
    first = decide(episode, config)
    second = decide(episode, config)
    assert first.acceptance_score == second.acceptance_score
    assert first.decision == second.decision
    assert [reason.code for reason in first.reasons] == [reason.code for reason in second.reasons]
    assert first.metadata_json() == second.metadata_json()
    assert [reason.code for reason in first.reasons] == [
        "NOT_SUCCESS",
        "JERK_TOO_HIGH",
        "SPARC_TOO_LOW",
        "PATH_TOO_LONG",
    ]


def test_malformed_episodes_are_invalid_and_never_raise():
    times, positions = min_jerk_positions(2.0, 0.4)
    count = len(times)
    nan_joints = np.tile(np.asarray(MID_RANGE_ARM), (count, 1))
    nan_joints[3, 2] = np.nan
    non_monotonic = times.copy()
    non_monotonic[5] = non_monotonic[4]
    cases = {
        "NAN_VALUES": make_episode(positions=positions, joint_positions=nan_joints),
        "NON_MONOTONIC_TIME": make_episode(positions=positions, timestamps_s=non_monotonic),
        "SHAPE_MISMATCH": make_episode(positions=positions, joint_positions=np.zeros((count, 6))),
        "ACTION_DIM": make_episode(positions=positions, actions=np.zeros((count, 7))),
        "TOO_SHORT": make_episode(positions=positions[:10]),
    }
    for code, episode in cases.items():
        result = decide(episode, BASE)
        assert result.decision == contracts.DECISION_INVALID, (code, result.reasons)
        assert code in [reason.code for reason in result.reasons], code

    empty = make_episode(positions=np.zeros((0, 3)))
    assert decide(empty, BASE).decision == contracts.DECISION_INVALID

    pipeline = FilterPipeline(BASE)
    duplicate = make_episode(episode_id="same-id")
    assert pipeline.process(duplicate).decision == contracts.DECISION_ACCEPT
    second = pipeline.process(make_episode(episode_id="same-id"))
    assert second.decision == contracts.DECISION_INVALID
    assert "DUPLICATE_EPISODE" in [reason.code for reason in second.reasons]


def test_labels_do_not_affect_score_decision_or_reasons():
    times, positions = min_jerk_positions(2.0, 0.4)
    baseline = make_episode(positions=positions, timestamps_s=times)
    variants = [
        replace(baseline, is_perturbed=True),
        replace(baseline, is_perturbed=True, perturbation_name="jitter"),
        replace(baseline, expected_quality="failure"),
        replace(
            baseline,
            is_perturbed=True,
            perturbation_name="bad_grasp",
            expected_quality="failure",
        ),
    ]
    expected = decide(baseline, BASE)
    for variant in variants:
        result = decide(variant, BASE)
        assert result.decision == expected.decision
        assert result.acceptance_score == expected.acceptance_score
        assert [reason.code for reason in result.reasons] == [
            reason.code for reason in expected.reasons
        ]
