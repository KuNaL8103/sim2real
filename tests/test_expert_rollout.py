"""Rollout runner: clean success, determinism, perturbations, failures."""

from __future__ import annotations

import numpy as np
import pytest

from sim2real import contracts
from sim2real.expert import (
    EpisodeRecord,
    load_expert_config,
    load_perturbations,
    parse_instruction,
    run_episode,
    run_episodes,
)
from sim2real.expert.ik import IKResult
from sim2real.expert.perturbations import select_perturbation
from sim2real.sim import create_backend
from sim2real.sim.fake_backend import FakeBackend

CONFIG_DIR = "configs/sim"
EXPERT_CONFIG_DIR = "configs/expert"
INSTRUCTION = "pick up the red cube and place it in the left zone"


@pytest.fixture(scope="module")
def config():
    return load_expert_config(EXPERT_CONFIG_DIR)


@pytest.fixture(scope="module")
def perturbations(config):
    return load_perturbations(config)


@pytest.fixture()
def backend():
    instance = create_backend("fake", CONFIG_DIR)
    yield instance
    instance.close()


def _task(backend, config):
    return parse_instruction(INSTRUCTION, backend.configs.scene, config.instructions_file)


def _run(backend, config, perturbation=None, seed=0, profile="none"):
    return run_episode(
        backend,
        _task(backend, config),
        seed,
        instruction=INSTRUCTION,
        episode_index=0,
        randomization_profile=profile,
        perturbation=perturbation,
        config=config,
    )


def test_clean_fake_rollout_succeeds(backend, config) -> None:
    record = _run(backend, config)
    assert record.outcome == contracts.OUTCOME_SUCCESS, record.failure_reason
    assert record.transitions[-1]["to"] == "DONE"
    assert record.success_metrics["xy_error_m"] <= record.success_metrics["allowed_radius_m"]
    assert record.steps
    for step in record.steps:
        assert step.joint_positions.shape == (7,)
        assert step.joint_velocities.shape == (7,)
        assert step.ee_pose.shape == (4, 4)
        assert step.action.shape == (8,)
        assert step.action_space == contracts.ACTION_JOINT_POSITION_TARGET


def test_same_seed_is_byte_identical(backend, config) -> None:
    first = _run(backend, config)
    second = _run(backend, config)
    assert first.outcome == second.outcome
    assert len(first.steps) == len(second.steps)
    assert first.transitions == second.transitions
    assert first.randomization_report == second.randomization_report
    for left, right in zip(first.steps, second.steps, strict=True):
        np.testing.assert_array_equal(left.joint_positions, right.joint_positions)
        np.testing.assert_array_equal(left.action, right.action)
        np.testing.assert_array_equal(left.ee_pose, right.ee_pose)
        assert left.stamp_ns == right.stamp_ns


def _energy(record: EpisodeRecord) -> float:
    q = np.asarray([step.action[:7] for step in record.steps])
    return float(np.mean(np.diff(q, n=2, axis=0) ** 2))


def test_bad_grasp_fails_with_reason(backend, config, perturbations) -> None:
    record = _run(backend, config, perturbations.get("bad_grasp"))
    assert record.outcome == contracts.OUTCOME_FAILURE
    assert record.is_perturbed
    assert record.perturbation_name == "bad_grasp"
    assert "grasp" in record.failure_reason.lower()


def test_jitter_has_more_high_frequency_energy(backend, config, perturbations) -> None:
    clean = _run(backend, config)
    jitter = _run(backend, config, perturbations.get("jitter"))
    assert jitter.outcome == contracts.OUTCOME_SUCCESS
    assert _energy(jitter) > 1.5 * _energy(clean)


def test_overspeed_is_shorter_than_clean(backend, config, perturbations) -> None:
    clean = _run(backend, config)
    fast = _run(backend, config, perturbations.get("overspeed"))
    assert fast.steps[-1].stamp_ns < clean.steps[-1].stamp_ns


def test_detour_and_hesitation_are_longer_than_clean(backend, config, perturbations) -> None:
    clean = _run(backend, config)
    detour = _run(backend, config, perturbations.get("detour"))
    hesitation = _run(backend, config, perturbations.get("hesitation"))
    assert detour.steps[-1].stamp_ns > clean.steps[-1].stamp_ns
    assert hesitation.steps[-1].stamp_ns > clean.steps[-1].stamp_ns


def test_low_sweep_transit_is_lower(backend, config, perturbations) -> None:
    clean = _run(backend, config)
    low = _run(backend, config, perturbations.get("low_sweep"))
    clean_transit = min(step.ee_pose[2, 3] for step in clean.steps if step.state == "TRANSIT")
    low_transit = min(step.ee_pose[2, 3] for step in low.steps if step.state == "TRANSIT")
    assert low_transit < clean_transit - 0.05


def test_perturbed_fraction_is_deterministic(perturbations) -> None:
    first = [
        select_perturbation(perturbations, 3, index, perturbed_fraction=0.5) for index in range(8)
    ]
    second = [
        select_perturbation(perturbations, 3, index, perturbed_fraction=0.5) for index in range(8)
    ]
    assert [(item.name if item else None) for item in first] == [
        (item.name if item else None) for item in second
    ]
    assert all(
        select_perturbation(perturbations, 3, i, perturbed_fraction=0.0) is None for i in range(5)
    )
    assert all(
        select_perturbation(perturbations, 3, i, perturbed_fraction=1.0) is not None
        for i in range(5)
    )


def test_randomization_is_seed_scoped_and_recorded(backend, config) -> None:
    first = _run(backend, config, seed=0, profile="default")
    repeat = _run(backend, config, seed=0, profile="default")
    other = _run(backend, config, seed=1, profile="default")
    first_pose = first.randomization_report["sample"]["object_pose"]["poses"]["cube_red"]
    repeat_pose = repeat.randomization_report["sample"]["object_pose"]["poses"]["cube_red"]
    other_pose = other.randomization_report["sample"]["object_pose"]["poses"]["cube_red"]
    assert first_pose == repeat_pose
    assert first_pose != other_pose
    assert first.randomization_seed == repeat.randomization_seed
    assert first.randomization_seed != other.randomization_seed
    assert first.randomization_report["report"]["groups"]["object_pose"]["status"] == "APPLIED"


def test_ik_failure_is_recorded_and_next_episode_runs(backend, config) -> None:
    class FailingIK:
        def solve(self, target_pose, q_init, panda, config=None):
            del target_pose, panda, config
            return IKResult(
                q=np.asarray(q_init, dtype=np.float64),
                success=False,
                position_error=float("inf"),
                orientation_error=float("inf"),
                iterations=0,
            )

    failed = run_episode(
        backend,
        _task(backend, config),
        0,
        instruction=INSTRUCTION,
        episode_index=0,
        randomization_profile="none",
        config=config,
        ik_solver=FailingIK(),
    )
    assert failed.outcome == contracts.OUTCOME_FAILURE
    assert "IK failed" in failed.failure_reason
    recovered = _run(backend, config)
    assert recovered.outcome == contracts.OUTCOME_SUCCESS


def test_backend_exception_is_aborted_and_batch_continues(backend, config) -> None:
    class ExplodingBackend(FakeBackend):
        def __init__(self) -> None:
            super().__init__(CONFIG_DIR)
            self.calls = 0
            self.exploded = False

        def step(self, joint_target, gripper_target):
            self.calls += 1
            if self.calls == 5 and not self.exploded:
                self.exploded = True
                raise RuntimeError("synthetic backend failure")
            return super().step(joint_target, gripper_target)

    exploding = ExplodingBackend()
    try:
        records = run_episodes(
            exploding,
            _task(exploding, config),
            [0, 1],
            instruction=INSTRUCTION,
            randomization_profile="none",
            config=config,
        )
    finally:
        exploding.close()
    assert records[0].outcome == contracts.OUTCOME_ABORTED
    assert "synthetic backend failure" in records[0].failure_reason
    assert records[1].outcome == contracts.OUTCOME_SUCCESS


def test_max_steps_yields_timeout(backend, config) -> None:
    record = run_episode(
        backend,
        _task(backend, config),
        0,
        instruction=INSTRUCTION,
        randomization_profile="none",
        config=config,
        max_steps=5,
    )
    assert record.outcome == contracts.OUTCOME_TIMEOUT
    assert "max_steps" in record.failure_reason
