"""Policy interface, closed-loop runner and paired-evaluation tests."""

from __future__ import annotations

import numpy as np

from sim2real.evaluation import (
    ActionChunk,
    NoisyExpertPolicy,
    PolicyObservation,
    ScriptedExpertPolicy,
    ZeroPolicy,
    load_evaluation_config,
    run_eval,
    run_paired_eval,
)
from sim2real.filtering.config import load_filter_config
from sim2real.sim import create_backend

CONFIG = load_evaluation_config()
FILTER = load_filter_config("configs/filtering/default.yaml")


def _expert_factory():
    return lambda: create_backend("fake", "configs/sim")


class CountingPolicy:
    name = "counting"
    needs_images = False

    def __init__(self) -> None:
        self.info = {"checkpoint_hash": "none"}
        self.calls = 0

    def reset(self, task_instruction: str, seed: int) -> None:
        del task_instruction, seed
        self.calls = 0

    def set_n_action_steps(self, n_action_steps: int) -> None:
        self._n = int(n_action_steps)

    def act(self, observation: PolicyObservation) -> ActionChunk:
        action = np.zeros((self._n, 8))
        action[:, :7] = observation.state[:7]
        action[:, 7] = observation.state[7]
        offsets = self.calls * self._n + np.arange(self._n)
        action[:, 0] += offsets * 1.0e-4
        self.calls += 1
        return ActionChunk(action)


class RaisingPolicy(CountingPolicy):
    name = "raising"

    def act(self, observation: PolicyObservation) -> ActionChunk:
        raise RuntimeError("boom")


class NaNPolicy(CountingPolicy):
    name = "nan"

    def act(self, observation: PolicyObservation) -> ActionChunk:
        action = np.zeros((1, 8))
        action[0, 0] = np.nan
        return ActionChunk(action)


class WrongShapePolicy(CountingPolicy):
    name = "wrong"

    def act(self, observation: PolicyObservation):
        return np.zeros((1, 7))


def test_timing_contract_holds_each_action_for_two_control_steps():
    from dataclasses import replace

    config = load_evaluation_config()
    config = replace(
        config,
        settings=replace(config.settings, n_action_steps=10, max_control_steps=40),
    )
    backend = create_backend("fake", "configs/sim")
    policy = CountingPolicy()
    try:
        result = run_eval(
            backend,
            policy,
            config,
            num_episodes=1,
            eval_seed=20000,
            run_id="timing",
            filter_config=FILTER,
        )
    finally:
        backend.close()
    record = result.records[0]
    assert record.termination_reason == "TIMEOUT"
    assert len(record.steps) == 40
    assert policy.calls == 2  # 10 policy steps x 2 control steps = 1 re-plan per 20 ticks
    for index in range(0, len(record.steps), 2):
        assert np.array_equal(record.steps[index].action, record.steps[index + 1].action)
    distinct = {tuple(step.action) for step in record.steps}
    assert len(distinct) == 20  # one command per policy step, 2 ticks each


def test_scripted_policy_reproduces_the_same_outcome_and_metrics():
    first_backend = create_backend("fake", "configs/sim")
    first_policy = ScriptedExpertPolicy(backend_factory=_expert_factory())
    second_backend = create_backend("fake", "configs/sim")
    second_policy = ScriptedExpertPolicy(backend_factory=_expert_factory())
    try:
        first = run_eval(
            first_backend,
            first_policy,
            CONFIG,
            num_episodes=3,
            eval_seed=20000,
            run_id="replay",
            filter_config=FILTER,
        )
        second = run_eval(
            second_backend,
            second_policy,
            CONFIG,
            num_episodes=3,
            eval_seed=20000,
            run_id="replay",
            filter_config=FILTER,
        )
    finally:
        first_backend.close()
        second_backend.close()
        first_policy.close()
        second_policy.close()
    assert [r.to_dict() for r in first.records] == [r.to_dict() for r in second.records]
    assert first.run.seeds == second.run.seeds
    assert first.run.aggregates == second.run.aggregates
    assert any(record.success for record in first.records)


def test_policy_errors_are_recorded_and_the_run_continues():
    for policy in (RaisingPolicy(), NaNPolicy(), WrongShapePolicy()):
        backend = create_backend("fake", "configs/sim")
        try:
            result = run_eval(
                backend,
                policy,
                CONFIG,
                num_episodes=3,
                eval_seed=20000,
                run_id=f"errors_{policy.name}",
                filter_config=FILTER,
            )
        finally:
            backend.close()
        assert len(result.records) == 3
        assert all(record.termination_reason == "POLICY_ERROR" for record in result.records)
        assert all(record.failure.get("reason") for record in result.records)


def test_paired_eval_reports_policy_only_failures():
    backend = create_backend("fake", "configs/sim")
    expert = ScriptedExpertPolicy(backend_factory=_expert_factory())
    candidate = ZeroPolicy()
    try:
        result = run_paired_eval(
            expert_backend_factory=None,
            backend=backend,
            expert_policy=expert,
            candidate_policy=candidate,
            config=CONFIG,
            num_episodes=5,
            eval_seed=20000,
            run_id="paired",
            filter_config=FILTER,
        )
    finally:
        backend.close()
        expert.close()
    assert result["num_pairs"] == 5
    assert result["expert"]["num_success"] >= 1
    assert result["policy"]["num_success"] == 0
    assert result["policy_only_failures"] == result["expert"]["num_success"]
    assert result["policy_success_given_expert_success"]["num_success"] == 0


def test_noisy_expert_is_deterministic_with_the_same_seed():
    observation = PolicyObservation(state=np.zeros(8), task_instruction="x")
    inner_a = ScriptedExpertPolicy(backend_factory=_expert_factory())
    inner_b = ScriptedExpertPolicy(backend_factory=_expert_factory())
    policy_a = NoisyExpertPolicy(inner_a, noise_sigma=0.01, dropout=0.1, seed=0)
    policy_b = NoisyExpertPolicy(inner_b, noise_sigma=0.01, dropout=0.1, seed=0)
    try:
        for policy in (policy_a, policy_b):
            policy.set_episode_index(0)
            policy.reset("pick up the red cube and place it in the left zone", 20000)
        chunk_a = policy_a.act(observation)
        chunk_b = policy_b.act(observation)
        assert policy_a.info["inner"] == "scripted_expert"
        assert np.array_equal(chunk_a.actions, chunk_b.actions)
        assert chunk_a.step_period_s == 0.1
    finally:
        policy_a.close()
        policy_b.close()
