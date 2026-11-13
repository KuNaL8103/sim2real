"""Execution contract, start-in-region flag and phase tests (hardware-free)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from sim2real.evaluation import (
    ActionChunk,
    PolicyObservation,
    load_evaluation_config,
    run_eval_episode,
    starts_in_success_region,
)
from sim2real.expert import load_expert_config, parse_instruction
from sim2real.filtering.config import load_filter_config
from sim2real.sim import create_backend

FILTER = load_filter_config("configs/filtering/default.yaml")
EXPERT = load_expert_config("configs/expert")


class SequencePolicy:
    name = "sequence"
    needs_images = False

    def __init__(self, home: np.ndarray) -> None:
        self.home = np.asarray(home, dtype=np.float64)
        self.info = {"checkpoint_hash": "none"}

    def reset(self, task_instruction: str, seed: int) -> None:
        del task_instruction, seed

    def set_n_action_steps(self, n_action_steps: int) -> None:
        self._n = int(n_action_steps)

    def act(self, observation: PolicyObservation) -> ActionChunk:
        del observation
        actions = np.tile(self.home, (self._n, 1))
        for index in range(self._n):
            actions[index, 0] += 0.01 * index
        return ActionChunk(actions)


def _run(execution: str):
    config = load_evaluation_config()
    config = replace(
        config,
        settings=replace(
            config.settings,
            execution=execution,
            n_action_steps=3,
            max_control_steps=6,
        ),
    )
    backend = create_backend("fake", "configs/sim")
    home = np.concatenate([backend.configs.panda.home_arm, [backend.configs.panda.home_gripper]])
    policy = SequencePolicy(home)
    try:
        record = run_eval_episode(
            backend,
            policy,
            seed=20000,
            episode_index=0,
            config=config,
            eval_run_id=f"exec_{execution}",
            config_hash=config.config_hash(),
            filter_config=FILTER,
            expert_config=EXPERT,
        )
    finally:
        backend.close()
    return record, home


def test_hold_applies_each_action_twice():
    record, home = _run("hold")
    assert len(record.steps) == 6
    values = [step.action[0] for step in record.steps]
    assert values == [
        home[0],
        home[0],
        home[0] + 0.01,
        home[0] + 0.01,
        home[0] + 0.02,
        home[0] + 0.02,
    ]


def test_linear_interpolates_and_holds_the_last_chunk_action():
    record, home = _run("linear")
    assert len(record.steps) == 6
    values = [step.action[0] for step in record.steps]
    expected = [
        home[0],
        home[0] + 0.005,
        home[0] + 0.01,
        home[0] + 0.015,
        home[0] + 0.02,
        home[0] + 0.02,
    ]
    assert np.allclose(values, expected, atol=1.0e-12)


def test_start_in_success_region_flag_uses_the_success_tolerance():
    backend = create_backend("fake", "configs/sim")
    try:
        backend.reset(0)
        observation = backend.get_observation()
        task = parse_instruction(
            "pick up the red cube and place it in the left zone",
            backend.configs.scene,
            EXPERT.instructions_file,
        )
        zone = backend.configs.scene.zones[task.zone_id]
        pose = np.eye(4)
        pose[:3, 3] = zone.position
        backend._object_poses[task.object_id] = pose
        observation = backend.get_observation()
        assert starts_in_success_region(backend, task, observation, EXPERT) is True
        far = np.eye(4)
        far[:3, 3] = [0.60, 0.20, zone.position[2]]
        backend._object_poses[task.object_id] = far
        observation = backend.get_observation()
        assert starts_in_success_region(backend, task, observation, EXPERT) is False
    finally:
        backend.close()
