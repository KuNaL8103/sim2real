"""MuJoCo smoke: 10 seeded clean episodes; assert validity, not success rate.

Skipped with a reason when MuJoCo or the Menagerie meshes are unavailable.
Physics step counts are printed so the measured throughput is visible in CI logs.
"""

from __future__ import annotations

import time

import pytest

from sim2real import contracts
from sim2real.expert import load_expert_config, parse_instruction, run_episode
from sim2real.sim import create_backend

CONFIG_DIR = "configs/sim"
EXPERT_CONFIG_DIR = "configs/expert"
INSTRUCTION = "pick up the red cube and place it in the left zone"
EPISODES = 10

pytestmark = pytest.mark.sim


def test_mujoco_ten_clean_episodes() -> None:
    config = load_expert_config(EXPERT_CONFIG_DIR)
    backend = create_backend("mujoco", CONFIG_DIR)
    try:
        task = parse_instruction(INSTRUCTION, backend.configs.scene, config.instructions_file)
        substeps = max(1, round(1.0 / (backend.control_rate_hz * backend.model.opt.timestep)))
        records = []
        start = time.perf_counter()
        for index in range(EPISODES):
            records.append(
                run_episode(
                    backend,
                    task,
                    seed=index,
                    instruction=INSTRUCTION,
                    episode_index=index,
                    randomization_profile="default",
                    config=config,
                )
            )
        elapsed = time.perf_counter() - start
    finally:
        backend.close()

    assert len(records) == EPISODES
    for record in records:
        assert record.steps, record.episode_id
        assert record.outcome in {
            contracts.OUTCOME_SUCCESS,
            contracts.OUTCOME_FAILURE,
            contracts.OUTCOME_TIMEOUT,
            contracts.OUTCOME_ABORTED,
        }
        assert record.transitions
    success = sum(record.outcome == contracts.OUTCOME_SUCCESS for record in records)
    control_steps = sum(len(record.steps) for record in records)
    print(
        f"MuJoCo smoke: success={success}/{EPISODES} "
        f"control_steps={control_steps} wall_s={elapsed:.3f} "
        f"control_steps_per_s={control_steps / elapsed:.1f} "
        f"physics_steps_per_s={control_steps * substeps / elapsed:.0f}"
    )
    for record in records:
        print(
            f"  {record.episode_id} seed={record.seed} outcome={record.outcome} "
            f"steps={len(record.steps)} reason={record.failure_reason or '-'}"
        )
