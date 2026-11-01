"""EpisodeRecord -> RolloutEpisode fields: presence, stamps, metadata."""

from __future__ import annotations

import copy
import dataclasses
import json

import pytest

from sim2real import contracts
from sim2real.expert import load_expert_config, parse_instruction, run_episode
from sim2real.expert.conversion import (
    ConversionError,
    episode_to_rollout_fields,
    summarize_outcomes,
)
from sim2real.metadata import validate_metadata_json
from sim2real.sim import create_backend

CONFIG_DIR = "configs/sim"
EXPERT_CONFIG_DIR = "configs/expert"
INSTRUCTION = "pick up the green cube and place it in the right zone"


@pytest.fixture(scope="module")
def record():
    config = load_expert_config(EXPERT_CONFIG_DIR)
    backend = create_backend("fake", CONFIG_DIR)
    try:
        task = parse_instruction(INSTRUCTION, backend.configs.scene, config.instructions_file)
        return run_episode(
            backend,
            task,
            4,
            instruction=INSTRUCTION,
            randomization_profile="none",
            config=config,
        )
    finally:
        backend.close()


def test_fields_match_the_wire_contract(record) -> None:
    fields = episode_to_rollout_fields(record)
    assert fields["episode_id"] == record.episode_id
    assert fields["schema_version"] == contracts.SCHEMA_VERSION
    assert fields["backend"] == "fake"
    assert fields["seed"] == 4
    assert fields["task_instruction"] == INSTRUCTION
    assert fields["outcome"] == contracts.OUTCOME_SUCCESS
    assert fields["is_perturbed"] is False
    assert fields["start_stamp_ns"] == record.steps[0].stamp_ns
    step = fields["steps"][0]
    assert set(step) >= {
        "stamp_ns",
        "step_index",
        "task_instruction",
        "joint_names",
        "joint_positions",
        "joint_velocities",
        "ee_pose",
        "ee_frame_id",
        "action",
        "action_space",
    }
    assert step["joint_names"] == list(contracts.CONTROLLED_JOINT_NAMES)
    assert step["ee_frame_id"] == contracts.PANDA_HAND_FRAME


def test_step_count_and_stamps_are_consistent(record) -> None:
    fields = episode_to_rollout_fields(record)
    assert len(fields["steps"]) == len(record.steps)
    stamps = [step["stamp_ns"] for step in fields["steps"]]
    assert stamps == sorted(stamps)
    metadata = json.loads(fields["metadata_json"])
    assert metadata["num_steps"] == len(record.steps)
    validate_metadata_json(fields["metadata_json"])
    assert "images" not in fields["metadata_json"]


def test_non_monotonic_stamps_are_rejected(record) -> None:
    broken = copy.deepcopy(record)
    broken.steps = list(broken.steps)
    broken.steps[0] = dataclasses.replace(broken.steps[0], stamp_ns=100)
    broken.steps[1] = dataclasses.replace(broken.steps[1], stamp_ns=50)
    with pytest.raises(ConversionError, match="non-monotonic"):
        episode_to_rollout_fields(broken)


def test_empty_episode_is_rejected(record) -> None:
    empty = copy.deepcopy(record)
    empty.steps = []
    with pytest.raises(ConversionError, match="no steps"):
        episode_to_rollout_fields(empty)


def test_outcome_summary(record) -> None:
    assert summarize_outcomes([record])["success"] == 1
    assert summarize_outcomes([record])["failure"] == 0
