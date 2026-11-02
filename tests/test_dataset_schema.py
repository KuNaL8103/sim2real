"""Unified schema and dataset validation."""

from __future__ import annotations

import numpy as np
import pytest

from sim2real.dataset_builder import (
    DataValidationError,
    build_dataset_index,
    validate_episode,
)
from tests.fixtures.dataset_builder import make_episode, panda_spec


def test_valid_episode_passes_and_index_is_contiguous():
    episodes = [make_episode(episode_id="a", frames=3), make_episode(episode_id="b", frames=2)]
    index = build_dataset_index(episodes, panda_spec(), "joint_position")
    assert index.num_episodes == 2
    assert index.total_frames == 5
    assert [row["start_index"] for row in index.entries] == [0, 3]
    assert [row["end_index"] for row in index.entries] == [3, 5]
    assert index.tasks[0]["instruction"] == "pick up the red cube"


def test_wrong_state_dimension_is_rejected():
    episode = make_episode()
    object.__setattr__(episode, "observation_state", np.zeros((4, 7), dtype=np.float32))
    with pytest.raises(DataValidationError, match="observation.state shape"):
        validate_episode(episode, panda_spec(), "joint_position")


def test_nan_and_nonmonotonic_time_are_rejected():
    episode = make_episode()
    state = episode.observation_state.copy()
    state[1, 0] = np.nan
    object.__setattr__(episode, "observation_state", state)
    with pytest.raises(DataValidationError, match="NaN"):
        validate_episode(episode, panda_spec(), "joint_position")

    episode = make_episode()
    timestamps = episode.timestamps.copy()
    timestamps[2] = timestamps[1]
    object.__setattr__(episode, "timestamps", timestamps)
    with pytest.raises(DataValidationError, match="strictly increasing"):
        validate_episode(episode, panda_spec(), "joint_position")


def test_missing_fps_empty_task_and_duplicate_key_are_rejected():
    episode = make_episode()
    object.__setattr__(episode, "fps", 0.0)
    with pytest.raises(DataValidationError, match="fps"):
        validate_episode(episode, panda_spec(), "joint_position")

    episode = make_episode(task=" ")
    with pytest.raises(DataValidationError, match="task string"):
        validate_episode(episode, panda_spec(), "joint_position")

    episodes = [make_episode(episode_id="same"), make_episode(episode_id="same")]
    with pytest.raises(DataValidationError, match="duplicate episode key"):
        build_dataset_index(episodes, panda_spec(), "joint_position")


def test_action_dimension_depends_on_representation():
    episode = make_episode()
    with pytest.raises(DataValidationError, match="action shape"):
        validate_episode(episode, panda_spec(), "ee_delta")
