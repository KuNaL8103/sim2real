"""Pure reset handler: seed echo, determinism, and metadata validity."""

from __future__ import annotations

import numpy as np
import pytest

from sim2real.metadata import validate_metadata_json
from sim2real.sim import create_backend
from sim2real.sim.reset_handler import reset_environment


@pytest.fixture()
def backend():
    instance = create_backend("fake", "configs/sim")
    yield instance
    instance.close()


def test_seed_is_echoed_and_json_is_valid(backend):
    outcome = reset_environment(backend, 3)
    assert outcome.success is True
    assert outcome.applied_seed == 3
    payload = validate_metadata_json(outcome.applied_config_json)
    assert payload["seed"] == 3
    assert payload["randomization"] == "ignored_task_3"


def test_reset_is_deterministic(backend):
    first = reset_environment(backend, 9)
    first_pose = backend.get_observation().object_poses["cube_red"]
    second = reset_environment(backend, 9)
    second_pose = backend.get_observation().object_poses["cube_red"]
    assert first.success and second.success
    np.testing.assert_array_equal(first_pose, second_pose)


def test_negative_and_non_integer_seed_fail(backend):
    negative = reset_environment(backend, -1)
    assert negative.success is False
    assert negative.applied_seed == 0
    validate_metadata_json(negative.applied_config_json)

    non_integer = reset_environment(backend, "3")  # type: ignore[arg-type]
    assert non_integer.success is False
    validate_metadata_json(non_integer.applied_config_json)


def test_randomization_fields_are_recorded_but_ignored(backend):
    outcome = reset_environment(
        backend,
        4,
        randomization_enabled=True,
        randomization_profile="lighting_only",
        task_instruction="pick the red cube",
        object_pose_override=np.eye(4),
    )
    payload = validate_metadata_json(outcome.applied_config_json)
    assert payload["randomization_enabled"] is True
    assert payload["randomization_profile"] == "lighting_only"
    assert payload["object_pose_override_ignored"] is True
    assert "randomization" in outcome.message
