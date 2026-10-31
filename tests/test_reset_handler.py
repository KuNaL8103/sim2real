"""Pure reset handler: seed echo, determinism, validation, and metadata."""

from __future__ import annotations

import numpy as np
import pytest

from sim2real.metadata import validate_metadata_json
from sim2real.sim import create_backend
from sim2real.sim.camera import make_pose
from sim2real.sim.reset_handler import reset_environment


@pytest.fixture()
def backend():
    instance = create_backend("fake", "configs/sim")
    yield instance
    instance.close()


def test_seed_is_echoed_and_nominal_json_is_valid(backend):
    outcome = reset_environment(backend, 3)
    assert outcome.success is True
    assert outcome.applied_seed == 3
    payload = validate_metadata_json(outcome.applied_config_json)
    assert payload["seed"] == 3
    assert payload["randomization_applied"] is False
    assert payload["sample"]["nominal"] is True
    assert payload["report"]["nominal"] is True


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


def test_randomization_profile_is_applied_and_recorded(backend):
    outcome = reset_environment(
        backend,
        4,
        randomization_enabled=True,
        randomization_profile="default",
        task_instruction="pick the red cube",
        object_pose_override=make_pose(np.array([0.45, 0.0, 0.44]), np.eye(3)),
    )
    assert outcome.success is True
    payload = validate_metadata_json(outcome.applied_config_json)
    assert payload["randomization_enabled"] is True
    assert payload["randomization_applied"] is True
    assert payload["profile"] == "default"
    assert payload["task_instruction"] == "pick the red cube"
    assert payload["object_pose_override"] == {
        "requested": True,
        "applied": True,
        "object": "cube_red",
    }
    assert payload["report"]["groups"]["object_pose"]["status"] == "APPLIED"
    assert payload["report"]["groups"]["lighting"]["status"] == "SKIPPED"
    assert "randomization" in outcome.message


def test_same_seed_twice_is_byte_identical(backend):
    first = reset_environment(
        backend, 7, randomization_enabled=True, randomization_profile="default"
    )
    second = reset_environment(
        backend, 7, randomization_enabled=True, randomization_profile="default"
    )
    assert first.success and second.success
    assert first.applied_config_json == second.applied_config_json


def test_unknown_profile_and_out_of_bounds_override_fail_without_crashing(backend):
    unknown = reset_environment(
        backend, 7, randomization_enabled=True, randomization_profile="does_not_exist"
    )
    assert unknown.success is False
    assert "does_not_exist" in unknown.message
    validate_metadata_json(unknown.applied_config_json)

    outside = reset_environment(
        backend,
        7,
        object_pose_override=make_pose(np.array([2.0, 0.0, 0.44]), np.eye(3)),
    )
    assert outside.success is False
    assert "workspace" in outside.message
    validate_metadata_json(outside.applied_config_json)
