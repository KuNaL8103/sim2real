"""FakeBackend determinism and rendering contract."""

from __future__ import annotations

import numpy as np
import pytest

from sim2real.sim import create_backend

CONFIG_DIR = "configs/sim"


@pytest.fixture()
def backend():
    instance = create_backend("fake", CONFIG_DIR)
    yield instance
    instance.close()


def test_same_seed_produces_identical_observations(backend):
    first = backend.reset(seed=11)
    frames_first = {camera: backend.render(camera) for camera in ("overhead", "wrist")}
    second = backend.reset(seed=11)
    frames_second = {camera: backend.render(camera) for camera in ("overhead", "wrist")}

    np.testing.assert_array_equal(
        first.robot_state.joint_positions, second.robot_state.joint_positions
    )
    for name in first.object_poses:
        np.testing.assert_array_equal(first.object_poses[name], second.object_poses[name])
    for camera in frames_first:
        np.testing.assert_array_equal(frames_first[camera].rgb, frames_second[camera].rgb)
        np.testing.assert_array_equal(frames_first[camera].depth, frames_second[camera].depth)


def test_reset_is_nominal_and_deterministic(backend):
    first = backend.reset(seed=1)
    second = backend.reset(seed=2)
    # Task 3 moved seed-driven pose variation into the sampler; reset itself is
    # always nominal so the applier can start from a known scene.
    for name, cube in backend.configs.scene.cubes.items():
        np.testing.assert_allclose(first.object_poses[name][:3, 3], cube.position)
        np.testing.assert_array_equal(first.object_poses[name], second.object_poses[name])


def test_observation_shapes_and_dtypes(backend):
    observation = backend.reset(seed=3)
    assert observation.robot_state.joint_positions.shape == (7,)
    assert observation.robot_state.joint_velocities.shape == (7,)
    assert observation.robot_state.ee_pose.shape == (4, 4)
    assert isinstance(observation.robot_state.gripper_position, float)
    for camera in ("overhead", "wrist"):
        frame = backend.render(camera)
        assert frame.rgb.shape == (256, 256, 3)
        assert frame.rgb.dtype == np.uint8
        assert frame.depth.shape == (256, 256)
        assert frame.depth.dtype == np.float32
        assert np.isfinite(frame.depth).any()
        assert frame.frame_id == f"camera_{camera}_optical_frame"


def test_step_clamps_to_joint_limits(backend):
    limits = backend.joint_limits
    result = backend.step(limits[:, 1] + 1.0, gripper_target=10.0)
    np.testing.assert_allclose(result.observation.robot_state.joint_positions, limits[:, 1])
    assert result.observation.robot_state.gripper_position == pytest.approx(
        backend.configs.panda.gripper_limits[1]
    )


def test_capabilities_have_reasons(backend):
    capabilities = backend.capabilities()
    assert capabilities.rgbd is True
    assert capabilities.physics_mass is False
    assert set(capabilities.as_dict()) == set(capabilities.reasons)
