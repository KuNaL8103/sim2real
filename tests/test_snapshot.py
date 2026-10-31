"""Camera stamps come from the snapshot, not from the current simulator time."""

from __future__ import annotations

import numpy as np

from sim2real.sim import create_backend

CONFIG_DIR = "configs/sim"


def test_fake_frame_stamp_equals_snapshot_stamp():
    backend = create_backend("fake", CONFIG_DIR)
    try:
        backend.reset(seed=1)
        backend.step(np.asarray(backend.configs.panda.home_arm), backend.configs.panda.home_gripper)
        snapshot = backend.capture_snapshot()
        assert snapshot.stamp_ns == snapshot.observation.stamp_ns

        frame = backend.render_from_snapshot("overhead", snapshot)
        assert frame.stamp_ns == snapshot.stamp_ns

        # Advancing the backend must not change the stamp of an old snapshot.
        backend.step(np.asarray(backend.configs.panda.home_arm), backend.configs.panda.home_gripper)
        again = backend.render_from_snapshot("overhead", snapshot)
        assert again.stamp_ns == snapshot.stamp_ns
        np.testing.assert_array_equal(frame.rgb, again.rgb)
        np.testing.assert_array_equal(frame.depth, again.depth)
    finally:
        backend.close()


def test_snapshot_observation_is_a_copy():
    backend = create_backend("fake", CONFIG_DIR)
    try:
        backend.reset(seed=2)
        snapshot = backend.capture_snapshot()
        positions = snapshot.observation.robot_state.joint_positions.copy()
        backend.step(np.full(7, 0.2), 0.0)
        np.testing.assert_array_equal(snapshot.observation.robot_state.joint_positions, positions)
    finally:
        backend.close()
