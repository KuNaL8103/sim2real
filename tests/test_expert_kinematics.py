"""Panda FK/IK: reachable, unreachable, limits, single source, MuJoCo check."""

from __future__ import annotations

import numpy as np
import pytest

from sim2real.expert.config import load_expert_config
from sim2real.expert.ik import DampedLeastSquaresIK
from sim2real.sim.config import load_sim_configs
from sim2real.sim.kinematics import (
    clamp_joints,
    forward_kinematics,
    hand_pose,
    tcp_pose,
)

CONFIG_DIR = "configs/sim"
EXPERT_CONFIG_DIR = "configs/expert"


@pytest.fixture(scope="module")
def panda():
    return load_sim_configs(CONFIG_DIR).panda


def test_home_fk_is_finite_and_matches_the_model(panda) -> None:
    pose = hand_pose(panda.home_arm, panda)
    assert np.all(np.isfinite(pose))
    assert pose.shape == (4, 4)
    # Values from the Menagerie MJCF at the configured home configuration.
    np.testing.assert_allclose(pose[:3, 3], [0.5545, 0.0, 0.6245], atol=1e-4)
    np.testing.assert_allclose(pose[2, :3], [0.0, 0.0, -1.0], atol=1e-6)


def test_ik_reaches_a_reachable_target(panda) -> None:
    config = load_expert_config(EXPERT_CONFIG_DIR)
    solver = DampedLeastSquaresIK(config.ik)
    home = np.asarray(panda.home_arm, dtype=np.float64)
    target = tcp_pose(home, panda).copy()
    target[:3, 3] += np.array([-0.04, 0.03, -0.02])
    result = solver.solve(target, home, panda)
    assert result.success, result.to_dict()
    assert result.iterations <= config.ik.max_iterations
    reached = tcp_pose(result.q, panda)
    assert np.linalg.norm(reached[:3, 3] - target[:3, 3]) <= config.ik.position_tolerance_m


def test_unreachable_target_fails_without_raising(panda) -> None:
    config = load_expert_config(EXPERT_CONFIG_DIR)
    solver = DampedLeastSquaresIK(config.ik)
    home = np.asarray(panda.home_arm, dtype=np.float64)
    target = tcp_pose(home, panda).copy()
    target[:3, 3] = [2.0, 2.0, 2.0]
    result = solver.solve(target, home, panda)
    assert result.success is False
    assert result.iterations <= config.ik.max_iterations
    assert result.position_error > config.ik.position_tolerance_m


def test_ik_respects_joint_limits(panda) -> None:
    config = load_expert_config(EXPERT_CONFIG_DIR)
    solver = DampedLeastSquaresIK(config.ik)
    outside = np.asarray(panda.joint_limits[:, 1] + 1.0, dtype=np.float64)
    target = tcp_pose(panda.home_arm, panda).copy()
    result = solver.solve(target, outside, panda)
    limits = np.asarray(panda.joint_limits, dtype=np.float64)
    assert np.all(result.q >= limits[:, 0] - 1e-12)
    assert np.all(result.q <= limits[:, 1] + 1e-12)
    np.testing.assert_allclose(clamp_joints(outside, panda), limits[:, 1])


def test_expert_and_tf_share_one_fk_function(panda) -> None:
    from sim2real.expert import ik as expert_ik
    from sim2real.sim import fake_backend, kinematics

    # The fake backend (the TF source in CI) and the expert use the same
    # function objects, not re-implementations.
    assert fake_backend.forward_kinematics is kinematics.forward_kinematics
    assert fake_backend.hand_pose is kinematics.hand_pose
    assert fake_backend.tcp_pose is kinematics.tcp_pose
    assert expert_ik.tcp_pose is kinematics.tcp_pose
    assert expert_ik.numeric_jacobian is kinematics.numeric_jacobian

    fake_link_poses = fake_backend.FakeBackend(CONFIG_DIR).link_poses()
    q = np.asarray(panda.home_arm, dtype=np.float64)
    expected = forward_kinematics(q, panda)
    assert set(fake_link_poses) == set(expected)
    for frame, (parent, pose) in expected.items():
        assert fake_link_poses[frame][0] == parent
        np.testing.assert_allclose(fake_link_poses[frame][1], pose)


@pytest.mark.sim
def test_mujoco_fk_cross_check(panda) -> None:
    import mujoco

    from sim2real.sim.mujoco_backend import MuJoCoBackend

    backend = MuJoCoBackend(CONFIG_DIR)
    try:
        max_error = 0.0
        limits = np.asarray(panda.joint_limits, dtype=np.float64)
        rng = np.random.default_rng(2026)
        for _ in range(5):
            q = rng.uniform(limits[:, 0], limits[:, 1])
            backend.data.qpos[:] = 0.0
            for index, name in enumerate(panda.mujoco_joint_names):
                address = backend.model.jnt_qposadr[backend._joint_id(name)]
                backend.data.qpos[address] = q[index]
            mujoco.mj_kinematics(backend.model, backend.data)
            poses = forward_kinematics(q, panda)
            for ros_name, (_, pose) in poses.items():
                mujoco_name = dict(panda.body_frames)[ros_name]
                body = backend.model.body(mujoco_name)
                max_error = max(
                    max_error,
                    float(np.max(np.abs(pose[:3, 3] - backend.data.xpos[body.id]))),
                    float(np.max(np.abs(pose[:3, :3] - backend.data.xmat[body.id].reshape(3, 3)))),
                )
        print(f"measured max FK discrepancy vs MuJoCo: {max_error:.3e}")
        assert max_error < 1e-6
    finally:
        backend.close()
