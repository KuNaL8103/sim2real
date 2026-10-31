"""SimBackend contract, parametrized over fake and MuJoCo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sim2real.sim import create_backend
from sim2real.sim.config import load_sim_configs

CONFIG_DIR = Path("configs/sim")
REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS = REPO_ROOT / "assets" / "mujoco" / "franka_emika_panda"


def _make(name: str):
    if name == "mujoco":
        pytest.importorskip("mujoco")
        if not (ASSETS / "panda.xml").is_file():
            pytest.skip("Menagerie assets missing; run scripts/fetch_assets.sh")
    backend = create_backend(name, CONFIG_DIR)
    return backend


@pytest.mark.parametrize("name", ["fake", "mujoco"])
def test_backend_contract(name):
    backend = _make(name)
    try:
        configs = load_sim_configs(CONFIG_DIR)
        observation = backend.reset(seed=7)

        assert backend.joint_names == configs.panda.joint_names
        assert backend.joint_limits.shape == (7, 2)
        assert np.all(np.isfinite(backend.joint_limits))
        assert np.all(backend.joint_limits[:, 0] < backend.joint_limits[:, 1])

        assert observation.robot_state.joint_positions.shape == (7,)
        assert observation.robot_state.joint_velocities.shape == (7,)
        assert observation.robot_state.ee_pose.shape == (4, 4)
        assert np.all(np.isfinite(observation.robot_state.joint_positions))
        assert observation.stamp_ns == 0

        for camera in ("overhead", "wrist"):
            intrinsics = backend.get_camera_info(camera)
            assert intrinsics.width == 256
            assert intrinsics.height == 256
            assert intrinsics.fx > 0.0 and intrinsics.fy > 0.0

        first = backend.reset(seed=7)
        second = backend.reset(seed=7)
        np.testing.assert_array_equal(
            first.robot_state.joint_positions, second.robot_state.joint_positions
        )
        for frame, pose in first.object_poses.items():
            np.testing.assert_array_equal(pose, second.object_poses[frame])

        result = backend.step(backend.joint_limits[:, 1] + 1.0, gripper_target=99.0)
        positions = result.observation.robot_state.joint_positions
        assert np.all(positions <= backend.joint_limits[:, 1] + 1e-3)
        assert np.all(positions >= backend.joint_limits[:, 0] - 1e-3)

        report = backend.collision_report()
        assert isinstance(report.num_unintended_contacts, int)
        assert report.num_unintended_contacts >= 0
    finally:
        backend.close()


def test_mujoco_limits_match_loaded_model():
    pytest.importorskip("mujoco")
    if not (ASSETS / "panda.xml").is_file():
        pytest.skip("Menagerie assets missing; run scripts/fetch_assets.sh")
    import mujoco

    configs = load_sim_configs(CONFIG_DIR)
    model = mujoco.MjModel.from_xml_path(str(configs.panda.model))
    for index, joint_name in enumerate(configs.panda.mujoco_joint_names):
        joint = model.joint(joint_name)
        np.testing.assert_allclose(
            configs.panda.joint_limits[index],
            model.jnt_range[joint.id],
            atol=1e-6,
            err_msg=f"limits for {joint_name} drifted from the MJCF",
        )
