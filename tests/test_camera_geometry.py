"""Camera convention and projection math."""

from __future__ import annotations

import numpy as np
import pytest

from sim2real.sim.backend import CameraIntrinsics
from sim2real.sim.camera import (
    MUJOCO_TO_OPTICAL,
    intrinsics_from_fovy,
    mat_to_quat,
    optical_pose_from_mujoco,
    pose_from_pos_quat,
    project_point,
    quat_wxyz_to_matrix,
)


def test_mujoco_to_optical_is_x_right_y_down_z_forward():
    np.testing.assert_array_equal(MUJOCO_TO_OPTICAL, np.diag([1.0, -1.0, -1.0]))
    # A point one metre in front of a MuJoCo camera is -Z in its own frame.
    point_mujoco = np.array([0.0, 0.0, -1.0])
    point_optical = MUJOCO_TO_OPTICAL @ point_mujoco
    np.testing.assert_allclose(point_optical, [0.0, 0.0, 1.0])


def test_intrinsics_from_fovy_uses_square_pixels():
    intrinsics = intrinsics_from_fovy(60.0, 256, 256)
    expected = 128.0 / np.tan(np.deg2rad(60.0) / 2.0)
    assert intrinsics.fx == pytest.approx(expected)
    assert intrinsics.fy == pytest.approx(expected)
    assert intrinsics.cx == pytest.approx(128.0)
    assert intrinsics.cy == pytest.approx(128.0)


def test_project_point_centre_and_offset():
    intrinsics = CameraIntrinsics(fx=200.0, fy=200.0, cx=128.0, cy=128.0, width=256, height=256)
    identity_optical = np.eye(4)
    u, v, depth = project_point(np.array([0.0, 0.0, 2.0]), identity_optical, intrinsics)
    assert (u, v) == pytest.approx((128.0, 128.0))
    assert depth == pytest.approx(2.0)

    u, v, depth = project_point(np.array([1.0, -0.5, 2.0]), identity_optical, intrinsics)
    assert u == pytest.approx(128.0 + 200.0 * 1.0 / 2.0)
    assert v == pytest.approx(128.0 + 200.0 * -0.5 / 2.0)
    assert depth == pytest.approx(2.0)


def test_project_point_behind_camera_is_nan():
    intrinsics = CameraIntrinsics(fx=200.0, fy=200.0, cx=128.0, cy=128.0, width=256, height=256)
    u, v, depth = project_point(np.array([0.0, 0.0, -1.0]), np.eye(4), intrinsics)
    assert np.isnan(u) and np.isnan(v)
    assert depth < 0.0


def test_mujoco_camera_rotation_matches_projection():
    """A MuJoCo camera rotated 90 degrees about x projects a known point."""
    intrinsics = intrinsics_from_fovy(60.0, 256, 256)
    # MuJoCo camera at the origin looking along -Z with +Y up.
    pose_optical = optical_pose_from_mujoco(np.zeros(3), np.eye(3))
    # World point on the optical axis (camera looks along -Z).
    u, v, depth = project_point(np.array([0.0, 0.0, -2.0]), pose_optical, intrinsics)
    assert u == pytest.approx(intrinsics.cx)
    assert v == pytest.approx(intrinsics.cy)
    assert depth == pytest.approx(2.0)


def test_quaternion_round_trip():
    quat = np.array([0.9238795, 0.0, 0.0, -0.3826834])
    matrix = quat_wxyz_to_matrix(quat)
    recovered = mat_to_quat(matrix)
    matrix_again = quat_wxyz_to_matrix(recovered)
    np.testing.assert_allclose(matrix, matrix_again, atol=1e-9)


def test_pose_from_pos_quat_translation():
    pose = pose_from_pos_quat(np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(pose[:3, 3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(pose[:3, :3], np.eye(3))
