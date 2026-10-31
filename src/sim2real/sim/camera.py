"""Camera geometry shared by the fake and MuJoCo backends.

Two conventions meet here:

* MuJoCo cameras look along local ``-Z`` with ``+Y`` up.
* ROS optical frames (REP-103/REP-105) are ``x`` right, ``y`` down, ``z``
  forward.

``MUJOCO_TO_OPTICAL`` converts a MuJoCo camera frame into a ROS optical frame.
All poses returned by backends use the ROS optical convention, so a point at
``(0, 0, 1)`` in camera coordinates is one metre in front of the lens.
"""

from __future__ import annotations

import numpy as np

from sim2real.sim.backend import CameraIntrinsics

# x right, y down, z forward = (x, -y, -z) of the MuJoCo camera frame.
MUJOCO_TO_OPTICAL: np.ndarray = np.diag([1.0, -1.0, -1.0])


def quat_wxyz_to_matrix(quat: tuple[float, float, float, float] | np.ndarray) -> np.ndarray:
    """Return a 3x3 rotation matrix for a ``(w, x, y, z)`` unit quaternion."""
    w, x, y, z = (float(value) for value in quat)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError("quaternion must be non-zero")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def make_pose(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from a 3-vector and 3x3 rotation."""
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = position
    return pose


def euler_xyz_to_matrix(angles: np.ndarray | tuple[float, float, float]) -> np.ndarray:
    """Return ``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` for ``(roll, pitch, yaw)`` radians."""
    roll, pitch, yaw = (float(value) for value in angles)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rotate_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    rotate_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rotate_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rotate_z @ rotate_y @ rotate_x


def pose_from_pos_quat(position: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    return make_pose(np.asarray(position, dtype=np.float64), quat_wxyz_to_matrix(quat_wxyz))


def invert_pose(pose: np.ndarray) -> np.ndarray:
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def optical_pose_from_mujoco(cam_pos: np.ndarray, cam_mat: np.ndarray) -> np.ndarray:
    """Convert MuJoCo ``cam_xpos``/``cam_xmat`` into a ROS optical pose."""
    return make_pose(np.asarray(cam_pos), np.asarray(cam_mat) @ MUJOCO_TO_OPTICAL)


def intrinsics_from_fovy(fovy_deg: float, width: int, height: int) -> CameraIntrinsics:
    """Pinhole intrinsics for square pixels from a vertical field of view."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    fy = (height / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
    return CameraIntrinsics(
        fx=float(fy), fy=float(fy), cx=width / 2.0, cy=height / 2.0, width=width, height=height
    )


def project_point(
    point_world: np.ndarray, pose_world_optical: np.ndarray, intrinsics: CameraIntrinsics
) -> tuple[float, float, float]:
    """Project a world point into pixels. Returns ``(u, v, depth_m)``.

    ``depth_m`` is the optical-frame z coordinate; a non-positive value means
    the point is behind the camera and must not be drawn.
    """
    point = np.asarray(point_world, dtype=np.float64)
    point_h = np.append(point, 1.0)
    point_camera = (invert_pose(pose_world_optical) @ point_h)[:3]
    depth = float(point_camera[2])
    if depth <= 0.0:
        return (np.nan, np.nan, depth)
    u = intrinsics.fx * point_camera[0] / depth + intrinsics.cx
    v = intrinsics.fy * point_camera[1] / depth + intrinsics.cy
    return (float(u), float(v), depth)


def look_at_quat(
    position: np.ndarray, target: np.ndarray, up: np.ndarray | None = None
) -> np.ndarray:
    """Return a MuJoCo camera quaternion that looks from ``position`` at ``target``.

    MuJoCo cameras look along ``-Z``; the returned quaternion is ``(w, x, y, z)``.
    """
    if up is None:
        up = np.array([0.0, 0.0, 1.0])
    position = np.asarray(position, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - position
    norm = np.linalg.norm(forward)
    if norm == 0.0:
        raise ValueError("position and target must differ")
    forward /= norm
    up = np.asarray(up, dtype=np.float64)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-9:
        raise ValueError("up vector is parallel to the view direction")
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    # Columns are the camera axes in world coordinates: +X right, +Y up, -Z forward.
    rotation = np.column_stack((right, true_up, -forward))
    return mat_to_quat(rotation)


def mat_to_quat(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a ``(w, x, y, z)`` quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = [
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        ]
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        quat = [
            (matrix[2, 1] - matrix[1, 2]) / scale,
            0.25 * scale,
            (matrix[0, 1] + matrix[1, 0]) / scale,
            (matrix[0, 2] + matrix[2, 0]) / scale,
        ]
    elif matrix[1, 1] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        quat = [
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[0, 1] + matrix[1, 0]) / scale,
            0.25 * scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
        ]
    else:
        scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        quat = [
            (matrix[1, 0] - matrix[0, 1]) / scale,
            (matrix[0, 2] + matrix[2, 0]) / scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
            0.25 * scale,
        ]
    return np.asarray(quat, dtype=np.float64)
