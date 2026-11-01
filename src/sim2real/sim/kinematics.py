"""Single-source Franka Panda forward kinematics used for TF and the expert.

Why this module exists
----------------------
The sim node publishes dynamic TF from ``SimBackend.link_poses()`` driven by
``tf_source=backend_fk``. ``FakeBackend`` had a *placeholder* analytic FK
(three sine/cosine terms) that did not represent the Panda. The expert needs a
physically correct TCP pose for IK, and duplicating a second FK would let TF and
the planner disagree silently.

This module is therefore the one pure-NumPy Panda FK. It mirrors the pinned
Menagerie MJCF body tree exactly (``assets/mujoco/franka_emika_panda/panda.xml``)
and is used by both ``FakeBackend`` (TF, end-effector) and the expert planner.
``MuJoCoBackend`` keeps the compiled model's own FK as ground truth; a test
cross-validates this implementation against it at random configurations.

Nothing here imports MuJoCo or ROS.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from sim2real.sim.camera import make_pose, quat_wxyz_to_matrix
from sim2real.sim.config import PandaConfig

_SQRT1_2 = float(np.sqrt(0.5))
_HALF_PI_QUARTER = np.deg2rad(22.5)
_HAND_COS = float(np.cos(_HALF_PI_QUARTER))
_HAND_SIN = float(np.sin(_HALF_PI_QUARTER))

# (mujoco_body, parent_mujoco_body, position_xyz, quat_wxyz, joint_index|None)
# Values copied verbatim from the pinned Menagerie MJCF. Every hinge rotates
# about the body-local +Z axis; the body quaternions re-orient the frames.
PANDA_BODY_CHAIN: tuple[
    tuple[
        str, str | None, tuple[float, float, float], tuple[float, float, float, float], int | None
    ],
    ...,
] = (
    ("link0", None, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), None),
    ("link1", "link0", (0.0, 0.0, 0.333), (1.0, 0.0, 0.0, 0.0), 0),
    ("link2", "link1", (0.0, 0.0, 0.0), (_SQRT1_2, -_SQRT1_2, 0.0, 0.0), 1),
    ("link3", "link2", (0.0, -0.316, 0.0), (_SQRT1_2, _SQRT1_2, 0.0, 0.0), 2),
    ("link4", "link3", (0.0825, 0.0, 0.0), (_SQRT1_2, _SQRT1_2, 0.0, 0.0), 3),
    ("link5", "link4", (-0.0825, 0.384, 0.0), (_SQRT1_2, -_SQRT1_2, 0.0, 0.0), 4),
    ("link6", "link5", (0.0, 0.0, 0.0), (_SQRT1_2, _SQRT1_2, 0.0, 0.0), 5),
    ("link7", "link6", (0.088, 0.0, 0.0), (_SQRT1_2, _SQRT1_2, 0.0, 0.0), 6),
    ("hand", "link7", (0.0, 0.0, 0.107), (_HAND_COS, 0.0, 0.0, -_HAND_SIN), None),
)


class KinematicsError(ValueError):
    """Raised when a joint vector is malformed or the model is unmapped."""


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _joint_vector(q: np.ndarray) -> np.ndarray:
    vector = np.asarray(q, dtype=np.float64)
    if vector.shape != (7,):
        raise KinematicsError(f"joint vector must have shape (7,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise KinematicsError("joint vector must be finite")
    return vector


def _chain_for(
    panda: PandaConfig,
) -> tuple[tuple[str, str | None, np.ndarray, np.ndarray, int | None], ...]:
    """Map the MJCF chain onto the ROS frame names in ``panda.body_frames``."""
    mujoco_to_ros = {mujoco: ros for ros, mujoco in panda.body_frames}
    entries = []
    for mujoco_name, parent_mujoco, position, quat, joint in PANDA_BODY_CHAIN:
        if mujoco_name not in mujoco_to_ros:
            continue
        parent = None if parent_mujoco is None else mujoco_to_ros.get(parent_mujoco)
        entries.append(
            (
                mujoco_to_ros[mujoco_name],
                parent,
                np.asarray(position, dtype=np.float64),
                quat_wxyz_to_matrix(quat),
                joint,
            )
        )
    if not entries or entries[-1][0] != "panda_hand":
        raise KinematicsError("panda.body_frames does not map the full chain to panda_hand")
    return tuple(entries)


def forward_kinematics(
    q: np.ndarray, panda: PandaConfig
) -> dict[str, tuple[str | None, np.ndarray]]:
    """Return ``ros_frame -> (parent_frame, world_pose)`` for the Panda chain.

    This is the same dict shape as ``SimBackend.link_poses()``, so ``FakeBackend``
    can return it directly and the sim node's TF path consumes it unchanged.
    """
    angles = _joint_vector(q)
    poses: dict[str, tuple[str | None, np.ndarray]] = {}
    for ros_name, parent, position, rotation, joint in _chain_for(panda):
        local = make_pose(position, rotation)
        if joint is not None:
            local = local @ make_pose(np.zeros(3), _rotation_z(float(angles[joint])))
        world = local if parent is None else poses[parent][1] @ local
        poses[ros_name] = (parent, world)
    return poses


def hand_pose(q: np.ndarray, panda: PandaConfig) -> np.ndarray:
    """World pose of ``panda_hand`` for joint configuration ``q``."""
    return forward_kinematics(q, panda)["panda_hand"][1]


def tcp_pose(q: np.ndarray, panda: PandaConfig) -> np.ndarray:
    """World pose of the tool centre point (hand pose plus ``tcp_offset``)."""
    offset = np.asarray(panda.tcp_offset, dtype=np.float64)
    return hand_pose(q, panda) @ make_pose(offset, np.eye(3))


def clamp_joints(q: np.ndarray, panda: PandaConfig) -> np.ndarray:
    """Clamp ``q`` to the configured Panda joint limits."""
    limits = np.asarray(panda.joint_limits, dtype=np.float64)
    return np.clip(_joint_vector(q), limits[:, 0], limits[:, 1])


def at_joint_limit(q: np.ndarray, panda: PandaConfig, tolerance: float = 1e-9) -> np.ndarray:
    """Boolean mask of joints within ``tolerance`` of a limit."""
    limits = np.asarray(panda.joint_limits, dtype=np.float64)
    values = _joint_vector(q)
    return (np.abs(values - limits[:, 0]) <= tolerance) | (
        np.abs(values - limits[:, 1]) <= tolerance
    )


def rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """Axis-angle vector (``so(3)`` log) for a 3x3 rotation matrix."""
    matrix = np.asarray(rotation, dtype=np.float64)
    cosine = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-12:
        return np.zeros(3, dtype=np.float64)
    axis = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        # Near pi: derive the axis from the diagonal.
        diagonal = np.clip((np.diag(matrix) + 1.0) / 2.0, 0.0, 1.0)
        axis = np.sqrt(diagonal)
        axis[1] *= np.sign(matrix[0, 1] + matrix[1, 0]) or 1.0
        axis[2] *= np.sign(matrix[0, 2] + matrix[2, 0]) or 1.0
        norm = float(np.linalg.norm(axis))
        if norm < 1e-12:
            return np.zeros(3, dtype=np.float64)
    return axis / norm * angle


def pose_error(current: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Position error (target - current) and orientation error in the world frame."""
    current = np.asarray(current, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    position = target[:3, 3] - current[:3, 3]
    rotation = rotation_vector(target[:3, :3] @ current[:3, :3].T)
    return position, rotation


def numeric_jacobian(q: np.ndarray, panda: PandaConfig, step_rad: float = 1e-6) -> np.ndarray:
    """Finite-difference TCP Jacobian, shape ``(6, 7)``.

    Rows 0-2 are linear velocity and rows 3-5 angular velocity. The step is
    central where both neighbours are inside the joint limits and one-sided at
    a limit. ``step_rad`` is documented in ``configs/expert/default.yaml``.
    """
    if step_rad <= 0.0:
        raise KinematicsError("jacobian step must be positive")
    values = _joint_vector(q)
    limits = np.asarray(panda.joint_limits, dtype=np.float64)
    jacobian = np.zeros((6, 7), dtype=np.float64)
    for joint in range(7):
        plus = values.copy()
        minus = values.copy()
        delta = float(step_rad)
        if values[joint] + delta > limits[joint, 1]:
            plus[joint] = values[joint]
            minus[joint] = values[joint] - delta
            denominator = delta
        elif values[joint] - delta < limits[joint, 0]:
            plus[joint] = values[joint] + delta
            minus[joint] = values[joint]
            denominator = delta
        else:
            plus[joint] += delta
            minus[joint] -= delta
            denominator = 2.0 * delta
        plus_pose = tcp_pose(plus, panda)
        minus_pose = tcp_pose(minus, panda)
        jacobian[:3, joint] = (plus_pose[:3, 3] - minus_pose[:3, 3]) / denominator
        relative = plus_pose[:3, :3] @ minus_pose[:3, :3].T
        jacobian[3:, joint] = rotation_vector(relative) / denominator
    return jacobian


def describe_chain(panda: PandaConfig) -> Mapping[str, str]:
    """Human-readable frame chain, used by diagnostics and tests."""
    return {
        ros_name: (parent if parent is not None else "<world>")
        for ros_name, parent, _, _, _ in _chain_for(panda)
    }
