"""Pure helpers that turn joint state into the policy's 8-D observation vector.

The closed-loop evaluation runner, the ROS VLA inference node, and the dataset
side all need the same layout: seven arm joint positions followed by one
gripper coordinate. Keeping the concatenation here (not in a ROS adapter) means
there is exactly one implementation.

The gripper value is the ``panda_finger_joint1`` slide coordinate in metres,
range ``[0, 0.04]``; it is the value the backend reports as
``RobotState.gripper_position`` and the value the dataset stores in state
dimension 8. It is **not** the sum of the two finger positions.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from sim2real import contracts

STATE_DIM = 8


def state_from_parts(joint_positions: Sequence[float], gripper_position: float) -> np.ndarray:
    """Return ``[q0..q6, gripper]`` as float64.

    This is deliberately a plain concatenation. Structural validation lives in
    :mod:`sim2real.inference.observation` and in ``PolicyObservation`` so the
    evaluation runner keeps its historical error behaviour.
    """
    joints = np.asarray(joint_positions, dtype=np.float64)
    return np.concatenate([joints, [float(gripper_position)]])


def positions_from_named_state(
    names: Sequence[str],
    positions: Sequence[float],
    *,
    expected_names: Sequence[str] = contracts.CONTROLLED_JOINT_NAMES,
) -> tuple[np.ndarray, float]:
    """Extract the seven arm positions and gripper value by joint name.

    Raises ``ValueError`` when a required name is missing, a name is duplicated,
    or the two arrays disagree in length. Values are returned in
    ``expected_names`` order regardless of the publisher's order.
    """
    if len(names) != len(positions):
        raise ValueError(f"names and positions differ in length: {len(names)} != {len(positions)}")
    index: dict[str, int] = {}
    for offset, name in enumerate(names):
        if name in index:
            raise ValueError(f"duplicate joint name {name!r}")
        index[str(name)] = offset
    missing = [name for name in expected_names if name not in index]
    if missing:
        raise ValueError(f"missing joint names {missing}")
    values = [float(positions[index[name]]) for name in expected_names]
    return np.asarray(values[:-1], dtype=np.float64), values[-1]


__all__ = ["STATE_DIM", "positions_from_named_state", "state_from_parts"]
