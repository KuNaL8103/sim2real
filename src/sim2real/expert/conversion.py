"""Pure conversion from :class:`EpisodeRecord` to ROS message field dictionaries.

This module must stay importable without ``rclpy``: the ROS node turns these
dictionaries into ``sim2real_msgs`` messages, while tests validate the contract
without a ROS build.

Contract rules enforced here:

* step stamps are monotonic non-decreasing;
* every step carries all seven arm joints plus the gripper in the action;
* ``metadata_json`` passes :func:`sim2real.metadata.validate_metadata_json`;
* images are never embedded (they stay on disk, referenced from metadata).
"""

from __future__ import annotations

import json
from typing import Any

from sim2real import contracts
from sim2real.expert.rollout import EpisodeRecord
from sim2real.metadata import validate_metadata_json


class ConversionError(ValueError):
    """Raised when a record cannot be represented by the wire contract."""


def _pose_matrix_to_fields(pose: Any) -> dict[str, list[float]]:
    position = [float(pose[0, 3]), float(pose[1, 3]), float(pose[2, 3])]
    # Column-major-agnostic quaternion is not needed here; the ROS adapter does
    # the matrix -> quaternion conversion. Keep the matrix as the source of truth.
    return {
        "position": position,
        "orientation_matrix": [[float(value) for value in row[:3]] for row in pose[:3, :3]],
    }


def step_to_fields(record: EpisodeRecord, step: Any) -> dict[str, Any]:
    return {
        "stamp_ns": int(step.stamp_ns),
        "step_index": int(step.step_index),
        "task_instruction": record.task_instruction,
        "joint_names": list(contracts.CONTROLLED_JOINT_NAMES),
        "joint_positions": [float(value) for value in step.joint_positions],
        "joint_velocities": [float(value) for value in step.joint_velocities],
        "ee_pose": _pose_matrix_to_fields(step.ee_pose),
        "ee_frame_id": contracts.PANDA_HAND_FRAME,
        "action": [float(value) for value in step.action],
        "action_space": int(step.action_space),
        "state": step.state,
        "unintended_collision": bool(step.unintended_collision),
    }


def metadata_json(record: EpisodeRecord) -> str:
    text = json.dumps(record.metadata_dict(), sort_keys=True, allow_nan=False)
    validate_metadata_json(text)
    return text


def episode_to_rollout_fields(record: EpisodeRecord) -> dict[str, Any]:
    """Return fields matching ``sim2real_msgs/msg/RolloutEpisode``.

    ``ee_pose`` in each step is returned as ``{position, orientation_matrix}``;
    the ROS adapter converts it to ``geometry_msgs/Pose``.
    """
    if not record.steps:
        raise ConversionError(f"episode {record.episode_id!r} has no steps")
    stamps = [int(step.stamp_ns) for step in record.steps]
    if any(current < previous for previous, current in zip(stamps, stamps[1:], strict=False)):
        raise ConversionError(f"episode {record.episode_id!r} has non-monotonic step stamps")
    if record.outcome not in {
        contracts.OUTCOME_UNKNOWN,
        contracts.OUTCOME_SUCCESS,
        contracts.OUTCOME_FAILURE,
        contracts.OUTCOME_TIMEOUT,
        contracts.OUTCOME_ABORTED,
    }:
        raise ConversionError(f"episode {record.episode_id!r} has invalid outcome")

    metadata = metadata_json(record)
    declared_steps = int(json.loads(metadata).get("num_steps", -1))
    if declared_steps != len(record.steps):
        raise ConversionError(
            f"metadata num_steps={declared_steps} but episode has {len(record.steps)} steps"
        )

    return {
        "episode_id": record.episode_id,
        "schema_version": contracts.SCHEMA_VERSION,
        "backend": record.backend,
        "seed": int(record.seed),
        "task_instruction": record.task_instruction,
        "start_stamp_ns": int(record.start_stamp_ns),
        "control_rate_hz": float(record.control_rate_hz),
        "is_perturbed": bool(record.is_perturbed),
        "outcome": int(record.outcome),
        "failure_reason": record.failure_reason,
        "steps": [step_to_fields(record, step) for step in record.steps],
        "metadata_json": metadata,
    }


def summarize_outcomes(records: list[EpisodeRecord]) -> dict[str, int]:
    counts = {
        "success": 0,
        "failure": 0,
        "timeout": 0,
        "aborted": 0,
        "unknown": 0,
    }
    mapping = {
        contracts.OUTCOME_SUCCESS: "success",
        contracts.OUTCOME_FAILURE: "failure",
        contracts.OUTCOME_TIMEOUT: "timeout",
        contracts.OUTCOME_ABORTED: "aborted",
        contracts.OUTCOME_UNKNOWN: "unknown",
    }
    for record in records:
        counts[mapping.get(record.outcome, "unknown")] += 1
    return counts
