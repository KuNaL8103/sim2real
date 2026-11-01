"""Pure conversion between ROS-style field dictionaries and ``EpisodeRecord``.

The ROS adapter extracts message fields into plain dictionaries and calls this
module; tests cover the same path without ``rclpy``. Conversion is deliberately
strict — a malformed message raises ``ConversionError`` and the node catches it
and publishes an ``INVALID`` decision instead of crashing.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from sim2real import contracts
from sim2real.expert.rollout import EpisodeRecord, StepRecord
from sim2real.sim.camera import make_pose, quat_wxyz_to_matrix


class ConversionError(ValueError):
    """Raised when message fields cannot be converted to an episode record."""


def _pose_matrix(raw: Any) -> np.ndarray:
    if not isinstance(raw, dict):
        raise ConversionError("ee_pose must be a mapping")
    position = np.asarray(raw.get("position"), dtype=np.float64)
    if position.shape != (3,):
        raise ConversionError("ee_pose.position must have three values")
    if "orientation_matrix" in raw:
        rotation = np.asarray(raw["orientation_matrix"], dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ConversionError("ee_pose.orientation_matrix must be 3x3")
    elif "orientation" in raw:
        quaternion = np.asarray(raw["orientation"], dtype=np.float64).reshape(-1)
        if quaternion.size != 4:
            raise ConversionError("ee_pose.orientation must have four values (x, y, z, w)")
        x, y, z, w = (float(value) for value in quaternion)
        rotation = quat_wxyz_to_matrix((w, x, y, z))
    else:
        raise ConversionError("ee_pose has no orientation")
    return make_pose(position, rotation)


def _step(raw: Any, index: int) -> StepRecord:
    if not isinstance(raw, dict):
        raise ConversionError(f"step {index} must be a mapping")
    try:
        stamp_ns = int(raw["stamp_ns"])
        action = np.asarray(raw["action"], dtype=np.float64)
        gripper = float(raw["gripper_position"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConversionError(f"step {index} is missing or has an invalid required field") from exc
    return StepRecord(
        stamp_ns=stamp_ns,
        step_index=int(raw.get("step_index", index)),
        joint_positions=np.asarray(raw["joint_positions"], dtype=np.float64),
        joint_velocities=np.asarray(raw["joint_velocities"], dtype=np.float64),
        ee_pose=_pose_matrix(raw.get("ee_pose")),
        action=action,
        action_space=int(raw.get("action_space", contracts.ACTION_JOINT_POSITION_TARGET)),
        gripper_position=gripper,
        unintended_collision=bool(raw.get("unintended_collision", False)),
        state=str(raw.get("state", "UNKNOWN")),
    )


def _metadata(fields: dict[str, Any]) -> dict[str, Any]:
    raw = fields.get("metadata_json")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise ConversionError("metadata_json is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ConversionError("metadata_json must be a JSON object")
    return parsed


def fields_to_episode(fields: dict[str, Any]) -> EpisodeRecord:
    """Build an ``EpisodeRecord`` from wire-style fields."""
    if not isinstance(fields, dict):
        raise ConversionError("episode fields must be a mapping")
    try:
        episode_id = str(fields["episode_id"])
        steps_raw = fields["steps"]
    except KeyError as exc:
        raise ConversionError(f"episode is missing {exc.args[0]!r}") from exc
    if not isinstance(steps_raw, list):
        raise ConversionError("steps must be a list")
    metadata = _metadata(fields)
    steps = [_step(raw, index) for index, raw in enumerate(steps_raw)]
    return EpisodeRecord(
        episode_id=episode_id,
        run_id=str(fields.get("run_id") or metadata.get("run_id") or ""),
        episode_index=int(fields.get("episode_index", metadata.get("episode_index", 0)) or 0),
        seed=int(fields.get("seed", 0) or 0),
        randomization_seed=int(
            fields.get("randomization_seed", metadata.get("randomization_seed", 0)) or 0
        ),
        backend=str(fields.get("backend", "")),
        task_instruction=str(fields.get("task_instruction", "")),
        task_object_id=str(metadata.get("task_object_id", "")),
        task_zone_id=str(metadata.get("task_zone_id", "")),
        control_rate_hz=float(fields.get("control_rate_hz", 20.0) or 20.0),
        outcome=int(fields.get("outcome", contracts.OUTCOME_UNKNOWN) or 0),
        failure_reason=str(fields.get("failure_reason", "")),
        is_perturbed=bool(fields.get("is_perturbed", False)),
        perturbation_name=metadata.get("perturbation_name"),
        expected_quality=str(metadata.get("expected_quality", "clean")),
        steps=steps,
        transitions=list(metadata.get("transitions") or []),
        randomization_report=dict(metadata.get("randomization") or {}),
        success_metrics=dict(metadata.get("success_metrics") or {}),
        metadata=dict(metadata),
    )


__all__ = ["ConversionError", "fields_to_episode"]
