"""Episode validation for trajectory filtering.

Validation never raises for malformed data: it returns a list of structured
issues, and the decision engine maps a non-empty list to ``INVALID``. This keeps
one bad record from stopping a batch or crashing a ROS callback.

The validator reads only recorded state (joint positions/velocities, end-effector
pose, timestamps, actions, collision flags, gripper state). It deliberately does
**not** read ``is_perturbed``, ``perturbation_name``, or ``expected_quality``;
those are diagnostic labels and must never influence a decision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from sim2real.filtering.config import ValidationConfig

MALFORMED_EPISODE = "MALFORMED_EPISODE"
NAN_VALUES = "NAN_VALUES"
TOO_SHORT = "TOO_SHORT"
NON_MONOTONIC_TIME = "NON_MONOTONIC_TIME"
DT_JITTER = "DT_JITTER"
DUPLICATE_EPISODE = "DUPLICATE_EPISODE"
SHAPE_MISMATCH = "SHAPE_MISMATCH"
ACTION_DIM = "ACTION_DIM"
MISSING_FIELD = "MISSING_FIELD"

#: Deterministic order used whenever issue codes are reported.
ISSUE_ORDER = (
    MALFORMED_EPISODE,
    MISSING_FIELD,
    SHAPE_MISMATCH,
    ACTION_DIM,
    NAN_VALUES,
    TOO_SHORT,
    NON_MONOTONIC_TIME,
    DT_JITTER,
    DUPLICATE_EPISODE,
)

_REQUIRED_STEP_FIELDS = (
    "stamp_ns",
    "joint_positions",
    "joint_velocities",
    "ee_pose",
    "action",
    "gripper_position",
    "unintended_collision",
)


@dataclass(frozen=True)
class ValidationIssue:
    """One reason an episode is not usable."""

    code: str
    message: str
    field: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "field": self.field}


def _issue(code: str, message: str, field: str | None = None) -> ValidationIssue:
    return ValidationIssue(code=code, message=message, field=field)


def _finite(array: Any, name: str, issues: list[ValidationIssue]) -> np.ndarray | None:
    try:
        values = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError):
        issues.append(_issue(SHAPE_MISMATCH, f"{name} is not numeric", name))
        return None
    if values.size == 0:
        issues.append(_issue(SHAPE_MISMATCH, f"{name} is empty", name))
        return None
    if not np.all(np.isfinite(values)):
        issues.append(_issue(NAN_VALUES, f"{name} contains NaN or Infinity", name))
        return None
    return values


def validate_episode(
    record: Any,
    config: ValidationConfig | None = None,
    *,
    seen_ids: set[str] | None = None,
) -> list[ValidationIssue]:
    """Return every validation issue; an empty list means the record is usable."""
    settings = config or ValidationConfig()
    issues: list[ValidationIssue] = []
    try:
        return _validate(record, settings, seen_ids, issues)
    except Exception as exc:  # noqa: BLE001 - malformed input must never raise
        return issues + [_issue(MALFORMED_EPISODE, f"{type(exc).__name__}: {exc}", None)]


def _validate(
    record: Any,
    config: ValidationConfig,
    seen_ids: set[str] | None,
    issues: list[ValidationIssue],
) -> list[ValidationIssue]:
    if record is None:
        return [_issue(MALFORMED_EPISODE, "record is None")]

    episode_id = getattr(record, "episode_id", None)
    if not isinstance(episode_id, str) or not episode_id.strip():
        issues.append(_issue(MALFORMED_EPISODE, "episode_id must be a non-empty string"))
    elif seen_ids is not None and episode_id in seen_ids:
        issues.append(
            _issue(
                DUPLICATE_EPISODE,
                f"episode_id {episode_id!r} was already processed in this run",
                "episode_id",
            )
        )

    steps = getattr(record, "steps", None)
    if not isinstance(steps, list) or not steps:
        issues.append(_issue(MALFORMED_EPISODE, "record has no steps", "steps"))
        return _sorted(issues)
    if len(steps) < config.min_steps:
        issues.append(
            _issue(
                TOO_SHORT,
                f"episode has {len(steps)} steps; minimum is {config.min_steps}",
                "steps",
            )
        )

    stamps: list[int] = []
    for index, step in enumerate(steps):
        missing = [name for name in _REQUIRED_STEP_FIELDS if not hasattr(step, name)]
        if missing:
            issues.append(
                _issue(
                    MISSING_FIELD,
                    f"step {index} is missing {missing}",
                    f"steps[{index}]",
                )
            )
            continue

        stamp = getattr(step, "stamp_ns", None)
        if isinstance(stamp, bool) or not isinstance(stamp, int):
            issues.append(
                _issue(SHAPE_MISMATCH, f"step {index} stamp_ns is not an int", "stamp_ns")
            )
        else:
            stamps.append(stamp)

        positions = _finite(step.joint_positions, f"steps[{index}].joint_positions", issues)
        if positions is not None and positions.shape != (7,):
            issues.append(
                _issue(
                    SHAPE_MISMATCH,
                    f"step {index} joint_positions must have 7 values, got {positions.shape}",
                    "joint_positions",
                )
            )
        velocities = _finite(step.joint_velocities, f"steps[{index}].joint_velocities", issues)
        if velocities is not None and velocities.shape != (7,):
            issues.append(
                _issue(
                    SHAPE_MISMATCH,
                    f"step {index} joint_velocities must have 7 values, got {velocities.shape}",
                    "joint_velocities",
                )
            )
        pose = _finite(step.ee_pose, f"steps[{index}].ee_pose", issues)
        if pose is not None and pose.shape != (4, 4):
            issues.append(
                _issue(
                    SHAPE_MISMATCH,
                    f"step {index} ee_pose must be 4x4, got {pose.shape}",
                    "ee_pose",
                )
            )
        action = _finite(step.action, f"steps[{index}].action", issues)
        if action is not None and action.shape != (8,):
            issues.append(
                _issue(
                    ACTION_DIM,
                    f"step {index} action must have 8 values (7 joints + gripper), "
                    f"got {action.shape}",
                    "action",
                )
            )
        gripper = _finite(step.gripper_position, f"steps[{index}].gripper_position", issues)
        if gripper is not None and gripper.shape != ():
            issues.append(
                _issue(
                    SHAPE_MISMATCH,
                    f"step {index} gripper_position must be scalar, got {gripper.shape}",
                    "gripper_position",
                )
            )
        collision = getattr(step, "unintended_collision", None)
        if not isinstance(collision, (bool, np.bool_, int)):
            issues.append(
                _issue(
                    SHAPE_MISMATCH,
                    f"step {index} unintended_collision must be boolean",
                    "unintended_collision",
                )
            )

    if len(stamps) == len(steps):
        _validate_time(stamps, config, issues)

    if issues:
        return _sorted(issues)
    return []


def _validate_time(
    stamps: list[int], config: ValidationConfig, issues: list[ValidationIssue]
) -> None:
    if any(current <= previous for previous, current in zip(stamps, stamps[1:], strict=False)):
        issues.append(
            _issue(
                NON_MONOTONIC_TIME,
                "step timestamps must be strictly increasing",
                "stamp_ns",
            )
        )
        return
    deltas = np.diff(np.asarray(stamps, dtype=np.float64)) * 1e-9
    if deltas.size == 0:
        return
    nominal = float(np.median(deltas))
    if not math.isfinite(nominal) or nominal <= 0:
        issues.append(_issue(NON_MONOTONIC_TIME, "invalid timestamp spacing", "stamp_ns"))
        return
    worst = float(np.max(np.abs(deltas - nominal)))
    if worst > config.dt_jitter_tolerance_s:
        issues.append(
            _issue(
                DT_JITTER,
                f"dt jitter {worst:.6f} s exceeds {config.dt_jitter_tolerance_s:.6f} s",
                "stamp_ns",
            )
        )


def _sorted(issues: list[ValidationIssue]) -> list[ValidationIssue]:
    order = {code: index for index, code in enumerate(ISSUE_ORDER)}
    return sorted(issues, key=lambda item: (order.get(item.code, len(order)), item.field or ""))


__all__ = [
    "ACTION_DIM",
    "DT_JITTER",
    "DUPLICATE_EPISODE",
    "ISSUE_ORDER",
    "MALFORMED_EPISODE",
    "MISSING_FIELD",
    "NAN_VALUES",
    "NON_MONOTONIC_TIME",
    "SHAPE_MISMATCH",
    "TOO_SHORT",
    "ValidationIssue",
    "validate_episode",
]
