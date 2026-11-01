"""Transform lookup abstraction and static-transform change detection.

The core mapper never imports ``rclpy`` or ``tf2_ros``.  It receives a
:class:`TransformProvider`; the ROS adapter supplies a tf2-backed provider and
the tests supply :class:`StaticTransformProvider`.

A missing transform (no data, extrapolation, disconnected frame) returns
``None``.  The mapper counts the frame as ``skipped_no_tf`` and never raises.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


class TransformProvider(Protocol):
    def lookup(
        self, target_frame: str, source_frame: str, stamp: float | None = None
    ) -> np.ndarray | None:
        """Return ``T_target_source`` (4x4) or ``None`` if unavailable."""


@dataclass
class StaticTransformProvider:
    """Deterministic test double that looks transforms up in a dictionary."""

    transforms: Mapping[tuple[str, str], np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.transforms = {
            (target, source): np.asarray(value, dtype=np.float64).copy()
            for (target, source), value in self.transforms.items()
        }

    def lookup(
        self, target_frame: str, source_frame: str, stamp: float | None = None
    ) -> np.ndarray | None:
        value = self.transforms.get((target_frame, source_frame))
        return None if value is None else value.copy()

    def set(self, target_frame: str, source_frame: str, transform: np.ndarray) -> None:
        self.transforms[(target_frame, source_frame)] = np.asarray(
            transform, dtype=np.float64
        ).copy()


def rotation_angle_rad(rotation: np.ndarray) -> float:
    """Angle of a rotation matrix, robust to tiny numerical drift."""
    trace = float(np.trace(rotation))
    return float(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)))


def transform_delta(previous: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Return ``(translation_m, rotation_rad)`` between two 4x4 transforms."""
    a = np.asarray(previous, dtype=np.float64)
    b = np.asarray(current, dtype=np.float64)
    translation = float(np.linalg.norm(b[:3, 3] - a[:3, 3]))
    rotation = rotation_angle_rad(a[:3, :3].T @ b[:3, :3])
    return translation, rotation


def camera_static_transforms(
    transforms: Mapping[tuple[str, str], np.ndarray],
    camera_frames: tuple[str, ...],
) -> dict[tuple[str, str], np.ndarray]:
    """Keep only static transforms whose child is a configured camera frame."""
    return {
        key: np.asarray(value, dtype=np.float64).copy()
        for key, value in transforms.items()
        if key[1] in camera_frames
    }


def static_transforms_changed(
    previous: Mapping[tuple[str, str], np.ndarray] | None,
    current: Mapping[tuple[str, str], np.ndarray],
    *,
    camera_frames: tuple[str, ...],
    translation_epsilon_m: float,
    rotation_epsilon_rad: float,
) -> bool:
    """Compare camera static transforms.

    ``previous is None`` means "first observation" and never counts as a
    change.  A camera transform that disappears counts as a change.  A change
    at or below the epsilon is ignored.  tf2 keeps only the latest static
    transform per frame pair, which is exactly why the ROS adapter subscribes
    to ``/tf_static`` directly instead of querying the static buffer.
    """
    current_cameras = camera_static_transforms(current, camera_frames)
    if previous is None:
        return False
    previous_cameras = camera_static_transforms(previous, camera_frames)
    keys = set(previous_cameras) | set(current_cameras)
    for key in keys:
        if key not in previous_cameras or key not in current_cameras:
            return True
        translation, rotation = transform_delta(previous_cameras[key], current_cameras[key])
        if translation > translation_epsilon_m or rotation > rotation_epsilon_rad:
            return True
    return False
