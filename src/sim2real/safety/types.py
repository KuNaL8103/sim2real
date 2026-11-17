"""Shared safety value objects and reason codes (ROS-free)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Chunk / input rejection reasons.
EMPTY_CHUNK = "EMPTY_CHUNK"
TOO_MANY_POINTS = "TOO_MANY_POINTS"
NON_FINITE = "NON_FINITE"
JOINT_NAMES_MISMATCH = "JOINT_NAMES_MISMATCH"
NON_INCREASING_TIME = "NON_INCREASING_TIME"
INVALID_TIME = "INVALID_TIME"
FUTURE_STAMP = "FUTURE_STAMP"
STALE_CHUNK = "STALE_CHUNK"
CONTINUITY = "CONTINUITY"
WORKSPACE_VIOLATION = "WORKSPACE_VIOLATION"
POSITION_LIMIT = "POSITION_LIMIT"
VELOCITY_LIMIT = "VELOCITY_LIMIT"
ACCELERATION_LIMIT = "ACCELERATION_LIMIT"
JERK_LIMIT = "JERK_LIMIT"
OBSTACLE_TOO_CLOSE = "OBSTACLE_TOO_CLOSE"
MAP_UNAVAILABLE = "MAP_UNAVAILABLE"
MAP_STALE = "MAP_STALE"
MAP_NO_DATA = "MAP_NO_DATA"
MAP_LOW_COVERAGE = "MAP_LOW_COVERAGE"
OUTSIDE_MAP = "OUTSIDE_MAP"
TRANSFORM_UNAVAILABLE = "TRANSFORM_UNAVAILABLE"
JOINT_STATE_UNAVAILABLE = "JOINT_STATE_UNAVAILABLE"
MAP_TOO_LARGE = "MAP_TOO_LARGE"
MAP_INVALID = "MAP_INVALID"

# Monitor/state reasons (not chunk decisions).
WATCHDOG_JOINT_STATE_STALE = "WATCHDOG_JOINT_STATE_STALE"
WATCHDOG_MAP_STALE = "WATCHDOG_MAP_STALE"
WATCHDOG_MAP_UNAVAILABLE = "WATCHDOG_MAP_UNAVAILABLE"
WATCHDOG_INPUT_OUTAGE = "WATCHDOG_INPUT_OUTAGE"
ESTOP_LATCHED = "ESTOP_LATCHED"
STARTING = "STARTING"

# Decision labels.
DECISION_ACCEPTED = "ACCEPTED"
DECISION_REJECTED = "REJECTED"
DECISION_NONE = "NONE"


@dataclass(frozen=True)
class LimitRatios:
    """Maximum ratio over all joints/samples for each checked quantity.

    ``1.0`` means exactly at the configured (margin-shrunk) limit. A value is
    ``nan`` when the quantity cannot be computed (for example acceleration for
    a one-point command).
    """

    position: float = float("nan")
    velocity: float = float("nan")
    acceleration: float = float("nan")
    jerk: float = float("nan")

    def worst(self) -> float:
        values = [
            v
            for v in (self.position, self.velocity, self.acceleration, self.jerk)
            if np.isfinite(v)
        ]
        return max(values) if values else float("nan")

    def as_dict(self) -> dict[str, float | None]:
        return {
            "position": None if not np.isfinite(self.position) else float(self.position),
            "velocity": None if not np.isfinite(self.velocity) else float(self.velocity),
            "acceleration": (
                None if not np.isfinite(self.acceleration) else float(self.acceleration)
            ),
            "jerk": None if not np.isfinite(self.jerk) else float(self.jerk),
        }


@dataclass(frozen=True)
class RawChunk:
    """ROS-independent raw action chunk received from ``/vla/raw_action``."""

    stamp_ns: int
    joint_names: tuple[str, ...]
    positions: np.ndarray  # (n, 8) float64
    time_from_start_s: tuple[float, ...]
    frame_id: str = ""

    @property
    def point_count(self) -> int:
        return int(self.positions.shape[0])


@dataclass(frozen=True)
class SafetyCommand:
    """The final, validated command sent to the controller."""

    joint_names: tuple[str, ...]
    positions: np.ndarray
    time_from_start_s: tuple[float, ...]
    header_stamp_ns: int = 0
    source_stamp_ns: int = 0
    retimed: bool = False
    dropped_points: int = 0


@dataclass(frozen=True)
class ChunkDecision:
    accepted: bool
    reason: str | None
    command: SafetyCommand | None
    ratios: LimitRatios = field(default_factory=LimitRatios)
    min_obstacle_distance_m: float = float("nan")
    map_valid: bool = False
    finger_clipped_points: int = 0
    finger_clipped: bool = False
    age_s: float = float("nan")
    retimed: bool = False
    dropped_points: int = 0
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "finger_clipped_points": self.finger_clipped_points,
            "finger_clipped": self.finger_clipped,
            "age_s": None if not np.isfinite(self.age_s) else float(self.age_s),
            "retimed": self.retimed,
            "dropped_points": self.dropped_points,
            "min_obstacle_distance_m": (
                None
                if not np.isfinite(self.min_obstacle_distance_m)
                else float(self.min_obstacle_distance_m)
            ),
            "map_valid": self.map_valid,
            "ratios": self.ratios.as_dict(),
            "command_points": 0 if self.command is None else self.command.positions.shape[0],
            "detail": self.detail,
        }
