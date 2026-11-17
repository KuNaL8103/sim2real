"""Joint position/velocity/acceleration/jerk checks (ROS-free).

All derivatives are finite differences over the *final* command times, with the
measured state prepended at ``t = 0`` because the controller interpolates from
the current state to the first command point.  Checking only the raw chunk
would miss the first jump and would validate a trajectory that is not the one
that is actually sent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SafetyConfig
from .types import (
    ACCELERATION_LIMIT,
    JERK_LIMIT,
    POSITION_LIMIT,
    VELOCITY_LIMIT,
    LimitRatios,
)

_EPS = 1e-12


@dataclass(frozen=True)
class LimitCheck:
    ok: bool
    reason: str | None
    ratios: LimitRatios
    detail: str = ""


def _velocity(positions: np.ndarray, times: np.ndarray) -> np.ndarray:
    if len(times) < 2:
        return np.zeros((0, positions.shape[1]), dtype=np.float64)
    dt = np.diff(times)
    return np.diff(positions, axis=0) / dt[:, None]


def _acceleration(positions: np.ndarray, times: np.ndarray) -> np.ndarray:
    if len(times) < 3:
        return np.zeros((0, positions.shape[1]), dtype=np.float64)
    velocity = _velocity(positions, times)
    h0 = np.diff(times)[:-1]
    h1 = np.diff(times)[1:]
    # Second derivative at the interior sample: 2 * (v_right - v_left) / (h0+h1).
    return 2.0 * (velocity[1:] - velocity[:-1]) / (h0 + h1)[:, None]


def _jerk(positions: np.ndarray, times: np.ndarray) -> np.ndarray:
    acceleration = _acceleration(positions, times)
    if len(acceleration) < 2:
        return np.zeros((0, positions.shape[1]), dtype=np.float64)
    # Acceleration sample i is at time i+1; the spacing between samples is the
    # interval between those times.
    dt = np.diff(times)[1:-1]
    return np.diff(acceleration, axis=0) / dt[:, None]


def _position_ratio(positions: np.ndarray, config: SafetyConfig) -> float:
    lower = config.effective_joint_lower_rad
    upper = config.effective_joint_upper_rad
    centre = config.joint_centre_rad
    half = 0.5 * (upper - lower)
    below = (centre - positions) / half
    above = (positions - centre) / half
    return float(np.max(np.maximum(below, above)))


def _ratio(values: np.ndarray, limits: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.max(np.abs(values) / limits))


def check_joint_limits(
    command_positions: np.ndarray,
    command_times_s: tuple[float, ...] | list[float],
    initial_positions: np.ndarray,
    config: SafetyConfig,
) -> LimitCheck:
    """Check a final commanded trajectory against margin-shrunk FER limits.

    ``command_positions`` has shape ``(n, 8)``: seven arm joints (rad) and the
    finger (m). ``initial_positions`` is the measured 8-D state at ``t=0``.
    """
    command = np.asarray(command_positions, dtype=np.float64)
    initial = np.asarray(initial_positions, dtype=np.float64)
    if command.ndim != 2 or command.shape[1] != 8:
        raise ValueError(f"command_positions must have shape (n, 8), got {command.shape}")
    if initial.shape != (8,):
        raise ValueError(f"initial_positions must have shape (8,), got {initial.shape}")
    if len(command_times_s) != command.shape[0]:
        raise ValueError("command_times_s length must match command_positions")
    if command.shape[0] == 0:
        raise ValueError("command_positions must not be empty")
    if not np.all(np.isfinite(command)) or not np.all(np.isfinite(initial)):
        return LimitCheck(False, POSITION_LIMIT, LimitRatios(), "non-finite position")

    times = np.asarray((0.0, *command_times_s), dtype=np.float64)
    if np.any(np.diff(times) <= 0.0):
        return LimitCheck(False, VELOCITY_LIMIT, LimitRatios(), "non-increasing command times")

    arm = command[:, :7]
    ratio_position = _position_ratio(arm, config)
    # The finger has no FER velocity/acceleration/jerk limit in this project;
    # it is clipped into [0, 0.04] m before this check. Derivatives therefore
    # use the seven arm joints only.
    all_positions = np.vstack([initial, command])[:, :7]
    ratio_velocity = _ratio(_velocity(all_positions, times), config.effective_velocity_limits_rad_s)
    ratio_acceleration = _ratio(
        _acceleration(all_positions, times), config.effective_acceleration_limits_rad_s2
    )
    ratio_jerk = _ratio(_jerk(all_positions, times), config.effective_jerk_limits_rad_s3)
    ratios = LimitRatios(
        position=ratio_position,
        velocity=ratio_velocity,
        acceleration=ratio_acceleration,
        jerk=ratio_jerk,
    )

    for reason, value in (
        (POSITION_LIMIT, ratio_position),
        (VELOCITY_LIMIT, ratio_velocity),
        (ACCELERATION_LIMIT, ratio_acceleration),
        (JERK_LIMIT, ratio_jerk),
    ):
        if np.isfinite(value) and value > 1.0 + _EPS:
            return LimitCheck(False, reason, ratios, f"{reason.lower()} ratio {value:.6g} > 1")

    return LimitCheck(True, None, ratios)


__all__ = ["LimitCheck", "check_joint_limits"]
