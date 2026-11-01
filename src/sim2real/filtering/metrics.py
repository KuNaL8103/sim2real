"""Trajectory-quality metrics for recorded rollouts.

Everything here uses the **recorded executed state** — the end-effector poses and
joint positions/velocities that the backend reported — not the commanded action
vector. A smooth command does not prove a smooth motion, so commands are ignored
except for validation (the action dimension).

Formulas are documented in ``docs/FILTER_METRICS.md``. The short version:

* dimensionless squared jerk ``J_d = (T^5 / L^2) * integral ||x'''||^2 dt``;
* SPARC from the speed profile (negative; more negative means less smooth);
* path length in metres (end effector) and radians (joint space);
* normalised joint-position and velocity-limit margins;
* collision count as the number of maximal runs of collision steps.

Derivatives are computed on uniformly resampled samples. The default method is a
Savitzky-Golay filter whose polynomial order (5) exactly differentiates a
quintic minimum-jerk profile while suppressing 20 Hz measurement noise. A
finite-difference method is available for comparison.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import savgol_filter

from sim2real import contracts
from sim2real.filtering.config import FilterConfig, SparcConfig, ValidationConfig


class DegenerateTrajectoryError(ValueError):
    """Raised when a trajectory has no meaningful jerk or speed profile."""


@dataclass(frozen=True)
class TrajectoryMetrics:
    episode_id: str
    num_steps: int
    duration_s: float
    dimensionless_squared_jerk: float
    sparc: float
    path_length_m: float
    joint_path_length_rad: float
    min_joint_limit_margin: float
    min_velocity_limit_margin: float
    collision_count: int
    task_success: bool
    derivative_method: str
    sparc_cutoff_hz: float
    sparc_sampling_rate_hz: float
    velocity_crosscheck_max_abs_diff_rad_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "num_steps": self.num_steps,
            "duration_s": self.duration_s,
            "dimensionless_squared_jerk": self.dimensionless_squared_jerk,
            "sparc": self.sparc,
            "path_length_m": self.path_length_m,
            "joint_path_length_rad": self.joint_path_length_rad,
            "min_joint_limit_margin": self.min_joint_limit_margin,
            "min_velocity_limit_margin": self.min_velocity_limit_margin,
            "collision_count": self.collision_count,
            "task_success": self.task_success,
            "derivative_method": self.derivative_method,
            "sparc_cutoff_hz": self.sparc_cutoff_hz,
            "sparc_sampling_rate_hz": self.sparc_sampling_rate_hz,
            "velocity_crosscheck_max_abs_diff_rad_s": (self.velocity_crosscheck_max_abs_diff_rad_s),
        }


def _trapezoid(values: np.ndarray, coordinates: np.ndarray) -> float:
    """Trapezoidal integral that works on NumPy 1.26 and 2.x."""
    if values.size < 2:
        return 0.0
    return float(np.sum((values[1:] + values[:-1]) * 0.5 * np.diff(coordinates)))


def _timestamps_s(record: Any) -> np.ndarray:
    stamps = np.asarray([int(step.stamp_ns) for step in record.steps], dtype=np.float64)
    return (stamps - stamps[0]) * 1e-9


def _positions(steps: list[Any]) -> np.ndarray:
    return np.asarray([np.asarray(step.ee_pose, dtype=np.float64)[:3, 3] for step in steps])


def _joint_positions(steps: list[Any]) -> np.ndarray:
    return np.asarray([step.joint_positions for step in steps], dtype=np.float64)


def _joint_velocities(steps: list[Any]) -> np.ndarray:
    return np.asarray([step.joint_velocities for step in steps], dtype=np.float64)


def _uniform_grid(times: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    duration = float(times[-1] - times[0])
    if duration <= 0:
        raise DegenerateTrajectoryError("trajectory duration must be positive")
    uniform = np.linspace(0.0, duration, len(times))
    resampled = np.column_stack(
        [np.interp(uniform, times, values[:, axis]) for axis in range(values.shape[1])]
    )
    return uniform, resampled


def _derivative(
    values: np.ndarray,
    times: np.ndarray,
    order: int,
    config: ValidationConfig,
) -> np.ndarray:
    """Return the ``order``-th time derivative on a uniform grid."""
    count = len(times)
    if count < 3:
        raise DegenerateTrajectoryError("need at least three samples for a derivative")
    dt = float(times[1] - times[0])
    if dt <= 0:
        raise DegenerateTrajectoryError("non-positive sample spacing")
    if config.derivative_method == "savgol":
        if count < config.savgol_window:
            raise DegenerateTrajectoryError(
                f"episode has {count} samples; Savitzky-Golay window is {config.savgol_window}"
            )
        return np.asarray(
            savgol_filter(
                values,
                window_length=config.savgol_window,
                polyorder=config.savgol_polyorder,
                deriv=order,
                delta=dt,
                axis=0,
                mode="interp",
            ),
            dtype=np.float64,
        )
    if config.derivative_method == "finite_difference":
        current = np.asarray(values, dtype=np.float64)
        for _ in range(order):
            current = np.gradient(current, dt, axis=0, edge_order=2)
        return current
    raise DegenerateTrajectoryError(f"unknown derivative method {config.derivative_method!r}")


def dimensionless_squared_jerk(
    positions: np.ndarray,
    times: np.ndarray,
    config: ValidationConfig,
) -> float:
    """``J_d = (T^5 / L^2) * integral ||x'''||^2 dt`` using executed positions."""
    duration = float(times[-1])
    if duration <= 0:
        raise DegenerateTrajectoryError("trajectory duration must be positive")
    path = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
    if path < config.min_path_length_m:
        raise DegenerateTrajectoryError(
            f"end-effector path length {path:.6f} m is below "
            f"validation.min_path_length_m={config.min_path_length_m}"
        )
    uniform, resampled = _uniform_grid(times, positions)
    jerk = _derivative(resampled, uniform, 3, config)
    integral = _trapezoid(np.sum(jerk * jerk, axis=1), uniform)
    if not math.isfinite(integral) or integral < 0:
        raise DegenerateTrajectoryError("jerk integral is not finite")
    value = (duration**5 / path**2) * integral
    if not math.isfinite(value):
        raise DegenerateTrajectoryError("dimensionless jerk is not finite")
    return float(value)


def _speed_profile(
    positions: np.ndarray, times: np.ndarray, config: ValidationConfig
) -> tuple[np.ndarray, np.ndarray, float]:
    uniform, resampled = _uniform_grid(times, positions)
    velocity = _derivative(resampled, uniform, 1, config)
    speed = np.linalg.norm(velocity, axis=1)
    if speed.size == 0 or not np.all(np.isfinite(speed)):
        raise DegenerateTrajectoryError("speed profile is not finite")
    if float(np.max(speed)) <= 0:
        raise DegenerateTrajectoryError("speed profile is all zeros")
    dt = float(uniform[1] - uniform[0])
    if dt <= 0:
        raise DegenerateTrajectoryError("non-positive sample spacing")
    return uniform, speed, 1.0 / dt


def sparc(
    positions: np.ndarray,
    times: np.ndarray,
    config: ValidationConfig,
    sparc_config: SparcConfig | None = None,
) -> tuple[float, float, float]:
    """Return ``(sparc, cutoff_hz, sampling_rate_hz)`` for the executed path."""
    settings = sparc_config or SparcConfig()
    _uniform, speed, sampling_rate = _speed_profile(positions, times, config)
    count = len(speed)
    nfft = 1
    while nfft < count:
        nfft *= 2
    nfft *= 2**settings.padlevel
    spectrum = np.abs(np.fft.rfft(speed, n=nfft))
    if spectrum[0] <= 0 or not math.isfinite(float(spectrum[0])):
        raise DegenerateTrajectoryError("speed spectrum has no DC component")
    normalised = spectrum / spectrum[0]
    frequencies = np.fft.rfftfreq(nfft, d=1.0 / sampling_rate)
    cutoff = settings.effective_cutoff_hz(sampling_rate)
    candidates = (frequencies > 0) & (frequencies <= cutoff)
    if not np.any(candidates):
        raise DegenerateTrajectoryError("no frequency bins below the cutoff")
    above = candidates & (normalised >= settings.amplitude_threshold)
    if np.any(above):
        cutoff_star = float(np.max(frequencies[above]))
    else:
        cutoff_star = float(np.min(frequencies[candidates]))
    if cutoff_star <= 0:
        raise DegenerateTrajectoryError("adaptive SPARC cutoff is not positive")
    selected = frequencies <= cutoff_star
    selected_frequencies = frequencies[selected]
    selected_amplitudes = normalised[selected]
    if selected_frequencies.size >= 2:
        derivative = np.gradient(selected_amplitudes, selected_frequencies)
    else:
        derivative = np.zeros_like(selected_frequencies)
    integrand = np.sqrt((1.0 / cutoff_star) ** 2 + derivative**2)
    value = -_trapezoid(integrand, selected_frequencies)
    if not math.isfinite(value):
        raise DegenerateTrajectoryError("SPARC is not finite")
    return float(value), cutoff_star, sampling_rate


def path_lengths(positions: np.ndarray, joints: np.ndarray) -> tuple[float, float]:
    """Executed end-effector (m) and joint-space (rad) path lengths."""
    ee = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
    joint = float(np.sum(np.linalg.norm(np.diff(joints, axis=0), axis=1)))
    return ee, joint


def limit_margins(
    positions: np.ndarray,
    velocities: np.ndarray,
    limits: Any,
) -> tuple[float, float]:
    """Minimum normalised joint-position and velocity margins over joints/time."""
    joint_limits = np.asarray(limits.joint_limits, dtype=np.float64)
    velocity_limits = np.asarray(limits.velocity_limits, dtype=np.float64)
    half_range = (joint_limits[:, 1] - joint_limits[:, 0]) / 2.0
    if np.any(half_range <= 0):
        raise DegenerateTrajectoryError("joint limits have a non-positive half range")
    position_margin = (
        np.minimum(
            positions - joint_limits[:, 0],
            joint_limits[:, 1] - positions,
        )
        / half_range
    )
    velocity_margin = 1.0 - np.abs(velocities) / velocity_limits
    return float(np.min(position_margin)), float(np.min(velocity_margin))


def collision_run_count(flags: Any) -> int:
    """Number of maximal runs of consecutive true collision flags."""
    values = np.asarray(flags, dtype=bool).reshape(-1)
    if values.size == 0:
        return 0
    padded = np.concatenate(([False], values, [False]))
    return int(np.count_nonzero(np.logical_and(~padded[:-1], padded[1:])))


def velocity_crosscheck(positions: np.ndarray, times: np.ndarray, recorded: np.ndarray) -> float:
    """Max absolute difference between recorded and position-derived joint speed."""
    if len(times) < 3:
        return 0.0
    derived = np.gradient(positions, times, axis=0, edge_order=2)
    return float(np.max(np.abs(derived - recorded)))


def compute_metrics(record: Any, config: FilterConfig) -> TrajectoryMetrics:
    """Compute all metrics; raise ``DegenerateTrajectoryError`` when undefined."""
    steps = list(record.steps)
    times = _timestamps_s(record)
    if len(steps) < 2 or times[-1] <= 0:
        raise DegenerateTrajectoryError("episode is too short for metrics")
    positions = _positions(steps)
    joints = _joint_positions(steps)
    velocities = _joint_velocities(steps)
    if not (
        np.all(np.isfinite(positions))
        and np.all(np.isfinite(joints))
        and np.all(np.isfinite(velocities))
    ):
        raise DegenerateTrajectoryError("episode contains non-finite state")

    jerk = dimensionless_squared_jerk(positions, times, config.validation)
    sparc_value, cutoff, rate = sparc(positions, times, config.validation, config.sparc)
    ee_length, joint_length = path_lengths(positions, joints)
    position_margin, velocity_margin = limit_margins(joints, velocities, config.limits)
    collisions = collision_run_count([step.unintended_collision for step in steps])
    outcome = int(getattr(record, "outcome", contracts.OUTCOME_UNKNOWN))
    success = outcome == contracts.OUTCOME_SUCCESS
    return TrajectoryMetrics(
        episode_id=str(record.episode_id),
        num_steps=len(steps),
        duration_s=float(times[-1]),
        dimensionless_squared_jerk=jerk,
        sparc=sparc_value,
        path_length_m=ee_length,
        joint_path_length_rad=joint_length,
        min_joint_limit_margin=position_margin,
        min_velocity_limit_margin=velocity_margin,
        collision_count=collisions,
        task_success=success,
        derivative_method=config.validation.derivative_method,
        sparc_cutoff_hz=cutoff,
        sparc_sampling_rate_hz=rate,
        velocity_crosscheck_max_abs_diff_rad_s=velocity_crosscheck(joints, times, velocities),
    )


__all__ = [
    "DegenerateTrajectoryError",
    "TrajectoryMetrics",
    "collision_run_count",
    "compute_metrics",
    "dimensionless_squared_jerk",
    "limit_margins",
    "path_lengths",
    "sparc",
    "velocity_crosscheck",
]
