"""Strict configuration for the safety watchdog.

The safety layer must never quietly fall back to a permissive default.  Every
numeric bound is therefore required, unit-suffixed, finite, and range-checked;
unknown YAML keys are rejected.  Hardware position/velocity/acceleration/jerk
limits come from ``configs/sim/panda.yaml`` so there is one source of truth.

Units: metres (m), radians (rad), seconds (s), hertz (Hz), fractions are
dimensionless.  ``safety_margin_fraction`` shrinks the hardware limits toward
zero (velocity/acceleration/jerk/effort) or toward each joint's range centre
(position).  It is a reserve for tracking error and model mismatch, not a
functional-safety claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sim2real.sim.config import PandaConfig, load_panda_config

SCHEMA_VERSION = 1
STALE_POLICIES = ("reject", "retime_if_continuous")
COMMAND_TIME_CONVENTIONS = ("zero_stamp_now_plus_dt",)


class SafetyConfigError(ValueError):
    """Raised when the safety config is missing, unknown, or malformed."""


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SafetyConfigError(f"{path}: expected a mapping")
    return value


def _unknown(mapping: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise SafetyConfigError(f"{path}: unknown keys {unknown}")


def _number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SafetyConfigError(f"{path}: expected a number")
    number = float(value)
    if not math.isfinite(number):
        raise SafetyConfigError(f"{path}: expected a finite number")
    if minimum is not None:
        if exclusive_minimum and number <= minimum:
            raise SafetyConfigError(f"{path}: expected > {minimum}")
        if not exclusive_minimum and number < minimum:
            raise SafetyConfigError(f"{path}: expected >= {minimum}")
    if maximum is not None and number > maximum:
        raise SafetyConfigError(f"{path}: expected <= {maximum}")
    return number


def _integer(value: Any, path: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SafetyConfigError(f"{path}: expected an integer")
    if not minimum <= value <= maximum:
        raise SafetyConfigError(f"{path}: expected {minimum}..{maximum}, got {value}")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise SafetyConfigError(f"{path}: expected true or false")
    return value


def _text(value: Any, path: str, *, choices: tuple[str, ...] | None = None) -> str:
    if not isinstance(value, str):
        raise SafetyConfigError(f"{path}: expected a string")
    text = value.strip()
    if not text:
        raise SafetyConfigError(f"{path}: expected a non-empty string")
    if choices is not None and text not in choices:
        raise SafetyConfigError(f"{path}: expected one of {list(choices)}, got {text!r}")
    return text


def _vector(value: Any, path: str, length: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SafetyConfigError(f"{path}: expected numeric values") from exc
    if array.shape != (length,):
        raise SafetyConfigError(f"{path}: expected {length} values, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise SafetyConfigError(f"{path}: expected finite values")
    return array


@dataclass(frozen=True)
class SafetyConfig:
    """All safety settings, with the unit in the field name."""

    safety_margin_fraction: float
    workspace_min_m: np.ndarray
    workspace_max_m: np.ndarray
    safety_frame: str
    min_obstacle_distance_m: float
    obstacle_safety_radius_m: float
    map_required: bool
    max_map_age_s: float
    min_map_coverage_ratio: float
    map_query_resolution_m: float
    map_max_query_distance_m: float
    max_joint_state_age_s: float
    max_chunk_age_s: float
    stale_chunk_policy: str
    max_first_point_distance_rad: float
    ema_alpha: float
    watchdog_timeout_s: float
    estop_reset_hold_s: float
    finger_min_m: float
    finger_max_m: float
    max_chunk_points: int
    command_period_s: float
    command_time_convention: str
    future_stamp_tolerance_s: float
    hold_duration_s: float
    status_rate_hz: float
    diagnostics_rate_hz: float
    pointcloud_max_points: int
    pointcloud_max_bytes: int
    joint_lower_rad: np.ndarray
    joint_upper_rad: np.ndarray
    velocity_limits_rad_s: np.ndarray
    acceleration_limits_rad_s2: np.ndarray
    jerk_limits_rad_s3: np.ndarray
    effort_limits_nm: np.ndarray

    @property
    def joint_centre_rad(self) -> np.ndarray:
        return 0.5 * (self.joint_lower_rad + self.joint_upper_rad)

    @property
    def effective_joint_lower_rad(self) -> np.ndarray:
        half = 0.5 * (self.joint_upper_rad - self.joint_lower_rad) * self.safety_margin_fraction
        return self.joint_centre_rad - half

    @property
    def effective_joint_upper_rad(self) -> np.ndarray:
        half = 0.5 * (self.joint_upper_rad - self.joint_lower_rad) * self.safety_margin_fraction
        return self.joint_centre_rad + half

    @property
    def effective_velocity_limits_rad_s(self) -> np.ndarray:
        return self.velocity_limits_rad_s * self.safety_margin_fraction

    @property
    def effective_acceleration_limits_rad_s2(self) -> np.ndarray:
        return self.acceleration_limits_rad_s2 * self.safety_margin_fraction

    @property
    def effective_jerk_limits_rad_s3(self) -> np.ndarray:
        return self.jerk_limits_rad_s3 * self.safety_margin_fraction

    def as_dict(self) -> dict[str, Any]:
        return {
            "safety_margin_fraction": self.safety_margin_fraction,
            "workspace_min_m": self.workspace_min_m.tolist(),
            "workspace_max_m": self.workspace_max_m.tolist(),
            "safety_frame": self.safety_frame,
            "min_obstacle_distance_m": self.min_obstacle_distance_m,
            "obstacle_safety_radius_m": self.obstacle_safety_radius_m,
            "map_required": self.map_required,
            "max_map_age_s": self.max_map_age_s,
            "min_map_coverage_ratio": self.min_map_coverage_ratio,
            "map_query_resolution_m": self.map_query_resolution_m,
            "map_max_query_distance_m": self.map_max_query_distance_m,
            "max_joint_state_age_s": self.max_joint_state_age_s,
            "max_chunk_age_s": self.max_chunk_age_s,
            "stale_chunk_policy": self.stale_chunk_policy,
            "max_first_point_distance_rad": self.max_first_point_distance_rad,
            "ema_alpha": self.ema_alpha,
            "watchdog_timeout_s": self.watchdog_timeout_s,
            "estop_reset_hold_s": self.estop_reset_hold_s,
            "finger_min_m": self.finger_min_m,
            "finger_max_m": self.finger_max_m,
            "max_chunk_points": self.max_chunk_points,
            "command_period_s": self.command_period_s,
            "command_time_convention": self.command_time_convention,
            "future_stamp_tolerance_s": self.future_stamp_tolerance_s,
            "hold_duration_s": self.hold_duration_s,
            "status_rate_hz": self.status_rate_hz,
            "diagnostics_rate_hz": self.diagnostics_rate_hz,
            "pointcloud_max_points": self.pointcloud_max_points,
            "pointcloud_max_bytes": self.pointcloud_max_bytes,
            "joint_limits": np.column_stack([self.joint_lower_rad, self.joint_upper_rad]).tolist(),
            "velocity_limits": self.velocity_limits_rad_s.tolist(),
            "acceleration_limits": self.acceleration_limits_rad_s2.tolist(),
            "jerk_limits": self.jerk_limits_rad_s3.tolist(),
            "effort_limits": self.effort_limits_nm.tolist(),
        }


_ALLOWED = (
    set(SafetyConfig.__dataclass_fields__)
    - {
        "joint_lower_rad",
        "joint_upper_rad",
        "velocity_limits_rad_s",
        "acceleration_limits_rad_s2",
        "jerk_limits_rad_s3",
        "effort_limits_nm",
    }
) | {"workspace_bounds_m"}


def _from_mapping(data: dict[str, Any], panda: PandaConfig, path: str) -> SafetyConfig:
    _unknown(data, _ALLOWED, path)

    def number(name: str, **kwargs: Any) -> float:
        if name not in data:
            raise SafetyConfigError(f"{path}.{name}: required")
        return _number(data[name], f"{path}.{name}", **kwargs)

    def integer(name: str, **kwargs: Any) -> int:
        if name not in data:
            raise SafetyConfigError(f"{path}.{name}: required")
        return _integer(data[name], f"{path}.{name}", **kwargs)

    def boolean(name: str) -> bool:
        if name not in data:
            raise SafetyConfigError(f"{path}.{name}: required")
        return _boolean(data[name], f"{path}.{name}")

    def text(name: str, **kwargs: Any) -> str:
        if name not in data:
            raise SafetyConfigError(f"{path}.{name}: required")
        return _text(data[name], f"{path}.{name}", **kwargs)

    if "workspace_bounds_m" not in data:
        raise SafetyConfigError(f"{path}.workspace_bounds_m: required")
    bounds = _mapping(data["workspace_bounds_m"], f"{path}.workspace_bounds_m")
    _unknown(bounds, {"min", "max"}, f"{path}.workspace_bounds_m")
    workspace_min = _vector(bounds.get("min"), f"{path}.workspace_bounds_m.min", 3)
    workspace_max = _vector(bounds.get("max"), f"{path}.workspace_bounds_m.max", 3)
    if np.any(workspace_min >= workspace_max):
        raise SafetyConfigError(f"{path}.workspace_bounds_m: every min must be < max")

    if panda.jerk_limits is None or panda.effort_limits is None:
        raise SafetyConfigError(
            "configs/sim/panda.yaml: jerk_limits and effort_limits are required by the "
            "safety layer (official FER values; do not invent them)"
        )

    finger_min = number("finger_min_m")
    finger_max = number("finger_max_m")
    if finger_min >= finger_max:
        raise SafetyConfigError(f"{path}: finger_min_m must be < finger_max_m")
    if not (
        math.isclose(finger_min, float(panda.gripper_limits[0]), abs_tol=1e-12)
        and math.isclose(finger_max, float(panda.gripper_limits[1]), abs_tol=1e-12)
    ):
        raise SafetyConfigError(
            f"{path}: finger bounds {finger_min}..{finger_max} must match "
            f"panda.gripper_limits {panda.gripper_limits}"
        )

    margin = number("safety_margin_fraction", minimum=0.0, maximum=1.0, exclusive_minimum=True)
    alpha = number("ema_alpha", minimum=0.0, maximum=1.0, exclusive_minimum=True)

    stale_policy = text("stale_chunk_policy", choices=STALE_POLICIES)
    convention = text("command_time_convention", choices=COMMAND_TIME_CONVENTIONS)

    return SafetyConfig(
        safety_margin_fraction=margin,
        workspace_min_m=workspace_min,
        workspace_max_m=workspace_max,
        safety_frame=text("safety_frame"),
        min_obstacle_distance_m=number("min_obstacle_distance_m", minimum=0.0),
        obstacle_safety_radius_m=number("obstacle_safety_radius_m", minimum=0.0),
        map_required=boolean("map_required"),
        max_map_age_s=number("max_map_age_s", minimum=0.0, exclusive_minimum=True),
        min_map_coverage_ratio=number("min_map_coverage_ratio", minimum=0.0, maximum=1.0),
        map_query_resolution_m=number(
            "map_query_resolution_m", minimum=0.0, maximum=1.0, exclusive_minimum=True
        ),
        map_max_query_distance_m=number(
            "map_max_query_distance_m", minimum=0.0, maximum=10.0, exclusive_minimum=True
        ),
        max_joint_state_age_s=number("max_joint_state_age_s", minimum=0.0, exclusive_minimum=True),
        max_chunk_age_s=number("max_chunk_age_s", minimum=0.0, exclusive_minimum=True),
        stale_chunk_policy=stale_policy,
        max_first_point_distance_rad=number("max_first_point_distance_rad", minimum=0.0),
        ema_alpha=alpha,
        watchdog_timeout_s=number("watchdog_timeout_s", minimum=0.0, exclusive_minimum=True),
        estop_reset_hold_s=number("estop_reset_hold_s", minimum=0.0),
        finger_min_m=finger_min,
        finger_max_m=finger_max,
        max_chunk_points=integer("max_chunk_points", minimum=1, maximum=1000),
        command_period_s=number(
            "command_period_s", minimum=0.0, maximum=1.0, exclusive_minimum=True
        ),
        command_time_convention=convention,
        future_stamp_tolerance_s=number("future_stamp_tolerance_s", minimum=0.0),
        hold_duration_s=number(
            "hold_duration_s", minimum=0.0, maximum=10.0, exclusive_minimum=True
        ),
        status_rate_hz=number("status_rate_hz", minimum=0.0, maximum=100.0, exclusive_minimum=True),
        diagnostics_rate_hz=number(
            "diagnostics_rate_hz", minimum=0.0, maximum=100.0, exclusive_minimum=True
        ),
        pointcloud_max_points=integer("pointcloud_max_points", minimum=1, maximum=5_000_000),
        pointcloud_max_bytes=integer("pointcloud_max_bytes", minimum=1, maximum=512_000_000),
        joint_lower_rad=np.asarray(panda.joint_limits, dtype=np.float64)[:, 0].copy(),
        joint_upper_rad=np.asarray(panda.joint_limits, dtype=np.float64)[:, 1].copy(),
        velocity_limits_rad_s=np.asarray(panda.velocity_limits, dtype=np.float64).copy(),
        acceleration_limits_rad_s2=np.asarray(panda.acceleration_limits, dtype=np.float64).copy(),
        jerk_limits_rad_s3=np.asarray(panda.jerk_limits, dtype=np.float64).copy(),
        effort_limits_nm=np.asarray(panda.effort_limits, dtype=np.float64).copy(),
    )


def load_safety_config(
    path: str | Path,
    *,
    panda_config_path: str | Path | None = None,
) -> tuple[SafetyConfig, PandaConfig]:
    """Load ``configs/safety/default.yaml`` and the matching Panda config.

    Returns ``(safety, panda)``.  The caller passes ``panda`` to the validator so
    FK and joint limits use the same single-source configuration.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise SafetyConfigError(f"{config_path}: not found")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SafetyConfigError(f"{config_path}: cannot read YAML: {exc}") from exc
    root = _mapping(raw, str(config_path))
    _unknown(root, {"schema_version", "safety"}, str(config_path))
    if root.get("schema_version") != SCHEMA_VERSION:
        raise SafetyConfigError(f"{config_path}: schema_version must be {SCHEMA_VERSION}")
    safety_data = _mapping(root.get("safety"), f"{config_path}:safety")

    if panda_config_path is None:
        panda_config_path = config_path.parent.parent / "sim" / "panda.yaml"
    panda = load_panda_config(panda_config_path)
    safety = _from_mapping(safety_data, panda, f"{config_path}:safety")
    return safety, panda
