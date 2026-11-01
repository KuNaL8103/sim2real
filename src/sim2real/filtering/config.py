"""Strict configuration for trajectory filtering.

The filter config is the only place where validation bounds, metric parameters,
hard gates, soft thresholds, weights, and joint limits live. Joint and velocity
limits are **not** repeated here: they are read from ``configs/sim/panda.yaml``
so the filter and the simulator/planner share one source.

Every field is validated. Unknown keys, wrong types, non-finite values, an
unknown derivative method, and an even Savitzky-Golay window are rejected with a
path-qualified error instead of being silently ignored.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DERIVATIVE_METHODS = ("savgol", "finite_difference")


class FilterConfigError(ValueError):
    """Raised when a filter config or its simulator limit source is invalid."""


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FilterConfigError(f"{path}: expected a mapping")
    return value


def _unknown(mapping: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise FilterConfigError(f"{path}: unknown keys {unknown}")


def _get(mapping: dict[str, Any], key: str, default: Any) -> Any:
    return mapping.get(key, default)


def _number(value: Any, path: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FilterConfigError(f"{path}: expected a number")
    number = float(value)
    if not math.isfinite(number):
        raise FilterConfigError(f"{path}: expected a finite number")
    if minimum is not None and number < minimum:
        raise FilterConfigError(f"{path}: expected >= {minimum}")
    return number


def _integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FilterConfigError(f"{path}: expected an integer")
    if minimum is not None and value < minimum:
        raise FilterConfigError(f"{path}: expected >= {minimum}")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise FilterConfigError(f"{path}: expected true or false")
    return value


@dataclass(frozen=True)
class ValidationConfig:
    min_steps: int = 20
    dt_jitter_tolerance_s: float = 0.005
    min_path_length_m: float = 0.01
    derivative_method: str = "savgol"
    savgol_window: int = 9
    savgol_polyorder: int = 5

    def __post_init__(self) -> None:
        if self.min_steps < 3:
            raise FilterConfigError("validation.min_steps: must be >= 3")
        if self.min_path_length_m < 0:
            raise FilterConfigError("validation.min_path_length_m: must be >= 0")
        if self.derivative_method not in DERIVATIVE_METHODS:
            raise FilterConfigError(
                f"validation.derivative_method: expected one of {list(DERIVATIVE_METHODS)}"
            )
        if self.savgol_window < 3 or self.savgol_window % 2 == 0:
            raise FilterConfigError("validation.savgol_window: must be odd and >= 3")
        if self.savgol_polyorder < 3:
            raise FilterConfigError(
                "validation.savgol_polyorder: must be >= 3 for a third derivative"
            )
        if self.savgol_polyorder >= self.savgol_window:
            raise FilterConfigError(
                "validation.savgol_polyorder: must be smaller than savgol_window"
            )
        if self.savgol_window > self.min_steps:
            raise FilterConfigError(
                "validation.savgol_window: must be <= min_steps so short episodes are "
                "rejected before the derivative is computed"
            )


@dataclass(frozen=True)
class SparcConfig:
    f_c_hz: float = 10.0
    nyquist_fraction: float = 0.8
    padlevel: int = 4
    amplitude_threshold: float = 0.05

    def __post_init__(self) -> None:
        if self.f_c_hz <= 0:
            raise FilterConfigError("metrics.sparc.f_c_hz: must be > 0")
        if not (0 < self.nyquist_fraction <= 1):
            raise FilterConfigError("metrics.sparc.nyquist_fraction: must be in (0, 1]")
        if self.padlevel < 0:
            raise FilterConfigError("metrics.sparc.padlevel: must be >= 0")
        if not 0 <= self.amplitude_threshold < 1:
            raise FilterConfigError("metrics.sparc.amplitude_threshold: must be in [0, 1)")

    def effective_cutoff_hz(self, sampling_rate_hz: float) -> float:
        """Cutoff strictly below Nyquist, derived from the actual sampling rate."""
        if sampling_rate_hz <= 0:
            raise FilterConfigError("sampling rate must be positive")
        return min(self.f_c_hz, self.nyquist_fraction * 0.5 * sampling_rate_hz)


@dataclass(frozen=True)
class HardGates:
    require_success: bool = True
    max_collisions: int = 0
    min_joint_margin: float = 0.0
    min_velocity_margin: float = 0.0


@dataclass(frozen=True)
class SoftThresholds:
    max_dimensionless_squared_jerk: float = 1.0e6
    min_sparc: float = -5.0
    max_path_length_m: float = 1.5


@dataclass(frozen=True)
class MetricWeights:
    dimensionless_squared_jerk: float = 0.4
    sparc: float = 0.4
    path_length_m: float = 0.2

    def __post_init__(self) -> None:
        values = (
            self.dimensionless_squared_jerk,
            self.sparc,
            self.path_length_m,
        )
        if any(value < 0 for value in values):
            raise FilterConfigError("weights: every weight must be >= 0")
        if sum(values) <= 0:
            raise FilterConfigError("weights: at least one weight must be > 0")


@dataclass(frozen=True)
class LimitConfig:
    joint_limits: tuple[tuple[float, float], ...]
    velocity_limits: tuple[float, ...]
    source: str

    def __post_init__(self) -> None:
        if len(self.joint_limits) != 7:
            raise FilterConfigError("limits: expected seven joint limits")
        if len(self.velocity_limits) != 7:
            raise FilterConfigError("limits: expected seven velocity limits")
        for index, (low, high) in enumerate(self.joint_limits):
            if not low < high:
                raise FilterConfigError(f"limits.joint_limits[{index}]: expected low < high")
        if any(value <= 0 for value in self.velocity_limits):
            raise FilterConfigError("limits.velocity_limits: every value must be > 0")


@dataclass(frozen=True)
class FilterConfig:
    schema_version: int
    validation: ValidationConfig
    sparc: SparcConfig
    hard_gates: HardGates
    soft_thresholds: SoftThresholds
    weights: MetricWeights
    accept_score_threshold: float
    limits: LimitConfig
    source_path: str = ""
    raw: dict[str, Any] = field(default_factory=dict, compare=False)

    def config_hash(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "validation": self.validation.__dict__,
            "sparc": self.sparc.__dict__,
            "hard_gates": self.hard_gates.__dict__,
            "soft_thresholds": self.soft_thresholds.__dict__,
            "weights": self.weights.__dict__,
            "accept_score_threshold": self.accept_score_threshold,
            "limits": {
                "joint_limits": [list(pair) for pair in self.limits.joint_limits],
                "velocity_limits": list(self.limits.velocity_limits),
                "source": self.limits.source,
            },
        }
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "validation": dict(self.validation.__dict__),
            "metrics": {"sparc": dict(self.sparc.__dict__)},
            "hard_gates": dict(self.hard_gates.__dict__),
            "soft_thresholds": dict(self.soft_thresholds.__dict__),
            "weights": dict(self.weights.__dict__),
            "accept_score_threshold": self.accept_score_threshold,
            "limits_source": self.limits.source,
        }


def load_limit_config(sim_config_path: str | Path) -> LimitConfig:
    path = Path(sim_config_path).expanduser().resolve()
    if not path.is_file():
        raise FilterConfigError(f"missing simulator config: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise FilterConfigError(f"invalid YAML in {path}: {exc}") from exc
    root = _mapping(data, str(path))
    raw_limits = root.get("joint_limits")
    raw_velocity = root.get("velocity_limits")
    if not isinstance(raw_limits, list) or len(raw_limits) != 7:
        raise FilterConfigError(f"{path}: joint_limits must be a list of seven pairs")
    if not isinstance(raw_velocity, list) or len(raw_velocity) != 7:
        raise FilterConfigError(f"{path}: velocity_limits must be a list of seven values")
    limits: list[tuple[float, float]] = []
    for index, pair in enumerate(raw_limits):
        if not isinstance(pair, list) or len(pair) != 2:
            raise FilterConfigError(f"{path}: joint_limits[{index}] must be [low, high]")
        low = _number(pair[0], f"{path}: joint_limits[{index}][0]")
        high = _number(pair[1], f"{path}: joint_limits[{index}][1]")
        limits.append((low, high))
    velocities = tuple(
        _number(value, f"{path}: velocity_limits[{index}]", minimum=0.0)
        for index, value in enumerate(raw_velocity)
    )
    return LimitConfig(tuple(limits), velocities, str(path))


def load_filter_config(path: str | Path) -> FilterConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FilterConfigError(f"missing filter config: {config_path}")
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise FilterConfigError(f"invalid YAML in {config_path}: {exc}") from exc
    root = _mapping(data, str(config_path))
    _unknown(
        root,
        {
            "schema_version",
            "sim_config",
            "validation",
            "metrics",
            "hard_gates",
            "soft_thresholds",
            "weights",
            "accept_score_threshold",
        },
        str(config_path),
    )
    schema_version = _integer(root.get("schema_version"), f"{config_path}: schema_version")
    if schema_version != 1:
        raise FilterConfigError(f"{config_path}: schema_version must be 1")

    raw_validation = _mapping(_get(root, "validation", {}), f"{config_path}: validation")
    _unknown(
        raw_validation,
        {
            "min_steps",
            "dt_jitter_tolerance_s",
            "min_path_length_m",
            "derivative_method",
            "savgol_window",
            "savgol_polyorder",
        },
        f"{config_path}: validation",
    )
    validation = ValidationConfig(
        min_steps=_integer(
            _get(raw_validation, "min_steps", 20), f"{config_path}: validation.min_steps"
        ),
        dt_jitter_tolerance_s=_number(
            _get(raw_validation, "dt_jitter_tolerance_s", 0.005),
            f"{config_path}: validation.dt_jitter_tolerance_s",
            minimum=0.0,
        ),
        min_path_length_m=_number(
            _get(raw_validation, "min_path_length_m", 0.01),
            f"{config_path}: validation.min_path_length_m",
            minimum=0.0,
        ),
        derivative_method=str(_get(raw_validation, "derivative_method", "savgol")),
        savgol_window=_integer(
            _get(raw_validation, "savgol_window", 9),
            f"{config_path}: validation.savgol_window",
        ),
        savgol_polyorder=_integer(
            _get(raw_validation, "savgol_polyorder", 5),
            f"{config_path}: validation.savgol_polyorder",
        ),
    )

    raw_metrics = _mapping(_get(root, "metrics", {}), f"{config_path}: metrics")
    _unknown(raw_metrics, {"sparc"}, f"{config_path}: metrics")
    raw_sparc = _mapping(_get(raw_metrics, "sparc", {}), f"{config_path}: metrics.sparc")
    _unknown(
        raw_sparc,
        {"f_c_hz", "nyquist_fraction", "padlevel", "amplitude_threshold"},
        f"{config_path}: metrics.sparc",
    )
    sparc = SparcConfig(
        f_c_hz=_number(
            _get(raw_sparc, "f_c_hz", 10.0), f"{config_path}: metrics.sparc.f_c_hz", minimum=0.0
        ),
        nyquist_fraction=_number(
            _get(raw_sparc, "nyquist_fraction", 0.8),
            f"{config_path}: metrics.sparc.nyquist_fraction",
            minimum=0.0,
        ),
        padlevel=_integer(
            _get(raw_sparc, "padlevel", 4), f"{config_path}: metrics.sparc.padlevel", minimum=0
        ),
        amplitude_threshold=_number(
            _get(raw_sparc, "amplitude_threshold", 0.05),
            f"{config_path}: metrics.sparc.amplitude_threshold",
            minimum=0.0,
        ),
    )

    raw_hard = _mapping(_get(root, "hard_gates", {}), f"{config_path}: hard_gates")
    _unknown(
        raw_hard,
        {"require_success", "max_collisions", "min_joint_margin", "min_velocity_margin"},
        f"{config_path}: hard_gates",
    )
    hard = HardGates(
        require_success=_boolean(
            _get(raw_hard, "require_success", True), f"{config_path}: hard_gates.require_success"
        ),
        max_collisions=_integer(
            _get(raw_hard, "max_collisions", 0),
            f"{config_path}: hard_gates.max_collisions",
            minimum=0,
        ),
        min_joint_margin=_number(
            _get(raw_hard, "min_joint_margin", 0.0),
            f"{config_path}: hard_gates.min_joint_margin",
        ),
        min_velocity_margin=_number(
            _get(raw_hard, "min_velocity_margin", 0.0),
            f"{config_path}: hard_gates.min_velocity_margin",
        ),
    )

    raw_soft = _mapping(_get(root, "soft_thresholds", {}), f"{config_path}: soft_thresholds")
    _unknown(
        raw_soft,
        {
            "max_dimensionless_squared_jerk",
            "min_sparc",
            "max_path_length_m",
        },
        f"{config_path}: soft_thresholds",
    )
    soft = SoftThresholds(
        max_dimensionless_squared_jerk=_number(
            _get(raw_soft, "max_dimensionless_squared_jerk", 1.0e6),
            f"{config_path}: soft_thresholds.max_dimensionless_squared_jerk",
            minimum=0.0,
        ),
        min_sparc=_number(
            _get(raw_soft, "min_sparc", -5.0),
            f"{config_path}: soft_thresholds.min_sparc",
        ),
        max_path_length_m=_number(
            _get(raw_soft, "max_path_length_m", 1.5),
            f"{config_path}: soft_thresholds.max_path_length_m",
            minimum=0.0,
        ),
    )

    raw_weights = _mapping(_get(root, "weights", {}), f"{config_path}: weights")
    _unknown(
        raw_weights,
        {"dimensionless_squared_jerk", "sparc", "path_length_m"},
        f"{config_path}: weights",
    )
    weights = MetricWeights(
        dimensionless_squared_jerk=_number(
            _get(raw_weights, "dimensionless_squared_jerk", 0.4),
            f"{config_path}: weights.dimensionless_squared_jerk",
            minimum=0.0,
        ),
        sparc=_number(
            _get(raw_weights, "sparc", 0.4), f"{config_path}: weights.sparc", minimum=0.0
        ),
        path_length_m=_number(
            _get(raw_weights, "path_length_m", 0.2),
            f"{config_path}: weights.path_length_m",
            minimum=0.0,
        ),
    )
    accept_score = _number(
        _get(root, "accept_score_threshold", 0.5),
        f"{config_path}: accept_score_threshold",
        minimum=0.0,
    )
    if accept_score > 1:
        raise FilterConfigError(f"{config_path}: accept_score_threshold must be <= 1")

    sim_config = _get(root, "sim_config", "../sim/panda.yaml")
    if not isinstance(sim_config, str) or not sim_config:
        raise FilterConfigError(f"{config_path}: sim_config must be a non-empty path")
    sim_path = Path(sim_config)
    if not sim_path.is_absolute():
        sim_path = config_path.parent / sim_path
        if not sim_path.is_file():
            # Installed ROS layout: filtering configs live in
            # share/sim2real_sim/filtering while simulator configs live in
            # share/sim2real_sim/config. The repo layout needs no fallback.
            fallback = config_path.parent.parent / "config" / Path(sim_config).name
            if fallback.is_file():
                sim_path = fallback
    limits = load_limit_config(sim_path)

    return FilterConfig(
        schema_version=schema_version,
        validation=validation,
        sparc=sparc,
        hard_gates=hard,
        soft_thresholds=soft,
        weights=weights,
        accept_score_threshold=accept_score,
        limits=limits,
        source_path=str(config_path),
        raw=dict(root),
    )


__all__ = [
    "DERIVATIVE_METHODS",
    "FilterConfig",
    "FilterConfigError",
    "HardGates",
    "LimitConfig",
    "MetricWeights",
    "SoftThresholds",
    "SparcConfig",
    "ValidationConfig",
    "load_filter_config",
    "load_limit_config",
]
