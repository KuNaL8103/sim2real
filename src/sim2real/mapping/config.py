"""Strict configuration for RGB-D workspace mapping.

The mapping config is the only source of voxel sizes, truncation distances,
workspace bounds, query resolution, and memory budgets.  Every field is
validated with a path-qualified :class:`ConfigError`; unknown keys, non-finite
values, inconsistent truncation, and an over-budget memory estimate are
rejected instead of silently ignored.

The workspace bounds are derived from ``configs/sim/scene.yaml`` when they are
not given explicitly.  Points outside those bounds are discarded *before* they
reach the TSDF volume, so the map cannot silently grow outside the scene.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SCHEMA_VERSION = 1
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "mapping" / "default.yaml"
DEFAULT_SCENE_PATH = Path(__file__).resolve().parents[3] / "configs" / "sim" / "scene.yaml"

# Open3D's UniformTSDFVoxel stores float tsdf + float weight + 3x float colour.
# 32 bytes is a conservative upper bound including padding and per-voxel
# bookkeeping.  It is used only for the config memory estimate.
BYTES_PER_VOXEL = 32

# Sane lower/upper bounds for a tabletop map.  They are validation guards, not
# tuning knobs: a 1 mm voxel over a 0.6 m cube already needs 1.7M voxels.
MIN_VOXEL_SIZE_M = 0.002
MAX_VOXEL_SIZE_M = 0.05


class ConfigError(ValueError):
    """Raised when a mapping config is invalid.  Messages carry a key path."""


@dataclass(frozen=True)
class WorkspaceBounds:
    """Axis-aligned workspace in the ``world`` frame (metres)."""

    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]

    def __post_init__(self) -> None:
        if any(not math.isfinite(v) for v in (*self.minimum, *self.maximum)):
            raise ConfigError("mapping.workspace_bounds: expected finite numbers")
        if any(hi <= lo for lo, hi in zip(self.minimum, self.maximum, strict=True)):
            raise ConfigError("mapping.workspace_bounds: max must be greater than min")

    @property
    def extent(self) -> np.ndarray:
        return np.asarray(self.maximum) - np.asarray(self.minimum)

    @property
    def max_extent(self) -> float:
        return float(self.extent.max())

    @property
    def center(self) -> np.ndarray:
        return (np.asarray(self.minimum) + np.asarray(self.maximum)) * 0.5

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Return a boolean mask for points inside the closed bounds."""
        pts = np.asarray(points, dtype=np.float64)
        lo = np.asarray(self.minimum)
        hi = np.asarray(self.maximum)
        return np.all((pts >= lo) & (pts <= hi), axis=-1)


@dataclass(frozen=True)
class SelfFilterConfig:
    """Coarse robot self-filter: one sphere per link frame."""

    link_frames: tuple[str, ...]
    radii_m: dict[str, float]

    def radius(self, frame: str) -> float:
        return float(self.radii_m[frame])


@dataclass(frozen=True)
class MappingConfig:
    schema_version: int
    voxel_size_m: float
    sdf_trunc_m: float
    depth_trunc_m: float
    depth_min_m: float
    workspace_bounds: WorkspaceBounds
    query_resolution_m: float = 0.02
    max_query_distance_m: float = 0.50
    publish_rate_hz: float = 2.0
    max_triangles: int = 200_000
    max_integrations_before_reset: int = 500
    max_memory_mb: float = 256.0
    map_stale_after_s: float = 5.0
    min_coverage_ratio: float = 0.25
    sync_slop_s: float = 0.05
    tf_timeout_s: float = 0.10
    reset_on_static_tf_change: bool = True
    static_tf_translation_epsilon_m: float = 0.005
    static_tf_rotation_epsilon_rad: float = 0.01
    self_filter: SelfFilterConfig = field(
        default_factory=lambda: SelfFilterConfig(
            link_frames=("panda_link0", "panda_hand"),
            radii_m={"panda_link0": 0.08, "panda_hand": 0.06},
        )
    )

    @property
    def estimated_memory_mb(self) -> float:
        return estimate_memory_mb(self.workspace_bounds, self.voxel_size_m, self.sdf_trunc_m)

    @property
    def volume_extent_m(self) -> float:
        """Side of the cubic uniform volume (workspace + truncation margin)."""
        return self.workspace_bounds.max_extent + 2.0 * self.sdf_trunc_m

    @property
    def volume_resolution(self) -> int:
        return max(1, int(math.ceil(self.volume_extent_m / self.voxel_size_m)))

    @property
    def actual_voxel_size_m(self) -> float:
        return self.volume_extent_m / self.volume_resolution


def estimate_memory_mb(bounds: WorkspaceBounds, voxel_size_m: float, sdf_trunc_m: float) -> float:
    """Upper-bound memory for a cubic uniform TSDF volume over ``bounds``."""
    length = bounds.max_extent + 2.0 * sdf_trunc_m
    resolution = max(1, int(math.ceil(length / voxel_size_m)))
    return resolution**3 * BYTES_PER_VOXEL / 1_000_000.0


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{path}: expected a mapping")
    return value


def _unknown(mapping: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigError(f"{path}: unknown keys {unknown}")


def _number(value: Any, path: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path}: expected a number")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"{path}: expected a finite number")
    if minimum is not None and number < minimum:
        raise ConfigError(f"{path}: expected >= {minimum}")
    return number


def _integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path}: expected an integer")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{path}: expected >= {minimum}")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{path}: expected true or false")
    return value


def _vector3(value: Any, path: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ConfigError(f"{path}: expected three numbers [x, y, z]")
    return tuple(_number(v, f"{path}[{i}]") for i, v in enumerate(value))  # type: ignore[return-value]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"{path}: config file not found")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return _mapping(loaded, str(path))


def _scene_bounds(scene_path: Path) -> WorkspaceBounds:
    scene = _load_yaml(scene_path)
    workspace = _mapping(scene.get("workspace"), f"{scene_path}: workspace")
    return WorkspaceBounds(
        minimum=_vector3(workspace.get("min"), f"{scene_path}: workspace.min"),
        maximum=_vector3(workspace.get("max"), f"{scene_path}: workspace.max"),
    )


_ALLOWED_TOP = {"schema_version", "mapping"}
_ALLOWED_MAPPING = {
    "voxel_size_m",
    "sdf_trunc_m",
    "depth_trunc_m",
    "depth_min_m",
    "workspace_bounds",
    "query_resolution_m",
    "max_query_distance_m",
    "publish_rate_hz",
    "max_triangles",
    "max_integrations_before_reset",
    "max_memory_mb",
    "map_stale_after_s",
    "min_coverage_ratio",
    "sync_slop_s",
    "tf_timeout_s",
    "reset_on_static_tf_change",
    "static_tf_translation_epsilon_m",
    "static_tf_rotation_epsilon_rad",
    "self_filter",
}
_ALLOWED_SELF_FILTER = {"link_frames", "radii_m"}


def load_mapping_config(
    path: str | Path | None = None,
    *,
    scene_path: str | Path | None = None,
) -> MappingConfig:
    """Load and validate ``configs/mapping/default.yaml`` (or ``path``)."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw = _load_yaml(config_path)
    _unknown(raw, _ALLOWED_TOP, str(config_path))
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ConfigError(
            f"{config_path}: schema_version: expected {SCHEMA_VERSION}, "
            f"got {raw.get('schema_version')!r}"
        )
    body = _mapping(raw.get("mapping"), "mapping")
    _unknown(body, _ALLOWED_MAPPING, "mapping")

    if "workspace_bounds" in body:
        wb = _mapping(body["workspace_bounds"], "mapping.workspace_bounds")
        _unknown(wb, {"min", "max"}, "mapping.workspace_bounds")
        bounds = WorkspaceBounds(
            minimum=_vector3(wb.get("min"), "mapping.workspace_bounds.min"),
            maximum=_vector3(wb.get("max"), "mapping.workspace_bounds.max"),
        )
    else:
        bounds = _scene_bounds(Path(scene_path) if scene_path else DEFAULT_SCENE_PATH)

    self_raw = _mapping(body.get("self_filter", {}), "mapping.self_filter")
    _unknown(self_raw, _ALLOWED_SELF_FILTER, "mapping.self_filter")
    frames_raw = self_raw.get("link_frames", ["panda_link0", "panda_hand"])
    if not isinstance(frames_raw, (list, tuple)) or not frames_raw:
        raise ConfigError("mapping.self_filter.link_frames: expected a non-empty list")
    frames: list[str] = []
    for i, frame in enumerate(frames_raw):
        if not isinstance(frame, str) or not frame or frame.startswith("/"):
            raise ConfigError(
                f"mapping.self_filter.link_frames[{i}]: expected a relative frame name"
            )
        frames.append(frame)
    radii_raw = _mapping(
        self_raw.get("radii_m", {frame: 0.05 for frame in frames}),
        "mapping.self_filter.radii_m",
    )
    radii: dict[str, float] = {}
    for frame, radius in radii_raw.items():
        if frame not in frames:
            raise ConfigError(f"mapping.self_filter.radii_m.{frame}: frame is not in link_frames")
        radii[frame] = _number(radius, f"mapping.self_filter.radii_m.{frame}", minimum=0.0)
    for frame in frames:
        if frame not in radii:
            raise ConfigError(f"mapping.self_filter.radii_m.{frame}: missing radius")
        if radii[frame] <= 0.0:
            raise ConfigError(f"mapping.self_filter.radii_m.{frame}: expected > 0")

    config = MappingConfig(
        schema_version=SCHEMA_VERSION,
        voxel_size_m=_number(body.get("voxel_size_m", 0.01), "mapping.voxel_size_m", minimum=0.0),
        sdf_trunc_m=_number(body.get("sdf_trunc_m", 0.04), "mapping.sdf_trunc_m", minimum=0.0),
        depth_trunc_m=_number(body.get("depth_trunc_m", 2.0), "mapping.depth_trunc_m", minimum=0.0),
        depth_min_m=_number(body.get("depth_min_m", 0.05), "mapping.depth_min_m", minimum=0.0),
        workspace_bounds=bounds,
        query_resolution_m=_number(
            body.get("query_resolution_m", 0.02),
            "mapping.query_resolution_m",
            minimum=0.0,
        ),
        max_query_distance_m=_number(
            body.get("max_query_distance_m", 0.50),
            "mapping.max_query_distance_m",
            minimum=0.0,
        ),
        publish_rate_hz=_number(
            body.get("publish_rate_hz", 2.0), "mapping.publish_rate_hz", minimum=0.0
        ),
        max_triangles=_integer(
            body.get("max_triangles", 200_000), "mapping.max_triangles", minimum=1
        ),
        max_integrations_before_reset=_integer(
            body.get("max_integrations_before_reset", 500),
            "mapping.max_integrations_before_reset",
            minimum=1,
        ),
        max_memory_mb=_number(
            body.get("max_memory_mb", 256.0), "mapping.max_memory_mb", minimum=1.0
        ),
        map_stale_after_s=_number(
            body.get("map_stale_after_s", 5.0),
            "mapping.map_stale_after_s",
            minimum=0.0,
        ),
        min_coverage_ratio=_number(
            body.get("min_coverage_ratio", 0.25),
            "mapping.min_coverage_ratio",
            minimum=0.0,
        ),
        sync_slop_s=_number(body.get("sync_slop_s", 0.05), "mapping.sync_slop_s", minimum=0.0),
        tf_timeout_s=_number(body.get("tf_timeout_s", 0.10), "mapping.tf_timeout_s", minimum=0.0),
        reset_on_static_tf_change=_boolean(
            body.get("reset_on_static_tf_change", True),
            "mapping.reset_on_static_tf_change",
        ),
        static_tf_translation_epsilon_m=_number(
            body.get("static_tf_translation_epsilon_m", 0.005),
            "mapping.static_tf_translation_epsilon_m",
            minimum=0.0,
        ),
        static_tf_rotation_epsilon_rad=_number(
            body.get("static_tf_rotation_epsilon_rad", 0.01),
            "mapping.static_tf_rotation_epsilon_rad",
            minimum=0.0,
        ),
        self_filter=SelfFilterConfig(tuple(frames), radii),
    )
    validate_mapping_config(config)
    return config


def validate_mapping_config(config: MappingConfig) -> None:
    """Semantic checks that depend on more than one field."""
    if not (MIN_VOXEL_SIZE_M <= config.voxel_size_m <= MAX_VOXEL_SIZE_M):
        raise ConfigError(
            "mapping.voxel_size_m: expected "
            f"[{MIN_VOXEL_SIZE_M}, {MAX_VOXEL_SIZE_M}], got {config.voxel_size_m}"
        )
    if config.sdf_trunc_m < 2.0 * config.voxel_size_m:
        raise ConfigError(
            "mapping.sdf_trunc_m: expected >= 2 * voxel_size_m "
            f"({2.0 * config.voxel_size_m}), got {config.sdf_trunc_m}"
        )
    if config.depth_min_m >= config.depth_trunc_m:
        raise ConfigError(
            "mapping.depth_min_m: expected < depth_trunc_m "
            f"({config.depth_trunc_m}), got {config.depth_min_m}"
        )
    if config.query_resolution_m <= 0.0:
        raise ConfigError("mapping.query_resolution_m: expected > 0")
    if config.query_resolution_m > config.workspace_bounds.max_extent:
        raise ConfigError(
            "mapping.query_resolution_m: expected <= workspace max extent "
            f"({config.workspace_bounds.max_extent})"
        )
    if config.max_query_distance_m <= 0.0:
        raise ConfigError("mapping.max_query_distance_m: expected > 0")
    if config.publish_rate_hz <= 0.0:
        raise ConfigError("mapping.publish_rate_hz: expected > 0")
    if not (0.0 <= config.min_coverage_ratio <= 1.0):
        raise ConfigError("mapping.min_coverage_ratio: expected [0, 1]")
    if config.map_stale_after_s <= 0.0:
        raise ConfigError("mapping.map_stale_after_s: expected > 0")
    estimate = config.estimated_memory_mb
    if estimate > config.max_memory_mb:
        raise ConfigError(
            "mapping.max_memory_mb: estimated volume memory "
            f"{estimate:.1f} MB exceeds budget {config.max_memory_mb:.1f} MB"
        )
