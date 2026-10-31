"""Validated configuration for domain randomization.

A profile is a small YAML file with ``schema_version: 1`` and a top-level
``randomization`` mapping. Every value is type-checked, bounded, and checked
against the loaded scene (``configs/sim/scene.yaml``): object translations must
lie inside the table workspace, camera names must exist, and colour-temperature
ranges must stay inside the approximate ``kelvin_to_rgb`` domain.

Failures raise :class:`~sim2real.sim.backend.ConfigError` with a path-qualified
message such as::

    randomization.physics.object_mass_kg: expected [lo,hi] with 0 < lo <= hi

Unknown keys are rejected at every level, so a typo cannot be silently ignored.
Each parameter group also accepts ``enabled: true|false`` (default ``true``);
a disabled group is returned as nominal and its sampler stream is not consumed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sim2real import contracts
from sim2real.randomization.color import KELVIN_MAX, KELVIN_MIN, MATERIAL_POOL
from sim2real.sim.backend import ConfigError
from sim2real.sim.config import SceneConfig

SCHEMA_VERSION = 1
PROFILE_ALIASES = {"": "none", "nominal": "none"}

# Documented jitter ceilings. Larger values would move a camera outside the
# tabletop view or rotate a light far enough to break the scene's lighting.
MAX_CAMERA_TRANSLATION_JITTER_M = 0.10
MAX_CAMERA_ROTATION_JITTER_RAD = 0.35
MAX_LIGHT_DIRECTION_JITTER_RAD = 0.50

_PROFILE_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_GROUP_NAMES = ("object_pose", "texture", "lighting", "camera", "physics", "distractors")


@dataclass(frozen=True)
class SceneDefaults:
    """Scene values used to build nominal samples and derive object heights."""

    cube_names: tuple[str, ...]
    cube_center_z_m: Mapping[str, float]
    nominal_positions: Mapping[str, tuple[float, float, float]]
    nominal_colors: Mapping[str, tuple[float, float, float, float]]
    workspace_min: tuple[float, float, float]
    workspace_max: tuple[float, float, float]
    camera_names: tuple[str, ...]


@dataclass(frozen=True)
class ObjectPoseConfig:
    enabled: bool
    translation_bounds_m: tuple[tuple[float, float], tuple[float, float]]
    yaw_range_rad: tuple[float, float]
    min_separation_m: float


@dataclass(frozen=True)
class TextureConfig:
    enabled: bool
    mode: str
    rgba_range: tuple[tuple[float, float], ...] | None
    pool: tuple[str, ...] | None


@dataclass(frozen=True)
class LightingConfig:
    enabled: bool
    intensity_range: tuple[float, float]
    direction_jitter_rad: float
    color_temperature_range_k: tuple[float, float]


@dataclass(frozen=True)
class CameraConfig:
    enabled: bool
    translation_jitter_m: tuple[float, float, float]
    rotation_jitter_rad: tuple[float, float, float]
    apply_to: tuple[str, ...]


@dataclass(frozen=True)
class FrictionConfig:
    sliding: tuple[float, float]
    torsional: tuple[float, float]
    rolling: tuple[float, float]


@dataclass(frozen=True)
class PhysicsConfig:
    enabled: bool
    object_mass_kg: tuple[float, float]
    friction: FrictionConfig
    joint_damping_scale: tuple[float, float]


@dataclass(frozen=True)
class DistractorConfig:
    enabled: bool
    count_range: tuple[int, int]
    pool: tuple[str, ...]


@dataclass(frozen=True)
class RandomizationConfig:
    enabled: bool
    seed: int
    object_pose: ObjectPoseConfig
    texture: TextureConfig
    lighting: LightingConfig
    camera: CameraConfig
    physics: PhysicsConfig
    distractors: DistractorConfig
    scene: SceneDefaults
    source: str
    config_hash: str


# -- primitive validators ----------------------------------------------------


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__}")
    return value


def _keys(mapping: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigError(f"{path}: unknown key(s) {unknown}; allowed: {sorted(allowed)}")


def _bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{path}: expected a boolean, got {type(value).__name__}")
    return value


def _int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path}: expected an integer, got {type(value).__name__}")
    return value


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path}: expected a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"{path}: expected a finite number")
    return number


def _pair(
    value: Any,
    path: str,
    *,
    positive: bool = False,
    low: float | None = None,
    high: float | None = None,
) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigError(f"{path}: expected [lo,hi]")
    lo = _number(value[0], f"{path}[0]")
    hi = _number(value[1], f"{path}[1]")
    if lo > hi:
        raise ConfigError(f"{path}: expected [lo,hi] with lo <= hi")
    if positive and lo <= 0.0:
        raise ConfigError(f"{path}: expected [lo,hi] with 0 < lo <= hi")
    if low is not None and lo < low:
        raise ConfigError(f"{path}: lower bound must be >= {low:g}")
    if high is not None and hi > high:
        raise ConfigError(f"{path}: upper bound must be <= {high:g}")
    return (lo, hi)


def _vector3(
    value: Any,
    path: str,
    *,
    non_negative: bool = False,
    max_abs: float | None = None,
) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ConfigError(f"{path}: expected 3 numbers")
    result = tuple(_number(component, f"{path}[{index}]") for index, component in enumerate(value))
    if non_negative and any(component < 0.0 for component in result):
        raise ConfigError(f"{path}: values must be non-negative")
    if max_abs is not None and any(abs(component) > max_abs for component in result):
        raise ConfigError(f"{path}: each magnitude must be <= {max_abs:g}")
    return result


def _string_list(value: Any, path: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{path}: expected a list of strings")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ConfigError(f"{path}[{index}]: expected a string, got {type(item).__name__}")
        result.append(item)
    return tuple(result)


def _require(data: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in data:
        raise ConfigError(f"{path}.{key}: missing required key")
    return data[key]


# -- group parsers -----------------------------------------------------------


def _parse_object_pose(data: Any, path: str, scene: SceneDefaults) -> ObjectPoseConfig:
    mapping = _mapping(data, path)
    _keys(mapping, {"enabled", "translation_bounds_m", "yaw_range_rad", "min_separation_m"}, path)
    enabled = _bool(mapping.get("enabled", True), f"{path}.enabled")

    bounds = _require(mapping, "translation_bounds_m", path)
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ConfigError(f"{path}.translation_bounds_m: expected [[xmin,xmax],[ymin,ymax]]")
    x_bounds = _pair(bounds[0], f"{path}.translation_bounds_m[0]")
    y_bounds = _pair(bounds[1], f"{path}.translation_bounds_m[1]")

    x_min, x_max = scene.workspace_min[0], scene.workspace_max[0]
    y_min, y_max = scene.workspace_min[1], scene.workspace_max[1]
    if x_bounds[0] < x_min or x_bounds[1] > x_max:
        raise ConfigError(
            f"{path}.translation_bounds_m: x bounds {list(x_bounds)} lie outside the "
            f"scene workspace [{x_min:g}, {x_max:g}]"
        )
    if y_bounds[0] < y_min or y_bounds[1] > y_max:
        raise ConfigError(
            f"{path}.translation_bounds_m: y bounds {list(y_bounds)} lie outside the "
            f"scene workspace [{y_min:g}, {y_max:g}]"
        )

    yaw = _pair(
        _require(mapping, "yaw_range_rad", path),
        f"{path}.yaw_range_rad",
        low=-math.pi,
        high=math.pi,
    )
    separation = _number(_require(mapping, "min_separation_m", path), f"{path}.min_separation_m")
    if separation <= 0.0:
        raise ConfigError(f"{path}.min_separation_m: expected a value > 0")
    return ObjectPoseConfig(
        enabled=enabled,
        translation_bounds_m=(x_bounds, y_bounds),
        yaw_range_rad=yaw,
        min_separation_m=separation,
    )


def _parse_texture(data: Any, path: str) -> TextureConfig:
    mapping = _mapping(data, path)
    _keys(mapping, {"enabled", "mode", "rgba_range", "pool"}, path)
    enabled = _bool(mapping.get("enabled", True), f"{path}.enabled")
    mode = _require(mapping, "mode", path)
    if not isinstance(mode, str) or mode not in {"color_jitter", "material_pool"}:
        raise ConfigError(f"{path}.mode: expected 'color_jitter' or 'material_pool', got {mode!r}")

    if mode == "color_jitter":
        if "pool" in mapping:
            raise ConfigError(f"{path}.pool: not allowed when mode is 'color_jitter'")
        raw = _require(mapping, "rgba_range", path)
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            raise ConfigError(f"{path}.rgba_range: expected 4 [lo,hi] pairs (r, g, b, a)")
        rgba = tuple(
            _pair(raw[index], f"{path}.rgba_range[{index}]", low=0.0, high=1.0)
            for index in range(4)
        )
        return TextureConfig(enabled=enabled, mode=mode, rgba_range=rgba, pool=None)

    if "rgba_range" in mapping:
        raise ConfigError(f"{path}.rgba_range: not allowed when mode is 'material_pool'")
    pool = _string_list(_require(mapping, "pool", path), f"{path}.pool")
    if not pool:
        raise ConfigError(f"{path}.pool: must not be empty for mode 'material_pool'")
    unknown = sorted(set(pool) - set(MATERIAL_POOL))
    if unknown:
        raise ConfigError(
            f"{path}.pool: unknown material(s) {unknown}; known: {sorted(MATERIAL_POOL)}"
        )
    return TextureConfig(enabled=enabled, mode=mode, rgba_range=None, pool=pool)


def _parse_lighting(data: Any, path: str) -> LightingConfig:
    mapping = _mapping(data, path)
    _keys(
        mapping,
        {"enabled", "intensity_range", "direction_jitter_rad", "color_temperature_range_k"},
        path,
    )
    enabled = _bool(mapping.get("enabled", True), f"{path}.enabled")
    intensity = _pair(
        _require(mapping, "intensity_range", path), f"{path}.intensity_range", positive=True
    )
    jitter = _number(
        _require(mapping, "direction_jitter_rad", path), f"{path}.direction_jitter_rad"
    )
    if jitter < 0.0 or jitter > MAX_LIGHT_DIRECTION_JITTER_RAD:
        raise ConfigError(
            f"{path}.direction_jitter_rad: expected 0 <= value <= "
            f"{MAX_LIGHT_DIRECTION_JITTER_RAD:g}"
        )
    temperature = _pair(
        _require(mapping, "color_temperature_range_k", path),
        f"{path}.color_temperature_range_k",
        low=KELVIN_MIN,
        high=KELVIN_MAX,
    )
    return LightingConfig(
        enabled=enabled,
        intensity_range=intensity,
        direction_jitter_rad=jitter,
        color_temperature_range_k=temperature,
    )


def _parse_camera(data: Any, path: str) -> CameraConfig:
    mapping = _mapping(data, path)
    _keys(mapping, {"enabled", "translation_jitter_m", "rotation_jitter_rad", "apply_to"}, path)
    enabled = _bool(mapping.get("enabled", True), f"{path}.enabled")
    translation = _vector3(
        _require(mapping, "translation_jitter_m", path),
        f"{path}.translation_jitter_m",
        non_negative=True,
        max_abs=MAX_CAMERA_TRANSLATION_JITTER_M,
    )
    rotation = _vector3(
        _require(mapping, "rotation_jitter_rad", path),
        f"{path}.rotation_jitter_rad",
        non_negative=True,
        max_abs=MAX_CAMERA_ROTATION_JITTER_RAD,
    )
    apply_to = _string_list(_require(mapping, "apply_to", path), f"{path}.apply_to")
    if not apply_to:
        raise ConfigError(f"{path}.apply_to: must name at least one camera")
    unknown = sorted(set(apply_to) - set(contracts.CAMERAS))
    if unknown:
        raise ConfigError(
            f"{path}.apply_to: unknown camera(s) {unknown}; configured: {sorted(contracts.CAMERAS)}"
        )
    if len(set(apply_to)) != len(apply_to):
        raise ConfigError(f"{path}.apply_to: camera names must be unique")
    return CameraConfig(
        enabled=enabled,
        translation_jitter_m=translation,
        rotation_jitter_rad=rotation,
        apply_to=apply_to,
    )


def _parse_physics(data: Any, path: str) -> PhysicsConfig:
    mapping = _mapping(data, path)
    _keys(mapping, {"enabled", "object_mass_kg", "friction", "joint_damping_scale"}, path)
    enabled = _bool(mapping.get("enabled", True), f"{path}.enabled")
    mass = _pair(_require(mapping, "object_mass_kg", path), f"{path}.object_mass_kg", positive=True)
    friction_mapping = _mapping(_require(mapping, "friction", path), f"{path}.friction")
    _keys(friction_mapping, {"sliding", "torsional", "rolling"}, f"{path}.friction")
    friction = FrictionConfig(
        sliding=_pair(
            _require(friction_mapping, "sliding", f"{path}.friction"),
            f"{path}.friction.sliding",
            positive=True,
        ),
        torsional=_pair(
            _require(friction_mapping, "torsional", f"{path}.friction"),
            f"{path}.friction.torsional",
            positive=True,
        ),
        rolling=_pair(
            _require(friction_mapping, "rolling", f"{path}.friction"),
            f"{path}.friction.rolling",
            positive=True,
        ),
    )
    damping = _pair(
        _require(mapping, "joint_damping_scale", path),
        f"{path}.joint_damping_scale",
        positive=True,
    )
    return PhysicsConfig(
        enabled=enabled, object_mass_kg=mass, friction=friction, joint_damping_scale=damping
    )


def _parse_distractors(data: Any, path: str) -> DistractorConfig:
    mapping = _mapping(data, path)
    _keys(mapping, {"enabled", "count_range", "pool"}, path)
    enabled = _bool(mapping.get("enabled", True), f"{path}.enabled")
    raw = _require(mapping, "count_range", path)
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ConfigError(f"{path}.count_range: expected [lo,hi]")
    lo = _int(raw[0], f"{path}.count_range[0]")
    hi = _int(raw[1], f"{path}.count_range[1]")
    if lo < 0 or lo > hi:
        raise ConfigError(f"{path}.count_range: expected [lo,hi] with 0 <= lo <= hi")
    pool = _string_list(_require(mapping, "pool", path), f"{path}.pool")
    if enabled and not pool:
        raise ConfigError(f"{path}.pool: must not be empty when distractors are enabled")
    return DistractorConfig(enabled=enabled, count_range=(lo, hi), pool=pool)


# -- public loading API ------------------------------------------------------


def scene_defaults(scene: SceneConfig) -> SceneDefaults:
    """Build the nominal values a sample needs from a validated scene config."""
    table_top = scene.table_position[2] + scene.table_size[2] / 2.0
    center_z = {name: table_top + cube.size / 2.0 for name, cube in scene.cubes.items()}
    return SceneDefaults(
        cube_names=tuple(scene.cubes),
        cube_center_z_m=center_z,
        nominal_positions={
            name: tuple(float(value) for value in cube.position)
            for name, cube in scene.cubes.items()
        },
        nominal_colors={
            name: (
                float(cube.color[0]),
                float(cube.color[1]),
                float(cube.color[2]),
                1.0,
            )
            for name, cube in scene.cubes.items()
        },
        workspace_min=tuple(float(value) for value in scene.workspace_min),
        workspace_max=tuple(float(value) for value in scene.workspace_max),
        camera_names=tuple(contracts.CAMERAS),
    )


def _config_hash(body: Mapping[str, Any]) -> str:
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _parse_randomization(
    data: Any, scene: SceneDefaults, source: str, config_hash: str
) -> RandomizationConfig:
    mapping = _mapping(data, "randomization")
    _keys(mapping, set(_GROUP_NAMES) | {"enabled", "seed"}, "randomization")
    for name in _GROUP_NAMES:
        if name not in mapping:
            raise ConfigError(f"randomization.{name}: missing required key")

    enabled = _bool(_require(mapping, "enabled", "randomization"), "randomization.enabled")
    seed = _int(_require(mapping, "seed", "randomization"), "randomization.seed")
    if seed < 0:
        raise ConfigError("randomization.seed: expected a non-negative integer")

    return RandomizationConfig(
        enabled=enabled,
        seed=seed,
        object_pose=_parse_object_pose(mapping["object_pose"], "randomization.object_pose", scene),
        texture=_parse_texture(mapping["texture"], "randomization.texture"),
        lighting=_parse_lighting(mapping["lighting"], "randomization.lighting"),
        camera=_parse_camera(mapping["camera"], "randomization.camera"),
        physics=_parse_physics(mapping["physics"], "randomization.physics"),
        distractors=_parse_distractors(mapping["distractors"], "randomization.distractors"),
        scene=scene,
        source=source,
        config_hash=config_hash,
    )


def load_config_file(path: str | Path, scene: SceneConfig | SceneDefaults) -> RandomizationConfig:
    """Load and validate one randomization profile file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"missing randomization config: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path}: must contain a YAML mapping at the top level")
    _keys(raw, {"schema_version", "randomization"}, str(config_path))
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ConfigError(f"{config_path}: schema_version must be {SCHEMA_VERSION}")
    body = _mapping(_require(raw, "randomization", str(config_path)), "randomization")
    defaults = scene_defaults(scene) if isinstance(scene, SceneConfig) else scene
    return _parse_randomization(body, defaults, str(config_path), _config_hash(body))


def load_profile(
    name: str,
    *,
    scene: SceneConfig | SceneDefaults,
    profiles_dir: str | Path,
) -> RandomizationConfig:
    """Load ``<profiles_dir>/<name>.yaml``, rejecting path-like names."""
    requested = name.strip()
    normalized = PROFILE_ALIASES.get(requested, requested)
    if not _PROFILE_RE.match(normalized):
        raise ConfigError(f"invalid randomization profile name {name!r}")
    directory = Path(profiles_dir)
    path = directory / f"{normalized}.yaml"
    if not path.is_file():
        available = sorted(item.stem for item in directory.glob("*.yaml"))
        raise ConfigError(
            f"unknown randomization profile {name!r}; available: {available or 'none'}"
        )
    return load_config_file(path, scene)
