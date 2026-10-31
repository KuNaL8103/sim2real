"""Typed loading and validation for ``configs/sim/*.yaml``.

Every failure raises :class:`ConfigError` with the file and field, so a typo in
a scene or camera file cannot silently produce a wrong simulation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sim2real import contracts
from sim2real.sim.backend import ConfigError

SCHEMA_VERSION = 1
REQUIRED_FILES = ("scene.yaml", "panda.yaml", "cameras.yaml")


@dataclass(frozen=True)
class CubeSpec:
    name: str
    size: float
    position: tuple[float, float, float]
    color: tuple[float, float, float]


@dataclass(frozen=True)
class ZoneSpec:
    name: str
    position: tuple[float, float, float]
    radius: float
    color: tuple[float, float, float]


@dataclass(frozen=True)
class SceneConfig:
    table_size: tuple[float, float, float]
    table_position: tuple[float, float, float]
    cubes: Mapping[str, CubeSpec]
    zones: Mapping[str, ZoneSpec]
    workspace_min: np.ndarray
    workspace_max: np.ndarray
    control_rate_hz: float
    physics_dt: float


@dataclass(frozen=True)
class PandaConfig:
    model: Path
    joint_names: tuple[str, ...]
    mujoco_joint_names: tuple[str, ...]
    gripper_joint: str
    mujoco_gripper_joint: str
    gripper_limits: tuple[float, float]
    home_arm: np.ndarray
    home_gripper: float
    joint_limits: np.ndarray
    tcp_offset: np.ndarray
    body_frames: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class CameraConfig:
    name: str
    width: int
    height: int
    rate_hz: float
    fovy_deg: float
    frame_id: str
    parent_frame: str
    position: np.ndarray
    quat_wxyz: np.ndarray


@dataclass(frozen=True)
class SimConfigs:
    scene: SceneConfig
    panda: PandaConfig
    cameras: Mapping[str, CameraConfig] = field(default_factory=dict)

    def camera(self, name: str) -> CameraConfig:
        try:
            return self.cameras[name]
        except KeyError as exc:
            raise ConfigError(
                f"unknown camera {name!r}; configured: {sorted(self.cameras)}"
            ) from exc


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"missing simulator config file: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ConfigError(f"{path}: schema_version must be {SCHEMA_VERSION}")
    return data


def _require(mapping: Mapping[str, Any], key: str, path: Path) -> Any:
    if key not in mapping:
        raise ConfigError(f"{path}: missing required key {key!r}")
    return mapping[key]


def _vec(value: Any, length: int, path: Path, key: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{path}: {key} must be numeric") from exc
    if array.shape != (length,):
        raise ConfigError(f"{path}: {key} must have {length} elements, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ConfigError(f"{path}: {key} must be finite")
    return array


def _positive(value: Any, path: Path, key: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{path}: {key} must be numeric") from exc
    if not np.isfinite(number) or number <= 0.0:
        raise ConfigError(f"{path}: {key} must be positive and finite")
    return number


def _load_scene(path: Path) -> SceneConfig:
    data = _load_yaml(path)
    table = _require(data, "table", path)
    table_size = _vec(_require(table, "size", path), 3, path, "table.size")
    table_position = _vec(_require(table, "position", path), 3, path, "table.position")
    if np.any(table_size <= 0.0):
        raise ConfigError(f"{path}: table.size values must be positive")

    cubes: dict[str, CubeSpec] = {}
    for name, spec in _require(data, "cubes", path).items():
        cubes[name] = CubeSpec(
            name=name,
            size=_positive(_require(spec, "size", path), path, f"cubes.{name}.size"),
            position=tuple(
                _vec(_require(spec, "position", path), 3, path, f"cubes.{name}.position")
            ),
            color=tuple(_vec(_require(spec, "color", path), 3, path, f"cubes.{name}.color")),
        )
    if not cubes:
        raise ConfigError(f"{path}: at least one cube is required")

    zones: dict[str, ZoneSpec] = {}
    for name, spec in _require(data, "zones", path).items():
        zones[name] = ZoneSpec(
            name=name,
            position=tuple(
                _vec(_require(spec, "position", path), 3, path, f"zones.{name}.position")
            ),
            radius=_positive(_require(spec, "radius", path), path, f"zones.{name}.radius"),
            color=tuple(_vec(_require(spec, "color", path), 3, path, f"zones.{name}.color")),
        )

    workspace = _require(data, "workspace", path)
    workspace_min = _vec(_require(workspace, "min", path), 3, path, "workspace.min")
    workspace_max = _vec(_require(workspace, "max", path), 3, path, "workspace.max")
    if np.any(workspace_min >= workspace_max):
        raise ConfigError(f"{path}: workspace.min must be strictly below workspace.max")

    return SceneConfig(
        table_size=tuple(table_size),
        table_position=tuple(table_position),
        cubes=cubes,
        zones=zones,
        workspace_min=workspace_min,
        workspace_max=workspace_max,
        control_rate_hz=_positive(_require(data, "control_rate_hz", path), path, "control_rate_hz"),
        physics_dt=_positive(_require(data, "physics_dt", path), path, "physics_dt"),
    )


def _resolve_model_path(config_path: Path, model: Path) -> Path:
    """Resolve the MJCF path from either the repo or an installed share dir."""
    if model.is_absolute():
        return model
    direct = (config_path.parent / model).resolve()
    if direct.is_file():
        return direct
    # Installed configs live under install/.../share/sim2real_sim/config, so
    # ../../assets no longer points at the repository. Search upwards for an
    # assets/ directory that contains the same suffix.
    parts = model.parts
    if "assets" in parts:
        tail = Path(*parts[parts.index("assets") + 1 :])
        for parent in config_path.parents:
            candidate = (parent / "assets" / tail).resolve()
            if candidate.is_file():
                return candidate
    return direct


def _load_panda(path: Path) -> PandaConfig:
    data = _load_yaml(path)
    model = _resolve_model_path(path, Path(str(_require(data, "model", path))))

    joint_names = tuple(_require(data, "joint_names", path))
    mujoco_joint_names = tuple(_require(data, "mujoco_joint_names", path))
    if len(joint_names) != 7 or len(mujoco_joint_names) != 7:
        raise ConfigError(f"{path}: joint_names and mujoco_joint_names must each have 7 entries")
    if len(set(joint_names)) != 7 or len(set(mujoco_joint_names)) != 7:
        raise ConfigError(f"{path}: joint names must be unique")

    limits = np.asarray(_require(data, "joint_limits", path), dtype=np.float64)
    if limits.shape != (7, 2):
        raise ConfigError(f"{path}: joint_limits must be a 7x2 array")
    if not np.all(np.isfinite(limits)):
        raise ConfigError(f"{path}: joint_limits must be finite")
    if np.any(limits[:, 0] >= limits[:, 1]):
        raise ConfigError(f"{path}: every joint limit must have lower < upper")

    home = _vec(_require(data, "home_arm", path), 7, path, "home_arm")
    outside = (home < limits[:, 0]) | (home > limits[:, 1])
    if np.any(outside):
        raise ConfigError(
            f"{path}: home_arm is outside joint_limits at indices {np.flatnonzero(outside)}"
        )

    gripper = _vec(_require(data, "gripper_limits", path), 2, path, "gripper_limits")
    if gripper[0] >= gripper[1]:
        raise ConfigError(f"{path}: gripper_limits must have lower < upper")
    home_gripper = float(_require(data, "home_gripper", path))
    if not gripper[0] <= home_gripper <= gripper[1]:
        raise ConfigError(f"{path}: home_gripper is outside gripper_limits")

    body_frames: list[tuple[str, str]] = []
    for entry in _require(data, "body_frames", path):
        body_frames.append(
            (str(_require(entry, "ros", path)), str(_require(entry, "mujoco", path)))
        )

    return PandaConfig(
        model=model,
        joint_names=joint_names,
        mujoco_joint_names=mujoco_joint_names,
        gripper_joint=str(_require(data, "gripper_joint", path)),
        mujoco_gripper_joint=str(_require(data, "mujoco_gripper_joint", path)),
        gripper_limits=(float(gripper[0]), float(gripper[1])),
        home_arm=home,
        home_gripper=home_gripper,
        joint_limits=limits,
        tcp_offset=_vec(_require(data, "tcp_offset", path), 3, path, "tcp_offset"),
        body_frames=tuple(body_frames),
    )


def _load_cameras(path: Path) -> dict[str, CameraConfig]:
    data = _load_yaml(path)
    cameras: dict[str, CameraConfig] = {}
    for name, spec in _require(data, "cameras", path).items():
        resolution = _require(spec, "resolution", path)
        if not isinstance(resolution, list) or len(resolution) != 2:
            raise ConfigError(f"{path}: cameras.{name}.resolution must be [width, height]")
        width, height = int(resolution[0]), int(resolution[1])
        if width <= 0 or height <= 0:
            raise ConfigError(f"{path}: cameras.{name}.resolution must be positive")
        fovy = _positive(_require(spec, "fovy_deg", path), path, f"cameras.{name}.fovy_deg")
        if fovy >= 180.0:
            raise ConfigError(f"{path}: cameras.{name}.fovy_deg must be below 180")
        quat = _vec(_require(spec, "quat_wxyz", path), 4, path, f"cameras.{name}.quat_wxyz")
        if np.linalg.norm(quat) == 0.0:
            raise ConfigError(f"{path}: cameras.{name}.quat_wxyz must be non-zero")
        cameras[name] = CameraConfig(
            name=name,
            width=width,
            height=height,
            rate_hz=_positive(_require(spec, "rate_hz", path), path, f"cameras.{name}.rate_hz"),
            fovy_deg=fovy,
            frame_id=str(_require(spec, "frame_id", path)),
            parent_frame=str(_require(spec, "parent_frame", path)),
            position=_vec(_require(spec, "position", path), 3, path, f"cameras.{name}.position"),
            quat_wxyz=quat / np.linalg.norm(quat),
        )
    if set(cameras) != set(contracts.CAMERAS):
        raise ConfigError(f"{path}: cameras must be exactly {sorted(contracts.CAMERAS)}")
    for camera in cameras.values():
        if camera.frame_id != f"camera_{camera.name}_optical_frame":
            raise ConfigError(
                f"{path}: cameras.{camera.name}.frame_id must be camera_{camera.name}_optical_frame"
            )
    return cameras


def load_sim_configs(config_dir: str | Path) -> SimConfigs:
    """Load and validate ``scene.yaml``, ``panda.yaml`` and ``cameras.yaml``."""
    directory = Path(config_dir).expanduser().resolve()
    if not directory.is_dir():
        raise ConfigError(f"simulator config directory does not exist: {directory}")
    paths = {name: directory / name for name in REQUIRED_FILES}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ConfigError(f"missing simulator config files: {missing}")
    return SimConfigs(
        scene=_load_scene(paths["scene.yaml"]),
        panda=_load_panda(paths["panda.yaml"]),
        cameras=_load_cameras(paths["cameras.yaml"]),
    )
