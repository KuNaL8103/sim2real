"""Typed loading for ``configs/expert/*.yaml``.

The expert package is pure Python: this module imports no simulator and no ROS.
All numbers that affect motion, IK, success, or staging come from these files so
they can be reviewed without reading code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = 1
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_DIR = _REPO_ROOT / "configs" / "expert"


class ExpertConfigError(ValueError):
    """Raised for a missing or malformed expert config file."""


@dataclass(frozen=True)
class IKConfig:
    max_iterations: int = 200
    position_tolerance_m: float = 0.002
    orientation_tolerance_rad: float = 0.05
    damping: float = 0.05
    max_step_rad: float = 0.15
    jacobian_step_rad: float = 1.0e-6


@dataclass(frozen=True)
class PlannerConfig:
    control_rate_hz: float = 20.0
    min_duration_s: float = 0.3
    max_waypoints: int = 64
    # ``None`` means "use the Panda model limits from configs/sim/panda.yaml".
    velocity_limits: tuple[float, ...] | None = None
    acceleration_limits: tuple[float, ...] | None = None


@dataclass(frozen=True)
class MotionConfig:
    pregrasp_height_m: float = 0.12
    grasp_depth_m: float = 0.0
    lift_height_m: float = 0.14
    transit_clearance_m: float = 0.16
    place_depth_m: float = 0.0
    retreat_height_m: float = 0.12
    gripper_closed: float = 0.0
    gripper_open: float = 0.04
    grasp_hold_steps: int = 8
    release_hold_steps: int = 5
    settle_steps: int = 6
    reached_tolerance_m: float = 0.02
    grasp_confirm_distance_m: float = 0.06


@dataclass(frozen=True)
class SuccessConfig:
    zone_radius_tolerance_m: float = 0.005
    rest_height_tolerance_m: float = 0.01
    stable_steps: int = 5
    stability_tolerance_m: float = 0.002
    require_gripper_open: bool = True
    success_zone_scale: float = 1.0


@dataclass(frozen=True)
class StagingConfig:
    capture_images: bool = False
    image_decimation: int = 5
    resolution: tuple[int, int] = (256, 256)
    desired_render_seconds_per_frame: float = 0.25


@dataclass(frozen=True)
class ExpertConfig:
    control_rate_hz: float
    ik: IKConfig
    planner: PlannerConfig
    motion: MotionConfig
    success: SuccessConfig
    staging: StagingConfig
    state_timeouts_s: dict[str, float]
    max_steps: int = 600
    default_profile: str = "default"
    instructions_file: Path = DEFAULT_CONFIG_DIR / "instructions.yaml"
    perturbations_file: Path = DEFAULT_CONFIG_DIR / "perturbations.yaml"
    extra: dict[str, Any] = field(default_factory=dict)


def resolve_config_dir(config_dir: str | Path | None = None) -> Path:
    directory = Path(config_dir).expanduser() if config_dir else DEFAULT_CONFIG_DIR
    directory = directory.resolve()
    if not directory.is_dir():
        raise ExpertConfigError(f"expert config directory not found: {directory}")
    return directory


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ExpertConfigError(f"missing expert config file: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ExpertConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ExpertConfigError(f"{path} must contain a YAML mapping")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ExpertConfigError(f"{path}: schema_version must be {SCHEMA_VERSION}")
    return data


def _mapping(data: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ExpertConfigError(f"{path}: {key} must be a mapping")
    return value


def _positive(value: Any, name: str, path: Path, *, allow_zero: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ExpertConfigError(f"{path}: {name} must be numeric") from exc
    if number < 0.0 or (number == 0.0 and not allow_zero):
        raise ExpertConfigError(f"{path}: {name} must be positive")
    return number


def _integer(value: Any, name: str, path: Path, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExpertConfigError(f"{path}: {name} must be an integer")
    if value < minimum:
        raise ExpertConfigError(f"{path}: {name} must be >= {minimum}")
    return value


def _vector(value: Any, length: int, name: str, path: Path) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ExpertConfigError(f"{path}: {name} must have {length} entries")
    return tuple(_positive(entry, name, path) for entry in value)


def load_expert_config(config_dir: str | Path | None = None) -> ExpertConfig:
    directory = resolve_config_dir(config_dir)
    path = directory / "default.yaml"
    data = _load_yaml(path)

    ik_data = _mapping(data, "ik", path)
    ik = IKConfig(
        max_iterations=_integer(
            ik_data.get("max_iterations", 200), "ik.max_iterations", path, minimum=1
        ),
        position_tolerance_m=_positive(
            ik_data.get("position_tolerance_m", 0.002), "ik.position_tolerance_m", path
        ),
        orientation_tolerance_rad=_positive(
            ik_data.get("orientation_tolerance_rad", 0.05),
            "ik.orientation_tolerance_rad",
            path,
        ),
        damping=_positive(ik_data.get("damping", 0.05), "ik.damping", path),
        max_step_rad=_positive(ik_data.get("max_step_rad", 0.15), "ik.max_step_rad", path),
        jacobian_step_rad=_positive(
            ik_data.get("jacobian_step_rad", 1.0e-6), "ik.jacobian_step_rad", path
        ),
    )

    planner_data = _mapping(data, "planner", path)
    velocity = planner_data.get("velocity_limits")
    acceleration = planner_data.get("acceleration_limits")
    planner = PlannerConfig(
        velocity_limits=(
            _vector(velocity, 7, "velocity_limits", path) if velocity is not None else None
        ),
        acceleration_limits=(
            _vector(acceleration, 7, "acceleration_limits", path)
            if acceleration is not None
            else None
        ),
        control_rate_hz=_positive(
            planner_data.get("control_rate_hz", 20.0), "planner.control_rate_hz", path
        ),
        min_duration_s=_positive(
            planner_data.get("min_duration_s", 0.3), "planner.min_duration_s", path
        ),
        max_waypoints=_integer(
            planner_data.get("max_waypoints", 64), "planner.max_waypoints", path, minimum=1
        ),
    )
    if planner.control_rate_hz <= 0.0:
        raise ExpertConfigError(f"{path}: planner.control_rate_hz must be positive")

    motion_data = _mapping(data, "motion", path)
    motion = MotionConfig(
        pregrasp_height_m=_positive(
            motion_data.get("pregrasp_height_m", 0.12), "motion.pregrasp_height_m", path
        ),
        grasp_depth_m=float(motion_data.get("grasp_depth_m", 0.0)),
        lift_height_m=_positive(
            motion_data.get("lift_height_m", 0.14), "motion.lift_height_m", path
        ),
        transit_clearance_m=_positive(
            motion_data.get("transit_clearance_m", 0.16), "motion.transit_clearance_m", path
        ),
        place_depth_m=float(motion_data.get("place_depth_m", 0.0)),
        retreat_height_m=_positive(
            motion_data.get("retreat_height_m", 0.12), "motion.retreat_height_m", path
        ),
        gripper_closed=float(motion_data.get("gripper_closed", 0.0)),
        gripper_open=float(motion_data.get("gripper_open", 0.04)),
        grasp_hold_steps=_integer(
            motion_data.get("grasp_hold_steps", 8), "motion.grasp_hold_steps", path, minimum=1
        ),
        release_hold_steps=_integer(
            motion_data.get("release_hold_steps", 5), "motion.release_hold_steps", path, minimum=1
        ),
        settle_steps=_integer(
            motion_data.get("settle_steps", 6), "motion.settle_steps", path, minimum=0
        ),
        reached_tolerance_m=_positive(
            motion_data.get("reached_tolerance_m", 0.02), "motion.reached_tolerance_m", path
        ),
        grasp_confirm_distance_m=_positive(
            motion_data.get("grasp_confirm_distance_m", 0.06),
            "motion.grasp_confirm_distance_m",
            path,
        ),
    )

    success_data = _mapping(data, "success", path)
    success = SuccessConfig(
        zone_radius_tolerance_m=_positive(
            success_data.get("zone_radius_tolerance_m", 0.005),
            "success.zone_radius_tolerance_m",
            path,
            allow_zero=True,
        ),
        rest_height_tolerance_m=_positive(
            success_data.get("rest_height_tolerance_m", 0.01),
            "success.rest_height_tolerance_m",
            path,
        ),
        stable_steps=_integer(
            success_data.get("stable_steps", 5), "success.stable_steps", path, minimum=1
        ),
        stability_tolerance_m=_positive(
            success_data.get("stability_tolerance_m", 0.002),
            "success.stability_tolerance_m",
            path,
        ),
        require_gripper_open=bool(success_data.get("require_gripper_open", True)),
        success_zone_scale=_positive(
            success_data.get("success_zone_scale", 1.0), "success.success_zone_scale", path
        ),
    )

    staging_data = _mapping(data, "staging", path)
    resolution = staging_data.get("resolution", [256, 256])
    if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
        raise ExpertConfigError(f"{path}: staging.resolution must be [height, width]")
    staging = StagingConfig(
        capture_images=bool(staging_data.get("capture_images", False)),
        image_decimation=_integer(
            staging_data.get("image_decimation", 5), "staging.image_decimation", path, minimum=1
        ),
        resolution=(
            _integer(resolution[0], "staging.resolution[0]", path, minimum=1),
            _integer(resolution[1], "staging.resolution[1]", path, minimum=1),
        ),
        desired_render_seconds_per_frame=_positive(
            staging_data.get("desired_render_seconds_per_frame", 0.25),
            "staging.desired_render_seconds_per_frame",
            path,
        ),
    )

    timeouts_data = _mapping(data, "state_timeouts_s", path)
    if not timeouts_data:
        raise ExpertConfigError(f"{path}: state_timeouts_s must not be empty")
    timeouts = {
        str(state): _positive(value, f"state_timeouts_s.{state}", path)
        for state, value in timeouts_data.items()
    }

    instructions = data.get("instructions_file", "instructions.yaml")
    perturbations = data.get("perturbations_file", "perturbations.yaml")

    return ExpertConfig(
        control_rate_hz=_positive(data.get("control_rate_hz", 20.0), "control_rate_hz", path),
        ik=ik,
        planner=planner,
        motion=motion,
        success=success,
        staging=staging,
        state_timeouts_s=timeouts,
        max_steps=_integer(data.get("max_steps", 600), "max_steps", path, minimum=1),
        default_profile=str(data.get("default_profile", "default")),
        instructions_file=(directory / str(instructions)).resolve(),
        perturbations_file=(directory / str(perturbations)).resolve(),
    )
