"""Per-embodiment declarations loaded from ``configs/dataset/embodiments.yaml``.

The declarations are deliberately explicit: units, frames, camera names, and
action semantics are data, not hidden assumptions in the writer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .schema import UnifiedFeatureSpec


class EmbodimentError(ValueError):
    """Raised for an invalid embodiment config."""


@dataclass(frozen=True)
class ActionSpec:
    name: str
    dim: int
    dtype: str
    units: tuple[str, ...]
    frame: str
    normalized_gripper: bool = False


@dataclass(frozen=True)
class EmbodimentSpec:
    embodiment_id: str
    robot_type: str
    control_rate_hz: float
    decimation: int
    fps: float
    state: UnifiedFeatureSpec
    actions: dict[str, ActionSpec]
    cameras: tuple[str, ...]
    gripper_open_m: float
    gripper_closed_m: float
    license_tag: str
    notes: str = ""

    def action(self, name: str) -> ActionSpec:
        try:
            return self.actions[name]
        except KeyError as exc:
            raise EmbodimentError(
                f"{self.embodiment_id}: unknown action representation {name!r}; "
                f"known: {sorted(self.actions)}"
            ) from exc

    def require_cameras(self, cameras: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        unknown = sorted(set(cameras) - set(self.cameras))
        if unknown:
            raise EmbodimentError(
                f"{self.embodiment_id}: unknown cameras {unknown}; known: {list(self.cameras)}"
            )
        return tuple(cameras)


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EmbodimentError(f"{path}: expected a mapping")
    return value


def _unknown(mapping: dict[str, Any], allowed: set[str], path: str) -> None:
    extra = sorted(set(mapping) - allowed)
    if extra:
        raise EmbodimentError(f"{path}: unknown keys {extra}")


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EmbodimentError(f"{path}: expected a non-empty string")
    return value.strip()


def _number(value: Any, path: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EmbodimentError(f"{path}: expected a number")
    number = float(value)
    if positive and number <= 0:
        raise EmbodimentError(f"{path}: expected > 0")
    return number


def _integer(value: Any, path: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EmbodimentError(f"{path}: expected an integer >= {minimum}")
    return value


def _string_list(value: Any, path: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise EmbodimentError(f"{path}: expected a non-empty list")
    return tuple(_string(item, f"{path}[{index}]") for index, item in enumerate(value))


def _state_spec(data: Any, path: str) -> UnifiedFeatureSpec:
    body = _mapping(data, path)
    _unknown(body, {"dtype", "names", "units", "frame"}, path)
    names = _string_list(body.get("names"), f"{path}.names")
    units = _string_list(body.get("units"), f"{path}.units")
    if len(units) != len(names):
        raise EmbodimentError(f"{path}: units and names must have the same length")
    return UnifiedFeatureSpec(
        name="observation.state",
        dtype=_string(body.get("dtype", "float32"), f"{path}.dtype"),
        shape=(len(names),),
        names=names,
        units=units,
        frame=_string(body.get("frame"), f"{path}.frame"),
    )


def _action_specs(data: Any, path: str) -> dict[str, ActionSpec]:
    body = _mapping(data, path)
    if not body:
        raise EmbodimentError(f"{path}: expected at least one action representation")
    actions: dict[str, ActionSpec] = {}
    for name, raw in body.items():
        item_path = f"{path}.{name}"
        item = _mapping(raw, item_path)
        _unknown(item, {"dim", "dtype", "units", "frame", "normalized_gripper"}, item_path)
        dim = _integer(item.get("dim"), f"{item_path}.dim")
        units = _string_list(item.get("units"), f"{item_path}.units")
        if len(units) != dim:
            raise EmbodimentError(f"{item_path}: units must have {dim} entries")
        actions[str(name)] = ActionSpec(
            name=str(name),
            dim=dim,
            dtype=_string(item.get("dtype", "float32"), f"{item_path}.dtype"),
            units=units,
            frame=_string(item.get("frame"), f"{item_path}.frame"),
            normalized_gripper=bool(item.get("normalized_gripper", False)),
        )
    return actions


def load_embodiments(path: str | Path) -> dict[str, EmbodimentSpec]:
    config_path = Path(path)
    if not config_path.is_file():
        raise EmbodimentError(f"missing embodiment config: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise EmbodimentError(f"invalid YAML in {config_path}: {exc}") from exc
    root = _mapping(raw, str(config_path))
    _unknown(root, {"schema_version", "embodiments"}, str(config_path))
    if root.get("schema_version") != 1:
        raise EmbodimentError(f"{config_path}: schema_version must be 1")
    body = _mapping(root.get("embodiments"), "embodiments")
    if not body:
        raise EmbodimentError("embodiments: expected at least one entry")
    result: dict[str, EmbodimentSpec] = {}
    for embodiment_id, value in body.items():
        item_path = f"embodiments.{embodiment_id}"
        item = _mapping(value, item_path)
        _unknown(
            item,
            {
                "robot_type",
                "control_rate_hz",
                "decimation",
                "fps",
                "state",
                "actions",
                "cameras",
                "gripper",
                "license_tag",
                "notes",
            },
            item_path,
        )
        control_rate = _number(
            item.get("control_rate_hz"), f"{item_path}.control_rate_hz", positive=True
        )
        decimation = _integer(item.get("decimation", 1), f"{item_path}.decimation")
        derived_fps = control_rate / decimation
        if "fps" in item:
            fps = _number(item["fps"], f"{item_path}.fps", positive=True)
            if abs(fps - derived_fps) > 1e-9:
                raise EmbodimentError(
                    f"{item_path}.fps: {fps} does not equal control_rate_hz / decimation "
                    f"({derived_fps})"
                )
        else:
            fps = derived_fps
        gripper = _mapping(item.get("gripper"), f"{item_path}.gripper")
        _unknown(gripper, {"open_m", "closed_m"}, f"{item_path}.gripper")
        open_m = _number(gripper.get("open_m"), f"{item_path}.gripper.open_m", positive=True)
        closed_m = _number(gripper.get("closed_m"), f"{item_path}.gripper.closed_m")
        if closed_m >= open_m:
            raise EmbodimentError(f"{item_path}.gripper: closed_m must be smaller than open_m")
        state = _state_spec(item.get("state"), f"{item_path}.state")
        cameras = _string_list(item.get("cameras"), f"{item_path}.cameras")
        if len(set(cameras)) != len(cameras):
            raise EmbodimentError(f"{item_path}.cameras: duplicate entries")
        result[str(embodiment_id)] = EmbodimentSpec(
            embodiment_id=str(embodiment_id),
            robot_type=_string(item.get("robot_type"), f"{item_path}.robot_type"),
            control_rate_hz=control_rate,
            decimation=decimation,
            fps=fps,
            state=state,
            actions=_action_specs(item.get("actions"), f"{item_path}.actions"),
            cameras=cameras,
            gripper_open_m=open_m,
            gripper_closed_m=closed_m,
            license_tag=_string(item.get("license_tag"), f"{item_path}.license_tag"),
            notes=str(item.get("notes", "")),
        )
    return result


def get_embodiment(path: str | Path, embodiment_id: str) -> EmbodimentSpec:
    specs = load_embodiments(path)
    try:
        return specs[embodiment_id]
    except KeyError as exc:
        raise EmbodimentError(
            f"unknown embodiment {embodiment_id!r}; known: {sorted(specs)}"
        ) from exc


def state_feature_name(spec: EmbodimentSpec) -> str:
    return spec.state.name


def action_units(spec: EmbodimentSpec, representation: str) -> tuple[str, ...]:
    return spec.action(representation).units


__all__ = [
    "ActionSpec",
    "EmbodimentError",
    "EmbodimentSpec",
    "action_units",
    "get_embodiment",
    "load_embodiments",
    "state_feature_name",
]
