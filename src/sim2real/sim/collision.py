"""Pure contact classification for the simulator backends.

The backends extract raw contacts from the physics engine and call
:func:`classify_contacts`.  The rules live in ``configs/sim/collision.yaml`` so
the MuJoCo and fake backends share one definition and tests can exercise the
logic without MuJoCo.

Names are geometry names: scene geoms use their MJCF names (``table``,
``cube_red_geom``, ``zone_left``); unnamed robot collision geoms are named
``"<body>/geom<id>"`` by the backend.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .backend import CollisionInfo

DEFAULT_COLLISION_PATH = Path(__file__).resolve().parents[3] / "configs" / "sim" / "collision.yaml"

ROBOT_GROUPS = frozenset({"arm_links", "hand", "fingers"})
ALLOWED_RULES = frozenset({"finger_target_grasp", "cube_table", "cube_zone"})


class CollisionConfigError(ValueError):
    """Raised for an invalid collision config. Messages carry a key path."""


@dataclass(frozen=True)
class CollisionConfig:
    groups: Mapping[str, tuple[str, ...]]
    target_group: str
    allowed_contacts: tuple[str, ...]
    ignore_robot_self: bool
    penetration_only: bool

    def group_for(self, geom_name: str) -> str | None:
        for group, prefixes in self.groups.items():
            for prefix in prefixes:
                if geom_name == prefix or geom_name.startswith(prefix):
                    return group
        return None

    def target_geom_name(self, target_object: str | None) -> str | None:
        if not target_object:
            return None
        if target_object.endswith("_geom"):
            return target_object
        return f"{target_object}_geom"


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CollisionConfigError(f"{path}: expected a mapping")
    return value


def _string_list(value: Any, path: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise CollisionConfigError(f"{path}: expected a non-empty list")
    items: list[str] = []
    for i, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise CollisionConfigError(f"{path}[{i}]: expected a non-empty string")
        items.append(item.strip())
    if len(set(items)) != len(items):
        raise CollisionConfigError(f"{path}: duplicate entries")
    return tuple(items)


def load_collision_config(path: str | Path | None = None) -> CollisionConfig:
    config_path = Path(path) if path is not None else DEFAULT_COLLISION_PATH
    if not config_path.is_file():
        raise CollisionConfigError(f"{config_path}: collision config not found")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw = _mapping(raw, str(config_path))
    unknown_top = sorted(set(raw) - {"schema_version", "collision"})
    if unknown_top:
        raise CollisionConfigError(f"{config_path}: unknown keys {unknown_top}")
    if raw.get("schema_version") != 1:
        raise CollisionConfigError(f"{config_path}: schema_version: expected 1")
    body = _mapping(raw.get("collision"), "collision")
    unknown = sorted(
        set(body)
        - {
            "groups",
            "target_group",
            "allowed_contacts",
            "ignore_robot_self",
            "penetration_only",
        }
    )
    if unknown:
        raise CollisionConfigError(f"collision: unknown keys {unknown}")

    groups_raw = _mapping(body.get("groups"), "collision.groups")
    groups: dict[str, tuple[str, ...]] = {}
    for group, prefixes in groups_raw.items():
        groups[group] = _string_list(prefixes, f"collision.groups.{group}")
    if not groups:
        raise CollisionConfigError("collision.groups: expected at least one group")

    target_group = body.get("target_group")
    if target_group not in groups:
        raise CollisionConfigError(f"collision.target_group: expected one of {sorted(groups)}")
    allowed = _string_list(body.get("allowed_contacts"), "collision.allowed_contacts")
    unknown_rules = sorted(set(allowed) - ALLOWED_RULES)
    if unknown_rules:
        raise CollisionConfigError(f"collision.allowed_contacts: unknown rules {unknown_rules}")
    ignore_self = body.get("ignore_robot_self", True)
    penetration_only = body.get("penetration_only", True)
    if not isinstance(ignore_self, bool):
        raise CollisionConfigError("collision.ignore_robot_self: expected true or false")
    if not isinstance(penetration_only, bool):
        raise CollisionConfigError("collision.penetration_only: expected true or false")
    return CollisionConfig(
        groups=groups,
        target_group=str(target_group),
        allowed_contacts=tuple(allowed),
        ignore_robot_self=ignore_self,
        penetration_only=penetration_only,
    )


def _is(candidate: str | None, group: str) -> bool:
    return candidate == group


def classify_contacts(
    contacts: Iterable[tuple[str, str, float]],
    config: CollisionConfig,
    target_object: str | None,
) -> CollisionInfo:
    """Classify ``(geom1, geom2, penetration_depth)`` contacts.

    ``penetration_depth`` is MuJoCo's signed contact distance: negative means
    penetration.  Only penetrating unintended contacts increment
    ``num_unintended_contacts``.  ``details`` includes allowed, self, touch, and
    unintended contacts so the expert can still detect a target grasp; the
    runner uses the count, not the details, for the collision flag.
    """
    target_geom = config.target_geom_name(target_object)
    details: list[str] = []
    unintended = 0
    for geom1, geom2, dist in contacts:
        g1 = config.group_for(geom1)
        g2 = config.group_for(geom2)
        penetrating = float(dist) < 0.0
        if config.penetration_only and not penetrating:
            status = "touch"
        else:
            status = _classify_pair(g1, g2, geom1, geom2, target_geom, config)
        if status == "unintended":
            unintended += 1
        details.append(
            f"{status}: {geom1}<->{geom2} dist={float(dist):.5f} "
            f"groups={g1 or 'unknown'}/{g2 or 'unknown'}"
        )
    return CollisionInfo(num_unintended_contacts=unintended, details=tuple(details))


def _classify_pair(
    g1: str | None,
    g2: str | None,
    geom1: str,
    geom2: str,
    target_geom: str | None,
    config: CollisionConfig,
) -> str:
    robot1 = g1 in ROBOT_GROUPS
    robot2 = g2 in ROBOT_GROUPS
    cube1 = _is(g1, "cubes")
    cube2 = _is(g2, "cubes")
    table1 = _is(g1, "table")
    table2 = _is(g2, "table")
    zone1 = _is(g1, "zones")
    zone2 = _is(g2, "zones")

    # Allowed target grasp: a finger touching the current target cube.
    if "finger_target_grasp" in config.allowed_contacts and target_geom is not None:
        finger_target = (_is(g1, "fingers") and geom2 == target_geom) or (
            _is(g2, "fingers") and geom1 == target_geom
        )
        if finger_target:
            return "allowed"
    if "cube_table" in config.allowed_contacts and ((cube1 and table2) or (cube2 and table1)):
        return "allowed"
    if "cube_zone" in config.allowed_contacts and ((cube1 and zone2) or (cube2 and zone1)):
        return "allowed"

    # Robot-robot contacts are self-contacts, not environment collisions.
    if robot1 and robot2:
        return "self" if config.ignore_robot_self else "unintended"

    # Anything else involving a robot geom is unintended, including arm/hand
    # vs table, finger vs table, and finger vs a non-target cube.
    if robot1 or robot2:
        return "unintended"

    # Cube-cube is unintended (target vs non-target, or two non-target cubes).
    if cube1 and cube2:
        return "unintended"

    # Unknown names are reported conservatively.
    if g1 is None or g2 is None:
        return "unintended"

    # Non-robot environment contacts (for example table<->zone) are ignored.
    return "ignored"
