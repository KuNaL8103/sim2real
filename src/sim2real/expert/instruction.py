"""Template-based natural-language instruction parser.

Templates live in ``configs/expert/instructions.yaml``. A template uses only two
placeholders, ``{colour}`` and ``{side}``; every other character is literal. The
parser never guesses: an unknown colour, unknown zone, empty string, or text
that matches no template raises :class:`UnparseableInstructionError`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sim2real.sim.config import SceneConfig


class UnparseableInstructionError(ValueError):
    """Raised when an instruction cannot be mapped to a scene task."""


@dataclass(frozen=True)
class TaskSpec:
    """A resolved pick-and-place task."""

    object_id: str
    zone_id: str
    colour: str
    side: str


@dataclass(frozen=True)
class _Template:
    text: str
    pattern: re.Pattern[str]


def _load_templates(path: Path) -> tuple[_Template, ...]:
    if not path.is_file():
        raise UnparseableInstructionError(f"instruction templates not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise UnparseableInstructionError(f"invalid instruction templates: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise UnparseableInstructionError(f"{path}: schema_version must be 1")
    raw = data.get("templates")
    if not isinstance(raw, list) or not raw:
        raise UnparseableInstructionError(f"{path}: templates must be a non-empty list")

    templates: list[_Template] = []
    for text in raw:
        if not isinstance(text, str) or not text.strip():
            raise UnparseableInstructionError(f"{path}: every template must be a string")
        if "{" in text.replace("{colour}", "").replace("{side}", ""):
            raise UnparseableInstructionError(f"{path}: unsupported placeholder in {text!r}")
        pattern_text = (
            re.escape(" ".join(text.lower().split()))
            .replace(re.escape("{colour}"), r"(?P<colour>[a-z]+)")
            .replace(re.escape("{side}"), r"(?P<side>[a-z]+)")
        )
        templates.append(_Template(text=text, pattern=re.compile(rf"^{pattern_text}$")))
    return tuple(templates)


def _colours(scene: SceneConfig) -> set[str]:
    return {name.removeprefix("cube_") for name in scene.cubes if name.startswith("cube_")}


def _sides(scene: SceneConfig) -> set[str]:
    return {name.removeprefix("zone_") for name in scene.zones if name.startswith("zone_")}


def parse_instruction(
    text: str,
    scene: SceneConfig,
    templates_path: str | Path,
) -> TaskSpec:
    """Map an instruction to ``(object_id, zone_id)`` or raise.

    Matching is case-insensitive, collapses runs of whitespace, and ignores
    trailing punctuation. No default colour or zone is ever substituted.
    """
    if not isinstance(text, str):
        raise UnparseableInstructionError("instruction must be a string")
    normalized = " ".join(text.lower().strip().rstrip(".!?").split())
    if not normalized:
        raise UnparseableInstructionError("instruction is empty")

    templates = _load_templates(Path(templates_path))
    for template in templates:
        match = template.pattern.match(normalized)
        if match is None:
            continue
        colour = match.group("colour")
        side = match.group("side")
        if colour not in _colours(scene):
            raise UnparseableInstructionError(
                f"unknown colour {colour!r}; known colours: {sorted(_colours(scene))}"
            )
        if side not in _sides(scene):
            raise UnparseableInstructionError(
                f"unknown zone {side!r}; known zones: {sorted(_sides(scene))}"
            )
        object_id = f"cube_{colour}"
        zone_id = f"zone_{side}"
        if object_id not in scene.cubes:
            raise UnparseableInstructionError(f"scene has no object {object_id!r}")
        if zone_id not in scene.zones:
            raise UnparseableInstructionError(f"scene has no zone {zone_id!r}")
        return TaskSpec(object_id=object_id, zone_id=zone_id, colour=colour, side=side)

    raise UnparseableInstructionError(
        f"instruction {text!r} matched no template; expected e.g. "
        "'pick up the red cube and place it in the left zone'"
    )


def template_lines(path: str | Path) -> tuple[str, ...]:
    """Return the raw template strings (diagnostics and documentation)."""
    return tuple(template.text for template in _load_templates(Path(path)))


def describe_templates(path: str | Path) -> dict[str, Any]:
    """Small JSON-safe description used by CLI diagnostics."""
    return {"path": str(path), "templates": list(template_lines(path))}
