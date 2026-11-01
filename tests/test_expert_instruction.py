"""Instruction parser: valid, paraphrase, unknown colour/zone, empty text."""

from __future__ import annotations

import pytest

from sim2real.expert.instruction import (
    TaskSpec,
    UnparseableInstructionError,
    parse_instruction,
    template_lines,
)
from sim2real.sim.config import load_sim_configs

CONFIG_DIR = "configs/sim"
TEMPLATES = "configs/expert/instructions.yaml"


@pytest.fixture(scope="module")
def scene():
    return load_sim_configs(CONFIG_DIR).scene


def test_canonical_instruction(scene) -> None:
    task = parse_instruction("pick up the red cube and place it in the left zone", scene, TEMPLATES)
    assert task == TaskSpec(object_id="cube_red", zone_id="zone_left", colour="red", side="left")


def test_paraphrase_is_case_and_punctuation_insensitive(scene) -> None:
    task = parse_instruction("  MOVE the BLUE block to the RIGHT region!  ", scene, TEMPLATES)
    assert task.object_id == "cube_blue"
    assert task.zone_id == "zone_right"


def test_unknown_colour_raises(scene) -> None:
    with pytest.raises(UnparseableInstructionError):
        parse_instruction("pick up the purple cube and place it in the left zone", scene, TEMPLATES)


def test_unknown_zone_raises(scene) -> None:
    with pytest.raises(UnparseableInstructionError):
        parse_instruction("pick up the red cube and place it in the middle zone", scene, TEMPLATES)


@pytest.mark.parametrize("text", ["", "   ", "\n\t "])
def test_empty_or_whitespace_raises(scene, text: str) -> None:
    with pytest.raises(UnparseableInstructionError):
        parse_instruction(text, scene, TEMPLATES)


def test_templates_are_loaded() -> None:
    lines = template_lines(TEMPLATES)
    assert len(lines) >= 8
    assert all("{colour}" in line and "{side}" in line for line in lines)
