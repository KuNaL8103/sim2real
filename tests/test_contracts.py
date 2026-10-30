"""Contract constants must be unique, well-formed, and documented."""

from __future__ import annotations

import re
from pathlib import Path

from sim2real import contracts

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE = ROOT / "docs" / "ARCHITECTURE.md"
TYPE_RE = re.compile(r"^[a-z][a-z0-9_]*/(msg|srv)/[A-Z][A-Za-z0-9_]*$")


def parse_contract_table(text: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip().strip("`").strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 2 or not cells[0].startswith("/"):
            continue
        rows[cells[0]] = cells[1]
    return rows


def test_topic_names_are_unique_absolute_and_whitespace_free():
    names = list(contracts.TOPICS)
    assert len(names) == len(set(names))
    for name in names:
        assert name.startswith("/"), name
        assert not any(character.isspace() for character in name), name


def test_service_names_are_unique_absolute_and_whitespace_free():
    names = list(contracts.SERVICES)
    assert len(names) == len(set(names))
    for name in names:
        assert name.startswith("/"), name
        assert not any(character.isspace() for character in name), name


def test_type_strings_are_well_formed():
    for name, type_string in {**contracts.TOPICS, **contracts.SERVICES}.items():
        assert TYPE_RE.match(type_string), f"{name} has invalid type string {type_string!r}"


def test_frames_are_unique_relative_and_whitespace_free():
    assert len(contracts.FRAMES) == len(set(contracts.FRAMES))
    for frame in contracts.FRAMES:
        assert not frame.startswith("/"), f"ROS 2 frame IDs must be relative: {frame}"
        assert not any(character.isspace() for character in frame), frame


def test_every_constant_appears_in_architecture_table_with_same_type():
    documented = parse_contract_table(ARCHITECTURE.read_text(encoding="utf-8"))
    expected = {**contracts.TOPICS, **contracts.SERVICES}
    missing = sorted(set(expected) - set(documented))
    assert not missing, f"missing from docs/ARCHITECTURE.md contract table: {missing}"
    mismatched = {
        name: (expected[name], documented[name])
        for name in expected
        if expected[name] != documented[name]
    }
    assert not mismatched, f"type mismatches (constant, documented): {mismatched}"


def test_camera_names_and_topic_paths():
    assert contracts.CAMERAS == ("overhead", "wrist")
    for camera in contracts.CAMERAS:
        for suffix in ("color/image_raw", "depth/image_raw", "camera_info"):
            assert f"/camera/{camera}/{suffix}" in contracts.TOPICS
    assert contracts.COLOR_ENCODING == "rgb8"
    assert contracts.DEPTH_ENCODING == "32FC1"


def test_core_library_does_not_import_ros():
    for path in (ROOT / "src" / "sim2real").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "import rclpy" not in source, f"{path} must stay ROS-free"
        assert "from rclpy" not in source, f"{path} must stay ROS-free"


def test_controlled_joint_names_are_seven_arm_joints_plus_gripper():
    assert tuple(f"panda_joint{i}" for i in range(1, 8)) == contracts.PANDA_ARM_JOINTS
    assert contracts.GRIPPER_JOINT == "panda_finger_joint1"
    assert (*contracts.PANDA_ARM_JOINTS, "panda_finger_joint1") == contracts.CONTROLLED_JOINT_NAMES
