"""Package metadata must be real, complete, and match the directory layout."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "ros2_ws" / "src"
PACKAGE_DIRS = sorted(path for path in SRC.iterdir() if (path / "package.xml").is_file())


def package_root(path: Path) -> ET.Element:
    return ET.parse(path / "package.xml").getroot()


def test_expected_five_packages_exist():
    names = sorted(path.name for path in PACKAGE_DIRS)
    assert names == [
        "sim2real_control",
        "sim2real_msgs",
        "sim2real_perception",
        "sim2real_sim",
        "sim2real_vla",
    ]


@pytest.mark.parametrize("package_dir", PACKAGE_DIRS, ids=lambda path: path.name)
def test_package_name_matches_directory(package_dir: Path):
    root = package_root(package_dir)
    assert root.findtext("name") == package_dir.name


@pytest.mark.parametrize("package_dir", PACKAGE_DIRS, ids=lambda path: path.name)
def test_description_maintainer_and_license_are_non_empty(package_dir: Path):
    root = package_root(package_dir)
    description = (root.findtext("description") or "").strip()
    license_name = (root.findtext("license") or "").strip()
    maintainer = root.find("maintainer")
    assert description, f"{package_dir.name}: empty description"
    assert license_name, f"{package_dir.name}: empty license"
    assert maintainer is not None
    assert (maintainer.text or "").strip(), f"{package_dir.name}: empty maintainer"
    assert maintainer.get("email"), f"{package_dir.name}: maintainer email missing"


@pytest.mark.parametrize("package_dir", PACKAGE_DIRS, ids=lambda path: path.name)
def test_build_type_is_correct(package_dir: Path):
    root = package_root(package_dir)
    build_type = root.findtext("export/build_type")
    expected = "ament_cmake" if package_dir.name == "sim2real_msgs" else "ament_python"
    assert build_type == expected


def test_msgs_package_declares_rosidl_generator_tags():
    root = package_root(SRC / "sim2real_msgs")
    buildtools = {element.text for element in root.findall("buildtool_depend")}
    groups = {element.text for element in root.findall("member_of_group")}
    exec_depends = {element.text for element in root.findall("exec_depend")}
    depends = {element.text for element in root.findall("depend")}

    assert "rosidl_default_generators" in buildtools
    assert "rosidl_interface_packages" in groups
    assert "rosidl_default_runtime" in exec_depends
    assert {"builtin_interfaces", "geometry_msgs", "std_msgs"} <= depends
