"""License and provenance checks.

The project is Apache-2.0. ``LICENSE`` must be the official text (fetched from
apache.org, not paraphrased) and every metadata surface must agree.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_license_file_exists_and_has_apache_header() -> None:
    license_path = REPO_ROOT / "LICENSE"
    assert license_path.is_file(), "LICENSE is missing"
    text = license_path.read_text(encoding="utf-8")
    # The canonical apache.org file starts with a blank line, then the header.
    assert text.lstrip().startswith("Apache License"), "LICENSE lacks the Apache header"
    assert "Version 2.0, January 2004" in text
    assert "http://www.apache.org/licenses/" in text
    assert "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION" in text


def test_pyproject_declares_apache_2_0() -> None:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    assert data["project"]["license"]["text"] == "Apache-2.0"


def test_all_package_xml_declare_apache_2_0() -> None:
    package_files = sorted((REPO_ROOT / "ros2_ws" / "src").glob("*/package.xml"))
    assert package_files, "no package.xml files found"
    for package_file in package_files:
        text = package_file.read_text(encoding="utf-8")
        assert "<license>Apache-2.0</license>" in text, f"{package_file} is not Apache-2.0"


def test_readme_states_apache_2_0_and_points_at_license() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "Apache-2.0" in readme
    assert "LICENSE" in readme
