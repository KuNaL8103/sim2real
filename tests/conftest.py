"""Shared pytest configuration.

Two jobs:
1. Make source-only ament_python packages importable for their trivial import
   tests, without requiring a colcon build.
2. Skip ROS-marked tests with an explicit reason when either ROS prerequisite
   (`rclpy` or the built `sim2real_msgs`) is unavailable. Skipped is never
   reported as passed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ROS_SRC = REPO_ROOT / "ros2_ws" / "src"
ROS_PREREQUISITES = ("rclpy", "sim2real_msgs")

for package_dir in sorted(ROS_SRC.glob("*")):
    if (package_dir / "package.xml").is_file() and (package_dir / package_dir.name).is_dir():
        source_path = str(package_dir)
        if source_path not in sys.path:
            sys.path.insert(0, source_path)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    missing = [name for name in ROS_PREREQUISITES if importlib.util.find_spec(name) is None]
    if not missing:
        return
    reason = (
        f"ROS prerequisites missing ({', '.join(missing)}): source "
        "/opt/ros/jazzy/setup.bash and build/source ros2_ws before running ROS-marked tests"
    )
    skip_marker = pytest.mark.skip(reason=reason)
    for item in items:
        if "ros" in item.keywords:
            item.add_marker(skip_marker)
