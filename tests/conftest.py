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
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ROS_SRC = REPO_ROOT / "ros2_ws" / "src"
ROS_PREREQUISITES = ("rclpy", "sim2real_msgs")
MENAGERIE_ASSETS = REPO_ROOT / "assets" / "mujoco" / "franka_emika_panda"

for package_dir in sorted(ROS_SRC.glob("*")):
    if (package_dir / "package.xml").is_file() and (package_dir / package_dir.name).is_dir():
        source_path = str(package_dir)
        if source_path not in sys.path:
            sys.path.insert(0, source_path)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    _skip_ros(items)
    _skip_simulation(items)


def _skip_ros(items: list[pytest.Item]) -> None:
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


def _skip_simulation(items: list[pytest.Item]) -> None:
    assets_ready = (MENAGERIE_ASSETS / "panda.xml").is_file() and (
        MENAGERIE_ASSETS / "assets"
    ).is_dir()
    mujoco_ready = importlib.util.find_spec("mujoco") is not None and assets_ready
    if not mujoco_ready:
        reason = (
            "MuJoCo backend unavailable: install with `uv sync --group sim` and run "
            "`scripts/fetch_assets.sh`"
        )
        skip_marker = pytest.mark.skip(reason=reason)
        for item in items:
            if "sim" in item.keywords or "render" in item.keywords:
                item.add_marker(skip_marker)
        return

    if os.environ.get("SIM2REAL_SKIP_RENDER") == "1":
        skip_marker = pytest.mark.skip(reason="SIM2REAL_SKIP_RENDER=1 (CI: no GPU/display)")
        for item in items:
            if "render" in item.keywords:
                item.add_marker(skip_marker)
