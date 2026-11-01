"""Every ROS test module must import without rclpy (CI collection safety).

A module-level ``import rclpy`` once broke CI collection while local runs passed
because the system ``rclpy`` was visible.  This test imports each ROS test file
in the ROS-free core environment.  A module may raise ``pytest.skip`` (for
example ``pytest.importorskip``); any other exception is a collection failure.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ROS_TEST_FILES = sorted((ROOT / "ros2_ws" / "src").glob("*/test/test_*.py"))


def test_every_ros_test_module_imports_without_rclpy():
    assert ROS_TEST_FILES, "expected ROS test files under ros2_ws/src"
    failures: list[str] = []
    for path in ROS_TEST_FILES:
        module_name = f"_collection_{path.parent.parent.name}_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except pytest.skip.Exception:
            continue
        except Exception as exc:  # noqa: BLE001 - report the file, not the stack
            failures.append(f"{path}: {type(exc).__name__}: {exc}")
    assert not failures, "ROS test modules failed to import without rclpy:\n" + "\n".join(failures)


def test_mapper_node_module_is_guarded_by_the_ros_test():
    """The ROS test itself must guard its rclpy imports, not the core package."""
    path = ROOT / "ros2_ws" / "src" / "sim2real_perception" / "test" / "test_workspace_mapper.py"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "try:" in text
    assert "import rclpy" in text
    assert "except ImportError" in text
