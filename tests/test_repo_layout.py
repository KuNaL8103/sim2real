"""The repository layout promised by Task 1 must actually exist."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DIRS = [
    ".github/workflows",
    "configs",
    "data",
    "docs",
    "ros2_ws/src/sim2real_msgs/msg",
    "ros2_ws/src/sim2real_msgs/srv",
    "ros2_ws/src/sim2real_sim",
    "ros2_ws/src/sim2real_perception",
    "ros2_ws/src/sim2real_vla",
    "ros2_ws/src/sim2real_control",
    "scripts",
    "src/sim2real",
    "tests",
]

REQUIRED_FILES = [
    "CLAUDE.md",
    "README.md",
    "pyproject.toml",
    "uv.lock",
    "Dockerfile",
    ".dockerignore",
    ".gitignore",
    ".github/workflows/ci.yml",
    "docs/ARCHITECTURE.md",
    "docs/DATASETS.md",
    "docs/SAFETY.md",
    "docs/DEPLOYMENT.md",
    "docs/MASTER_SPEC.md",
    "scripts/dev_env_check.sh",
    "scripts/setup_envs.sh",
    "scripts/run_core_tests.sh",
    "scripts/build_and_test_ros.sh",
    "src/sim2real/__init__.py",
    "src/sim2real/contracts.py",
    "src/sim2real/metadata.py",
    "ros2_ws/src/sim2real_msgs/CMakeLists.txt",
    "ros2_ws/src/sim2real_msgs/package.xml",
    "ros2_ws/src/sim2real_msgs/msg/TrajectoryStep.msg",
    "ros2_ws/src/sim2real_msgs/msg/RolloutEpisode.msg",
    "ros2_ws/src/sim2real_msgs/msg/FilterMetrics.msg",
    "ros2_ws/src/sim2real_msgs/msg/SafetyStatus.msg",
    "ros2_ws/src/sim2real_msgs/srv/ResetEnv.srv",
    "ros2_ws/src/sim2real_msgs/srv/GenerateRollouts.srv",
]

CONFIG_NAMES = [
    "sim",
    "randomization",
    "expert",
    "filtering",
    "mapping",
    "dataset",
    "training",
    "evaluation",
    "safety",
]


def test_required_directories_exist():
    missing = [name for name in REQUIRED_DIRS if not (ROOT / name).is_dir()]
    assert not missing, f"missing directories: {missing}"


def test_required_files_exist():
    missing = [name for name in REQUIRED_FILES if not (ROOT / name).is_file()]
    assert not missing, f"missing files: {missing}"


def test_data_directory_keeps_only_gitkeep_tracked():
    # Generated staging/curated/rejected data is expected after Task 5 and is
    # git-ignored. The invariant is that only data/.gitkeep is tracked.
    assert (ROOT / "data" / ".gitkeep").is_file()
    tracked = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "data"],
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode == 0:
        assert tracked.stdout.split() == ["data/.gitkeep"]
    else:
        assert "data/**" in (ROOT / ".gitignore").read_text(encoding="utf-8")


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_config_default_is_valid_yaml_with_schema_version(name: str):
    path = ROOT / "configs" / name / "default.yaml"
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict), f"{path} must contain a YAML mapping"
    assert parsed.get("schema_version") == 1, f"{path} must declare schema_version: 1"
