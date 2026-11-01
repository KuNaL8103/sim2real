#!/usr/bin/env bash
# Canonical hardware-free verification for the core library.
#
# The ROS-free run is the official core baseline. A module-level ROS name once
# broke CI collection while local runs passed because the system rclpy was
# visible. Unsetting PYTHONPATH and AMENT_PREFIX_PATH reproduces CI exactly.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

uv sync --group sim
env -u PYTHONPATH -u AMENT_PREFIX_PATH uv run ruff check .
env -u PYTHONPATH -u AMENT_PREFIX_PATH uv run ruff format --check .
env -u PYTHONPATH -u AMENT_PREFIX_PATH uv run pytest -ra "$@"
