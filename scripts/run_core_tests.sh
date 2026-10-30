#!/usr/bin/env bash
# Canonical hardware-free verification for the core library.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

uv sync
# A sourced ROS environment exports PYTHONPATH, which would leak ROS pytest
# plugins into the core venv. Core tests must stay ROS-free.
unset PYTHONPATH
uv run ruff check .
uv run ruff format --check .
uv run pytest -ra "$@"
