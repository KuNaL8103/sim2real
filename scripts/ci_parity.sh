#!/usr/bin/env bash
# Run exactly what .github/workflows/ci.yml runs, as far as this machine allows.
#
# Core job: uv sync --group sim, fetch the pinned assets, ruff, ROS-free pytest.
# ROS job:  colcon build + colcon test under SYSTEM Python (the ros:jazzy
#           container has no scipy/open3d), with those packages shadowed by
#           ImportError stubs so a missing dependency is a skip, not a collection
#           error. The temporary directory is removed on exit.
#
# CI sets SIM2REAL_SKIP_RENDER=1 on the hosted runner; this script mirrors that.
#
# --clean-clone mirrors only the CI core job in a fresh local clone (no
# an ML venv, no CLAUDE.md, no data/). Local fetched assets are copied into the
# clone when present so the asset step does not need the network.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CLEAN_CLONE=0
if [[ "${1:-}" == "--clean-clone" ]]; then
  CLEAN_CLONE=1
  shift
fi
if [[ $# -gt 0 ]]; then
  echo "ERROR: unknown argument(s): $*" >&2
  echo "Usage: scripts/ci_parity.sh [--clean-clone]" >&2
  exit 2
fi

if [[ "$CLEAN_CLONE" -eq 1 ]]; then
  clone_root="$(mktemp -d)"
  cleanup_clone() {
    rm -rf "$clone_root"
  }
  trap cleanup_clone EXIT
  git clone --no-local "$ROOT" "$clone_root/repo"
  cd "$clone_root/repo"
  echo "== clean clone: $(git rev-parse HEAD) =="
  if [[ -d "$ROOT/assets/mujoco/franka_emika_panda" ]]; then
    echo "== clean clone: copying local pinned assets (no network) =="
    mkdir -p assets/mujoco
    cp -a "$ROOT/assets/mujoco/franka_emika_panda" assets/mujoco/
  fi

  echo "== clean clone core: uv sync --locked --group sim =="
  uv sync --locked --group sim

  echo "== clean clone core: fetch pinned Menagerie assets =="
  ./scripts/fetch_assets.sh

  echo "== clean clone core: ruff check / format =="
  uv run ruff check .
  uv run ruff format --check .

  echo "== clean clone core: ROS-free pytest =="
  env -u PYTHONPATH -u AMENT_PREFIX_PATH SIM2REAL_SKIP_RENDER=1 uv run pytest -ra

  echo "clean-clone CI core checks passed (ROS job is not run in this mode)."
  exit 0
fi

echo "== core: uv sync --locked --group sim =="
uv sync --locked --group sim

echo "== core: fetch pinned Menagerie assets =="
./scripts/fetch_assets.sh

echo "== core: ruff check / format =="
uv run ruff check .
uv run ruff format --check .

echo "== core: ROS-free pytest =="
env -u PYTHONPATH -u AMENT_PREFIX_PATH SIM2REAL_SKIP_RENDER=1 uv run pytest -ra

echo "== ros: colcon build/test with scipy and open3d shadowed =="
if [ ! -f /opt/ros/jazzy/setup.bash ]; then
  echo "ERROR: /opt/ros/jazzy/setup.bash not found; cannot mirror the ROS job." >&2
  exit 2
fi

shadow="$(mktemp -d)"
cleanup() {
  rm -rf "$shadow"
}
trap cleanup EXIT

mkdir -p "$shadow/scipy" "$shadow/open3d"
printf 'raise ImportError("ci_parity: scipy shadowed for ROS-job parity")\n' \
  > "$shadow/scipy/__init__.py"
printf 'raise ImportError("ci_parity: open3d shadowed for ROS-job parity")\n' \
  > "$shadow/open3d/__init__.py"

set +u
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
set -u

cd "$ROOT/ros2_ws"
colcon build --symlink-install
PYTHONPATH="$shadow:${PYTHONPATH:-}" colcon test
colcon test-result --verbose

echo "CI parity checks passed."
