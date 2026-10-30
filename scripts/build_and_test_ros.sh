#!/usr/bin/env bash
# Build and test the ROS 2 workspace. Requires a sourced installation of
# ROS 2 Jazzy at /opt/ros/jazzy; installs nothing.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -f /opt/ros/jazzy/setup.bash ]; then
  echo "ERROR: ROS 2 Jazzy is not installed at /opt/ros/jazzy." >&2
  echo "ROS build/test is PENDING; do not mark it as passing." >&2
  exit 2
fi

# shellcheck disable=SC1091
set +u
source /opt/ros/jazzy/setup.bash
set -u

cd "$ROOT/ros2_ws"
colcon build --symlink-install
colcon test
colcon test-result --verbose

# shellcheck disable=SC1091
set +u
source install/setup.bash
set -u
ros2 interface list | grep sim2real_msgs

cd "$ROOT"
if [ -x .venv-ros/bin/python ]; then
  .venv-ros/bin/python -m pytest -ra
else
  echo ".venv-ros is missing; run scripts/setup_envs.sh first." >&2
  exit 2
fi
