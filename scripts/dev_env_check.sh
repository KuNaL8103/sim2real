#!/usr/bin/env bash
# Read-only environment preflight. Installs nothing and never uses sudo.
set -uo pipefail

section() {
  printf '\n== %s ==\n' "$1"
}

section "Kernel / WSL"
uname -a
grep -i microsoft /proc/version || true
echo "WSL_DISTRO_NAME=${WSL_DISTRO_NAME:-<unset>}"

section "Distribution"
lsb_release -a 2>/dev/null || cat /etc/os-release
pwd

section "Toolchain"
python3 --version || true
git --version || true
echo "git user.name: $(git config user.name || echo '<unset>')"
echo "git user.email: $(git config user.email || echo '<unset>')"
uv --version || echo "uv: MISSING"
colcon version-check >/dev/null 2>&1 && echo "colcon: present" || echo "colcon: MISSING"

section "ROS 2"
echo "ROS_DISTRO=${ROS_DISTRO:-<unset>}"
ls /opt/ros 2>/dev/null || echo "/opt/ros: MISSING"
if [ -f /opt/ros/jazzy/setup.bash ]; then
  # shellcheck disable=SC1091
  set +u
  source /opt/ros/jazzy/setup.bash
  set -u
  python3 -c "import rclpy; print('rclpy:', rclpy.__file__)" || echo "rclpy: MISSING"
fi

section "Containers / GitHub CLI"
if docker --version >/dev/null 2>&1; then
  docker --version
else
  echo "docker: unavailable in this WSL distro"
fi
gh --version 2>/dev/null | head -n 1 || echo "gh: MISSING"

section "Resources"
echo "nproc: $(nproc)"
free -h
df -h "$HOME"

section "GPU (may fail; no CUDA installation is attempted)"
nvidia-smi || echo "nvidia-smi: FAILED (report honestly; do not install CUDA)"
