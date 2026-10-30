#!/usr/bin/env bash
# Create the two environments used in Task 1.
#   .venv      core/CI environment, no ROS
#   .venv-ros  ROS adapter/test environment built on the system Python 3.12
# .venv-ml is documented only (torch/LeRobot/PEFT) and is NOT created here.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not installed. Install it first (see docs/DEPLOYMENT.md)." >&2
  exit 1
fi
if [ ! -f /opt/ros/jazzy/setup.bash ]; then
  echo "ERROR: /opt/ros/jazzy/setup.bash not found. Install ROS 2 Jazzy first." >&2
  exit 1
fi

echo "==> Core environment (.venv)"
uv sync

echo "==> ROS environment (.venv-ros)"
# ROS setup scripts reference optional unset variables; relax nounset while
# sourcing them.
set +u
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
set -u

if [ ! -x .venv-ros/bin/python ]; then
  # system-site-packages is mandatory so the venv can see rclpy from /opt/ros.
  if ! uv venv --python /usr/bin/python3.12 --system-site-packages .venv-ros; then
    echo "uv venv failed; falling back to python3 -m venv --system-site-packages" >&2
    python3 -m venv --system-site-packages .venv-ros
  fi
fi

# Only small pure-Python dependencies. Do NOT install numpy/torch here: the
# system numpy 1.26.4 from ROS Jazzy must remain the visible version.
uv pip install --python .venv-ros/bin/python "pytest>=8,<9" "pyyaml>=6,<7" "ruff>=0.9"
# Editable core package so .venv-ros can run the same test suite. No deps.
uv pip install --python .venv-ros/bin/python --no-deps -e .

.venv-ros/bin/python -c "import rclpy; print('rclpy OK:', rclpy.__file__)"

cat <<'EOF'

==> Environment summary
.venv       core/CI, no ROS, managed by uv.
.venv-ros   system Python 3.12 + ROS Jazzy via --system-site-packages.
.venv-ml    NOT created. Planned for torch/LeRobot/PEFT (Task 8).
EOF
