#!/usr/bin/env bash
# Run the optional ML tests in .venv-ml.
#
# The bare `-m ml` selector collects unrelated modules that need scipy/open3d,
# which .venv-ml deliberately does not have. This runner selects the ML test
# files by path and puts the editable source tree on PYTHONPATH.
#
# Usage:
#   scripts/run_ml_tests.sh              # markers: ml and not gpu
#   scripts/run_ml_tests.sh --gpu        # markers: ml and gpu (after approval)
#   scripts/run_ml_tests.sh --all        # every marker in the ML files
#   scripts/run_ml_tests.sh -k test_name # extra pytest arguments are forwarded
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv-ml/bin/python"

if [[ ! -x "${PY}" ]]; then
  echo "error: ${ROOT}/.venv-ml is missing." >&2
  echo "Create it with: ./scripts/setup_envs.sh --with-ml" >&2
  exit 2
fi

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

MARKER="ml and not gpu"
if [[ "${1:-}" == "--gpu" ]]; then
  MARKER="ml and gpu"
  shift
elif [[ "${1:-}" == "--all" ]]; then
  MARKER="ml or gpu"
  shift
fi

TESTS=(
  "${ROOT}/tests/test_dataset_writer_ml.py"
  "${ROOT}/tests/test_training_ml.py"
  "${ROOT}/tests/test_training_gpu.py"
)

echo "==> .venv-ml pytest (-m '${MARKER}')"
exec "${PY}" -m pytest -ra -m "${MARKER}" "${TESTS[@]}" "$@"
