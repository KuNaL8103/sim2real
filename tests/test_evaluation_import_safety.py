"""The evaluation package and CLIs must import without torch or LeRobot."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    return environment


def test_evaluation_package_imports_without_ml_modules():
    code = (
        "import sys; import sim2real.evaluation; "
        "print('torch' in sys.modules, 'peft' in sys.modules, "
        "'lerobot' in sys.modules, 'rclpy' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=_env(),
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False False False False"


def test_evaluation_cli_help_works_without_torch():
    for script in ("run_eval.py", "fit_failure_distribution.py", "flywheel_status.py"):
        result = subprocess.run(
            [sys.executable, f"scripts/{script}", "--help"],
            cwd=ROOT,
            env=_env(),
            check=True,
            capture_output=True,
            text=True,
        )
        assert "usage:" in result.stdout.lower()


def test_policy_server_help_works_without_torch():
    result = subprocess.run(
        [sys.executable, "scripts/policy_server.py", "--help"],
        cwd=ROOT,
        env=_env(),
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--fake-model" in result.stdout
