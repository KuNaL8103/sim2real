"""Importing the training package (and its CLIs) must not require torch."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return env


def test_training_package_imports_without_optional_ml_modules():
    code = (
        "import sys; import sim2real.training; "
        "print('torch' in sys.modules, 'peft' in sys.modules, "
        "'lerobot' in sys.modules, 'wandb' in sys.modules, 'mlflow' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=_env(),
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False False False False False"


def test_train_cli_help_works_without_torch():
    result = subprocess.run(
        [sys.executable, "scripts/train_vla.py", "--help"],
        cwd=ROOT,
        env=_env(),
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--fake" in result.stdout
    assert "--smoke" in result.stdout


def test_list_lora_targets_fake_tree_works_without_torch():
    result = subprocess.run(
        [sys.executable, "scripts/list_lora_targets.py", "--fake-tree"],
        cwd=ROOT,
        env=_env(),
        check=True,
        capture_output=True,
        text=True,
    )
    assert "resolved" in result.stdout
    assert "model.state_proj" in result.stdout
    assert "resolver-negative-check" in result.stdout


def test_validate_checkpoint_help_works_without_torch():
    result = subprocess.run(
        [sys.executable, "scripts/validate_checkpoint.py", "--help"],
        cwd=ROOT,
        env=_env(),
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--checkpoint" in result.stdout
