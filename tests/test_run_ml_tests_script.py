"""Text checks for the .venv-ml runner and the opt-in CUDA setup lines."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_ml_tests.sh"
SETUP = ROOT / "scripts" / "setup_envs.sh"


def test_ml_runner_exists_and_is_executable():
    assert RUNNER.is_file()
    assert os.access(RUNNER, os.X_OK)
    text = RUNNER.read_text(encoding="utf-8")
    assert ".venv-ml/bin/python" in text
    assert "PYTHONPATH" in text
    assert "ml and not gpu" in text
    assert "--gpu" in text
    assert "setup_envs.sh --with-ml" in text
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)


def test_ml_runner_names_the_ml_test_files_by_path():
    text = RUNNER.read_text(encoding="utf-8")
    assert "tests/test_dataset_writer_ml.py" in text
    assert "tests/test_training_ml.py" in text
    assert "tests/test_training_gpu.py" in text


def test_setup_envs_has_the_opt_in_cuda_lines():
    text = SETUP.read_text(encoding="utf-8")
    assert "ML_TORCH_BACKEND" in text
    assert "cu126" in text
    assert "https://download.pytorch.org/whl/cu126" in text
    assert "torch==2.11.0+cu126" in text
    assert "peft>=0.18,<1" in text
    assert "accelerate>=1.14,<2" in text
    assert "transformers>=5.4,<5.6" in text
    assert "num2words>=0.5,<1" in text


def test_ci_never_installs_venv_ml():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert ".venv-ml" not in workflow
    assert "--with-ml" not in workflow
