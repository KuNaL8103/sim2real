"""GPU-marked SmolVLA smoke test (only when explicitly approved and enabled).

This test is excluded by ``scripts/run_ml_tests.sh`` (marker ``ml and not gpu``).
Set ``SIM2REAL_RUN_GPU_SMOKE=1`` after the resource report and explicit approval.
It downloads/loads the pinned model, runs one optimizer step, and asserts that
the loss is finite and a checkpoint was written.  It does not claim anything
about policy quality.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="optional .venv-ml torch environment is absent")
pytest.importorskip("peft", reason="optional .venv-ml PEFT package is absent")
pytest.importorskip("lerobot", reason="optional .venv-ml LeRobot environment is absent")

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.ml
@pytest.mark.gpu
def test_smolvla_gpu_smoke(tmp_path):
    if os.environ.get("SIM2REAL_RUN_GPU_SMOKE") != "1":
        pytest.skip("GPU smoke not approved in this session (set SIM2REAL_RUN_GPU_SMOKE=1)")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available in this environment")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_vla.py"),
            "--config",
            str(ROOT / "configs" / "training" / "smolvla_lora.yaml"),
            "--smoke",
            "--max-steps",
            "1",
            "--output-dir",
            str(tmp_path / "smoke"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    summary = json.loads(result.stdout)
    assert summary["run_kind"] == "smolvla"
    assert summary["final_step"] == 1
    assert summary["last_loss"] is not None
    assert math.isfinite(summary["last_loss"])
    checkpoint = Path(summary["checkpoint_dir"])
    assert (checkpoint / "checkpoint_metadata.json").is_file()
