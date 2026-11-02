"""Script and environment smoke checks that do not require .venv-ml."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        check=False,
    )


def test_script_help_paths():
    for script in (
        "scripts/build_lerobot_dataset.py",
        "scripts/compute_stats.py",
        "scripts/preview_mixture.py",
        "scripts/convert_bridge_subset.py",
    ):
        result = _run(script, "--help")
        assert result.returncode == 0, result.stderr


def test_preview_mixture_runs_on_example():
    result = _run(
        "scripts/preview_mixture.py",
        "configs/dataset/mixtures/example.yaml",
        "--draws",
        "1000",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["draws"] == 1000
    assert set(payload["proportions"]) == {"panda_sim", "widowx_bridge"}


def test_bridge_fixture_dry_run_and_download_refusal():
    result = _run(
        "scripts/convert_bridge_subset.py",
        "--source",
        "fixture",
        "--max-episodes",
        "2",
        "--out",
        "data/lerobot/bridge_fixture",
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["converted"] <= 2

    refused = _run(
        "scripts/convert_bridge_subset.py",
        "--source",
        "tfds",
        "--max-episodes",
        "1",
        "--out",
        "data/lerobot/bridge_fixture",
    )
    assert refused.returncode == 2
    assert "387.49" in refused.stderr


def test_setup_script_documents_opt_in_ml_environment_and_ci_skips_it():
    text = (ROOT / "scripts" / "setup_envs.sh").read_text(encoding="utf-8")
    assert "uv venv .venv-ml --python 3.12" in text
    assert "lerobot[dataset]==0.6.1" in text
    assert "download.pytorch.org/whl/cpu" in text
    assert "--with-ml" in text
    for path in (ROOT / ".github" / "workflows" / "ci.yml", ROOT / "scripts" / "ci_parity.sh"):
        assert ".venv-ml" not in path.read_text(encoding="utf-8")
        assert "lerobot" not in path.read_text(encoding="utf-8").lower()
