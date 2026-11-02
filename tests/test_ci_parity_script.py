"""The CI-parity script must exist and mirror the CI commands."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ci_parity.sh"


def test_ci_parity_script_exists_and_mentions_ci_steps():
    assert SCRIPT.is_file()
    text = SCRIPT.read_text(encoding="utf-8")
    assert "ruff check ." in text
    assert "ruff format --check ." in text
    assert "pytest -ra" in text
    assert "colcon build" in text
    assert "colcon test" in text
    assert "colcon test-result --verbose" in text
    assert "scipy" in text and "open3d" in text
    assert "mktemp -d" in text


def test_ci_parity_script_syncs_the_ci_group():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "uv sync --locked --group sim" in text
