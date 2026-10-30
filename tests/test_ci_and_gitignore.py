"""CI workflow and .gitignore must encode the CPU-only, least-privilege policy."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CI_PATH = ROOT / ".github" / "workflows" / "ci.yml"
GITIGNORE_PATH = ROOT / ".gitignore"


def load_ci() -> dict:
    parsed = yaml.safe_load(CI_PATH.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict), "ci.yml must be a YAML mapping"
    return parsed


def test_ci_is_valid_yaml_with_core_and_ros_jobs():
    ci = load_ci()
    assert set(ci["jobs"]) == {"core", "ros"}


def test_ci_runs_ruff_and_pytest():
    text = CI_PATH.read_text(encoding="utf-8")
    assert "ruff check ." in text
    assert "ruff format --check ." in text
    assert "pytest -ra" in text


def test_ci_is_least_privilege_and_has_concurrency():
    ci = load_ci()
    assert ci["permissions"] == {"contents": "read"}
    assert "concurrency" in ci
    assert ci["concurrency"]["group"]
    assert ci["concurrency"]["cancel-in-progress"] is True


def test_ci_has_no_gpu_runner_or_gpu_dependencies():
    ci = load_ci()
    text = CI_PATH.read_text(encoding="utf-8").lower()
    assert "nvidia" not in text
    for job_name, job in ci["jobs"].items():
        runs_on = job["runs-on"]
        assert isinstance(runs_on, str), f"{job_name}: runs-on must be a single label"
        assert "gpu" not in runs_on.lower(), f"{job_name}: GPU runner is not allowed"


def test_ci_uses_only_github_token_if_any_secret():
    text = CI_PATH.read_text(encoding="utf-8")
    used = set(re.findall(r"secrets\.([A-Za-z0-9_]+)", text))
    assert used <= {"GITHUB_TOKEN"}, f"unexpected secrets: {used}"


def test_ci_actions_are_pinned_to_a_version():
    text = CI_PATH.read_text(encoding="utf-8")
    for reference in re.findall(r"uses:\s*([^\s]+)", text):
        assert "@" in reference, f"unpinned action: {reference}"
        version = reference.split("@", 1)[1]
        assert re.fullmatch(r"v\d+(\.\d+){0,2}", version), f"unexpected ref: {reference}"


def test_gitignore_covers_required_patterns():
    text = GITIGNORE_PATH.read_text(encoding="utf-8")
    required = [
        "data/",
        ".venv",
        "build/",
        "install/",
        "log/",
        "__pycache__/",
        "*.pt",
        "*.pth",
        "*.safetensors",
        "*.ckpt",
        "*.npz",
        "wandb/",
        "mlruns/",
        ".env",
        "*.bag",
        "*.mcap",
        ".hydra/",
        "outputs/",
    ]
    missing = [pattern for pattern in required if pattern not in text]
    assert not missing, f"missing .gitignore patterns: {missing}"
