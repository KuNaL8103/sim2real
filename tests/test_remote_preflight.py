"""CPU-only tests for remote preflight, --check-only and --from-bundle."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "remote_train.sh"


def _run(cmd, cwd, env=None):
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _git(repo: Path, *args: str) -> str:
    result = _run(["git", *args], repo)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, manifest_commit: str | None = None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "scripts").mkdir()
    repo_script = repo / "scripts" / "remote_train.sh"
    repo_script.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    repo_script.chmod(0o755)
    (repo / ".gitignore").write_text("outputs/\ndata/**\n", encoding="utf-8")
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "fixture")
    head = _git(repo, "rev-parse", "HEAD")
    bundle = tmp_path / "repo.bundle"
    _git(repo, "bundle", "create", str(bundle), "HEAD")

    extract = tmp_path / "extract"
    payload = extract / "outputs" / "remote_bundle_payload"
    (payload / "configs" / "training").mkdir(parents=True)
    (payload / "requirements.txt").write_text("torch==1.2.3\n", encoding="utf-8")
    config = payload / "configs" / "training" / "smolvla_lora_overfit_remote.yaml"
    config.write_text("schema_version: 1\n", encoding="utf-8")
    (payload / "remote_train.sh").write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    files = {}
    for path in (payload / "requirements.txt", config, payload / "remote_train.sh"):
        relative = path.relative_to(extract).as_posix()
        files[relative] = {"sha256": _sha(path), "size": path.stat().st_size}
    manifest = {
        "schema_version": 1,
        "git_commit": manifest_commit or head,
        "git_dirty": False,
        "python": {
            "version": f"{sys.version_info.major}.{sys.version_info.minor}.0",
            "major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
            "executable": sys.executable,
            "lerobot_requires_python": ">=3.10",
        },
        "files": files,
    }
    (payload / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    return extract, payload, bundle, head


def _command(payload: Path, bundle: Path, workdir: Path, *extra: str):
    return [
        "bash",
        str(payload / "remote_train.sh"),
        "--check-only",
        "--bundle-dir",
        str(payload),
        "--config",
        str(payload / "configs/training/smolvla_lora_overfit_remote.yaml"),
        "--output-dir",
        str(workdir.parent / "out"),
        "--from-bundle",
        str(bundle),
        "--workdir",
        str(workdir),
        *extra,
    ]


def test_script_has_preflight_guards_in_order():
    text = SCRIPT.read_text(encoding="utf-8")
    for required in (
        "--check-only",
        "--from-bundle",
        "python.major_minor",
        "lerobot_requires_python",
        "driver_version",
        "nvidia-smi",
        "urllib.request",
        "torch.cuda.is_available",
        "torch.version.cuda",
        "repo.bundle",
        "preflight",
        "RUN_REPORT.json",
        "git status --porcelain",
    ):
        assert required in text
    assert text.index('if [[ "$CHECK_ONLY" -eq 1 ]]') < text.index(
        '"$VENV/bin/python" -m pip install'
    )
    assert text.index('if [[ "$CHECK_ONLY" -eq 1 ]]') < text.index("network check (HEAD requests")
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_check_only_uses_no_network_or_install_tools(tmp_path):
    extract, payload, bundle, _head = _fixture(tmp_path)
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    marker = tmp_path / "invoked.txt"
    for name in ("pip", "uv", "hf", "curl", "wget"):
        stub = stubs / name
        stub.write_text(
            f'#!/usr/bin/env bash\necho "{name}" >> "{marker}"\nexit 99\n',
            encoding="utf-8",
        )
        stub.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{stubs}:{env['PATH']}"
    result = _run(
        _command(payload, bundle, tmp_path / "clone"),
        cwd=extract,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CHECK ONLY" in result.stdout
    assert "code commit" in result.stdout and "checked out clean" in result.stdout
    assert "GPU:" in result.stdout or "nvidia-smi not found" in result.stdout
    assert not marker.exists(), marker.read_text() if marker.exists() else ""


def test_from_bundle_clones_and_rejects_wrong_commit(tmp_path):
    extract, payload, bundle, head = _fixture(tmp_path)
    result = _run(
        _command(payload, bundle, tmp_path / "clone"),
        cwd=extract,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "cloning repository from local bundle" in result.stdout
    assert head in result.stdout
    assert (tmp_path / "clone/.git").is_dir()

    extract2, payload2, bundle2, _ = _fixture(tmp_path / "second", manifest_commit="0" * 40)
    failed = _run(
        _command(payload2, bundle2, tmp_path / "clone2"),
        cwd=extract2,
    )
    assert failed.returncode != 0
    assert "!= manifest commit" in (failed.stdout + failed.stderr)


def test_python_version_mismatch_exits_nonzero(tmp_path):
    extract, payload, bundle, _head = _fixture(tmp_path)
    stub = tmp_path / "python-stub"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "${1:-}" == "-c" && "${2:-}" == *"sys.version_info.major"* ]]; then\n'
        '  echo "3.11"\n'
        "  exit 0\n"
        "fi\n"
        f'exec "{sys.executable}" "$@"\n',
        encoding="utf-8",
    )
    stub.chmod(0o755)
    result = _run(
        _command(payload, bundle, tmp_path / "clone", "--python", str(stub)),
        cwd=extract,
    )
    assert result.returncode != 0
    assert "does not match manifest" in (result.stdout + result.stderr)
