"""File and provenance helpers for the orchestrator."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(path: str | Path) -> str:
    """Hash relative file names and contents; directories are ordered."""
    root = Path(path)
    if root.is_file():
        return sha256_file(root)
    if not root.exists():
        return ""
    digest = hashlib.sha256()
    for child in sorted(root.rglob("*")):
        if not child.is_file():
            continue
        relative = child.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(child).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def tree_size_bytes(path: str | Path) -> int:
    root = Path(path)
    if root.is_file():
        return root.stat().st_size
    if not root.exists():
        return 0
    return sum(child.stat().st_size for child in root.rglob("*") if child.is_file())


def atomic_write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def quarantine(path: str | Path, quarantine_root: str | Path, label: str) -> Path:
    """Move a corrupt/incomplete artifact out of the way; never delete it."""
    source = Path(path)
    root = Path(quarantine_root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    target = root / f"{label}-{stamp}"
    if source.exists():
        shutil.move(str(source), str(target))
    else:  # pragma: no cover - caller normally checks existence first
        target.mkdir(parents=True, exist_ok=True)
        (target / "MISSING.txt").write_text(
            f"recorded artifact was already missing: {source}\n", encoding="utf-8"
        )
    return target


def git_state(repo_root: str | Path) -> tuple[str, bool]:
    """Return ``(commit, dirty)`` without raising in a non-git checkout."""
    root = str(repo_root)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True
    return commit, bool(status.strip())


__all__ = [
    "atomic_write_json",
    "git_state",
    "hash_tree",
    "quarantine",
    "read_json",
    "sha256_bytes",
    "sha256_file",
    "tree_size_bytes",
]
