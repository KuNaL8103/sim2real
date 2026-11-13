#!/usr/bin/env python3
"""Build a provenance-safe remote-training bundle (no upload, no network).

The archive contains only the training dataset, the training configs, a pinned
requirements file generated from the local ``.venv-ml``, and a MANIFEST.json. It
never contains credentials, caches, logs, checkpoints, or model weights: the
remote machine downloads the pinned base model itself.

Layout inside the archive (extract at the repository root)::

    data/lerobot/train_v1/...                  full dataset
    outputs/remote_bundle_payload/MANIFEST.json
    outputs/remote_bundle_payload/requirements.txt
    outputs/remote_bundle_payload/configs/training/...

``outputs/`` and ``data/`` are git-ignored, so extracting the bundle leaves the
remote checkout clean. The bundle is written under ``outputs/remote_bundle/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sim2real.training.checkpoint import (  # noqa: E402
    REQUIRED_FIELDS,
    dataset_fingerprint,
    file_sha256,
    git_provenance,
)
from sim2real.training.config import load_training_config  # noqa: E402

PAYLOAD_PREFIX = "outputs/remote_bundle_payload"
DATASET_ARCNAME = "data/lerobot/train_v1"
TORCH_INDEX = "https://download.pytorch.org/whl/cu126"
PINNED_PACKAGES = (
    "torch",
    "torchvision",
    "torchcodec",
    "triton",
    "peft",
    "transformers",
    "accelerate",
    "lerobot",
    "num2words",
    "datasets",
    "numpy",
    "safetensors",
    "huggingface-hub",
    "tokenizers",
    "pillow",
    "av",
    "pyarrow",
    "draccus",
    "einops",
    "gymnasium",
    "opencv-python-headless",
    "pandas",
    "regex",
    "requests",
    "tqdm",
    "filelock",
    "fsspec",
)
FORBIDDEN_SUFFIXES = (
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".bin",
    ".h5",
    ".onnx",
    ".pkl",
    ".pickle",
    ".npz",
    ".npy",
    ".tar",
    ".tgz",
    ".zip",
)
FORBIDDEN_NAMES = {
    ".env",
    ".netrc",
    ".git-credentials",
    ".docker/config.json",
    "credentials",
    "credentials.json",
    "id_rsa",
    "id_ed25519",
    "opencode.json",
    "opencode.jsonc",
    "claude.md",
    "agents.md",
}
FORBIDDEN_DIRS = {
    ".git",
    ".venv",
    ".venv-ml",
    ".venv-ros",
    "__pycache__",
    ".cache",
    ".huggingface",
    "wandb",
    "mlruns",
    "checkpoints",
}


class BundleError(RuntimeError):
    """Raised when the bundle cannot be made provenance-safe."""


@dataclass(frozen=True)
class BundleResult:
    archive: Path
    manifest: dict[str, Any]
    payload_bytes: int
    archive_bytes: int
    archive_sha256: str
    files: int
    commit: str


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def forbidden_reason(relative: str) -> str | None:
    """Return a human-readable reason when a path must not be bundled."""
    path = PurePosixPath(relative)
    parts = list(path.parts)
    if parts[:1] == ["outputs"]:
        # The payload wrapper lives under the git-ignored outputs tree; inspect
        # the logical path after it instead of rejecting the wrapper itself.
        parts = parts[2:]
    lowered = [part.lower() for part in parts]
    for part in lowered:
        if part in FORBIDDEN_DIRS:
            return f"forbidden directory component {part!r}"
        if part.startswith(".env"):
            return f"credentials-like file {part!r}"
        if "credential" in part or "secret" in part:
            return f"credentials-like file {part!r}"
    name = lowered[-1] if lowered else ""
    if name in FORBIDDEN_NAMES:
        return f"forbidden file name {name!r}"
    if name.endswith(FORBIDDEN_SUFFIXES):
        return f"forbidden model/checkpoint suffix in {name!r}"
    return None


def _python_versions(python: Path) -> dict[str, str]:
    names = list(PINNED_PACKAGES)
    code = (
        "import json,sys\n"
        "from importlib.metadata import PackageNotFoundError, version\n"
        "names=json.loads(sys.argv[1])\n"
        "out={}\n"
        "for name in names:\n"
        "    try:\n"
        "        out[name]=version(name)\n"
        "    except PackageNotFoundError:\n"
        "        out[name]=None\n"
        "print(json.dumps(out,sort_keys=True))\n"
    )
    result = subprocess.run(
        [str(python), "-c", code, json.dumps(names)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise BundleError(f"could not read pinned versions from {python}: {result.stderr.strip()}")
    versions = json.loads(result.stdout)
    missing = [name for name, value in versions.items() if not value]
    if missing:
        raise BundleError(f"{python} is missing pinned training packages: {missing}")
    return versions


def render_requirements(versions: dict[str, str], python: str | None = None) -> str:
    missing = [name for name in PINNED_PACKAGES if not versions.get(name)]
    if missing:
        raise BundleError(f"missing versions for pinned packages: {missing}")
    lines = [
        "# Generated by scripts/export_remote_bundle.py; do not edit by hand.",
        f"# Source environment: {python or 'local .venv-ml'}",
        f"# CUDA wheel index used locally: {TORCH_INDEX}",
        "# The remote driver must be compatible with the CUDA 12.6 runtime.",
        f"--extra-index-url {TORCH_INDEX}",
    ]
    lines.extend(f"{name}=={versions[name]}" for name in PINNED_PACKAGES)
    return "\n".join(lines) + "\n"


def _dataset_info(dataset_root: Path) -> dict:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise BundleError(f"{info_path}: dataset metadata not found")
    return json.loads(info_path.read_text(encoding="utf-8"))


def expected_checkpoint_metadata(
    *, config, dataset_root: Path, commit: str, config_hash: str
) -> dict:
    fingerprint = dataset_fingerprint(dataset_root)
    info = _dataset_info(dataset_root)
    features = info.get("features", {})
    state_shape = features.get("observation.state", {}).get("shape", [])
    action_shape = features.get("action", {}).get("shape", [])
    state_dim = int(state_shape[0]) if state_shape else 0
    action_dim = int(action_shape[0]) if action_shape else 0
    return {
        "required_fields": list(REQUIRED_FIELDS),
        "base_model_id": config.model.checkpoint,
        "base_model_revision": config.model.revision,
        "git_commit": commit,
        "git_dirty": False,
        "config_hash": config_hash,
        "dataset_root": config.data.dataset_root,
        "dataset_version": info.get("codebase_version"),
        "dataset_hash": fingerprint["dataset_hash"],
        "stats_file_hash": fingerprint["stats_file_hash"],
        "camera_rename_map": dict(config.data.rename_map),
        "action_representation": config.data.action_representation,
        "chunk_size": config.training.chunk_size,
        "n_action_steps": config.training.n_action_steps,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_state_dim": config.model.max_state_dim,
        "max_action_dim": config.model.max_action_dim,
        "fps": config.data.fps,
        "horizon_seconds": config.horizon_seconds,
    }


def _copy_dataset(source: Path, staging: Path) -> list[str]:
    destination = staging / DATASET_ARCNAME
    if not source.is_dir():
        raise BundleError(f"{source}: dataset directory not found")
    shutil.copytree(source, destination)
    relative: list[str] = []
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            relative.append(path.relative_to(staging).as_posix())
    if not relative:
        raise BundleError(f"{source}: dataset directory is empty")
    return relative


def _copy_config(source: Path, staging: Path) -> str:
    if not source.is_file():
        raise BundleError(f"{source}: training config not found")
    relative = PurePosixPath(PAYLOAD_PREFIX) / "configs/training" / source.name
    destination = staging / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return relative.as_posix()


def _manifest_files(staging: Path) -> dict[str, dict]:
    entries: dict[str, dict] = {}
    for path in sorted(staging.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(staging).as_posix()
        if relative.endswith("MANIFEST.json"):
            continue
        reason = forbidden_reason(relative)
        if reason is not None:
            raise BundleError(f"refusing to bundle {relative}: {reason}")
        entries[relative] = {
            "sha256": file_sha256(path),
            "size": path.stat().st_size,
        }
    return entries


def export_bundle(
    *,
    dataset_root: Path,
    config_paths: list[Path],
    output_dir: Path,
    python_executable: Path | None = None,
    versions: dict[str, str] | None = None,
    commit: str | None = None,
    timestamp: str | None = None,
    allow_dirty: bool = False,
) -> BundleResult:
    provenance = git_provenance(ROOT)
    resolved_commit = commit or provenance["git_commit"]
    if not resolved_commit:
        raise BundleError("could not resolve the git commit")
    dirty = bool(provenance["git_dirty"])
    if dirty and not allow_dirty:
        raise BundleError(
            "git working tree is dirty; commit the code first so the remote clone "
            "and checkpoint metadata can record git_dirty=false"
        )
    if versions is None:
        python = python_executable or (ROOT / ".venv-ml/bin/python")
        if not Path(python).is_file():
            raise BundleError(f"{python}: ML environment not found")
        versions = _python_versions(Path(python))
    requirements = render_requirements(versions, str(python_executable or "local .venv-ml"))

    configs = [load_training_config(path) for path in config_paths]
    dataset_info = _dataset_info(Path(dataset_root))

    with tempfile.TemporaryDirectory(prefix="remote_bundle_") as temporary:
        staging = Path(temporary)
        payload = staging / PAYLOAD_PREFIX
        payload.mkdir(parents=True)
        (payload / "requirements.txt").write_text(requirements, encoding="utf-8")
        _copy_dataset(Path(dataset_root), staging)
        config_entries: dict[str, dict] = {}
        for source, config in zip(config_paths, configs, strict=True):
            relative = _copy_config(Path(source), staging)
            config_entries[relative] = {
                "config_hash": config.config_hash,
                "sha256": file_sha256(staging / relative),
                "expected_checkpoint_metadata": expected_checkpoint_metadata(
                    config=config,
                    dataset_root=Path(dataset_root),
                    commit=resolved_commit,
                    config_hash=config.config_hash,
                ),
            }
        entries = _manifest_files(staging)
        total = sum(entry["size"] for entry in entries.values())
        print(
            f"bundle payload: {len(entries)} files, {total} bytes "
            f"({total / 1024**2:.2f} MiB) before writing the archive"
        )
        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "git_commit": resolved_commit,
            "git_dirty": dirty,
            "dataset": {
                "arcname": DATASET_ARCNAME,
                "root": configs[0].data.dataset_root,
                "version": dataset_info.get("codebase_version"),
                "episodes": dataset_info.get("total_episodes"),
                "frames": dataset_info.get("total_frames"),
                "fingerprint": dataset_fingerprint(Path(dataset_root)),
            },
            "configs": config_entries,
            "requirements": {
                "path": f"{PAYLOAD_PREFIX}/requirements.txt",
                "sha256": file_sha256(payload / "requirements.txt"),
                "torch_index": TORCH_INDEX,
                "packages": versions,
            },
            "model_downloads": {
                "lerobot/smolvla_base": "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6",
                "HuggingFaceTB/SmolVLM2-500M-Video-Instruct": (
                    "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
                ),
            },
            "files": entries,
        }
        (payload / "MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = timestamp or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        archive = output_dir / f"sim2real_remote_{resolved_commit[:7]}_{stamp}.tar.gz"
        with tarfile.open(archive, "w:gz") as handle:
            for relative in sorted(entries) + [f"{PAYLOAD_PREFIX}/MANIFEST.json"]:
                handle.add(staging / relative, arcname=relative)
        digest = file_sha256(archive)
        print(f"archive: {archive}")
        archive_bytes = archive.stat().st_size
        print(f"archive size: {archive_bytes} bytes ({archive_bytes / 1024**2:.2f} MiB)")
        print(f"archive sha256: {digest}")
        return BundleResult(
            archive=archive,
            manifest=manifest,
            payload_bytes=total,
            archive_bytes=archive.stat().st_size,
            archive_sha256=digest,
            files=len(entries),
            commit=resolved_commit,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/lerobot/train_v1")
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="training config to include (repeat for more than one)",
    )
    parser.add_argument("--output-dir", default="outputs/remote_bundle")
    parser.add_argument("--python", default=".venv-ml/bin/python")
    parser.add_argument("--commit", default=None)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = export_bundle(
            dataset_root=ROOT / args.dataset,
            config_paths=[ROOT / path for path in args.config],
            output_dir=ROOT / args.output_dir,
            python_executable=ROOT / args.python,
            commit=args.commit,
            timestamp=args.timestamp,
            allow_dirty=args.allow_dirty,
        )
    except BundleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"archive": str(result.archive), "sha256": result.archive_sha256}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
