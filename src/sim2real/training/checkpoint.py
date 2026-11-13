"""Checkpoint export with complete, verifiable provenance metadata.

A checkpoint is a directory:

```
<run>/checkpoint-00000050/
  adapter/                      adapter weights (fake NPZ or LeRobot/PEFT files)
  checkpoint_metadata.json      every provenance field below
```

The metadata is not decorative: ``load_checkpoint`` refuses a checkpoint whose
stats hash, action representation, or camera rename map differs from the caller's
expectations.  That makes a mismatched evaluation fail loudly instead of printing
plausible but meaningless L1 numbers.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

METADATA_NAME = "checkpoint_metadata.json"
REQUIRED_FIELDS = (
    "schema_version",
    "created_at",
    "base_model_id",
    "base_model_revision",
    "library_versions",
    "gpu_name",
    "git_commit",
    "git_dirty",
    "config_hash",
    "config",
    "dataset_root",
    "dataset_version",
    "dataset_hash",
    "split",
    "mixture_manifest",
    "stats_file_hash",
    "seed",
    "step",
    "lora",
    "camera_rename_map",
    "action_representation",
    "chunk_size",
    "n_action_steps",
    "state_dim",
    "action_dim",
    "max_state_dim",
    "max_action_dim",
    "fps",
    "horizon_seconds",
)


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is incomplete or does not match its caller."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def config_hash(config: dict[str, Any]) -> str:
    """Deterministic SHA-256 of a JSON-serialisable config."""
    return hashlib.sha256(canonical_json(config).encode()).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_tree_sha256(path: str | Path) -> str:
    """SHA-256 over every adapter file, addressing relative paths and bytes.

    Adapters may contain one safetensors file or several PEFT sidecars, so the
    hash covers the whole directory in sorted order rather than a fixed name.
    """
    target = Path(path)
    if not target.is_dir():
        raise CheckpointError(f"{target}: adapter directory not found")
    digest = hashlib.sha256()
    files = sorted(item for item in target.rglob("*") if item.is_file())
    if not files:
        raise CheckpointError(f"{target}: adapter directory is empty")
    for item in files:
        relative = item.relative_to(target).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def dataset_fingerprint(root: str | Path) -> dict[str, Any]:
    """Hash the dataset metadata files that determine training semantics."""
    dataset_root = Path(root)
    meta = dataset_root / "meta"
    files = {
        "info": meta / "info.json",
        "stats": meta / "stats.json",
        "split": meta / "sim2real_split.json",
    }
    hashes: dict[str, str | None] = {}
    info: dict[str, Any] = {}
    for name, path in files.items():
        hashes[name] = file_sha256(path) if path.is_file() else None
    if files["info"].is_file():
        try:
            info = json.loads(files["info"].read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            info = {}
    combined = hashlib.sha256(canonical_json(hashes).encode()).hexdigest()
    return {
        "dataset_root": str(dataset_root),
        "dataset_version": info.get("codebase_version"),
        "dataset_hash": combined,
        "stats_file_hash": hashes["stats"],
        "file_hashes": hashes,
    }


def git_provenance(root: str | Path = ".") -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"git_commit": commit, "git_dirty": bool(status) if status is not None else None}


def _distribution_version(name: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(name)
        except PackageNotFoundError:
            return None
    except Exception:  # pragma: no cover - defensive only
        return None


def library_versions() -> dict[str, Any]:
    """Library versions without importing torch unless it is installed."""
    torch_version = _distribution_version("torch")
    cuda_version = None
    cuda_available = None
    if torch_version is not None:
        try:
            import torch

            cuda_version = torch.version.cuda
            cuda_available = bool(torch.cuda.is_available())
        except Exception:  # pragma: no cover - broken torch is reported as missing CUDA
            cuda_version = None
            cuda_available = False
    return {
        "python": sys.version.split()[0],
        "torch": torch_version,
        "torch_cuda": cuda_version,
        "cuda_available": cuda_available,
        "peft": _distribution_version("peft"),
        "lerobot": _distribution_version("lerobot"),
        "accelerate": _distribution_version("accelerate"),
        "transformers": _distribution_version("transformers"),
        "numpy": _distribution_version("numpy"),
    }


def gpu_name() -> str | None:
    import os

    override = os.environ.get("SIM2REAL_GPU_NAME")
    if override:
        return override
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:
        return None
    return None


def build_metadata(
    *,
    config: Any,
    dataset: dict[str, Any],
    split: dict[str, Any],
    step: int,
    state_dim: int,
    action_dim: int,
    mixture_manifest: str | None = None,
    seed: int | None = None,
    repo_root: str | Path = ".",
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build every required provenance field from the config and dataset."""
    provenance = git_provenance(repo_root)
    config_dict = config.to_dict()
    metadata = {
        "schema_version": 1,
        "created_at": created_at or datetime.now(UTC).isoformat(),
        "base_model_id": config.model.checkpoint,
        "base_model_revision": config.model.revision,
        "library_versions": library_versions(),
        "gpu_name": gpu_name(),
        "git_commit": provenance["git_commit"],
        "git_dirty": provenance["git_dirty"],
        "config_hash": config_hash(config_dict),
        "config": config_dict,
        "dataset_root": dataset.get("dataset_root"),
        "dataset_version": dataset.get("dataset_version"),
        "dataset_hash": dataset.get("dataset_hash"),
        "split": split,
        "mixture_manifest": mixture_manifest,
        "stats_file_hash": dataset.get("stats_file_hash"),
        "seed": config.training.seed if seed is None else int(seed),
        "step": int(step),
        "lora": {
            "enabled": config.lora.enabled,
            "r": config.lora.r,
            "alpha": config.lora.alpha,
            "dropout": config.lora.dropout,
            "bias": config.lora.bias,
            "method_type": config.lora.method_type,
            "target_modules": list(config.lora.target_modules),
        },
        "camera_rename_map": dict(config.data.rename_map),
        "action_representation": config.data.action_representation,
        "chunk_size": config.training.chunk_size,
        "n_action_steps": config.training.n_action_steps,
        "state_dim": int(state_dim),
        "action_dim": int(action_dim),
        "max_state_dim": config.model.max_state_dim,
        "max_action_dim": config.model.max_action_dim,
        "fps": config.data.fps,
        "horizon_seconds": config.horizon_seconds,
    }
    return metadata


def save_checkpoint(
    path: str | Path,
    adapter: Any,
    metadata: dict[str, Any],
) -> Path:
    """Write adapter weights and validated metadata atomically enough for a run."""
    checkpoint_dir = Path(path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    missing = [field for field in REQUIRED_FIELDS if field not in metadata]
    if missing:
        raise CheckpointError(f"checkpoint metadata is missing required fields: {missing}")
    adapter.save_adapter(checkpoint_dir / "adapter")
    metadata = dict(metadata)
    metadata["adapter_hash"] = adapter_tree_sha256(checkpoint_dir / "adapter")
    (checkpoint_dir / METADATA_NAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return checkpoint_dir


def load_checkpoint(
    path: str | Path,
    *,
    expected_stats_hash: str | None = None,
    expected_action_representation: str | None = None,
    expected_rename_map: dict[str, str] | None = None,
    expected_config_hash: str | None = None,
) -> dict[str, Any]:
    """Read metadata and refuse a semantic mismatch."""
    checkpoint_dir = Path(path)
    metadata_path = checkpoint_dir / METADATA_NAME
    if not metadata_path.is_file():
        raise CheckpointError(f"{metadata_path}: checkpoint metadata not found")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    missing = [field for field in REQUIRED_FIELDS if field not in metadata]
    if missing:
        raise CheckpointError(f"{metadata_path}: missing required fields {missing}")
    if expected_stats_hash is not None and metadata.get("stats_file_hash") != expected_stats_hash:
        raise CheckpointError(
            "checkpoint stats file hash does not match the expected dataset stats "
            f"({metadata.get('stats_file_hash')} != {expected_stats_hash})"
        )
    if (
        expected_action_representation is not None
        and metadata.get("action_representation") != expected_action_representation
    ):
        raise CheckpointError(
            "checkpoint action representation does not match: "
            f"{metadata.get('action_representation')} != {expected_action_representation}"
        )
    if expected_rename_map is not None and metadata.get("camera_rename_map") != expected_rename_map:
        raise CheckpointError(
            "checkpoint camera rename map does not match: "
            f"{metadata.get('camera_rename_map')} != {expected_rename_map}"
        )
    if expected_config_hash is not None and metadata.get("config_hash") != expected_config_hash:
        raise CheckpointError(
            "checkpoint config hash does not match: "
            f"{metadata.get('config_hash')} != {expected_config_hash}"
        )
    adapter_dir = checkpoint_dir / "adapter"
    if not adapter_dir.exists():
        raise CheckpointError(f"{adapter_dir}: adapter weights not found")
    return metadata


__all__ = [
    "METADATA_NAME",
    "REQUIRED_FIELDS",
    "CheckpointError",
    "adapter_tree_sha256",
    "build_metadata",
    "canonical_json",
    "config_hash",
    "dataset_fingerprint",
    "file_sha256",
    "git_provenance",
    "gpu_name",
    "library_versions",
    "load_checkpoint",
    "save_checkpoint",
]
