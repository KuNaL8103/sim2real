"""CPU-only tests for the remote bundle, verifier, and remote shell script."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from sim2real.training.adapters import FakePolicyAdapter
from sim2real.training.checkpoint import (
    adapter_tree_sha256,
    build_metadata,
    dataset_fingerprint,
    file_sha256,
    save_checkpoint,
)
from sim2real.training.config import load_training_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_SOURCE = ROOT / "configs/training/smolvla_lora_overfit_remote.yaml"


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_dataset(root: Path) -> Path:
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "fps": 10,
        "features": {
            "observation.state": {"shape": [8]},
            "action": {"shape": [8]},
        },
        "total_episodes": 2,
        "total_frames": 4,
    }
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta/stats.json").write_text(
        json.dumps({"action": {"mean": [0.0] * 8, "std": [1.0] * 8}}),
        encoding="utf-8",
    )
    split = {
        "seed": 123,
        "train": [{"episode_id": "ep0", "index": 0, "source": "mujoco"}],
        "val": [{"episode_id": "ep1", "index": 1, "source": "mujoco"}],
        "by_source": {"mujoco": {"train": 1, "val": 1}},
    }
    (root / "meta/sim2real_split.json").write_text(json.dumps(split), encoding="utf-8")
    (root / "data/file-000.parquet").write_bytes(b"fixture parquet bytes")
    return root


def _write_config(tmp_path: Path, dataset: Path) -> Path:
    text = CONFIG_SOURCE.read_text(encoding="utf-8")
    text = text.replace(
        "dataset_root: data/lerobot/train_v1",
        f"dataset_root: {dataset.as_posix()}",
    )
    path = tmp_path / "configs" / "smolvla_lora_overfit_remote.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(text, encoding="utf-8")
    return path


def _versions(bundle_module) -> dict[str, str]:
    return {name: "1.2.3" for name in bundle_module.PINNED_PACKAGES}


def _checkpoint(tmp_path: Path, config_path: Path, dataset: Path) -> Path:
    config = load_training_config(config_path)
    split = json.loads((dataset / "meta/sim2real_split.json").read_text(encoding="utf-8"))
    metadata = build_metadata(
        config=config,
        dataset=dataset_fingerprint(dataset),
        split=split,
        step=100,
        state_dim=8,
        action_dim=8,
        created_at="2026-11-13T00:00:00+00:00",
        repo_root=ROOT,
    )
    metadata["git_commit"] = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    metadata["git_dirty"] = False
    adapter = FakePolicyAdapter(8, 8, seed=0)
    return save_checkpoint(tmp_path / "checkpoint-00000100", adapter, metadata)


def _run_report(checkpoint: Path) -> dict:
    metadata_path = checkpoint / "checkpoint_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "git_commit": metadata["git_commit"],
        "config_hash": metadata["config_hash"],
        "stats_file_hash": metadata["stats_file_hash"],
        "checkpoints": [
            {
                "step": metadata["step"],
                "path": str(checkpoint),
                "name": checkpoint.name,
                "metadata_sha256": file_sha256(metadata_path),
                "adapter_sha256": adapter_tree_sha256(checkpoint / "adapter"),
            }
        ],
    }


def test_bundle_hashes_files_prints_size_and_excludes_forbidden(tmp_path, capsys):
    bundle = _load_script("export_remote_bundle")
    dataset = _write_dataset(tmp_path / "dataset")
    config = _write_config(tmp_path, dataset)
    output = tmp_path / "bundle"
    result = bundle.export_bundle(
        dataset_root=dataset,
        config_paths=[config],
        output_dir=output,
        versions=_versions(bundle),
        commit="a" * 40,
        timestamp="20260101T000000Z",
        allow_dirty=True,
    )
    printed = capsys.readouterr().out
    assert "bundle payload" in printed and "MiB" in printed
    assert result.archive.is_file()
    assert result.files >= 4
    files = result.manifest["files"]
    assert "data/lerobot/train_v1/meta/info.json" in files
    assert "data/lerobot/train_v1/data/file-000.parquet" in files
    assert "outputs/remote_bundle_payload/requirements.txt" in files
    with tarfile.open(result.archive) as archive:
        names = set(archive.getnames())
        for relative, entry in files.items():
            handle = archive.extractfile(relative)
            assert handle is not None
            assert hashlib.sha256(handle.read()).hexdigest() == entry["sha256"]
    assert "outputs/remote_bundle_payload/MANIFEST.json" in names
    assert "outputs/remote_bundle_payload/requirements.txt" in names
    assert not any(name.endswith((".safetensors", ".pt", ".pth")) for name in names)
    assert not any(".env" in name for name in names)


def test_bundle_refuses_forbidden_files(tmp_path):
    bundle = _load_script("export_remote_bundle")
    dataset = _write_dataset(tmp_path / "dataset")
    (dataset / ".env").write_text("TOKEN=secret", encoding="utf-8")
    config = _write_config(tmp_path, dataset)
    with pytest.raises(bundle.BundleError, match="credentials"):
        bundle.export_bundle(
            dataset_root=dataset,
            config_paths=[config],
            output_dir=tmp_path / "bundle",
            versions=_versions(bundle),
            commit="a" * 40,
            allow_dirty=True,
        )


def test_bundle_refuses_model_weights(tmp_path):
    bundle = _load_script("export_remote_bundle")
    dataset = _write_dataset(tmp_path / "dataset")
    (dataset / "data/model.safetensors").write_bytes(b"weights")
    config = _write_config(tmp_path, dataset)
    with pytest.raises(bundle.BundleError, match="suffix"):
        bundle.export_bundle(
            dataset_root=dataset,
            config_paths=[config],
            output_dir=tmp_path / "bundle",
            versions=_versions(bundle),
            commit="a" * 40,
            allow_dirty=True,
        )


def test_verifier_accepts_good_checkpoint_and_run_report(tmp_path):
    verifier = _load_script("verify_remote_checkpoint")
    dataset = _write_dataset(tmp_path / "dataset")
    config = _write_config(tmp_path, dataset)
    checkpoint = _checkpoint(tmp_path, config, dataset)
    report_path = tmp_path / "RUN_REPORT.json"
    report_path.write_text(json.dumps(_run_report(checkpoint)), encoding="utf-8")
    summary = verifier.verify_checkpoint(
        checkpoint,
        config,
        run_report=report_path,
        handshake="skip",
    )
    assert summary["verified"] is True
    assert summary["git_dirty"] is False
    assert summary["handshake"] == "skip"


def test_verifier_handshake_accepts_fake_model(tmp_path):
    verifier = _load_script("verify_remote_checkpoint")
    dataset = _write_dataset(tmp_path / "dataset")
    config = _write_config(tmp_path, dataset)
    checkpoint = _checkpoint(tmp_path, config, dataset)
    summary = verifier.verify_checkpoint(
        checkpoint,
        config,
        handshake="fake",
        server_python=Path(sys.executable),
    )
    assert summary["handshake"] == "fake"


def test_verifier_rejects_dirty_checkpoint(tmp_path):
    verifier = _load_script("verify_remote_checkpoint")
    dataset = _write_dataset(tmp_path / "dataset")
    config = _write_config(tmp_path, dataset)
    checkpoint = _checkpoint(tmp_path, config, dataset)
    metadata_path = checkpoint / "checkpoint_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["git_dirty"] = True
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(verifier.VerificationError, match="git_dirty"):
        verifier.verify_checkpoint(checkpoint, config, handshake="skip")


def test_verifier_rejects_stats_mismatch(tmp_path):
    verifier = _load_script("verify_remote_checkpoint")
    dataset = _write_dataset(tmp_path / "dataset")
    config = _write_config(tmp_path, dataset)
    checkpoint = _checkpoint(tmp_path, config, dataset)
    metadata_path = checkpoint / "checkpoint_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["stats_file_hash"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(verifier.VerificationError, match="stats file hash"):
        verifier.verify_checkpoint(checkpoint, config, handshake="skip")


def test_verifier_rejects_unknown_commit(tmp_path):
    verifier = _load_script("verify_remote_checkpoint")
    dataset = _write_dataset(tmp_path / "dataset")
    config = _write_config(tmp_path, dataset)
    checkpoint = _checkpoint(tmp_path, config, dataset)
    metadata_path = checkpoint / "checkpoint_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["git_commit"] = "0" * 40
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(verifier.VerificationError, match="not present in local history"):
        verifier.verify_checkpoint(checkpoint, config, handshake="skip")


def test_verifier_rejects_adapter_hash_mismatch(tmp_path):
    verifier = _load_script("verify_remote_checkpoint")
    dataset = _write_dataset(tmp_path / "dataset")
    config = _write_config(tmp_path, dataset)
    checkpoint = _checkpoint(tmp_path, config, dataset)
    adapter_file = checkpoint / "adapter/fake_adapter.npz"
    adapter_file.write_bytes(adapter_file.read_bytes() + b"tamper")
    with pytest.raises(verifier.VerificationError, match="adapter directory hash"):
        verifier.verify_checkpoint(checkpoint, config, handshake="skip")


def test_remote_train_script_has_required_guards():
    script = ROOT / "scripts/remote_train.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text(encoding="utf-8")
    for required in (
        "nvidia-smi",
        "torch.cuda.is_available",
        "a @ b",
        "sha256",
        "git status --porcelain",
        "MANIFEST",
        "RUN_REPORT.json",
        "set -euo pipefail",
    ):
        assert required in text
