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


def _python_metadata() -> dict:
    return {
        "version": "3.12.3",
        "major_minor": "3.12",
        "executable": "/usr/bin/python3.12",
        "lerobot_requires_python": ">=3.10",
    }


def _head() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


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
        python_metadata=_python_metadata(),
        commit=_head(),
        timestamp="20260101T000000Z",
        allow_dirty=True,
    )
    printed = capsys.readouterr().out
    assert "bundle payload" in printed and "MiB" in printed
    assert result.archive.is_file()
    assert result.files >= 6
    files = result.manifest["files"]
    assert "data/lerobot/train_v1/meta/info.json" in files
    assert "data/lerobot/train_v1/data/file-000.parquet" in files
    assert "outputs/remote_bundle_payload/requirements.txt" in files
    assert "outputs/remote_bundle_payload/repo.bundle" in files
    assert "outputs/remote_bundle_payload/remote_train.sh" in files
    assert result.manifest["python"]["major_minor"] == "3.12"
    assert (
        result.manifest["repo_bundle"]["sha256"]
        == files["outputs/remote_bundle_payload/repo.bundle"]["sha256"]
    )
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
            python_metadata=_python_metadata(),
            commit=_head(),
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
            python_metadata=_python_metadata(),
            commit=_head(),
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
        allow_no_run_report=True,
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
        verifier.verify_checkpoint(checkpoint, config, handshake="skip", allow_no_run_report=True)


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
        verifier.verify_checkpoint(checkpoint, config, handshake="skip", allow_no_run_report=True)


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
        verifier.verify_checkpoint(checkpoint, config, handshake="skip", allow_no_run_report=True)


def test_verifier_rejects_adapter_hash_mismatch(tmp_path):
    verifier = _load_script("verify_remote_checkpoint")
    dataset = _write_dataset(tmp_path / "dataset")
    config = _write_config(tmp_path, dataset)
    checkpoint = _checkpoint(tmp_path, config, dataset)
    adapter_file = checkpoint / "adapter/fake_adapter.npz"
    adapter_file.write_bytes(adapter_file.read_bytes() + b"tamper")
    with pytest.raises(verifier.VerificationError, match="adapter directory hash"):
        verifier.verify_checkpoint(checkpoint, config, handshake="skip", allow_no_run_report=True)


def test_verifier_requires_run_report_unless_explicitly_overridden(tmp_path, capsys):
    verifier = _load_script("verify_remote_checkpoint")
    dataset = _write_dataset(tmp_path / "dataset")
    config = _write_config(tmp_path, dataset)
    checkpoint = _checkpoint(tmp_path, config, dataset)
    with pytest.raises(verifier.VerificationError, match="no RUN_REPORT"):
        verifier.verify_checkpoint(checkpoint, config, handshake="skip")
    summary = verifier.verify_checkpoint(
        checkpoint,
        config,
        handshake="skip",
        allow_no_run_report=True,
    )
    assert summary["run_report"] == "UNVERIFIED: no run report"
    assert "UNVERIFIED: no run report" in capsys.readouterr().err


def test_requirements_strip_local_version_labels():
    bundle = _load_script("export_remote_bundle")
    versions = {name: "1.2.3" for name in bundle.PINNED_PACKAGES}
    versions["torch"] = "2.11.0+cu126"
    versions["torchvision"] = "0.26.0+cu126"
    versions["torchcodec"] = "0.11.1+cpu"
    text = bundle.render_requirements(versions, "/x/.venv-ml/bin/python")
    pins = bundle.requirement_pins(text)
    assert pins["torch"] == "2.11.0"
    assert pins["torchvision"] == "0.26.0"
    assert pins["torchcodec"] == "0.11.1"
    assert not any("+" in value for value in pins.values())
    assert f"--extra-index-url {bundle.TORCH_INDEX}" in text
    # the index is documented as the CUDA wheel source, not a bare option
    assert "# The CUDA build comes from the wheel index on the next line." in text


def test_real_pin_file_has_no_local_version_pins(tmp_path):
    """The pins read from the real .venv-ml must render without +cpu/+cu126."""
    python = ROOT / ".venv-ml" / "bin" / "python"
    if not python.is_file():
        pytest.skip("optional .venv-ml is absent")
    bundle = _load_script("export_remote_bundle")
    pins = bundle.requirement_pins(bundle.render_requirements(bundle._python_versions(python)))
    offenders = {name: value for name, value in pins.items() if "+" in value}
    assert not offenders, offenders


def test_export_writes_bundle_info_sidecar(tmp_path):
    bundle = _load_script("export_remote_bundle")
    dataset = _write_dataset(tmp_path / "dataset")
    config = _write_config(tmp_path, dataset)
    result = bundle.export_bundle(
        dataset_root=dataset,
        config_paths=[config],
        output_dir=tmp_path / "bundle",
        versions=_versions(bundle),
        python_metadata=_python_metadata(),
        commit=_head(),
        timestamp="20260101T000000Z",
        allow_dirty=True,
    )
    assert result.bundle_info.is_file()
    info = json.loads(result.bundle_info.read_text(encoding="utf-8"))
    assert info["archive"]["name"] == result.archive.name
    assert info["archive"]["sha256"] == result.archive_sha256
    assert info["archive"]["size"] == result.archive_bytes
    with tarfile.open(result.archive) as archive:
        manifest_bytes = archive.extractfile("outputs/remote_bundle_payload/MANIFEST.json").read()
    assert info["manifest"]["sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert info["requirements"]["sha256"] == result.manifest["requirements"]["sha256"]
    assert info["git_commit"] == result.commit


def test_verifier_rejects_and_accepts_recorded_overrides(tmp_path, capsys):
    verifier = _load_script("verify_remote_checkpoint")
    dataset = _write_dataset(tmp_path / "dataset")
    config = _write_config(tmp_path, dataset)
    checkpoint = _checkpoint(tmp_path, config, dataset)
    report = _run_report(checkpoint)
    report["overrides"] = ["low_clock"]
    report_path = tmp_path / "RUN_REPORT.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(verifier.VerificationError, match="overrides"):
        verifier.verify_checkpoint(checkpoint, config, run_report=report_path, handshake="skip")
    summary = verifier.verify_checkpoint(
        checkpoint,
        config,
        run_report=report_path,
        handshake="skip",
        accept_overrides=True,
    )
    assert summary["overrides"] == ["low_clock"]
    assert "overrides: ['low_clock']" in capsys.readouterr().out


def test_verifier_bundle_info_paths(tmp_path, capsys):
    verifier = _load_script("verify_remote_checkpoint")
    dataset = _write_dataset(tmp_path / "dataset")
    config = _write_config(tmp_path, dataset)
    checkpoint = _checkpoint(tmp_path, config, dataset)
    report = _run_report(checkpoint)
    report["preflight"] = {
        "bundle": {
            "manifest": {"sha256": "a" * 64},
            "requirements": {"sha256": "b" * 64},
        }
    }
    report_path = tmp_path / "RUN_REPORT.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    sidecar = tmp_path / "archive.tar.gz.bundle_info.json"
    sidecar.write_text(
        json.dumps(
            {
                "archive": {"name": "archive.tar.gz", "sha256": "c" * 64},
                "manifest": {"sha256": "a" * 64},
                "requirements": {"sha256": "b" * 64},
            }
        ),
        encoding="utf-8",
    )
    summary = verifier.verify_checkpoint(
        checkpoint, config, run_report=report_path, bundle_info=sidecar, handshake="skip"
    )
    assert summary["bundle_provenance"] == "verified"
    assert summary["archive_sha256"] == "c" * 64

    bad = tmp_path / "bad.bundle_info.json"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["requirements"]["sha256"] = "9" * 64
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(verifier.VerificationError, match="does not match the exported"):
        verifier.verify_checkpoint(
            checkpoint, config, run_report=report_path, bundle_info=bad, handshake="skip"
        )

    unverified = verifier.verify_checkpoint(
        checkpoint, config, run_report=report_path, handshake="skip"
    )
    assert unverified["bundle_provenance"] == "UNVERIFIED: bundle provenance not checked"
    assert "UNVERIFIED: bundle provenance not checked" in capsys.readouterr().err


def test_remote_train_low_clock_override_is_explicit_and_memory_is_hard():
    text = (ROOT / "scripts" / "remote_train.sh").read_text(encoding="utf-8")
    assert "ALLOW_LOW_CLOCK=0" in text
    assert "--allow-low-clock" in text
    assert 'overrides.append("low_clock")' in text
    assert '"overrides": overrides' in text
    memory = text[
        text.index("if total_mib < min_memory_mib") : text.index("if median_clock < min_clock")
    ]
    assert "allow_low_clock" not in memory
    assert "min_memory_mib" in memory


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
