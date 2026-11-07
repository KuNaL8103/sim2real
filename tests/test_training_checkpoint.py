"""Checkpoint provenance: required fields, deterministic hash, refusal paths."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sim2real.training.adapters import FakePolicyAdapter
from sim2real.training.checkpoint import (
    REQUIRED_FIELDS,
    CheckpointError,
    build_metadata,
    config_hash,
    load_checkpoint,
    save_checkpoint,
)
from sim2real.training.config import load_training_config


def _metadata(config):
    dataset = {
        "dataset_root": "data/lerobot/train_v1",
        "dataset_version": "v3.0",
        "dataset_hash": "dataset-hash",
        "stats_file_hash": "stats-hash",
    }
    split = {"train": [{"episode_id": "a", "index": 0}], "val": [{"episode_id": "b", "index": 1}]}
    return build_metadata(
        config=config,
        dataset=dataset,
        split=split,
        step=7,
        state_dim=8,
        action_dim=8,
        created_at="2026-11-07T00:00:00+00:00",
    )


def _save(tmp_path: Path, config) -> Path:
    adapter = FakePolicyAdapter(8, 8, seed=3)
    metadata = _metadata(config)
    return save_checkpoint(tmp_path / "checkpoint-00000007", adapter, metadata)


def test_every_required_provenance_field_is_present(tmp_path):
    config = load_training_config("configs/training/smolvla_lora.yaml")
    checkpoint = _save(tmp_path, config)
    metadata = json.loads((checkpoint / "checkpoint_metadata.json").read_text())
    missing = [field for field in REQUIRED_FIELDS if field not in metadata]
    assert missing == []
    assert metadata["base_model_id"] == "lerobot/smolvla_base"
    assert metadata["base_model_revision"].startswith("5e8d12a")
    assert metadata["horizon_seconds"] == pytest.approx(5.0)
    assert metadata["camera_rename_map"]["observation.images.overhead"] == (
        "observation.images.camera1"
    )
    assert metadata["split"]["train"][0]["episode_id"] == "a"


def test_config_hash_is_deterministic_and_sensitive():
    config = load_training_config("configs/training/smolvla_lora.yaml")
    assert config.config_hash == config_hash(config.to_dict())
    assert config.config_hash == config_hash(config.to_dict())
    changed = config.to_dict()
    changed["lora"]["r"] = 8
    assert config_hash(changed) != config.config_hash


def test_load_refuses_mismatched_stats_hash(tmp_path):
    config = load_training_config("configs/training/smolvla_lora.yaml")
    checkpoint = _save(tmp_path, config)
    with pytest.raises(CheckpointError, match="stats file hash"):
        load_checkpoint(checkpoint, expected_stats_hash="different")


def test_load_refuses_other_action_representation(tmp_path):
    config = load_training_config("configs/training/smolvla_lora.yaml")
    checkpoint = _save(tmp_path, config)
    with pytest.raises(CheckpointError, match="action representation"):
        load_checkpoint(checkpoint, expected_action_representation="ee_delta")


def test_load_refuses_other_camera_rename_map(tmp_path):
    config = load_training_config("configs/training/smolvla_lora.yaml")
    checkpoint = _save(tmp_path, config)
    with pytest.raises(CheckpointError, match="camera rename map"):
        load_checkpoint(
            checkpoint,
            expected_rename_map={"observation.images.overhead": "observation.images.camera3"},
        )


def test_load_refuses_other_config_hash(tmp_path):
    config = load_training_config("configs/training/smolvla_lora.yaml")
    checkpoint = _save(tmp_path, config)
    with pytest.raises(CheckpointError, match="config hash"):
        load_checkpoint(checkpoint, expected_config_hash="not-the-hash")


def test_missing_adapter_directory_refused(tmp_path):
    config = load_training_config("configs/training/smolvla_lora.yaml")
    checkpoint = _save(tmp_path, config)
    for path in (checkpoint / "adapter").iterdir():
        path.unlink()
    (checkpoint / "adapter").rmdir()
    with pytest.raises(CheckpointError, match="adapter weights"):
        load_checkpoint(checkpoint)
