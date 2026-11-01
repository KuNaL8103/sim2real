"""Staging: atomic write, manifest sha256, corrupt skip, duplicate ids."""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from sim2real.expert import load_expert_config, parse_instruction, run_episode
from sim2real.expert.staging import (
    DuplicateEpisodeError,
    StagingBuffer,
    estimate_storage,
)
from sim2real.sim import create_backend

CONFIG_DIR = "configs/sim"
EXPERT_CONFIG_DIR = "configs/expert"
INSTRUCTION = "pick up the red cube and place it in the left zone"


@pytest.fixture(scope="module")
def record():
    config = load_expert_config(EXPERT_CONFIG_DIR)
    backend = create_backend("fake", CONFIG_DIR)
    try:
        task = parse_instruction(INSTRUCTION, backend.configs.scene, config.instructions_file)
        return run_episode(
            backend,
            task,
            0,
            instruction=INSTRUCTION,
            randomization_profile="none",
            config=config,
        )
    finally:
        backend.close()


def _clone(record, episode_id: str):
    clone = copy.deepcopy(record)
    clone.episode_id = episode_id
    return clone


def test_atomic_write_manifest_and_sha(tmp_path, record) -> None:
    buffer = StagingBuffer(tmp_path, "unit_run")
    staged = buffer.write(record)
    assert staged.path.is_file()
    assert staged.sidecar_path.is_file()
    digest = hashlib.sha256(staged.path.read_bytes()).hexdigest()
    assert digest == staged.sha256
    rows = buffer.manifest_rows()
    assert len(rows) == 1
    assert rows[0]["episode_id"] == record.episode_id
    assert rows[0]["sha256"] == digest
    assert set(rows[0]) == {
        "episode_id",
        "seed",
        "outcome",
        "path",
        "sha256",
        "written_at",
    }
    loaded = list(buffer.iter_records())
    assert len(loaded) == 1
    np.testing.assert_array_equal(
        loaded[0].steps[0].joint_positions, record.steps[0].joint_positions
    )
    assert buffer.verify()["valid"] == 1


def test_crash_between_tmp_and_rename_leaves_no_partial_file(tmp_path, record, monkeypatch) -> None:
    import sim2real.expert.staging as staging

    buffer = StagingBuffer(tmp_path, "crash_run")
    original = staging.os.replace

    def fail_once(source, destination):
        raise OSError("simulated crash before rename")

    monkeypatch.setattr(staging.os, "replace", fail_once)
    with pytest.raises(OSError):
        buffer.write(record)
    monkeypatch.setattr(staging.os, "replace", original)
    assert not buffer.episode_path(record.episode_id).exists()
    assert not buffer.sidecar_path(record.episode_id).exists()
    assert list(buffer.root.glob("*.tmp")) == []
    assert buffer.manifest_rows() == []


def test_duplicate_episode_id_is_rejected(tmp_path, record) -> None:
    buffer = StagingBuffer(tmp_path, "dup_run")
    buffer.write(record)
    with pytest.raises(DuplicateEpisodeError):
        buffer.write(record)
    # An explicit overwrite is allowed and stays a single manifest row.
    buffer.write(record, overwrite=True)
    assert len(buffer.manifest_rows()) == 1


def test_corrupt_entry_is_skipped_with_logged_error(tmp_path, record, caplog) -> None:
    buffer = StagingBuffer(tmp_path, "corrupt_run")
    buffer.write(_clone(record, "good"))
    buffer.write(_clone(record, "bad"))
    # Truncate the array file for the second episode.
    bad_path = buffer.episode_path("bad")
    bad_path.write_bytes(b"not an npz file")
    with caplog.at_level("ERROR"):
        loaded = list(buffer.iter_records())
    assert [item.episode_id for item in loaded] == ["good"]
    assert buffer.last_errors
    assert any("bad" in message for message in buffer.last_errors)
    verification = buffer.verify()
    assert verification["valid"] == 1
    assert verification["corrupt"] >= 1


def test_missing_sidecar_is_skipped(tmp_path, record) -> None:
    buffer = StagingBuffer(tmp_path, "missing_run")
    buffer.write(record)
    buffer.sidecar_path(record.episode_id).unlink()
    assert list(buffer.iter_records()) == []
    assert buffer.verify()["missing"] == 0  # sidecar missing is counted as corrupt


def test_manifest_sha256_mismatch_is_detected(tmp_path, record) -> None:
    buffer = StagingBuffer(tmp_path, "tamper_run")
    buffer.write(record)
    manifest = buffer.manifest_path.read_text(encoding="utf-8")
    row = json.loads(manifest)
    row["sha256"] = "0" * 64
    buffer.manifest_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert list(buffer.iter_records()) == []
    assert any("sha256" in message for message in buffer.last_errors)


def test_storage_estimate_scales_with_episodes() -> None:
    one = estimate_storage(100, 1, (256, 256), 5)
    two = estimate_storage(100, 2, (256, 256), 5)
    assert two["bytes"] == 2 * one["bytes"]
    assert one["gib"] > 0.0
    assert one["minutes"] == pytest.approx(one["seconds"] / 60.0)
