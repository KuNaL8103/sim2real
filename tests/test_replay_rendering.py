"""Replay determinism, fidelity, estimates, no-overwrite, and missing fields."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import render_replay  # noqa: E402

from sim2real.expert import load_expert_config, parse_instruction, run_episode  # noqa: E402
from sim2real.expert.staging import StagingBuffer  # noqa: E402
from sim2real.sim.fake_backend import FakeBackend  # noqa: E402


def _staged_record(tmp_path: Path, *, backend_name: str = "fake"):
    config = load_expert_config("configs/expert")
    backend = FakeBackend("configs/sim")
    task = parse_instruction(
        "pick up the red cube and place it in the left zone",
        backend.configs.scene,
        config.instructions_file,
    )
    record = run_episode(
        backend,
        task,
        seed=0,
        randomization_profile="none",
        randomization_dir="configs/randomization",
        config=config,
        max_steps=40,
    )
    buffer = StagingBuffer(tmp_path, "replay_test")
    buffer.write(record)
    loaded = next(buffer.iter_records())
    return backend, buffer, loaded


def test_fake_replay_is_deterministic_and_faithful(tmp_path):
    backend, _, loaded = _staged_record(tmp_path)
    try:
        first = render_replay.replay_record(loaded, backend, tmp_path / "a", decimation=5)
        second = render_replay.replay_record(loaded, backend, tmp_path / "b", decimation=5)
        assert first["frames"] > 0
        assert first["max_joint_error_rad"] < 1e-12
        assert first["max_object_error_m"] is not None
        assert first["max_object_error_m"] < 1e-9
        hashes_a = [
            (image["step_index"], image["camera"], image["sha256"]) for image in first["images"]
        ]
        hashes_b = [
            (image["step_index"], image["camera"], image["sha256"]) for image in second["images"]
        ]
        assert hashes_a == hashes_b
    finally:
        backend.close()


def test_estimate_and_unapproved_refusal():
    class FakeRecord:
        steps = [object()] * 20_000

    estimate = render_replay.estimate_replay([FakeRecord()], decimation=1)
    assert estimate.gib > 1.0
    with pytest.raises(render_replay.ReplayRefused):
        render_replay.ensure_approved(estimate, approved=False)
    render_replay.ensure_approved(estimate, approved=True)
    small = render_replay.estimate_replay([FakeRecord()], decimation=1, render_seconds=0.000001)
    assert small.frames == 40_000


def test_existing_images_are_never_overwritten(tmp_path):
    backend, _, loaded = _staged_record(tmp_path)
    try:
        first = render_replay.replay_record(loaded, backend, tmp_path / "out", decimation=5)
        target = tmp_path / "out" / first["images"][0]["path"]
        sentinel = b"DO NOT OVERWRITE"
        target.write_bytes(sentinel)
        second = render_replay.replay_record(loaded, backend, tmp_path / "out", decimation=5)
        assert target.read_bytes() == sentinel
        assert any(image["skipped"] for image in second["images"])
    finally:
        backend.close()


def test_missing_replay_fields_raise_clear_errors(tmp_path):
    backend, _, loaded = _staged_record(tmp_path)
    try:
        with pytest.raises(render_replay.ReplayError, match="no recorded steps"):
            render_replay.replay_record(replace(loaded, steps=[]), backend, tmp_path / "x")
        with pytest.raises(render_replay.ReplayError, match="missing episode_index"):
            render_replay.replay_record(
                replace(loaded, episode_index=None), backend, tmp_path / "x"
            )
    finally:
        backend.close()
