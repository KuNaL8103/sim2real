"""Policy-server protocol tests in fake-model mode (no torch, no GPU)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from sim2real.evaluation import CheckpointPolicy, PolicyError, PolicyObservation
from sim2real.evaluation.policy_protocol import PROTOCOL_VERSION, read_packet, write_packet
from sim2real.training.checkpoint import METADATA_NAME, REQUIRED_FIELDS

ROOT = Path(__file__).resolve().parents[1]


def _checkpoint(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "checkpoint-00000001"
    (checkpoint / "adapter").mkdir(parents=True)
    metadata = {field: None for field in REQUIRED_FIELDS}
    metadata.update(
        {
            "schema_version": 1,
            "chunk_size": 4,
            "n_action_steps": 4,
            "fps": 10.0,
            "state_dim": 8,
            "action_dim": 8,
            "camera_rename_map": {
                "observation.images.overhead": "observation.images.camera1",
                "observation.images.wrist": "observation.images.camera2",
            },
            "config": {
                "model": {
                    "expected_cameras": [
                        "observation.images.camera1",
                        "observation.images.camera2",
                    ]
                }
            },
            "dataset_root": str(tmp_path / "dataset"),
        }
    )
    (checkpoint / METADATA_NAME).write_text(json.dumps(metadata), encoding="utf-8")
    return checkpoint


def _observation() -> PolicyObservation:
    return PolicyObservation(
        state=np.arange(8, dtype=np.float64),
        task_instruction="pick up the red cube and place it in the left zone",
        images={
            "overhead": np.zeros((8, 8, 3), dtype=np.uint8),
            "wrist": np.zeros((8, 8, 3), dtype=np.uint8),
        },
    )


def _assert_handshake_failure(checkpoint: Path, **kwargs):
    policy = CheckpointPolicy(
        checkpoint, server_python=sys.executable, repo_root=ROOT, fake_model=True, **kwargs
    )
    with pytest.raises(PolicyError):
        policy.reset("pick", 20000)
    policy.close()


def test_fake_model_handshake_and_chunk(tmp_path):
    policy = CheckpointPolicy(
        _checkpoint(tmp_path),
        server_python=sys.executable,
        repo_root=ROOT,
        fake_model=True,
        timeout_s=10.0,
    )
    try:
        policy.reset("pick", 20000)
        assert policy.info["server_protocol"] == PROTOCOL_VERSION
        pid = policy._process.pid
        policy.reset("pick", 20001)
        assert policy._process.pid == pid  # a live server is reused, not reloaded
        chunk = policy.act(_observation())
        assert chunk.actions.shape == (4, 8)
        assert np.all(np.isfinite(chunk.actions))
        assert chunk.step_period_s == pytest.approx(0.1)
    finally:
        policy.close()


def test_checkpoint_hash_mismatch_is_refused(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    policy = CheckpointPolicy(
        checkpoint,
        server_python=sys.executable,
        repo_root=ROOT,
        fake_model=True,
        timeout_s=10.0,
    )
    metadata_path = checkpoint / METADATA_NAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["chunk_size"] = 99
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    try:
        with pytest.raises(PolicyError):
            policy.reset("pick", 20000)
    finally:
        policy.close()


def test_missing_checkpoint_metadata_is_refused(tmp_path):
    with pytest.raises(PolicyError):
        CheckpointPolicy(tmp_path / "missing", server_python=sys.executable)


def test_timeout_and_server_death_are_reported(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    sleeper = tmp_path / "sleeper.py"
    sleeper.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    _assert_handshake_failure(checkpoint, server_script=sleeper, timeout_s=0.3)

    dead = tmp_path / "dead.py"
    dead.write_text("raise SystemExit(1)\n", encoding="utf-8")
    _assert_handshake_failure(checkpoint, server_script=dead, timeout_s=0.5)


def test_malformed_frame_is_rejected(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    malformed = tmp_path / "malformed.py"
    malformed.write_text(
        "import sys, time\n"
        "sys.stdout.buffer.write(b'garbage')\n"
        "sys.stdout.flush()\n"
        "time.sleep(5)\n",
        encoding="utf-8",
    )
    _assert_handshake_failure(checkpoint, server_script=malformed, timeout_s=0.5)


def test_protocol_round_trip_and_no_pickle():
    import io

    stream = io.BytesIO()
    write_packet(stream, {"type": "hello", "value": 1}, [b"abc"])
    stream.seek(0)
    header, blobs = read_packet(stream)
    assert header == {"type": "hello", "value": 1}
    assert blobs == [b"abc"]

    for name in ("policy_protocol.py", "policy_server.py"):
        path = (
            ROOT / "src" / "sim2real" / "evaluation" / name
            if name == "policy_protocol.py"
            else ROOT / "scripts" / name
        )
        source = path.read_text(encoding="utf-8").lower()
        assert "import pickle" not in source
        assert "pickle.loads" not in source
        assert "pickle.dumps" not in source
    time.sleep(0)
