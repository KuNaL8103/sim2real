"""Async worker: latest-wins mailbox, latency, bounded restarts, shutdown.

Most tests use ``FakeChunkPolicy`` (fast, deterministic). The failure-injection
tests at the end use the real ``CheckpointPolicy`` client against a tiny server
that speaks the real length-prefixed protocol, so the client path is exercised
without torch.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from sim2real.evaluation import CheckpointPolicy, PolicyObservation
from sim2real.inference.async_worker import (
    FAILED,
    MODEL_UNAVAILABLE,
    READY,
    AsyncInferenceWorker,
)
from sim2real.inference.fake_policy import FakeChunkPolicy
from sim2real.training.checkpoint import METADATA_NAME, REQUIRED_FIELDS

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_SERVER = ROOT / "tests" / "fixtures" / "fake_policy_server.py"


def _observation(stamp_ns: int = 1_000_000_000) -> PolicyObservation:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    return PolicyObservation(
        state=np.zeros(8, dtype=np.float64),
        task_instruction="pick up the red cube",
        stamp_ns=stamp_ns,
        images={"overhead": image, "wrist": image},
    )


def _wait_for(predicate, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


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


def _real_client_factory(checkpoint: Path, **kwargs):
    def factory() -> CheckpointPolicy:
        return CheckpointPolicy(
            checkpoint,
            server_python=sys.executable,
            server_script=FIXTURE_SERVER,
            repo_root=ROOT,
            timeout_s=kwargs.get("timeout_s", 1.0),
            startup_timeout_s=kwargs.get("startup_timeout_s", 2.0),
        )

    return factory


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - same user in tests
        return True
    return True


def test_latest_observation_wins_and_request_ids_are_monotonic():
    results = []
    states = []
    ready = []
    policy = FakeChunkPolicy(
        chunk_size=4,
        step_period_s=0.1,
        delay_s=0.25,
        mode="fixed",
        fixed_actions=np.arange(4 * 8, dtype=np.float64).reshape(4, 8),
    )
    worker = AsyncInferenceWorker(
        lambda: policy,
        call_timeout_s=5.0,
        max_restarts=0,
        restart_backoff_s=0.0,
        on_result=results.append,
        on_state=lambda state, detail: (states.append(state), ready.append(state == READY)),
    )
    worker.start()
    try:
        assert _wait_for(lambda: READY in states)
        first = worker.submit(_observation())
        assert first is not None
        assert _wait_for(lambda: worker.in_flight == first)
        later = [worker.submit(_observation()) for _ in range(4)]
        assert all(request_id is not None for request_id in later)
        assert sorted(later) == later
        assert _wait_for(lambda: len(results) == 2)
        time.sleep(0.3)
        assert len(results) == 2
        assert [result.request_id for result in results] == [first, later[-1]]
        stats = worker.stats()
        assert stats["replaced"] == 3
        assert stats["max_waiting"] == 1
        assert stats["mailbox_waiting"] == 0
    finally:
        worker.shutdown()


def test_slow_call_does_not_block_submit_and_reports_latency():
    results = []
    delay_s = 0.3
    worker = AsyncInferenceWorker(
        lambda: FakeChunkPolicy(chunk_size=4, delay_s=delay_s),
        call_timeout_s=5.0,
        max_restarts=0,
        restart_backoff_s=0.0,
        on_result=results.append,
    )
    worker.start()
    try:
        time.sleep(0.05)
        started = time.monotonic()
        request_id = worker.submit(_observation())
        elapsed = time.monotonic() - started
        assert request_id is not None
        assert elapsed < delay_s / 5.0, f"submit blocked for {elapsed:.3f} s"
        assert _wait_for(lambda: len(results) == 1)
        assert results[0].latency_s >= delay_s * 0.8
        assert results[0].elapsed_since_observation_s() >= delay_s * 0.8
    finally:
        worker.shutdown()


def test_worker_reports_time_to_ready():
    def factory():
        time.sleep(0.05)
        return FakeChunkPolicy(chunk_size=4)

    worker = AsyncInferenceWorker(
        factory,
        call_timeout_s=2.0,
        max_restarts=0,
        restart_backoff_s=0.0,
    )
    worker.start()
    try:
        assert _wait_for(lambda: worker.state == READY)
        value = worker.stats()["time_to_ready_s"]
        assert value is not None
        assert value >= 0.05
        assert value < 2.0
    finally:
        worker.shutdown()


def test_policy_failures_are_bounded_and_terminal():
    results = []
    policy = FakeChunkPolicy(chunk_size=4, failure="raise")
    worker = AsyncInferenceWorker(
        lambda: policy,
        call_timeout_s=2.0,
        max_restarts=1,
        restart_backoff_s=0.01,
        on_result=results.append,
    )
    worker.start()
    try:
        assert worker.submit(_observation()) is not None
        assert _wait_for(lambda: worker.state == MODEL_UNAVAILABLE)
        assert worker.submit(_observation()) is not None
        assert _wait_for(lambda: worker.state == FAILED)
        stats = worker.stats()
        assert stats["failed_calls"] >= 2
        assert stats["restarts"] == 1
        assert results == []
        assert worker.submit(_observation()) is None
    finally:
        worker.shutdown()


def test_invalid_chunks_are_rejected_by_the_worker():
    for failure in ("nan", "wrong_shape", "empty", "wrong_dim"):
        results = []
        worker = AsyncInferenceWorker(
            lambda failure=failure: FakeChunkPolicy(chunk_size=4, failure=failure),
            call_timeout_s=2.0,
            max_restarts=0,
            restart_backoff_s=0.0,
            on_result=results.append,
        )
        worker.start()
        try:
            worker.submit(_observation())
            assert _wait_for(lambda worker=worker: worker.state == FAILED), failure
            assert results == []
            assert worker.stats()["failed_calls"] >= 1
        finally:
            worker.shutdown()


def test_start_failure_is_terminal_after_bounded_retries():
    calls = []

    def factory():
        calls.append(1)
        raise RuntimeError("no policy interpreter")

    worker = AsyncInferenceWorker(
        factory,
        call_timeout_s=1.0,
        max_restarts=1,
        restart_backoff_s=0.0,
    )
    worker.start()
    try:
        assert _wait_for(lambda: worker.state == FAILED)
        stats = worker.stats()
        assert stats["start_failures"] == 2
        assert stats["restarts"] == 1
        assert len(calls) == 2
    finally:
        worker.shutdown()


def test_hang_is_detected_as_timeout():
    worker = AsyncInferenceWorker(
        lambda: FakeChunkPolicy(chunk_size=4, failure="hang", hang_s=0.2),
        call_timeout_s=0.05,
        max_restarts=0,
        restart_backoff_s=0.0,
    )
    worker.start()
    try:
        worker.submit(_observation())
        assert _wait_for(lambda: worker.state == FAILED)
        assert worker.stats()["timed_out_calls"] >= 1
    finally:
        worker.shutdown()


def test_real_client_crash_restarts_then_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_POLICY_SERVER_MODE", "crash")
    results = []
    worker = AsyncInferenceWorker(
        _real_client_factory(_checkpoint(tmp_path)),
        call_timeout_s=1.0,
        max_restarts=1,
        restart_backoff_s=0.01,
        on_result=results.append,
    )
    worker.start()
    try:
        assert worker.submit(_observation()) is not None
        assert _wait_for(lambda: worker.state == MODEL_UNAVAILABLE)
        assert worker.submit(_observation()) is not None
        assert _wait_for(lambda: worker.state == FAILED, timeout_s=10.0)
        assert results == []
        assert worker.stats()["restarts"] == 1
    finally:
        worker.shutdown()


def test_real_client_startup_timeout_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_POLICY_SERVER_MODE", "slow-start")
    monkeypatch.setenv("FAKE_POLICY_SERVER_DELAY", "1.0")
    worker = AsyncInferenceWorker(
        _real_client_factory(_checkpoint(tmp_path), startup_timeout_s=0.2),
        call_timeout_s=1.0,
        max_restarts=0,
        restart_backoff_s=0.0,
    )
    worker.start()
    try:
        assert _wait_for(lambda: worker.state == FAILED, timeout_s=10.0)
        assert worker.stats()["start_failures"] >= 1
    finally:
        worker.shutdown()


def test_real_client_hang_times_out_and_is_terminal(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_POLICY_SERVER_MODE", "hang")
    worker = AsyncInferenceWorker(
        # The client's hello timeout and the worker's call timeout are separate
        # concerns; give the tiny fixture server 1 s to import and answer hello.
        _real_client_factory(_checkpoint(tmp_path), timeout_s=1.0),
        call_timeout_s=0.3,
        max_restarts=0,
        restart_backoff_s=0.0,
    )
    worker.start()
    try:
        worker.submit(_observation())
        assert _wait_for(lambda: worker.state == FAILED, timeout_s=10.0)
        assert worker.stats()["timed_out_calls"] >= 1
    finally:
        worker.shutdown()


def test_real_client_shutdown_leaves_no_child(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_POLICY_SERVER_MODE", "ok")
    results = []
    holder = {}

    def factory() -> CheckpointPolicy:
        policy = CheckpointPolicy(
            _checkpoint(tmp_path),
            server_python=sys.executable,
            server_script=FIXTURE_SERVER,
            repo_root=ROOT,
            timeout_s=2.0,
            startup_timeout_s=2.0,
        )
        holder["policy"] = policy
        return policy

    worker = AsyncInferenceWorker(
        factory,
        call_timeout_s=2.0,
        max_restarts=0,
        restart_backoff_s=0.0,
        on_result=results.append,
    )
    worker.start()
    try:
        assert _wait_for(lambda: worker.state == READY)
        worker.submit(_observation())
        assert _wait_for(lambda: len(results) == 1)
        process = holder["policy"]._process
        assert process is not None
        pid = process.pid
        assert _pid_alive(pid)
    finally:
        worker.shutdown()
    assert not _pid_alive(pid)
    # Shutdown aborts the server to unblock a call; that is not a restart.
    assert worker.stats()["restarts"] == 0
    assert worker.state == "STOPPED"


def test_real_client_passes_server_values_through(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_POLICY_SERVER_MODE", "ok")
    results = []
    worker = AsyncInferenceWorker(
        _real_client_factory(_checkpoint(tmp_path)),
        call_timeout_s=2.0,
        max_restarts=0,
        restart_backoff_s=0.0,
        on_result=results.append,
    )
    worker.start()
    try:
        worker.submit(_observation())
        assert _wait_for(lambda: len(results) == 1)
    finally:
        worker.shutdown()
    expected = (
        (np.arange(4 * 8, dtype=np.float32).reshape(4, 8) * np.float32(0.1) + np.float32(1.0))
        .astype(np.float32)
        .astype(np.float64)
    )
    np.testing.assert_array_equal(results[0].actions, expected)


@pytest.mark.parametrize("mode", ["bad-nan"])
def test_real_client_nonfinite_chunk_is_rejected(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("FAKE_POLICY_SERVER_MODE", mode)
    results = []
    worker = AsyncInferenceWorker(
        _real_client_factory(_checkpoint(tmp_path)),
        call_timeout_s=2.0,
        max_restarts=0,
        restart_backoff_s=0.0,
        on_result=results.append,
    )
    worker.start()
    try:
        worker.submit(_observation())
        assert _wait_for(lambda: worker.state == FAILED)
        assert results == []
    finally:
        worker.shutdown()
