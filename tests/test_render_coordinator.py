"""RenderCoordinator: one render in flight, reset quiesce, skip counters."""

from __future__ import annotations

import dataclasses
import threading
import time

from sim2real.sim import create_backend
from sim2real.sim.rendering import CameraPairScheduler, RenderCoordinator, round_robin_order


def test_reset_waits_for_inflight_render_and_blocks_new_ones():
    backend = create_backend("fake", "configs/sim")
    try:
        backend.render_delay_s = 0.3
        coordinator = RenderCoordinator(backend)
        snapshot = backend.capture_snapshot()

        finished = threading.Event()

        def slow_render() -> None:
            coordinator.render("overhead", snapshot)
            finished.set()

        thread = threading.Thread(target=slow_render)
        thread.start()
        time.sleep(0.05)

        started = time.perf_counter()
        coordinator.begin_reset()
        waited = time.perf_counter() - started
        assert finished.is_set() is False or waited > 0.0
        assert waited >= 0.2, f"reset did not wait for the in-flight render: {waited:.3f}s"

        before = coordinator.stats()
        assert coordinator.render("wrist", snapshot) is None
        after = coordinator.stats()
        assert after["skipped_reset"] == before["skipped_reset"] + 1

        coordinator.end_reset()
        # A snapshot captured before the reset is stale by design; capture and
        # tag a fresh one before rendering again.
        fresh = coordinator.tag(backend.capture_snapshot())
        assert coordinator.render("wrist", fresh) is not None
        thread.join(timeout=2.0)
        assert finished.is_set()
        assert coordinator.stats()["published"] >= 2
    finally:
        backend.close()


def test_round_robin_order_rotates():
    cameras = ("overhead", "wrist")
    assert round_robin_order(cameras, 0) == ("overhead", "wrist")
    assert round_robin_order(cameras, 1) == ("wrist", "overhead")
    assert round_robin_order(("a", "b", "c"), 2) == ("c", "a", "b")
    assert round_robin_order((), 0) == ()


def test_pair_scheduler_serves_both_cameras_from_one_snapshot():
    backend = create_backend("fake", "configs/sim")
    backend.render_delay_s = 0.3
    coordinator = RenderCoordinator(backend)
    scheduler = CameraPairScheduler(("overhead", "wrist"))
    served = {"overhead": 0, "wrist": 0}
    stamps: dict[str, list[int]] = {"overhead": [], "wrist": []}
    active = 0
    max_active = 0

    def on_frame(camera, frame):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        served[camera] += 1
        stamps[camera].append(frame.stamp_ns)
        time.sleep(0.01)
        active -= 1

    def submit(camera, snapshot):
        return coordinator.submit(camera, snapshot, lambda frame, cam=camera: on_frame(cam, frame))

    try:
        for tick in range(40):
            base = backend.capture_snapshot()
            snapshot = coordinator.tag(dataclasses.replace(base, stamp_ns=tick))
            scheduler.next(snapshot, submit)
            time.sleep(0.1)
        time.sleep(0.5)
    finally:
        coordinator.close()
        backend.close()

    assert served["overhead"] > 0
    assert served["wrist"] > 0
    # Both cameras share the snapshot stamp; a pair is never split.
    assert set(stamps["overhead"]) == set(stamps["wrist"])
    assert len(stamps["overhead"]) == len(set(stamps["overhead"]))
    # The slow renderer can serve roughly one camera per 0.1 s tick, so over 40
    # ticks the two served counts may differ by at most one completed pair.
    assert abs(served["overhead"] - served["wrist"]) <= 1
    # The single worker never renders two cameras concurrently.
    assert max_active == 1
    stats = coordinator.stats()
    assert stats["published_by_camera"]["overhead"] > 0
    assert stats["published_by_camera"]["wrist"] > 0


def test_scheduler_clear_drops_the_pending_pair():
    backend = create_backend("fake", "configs/sim")
    try:
        snapshot = backend.capture_snapshot()
        scheduler = CameraPairScheduler(("overhead", "wrist"))
        assert scheduler.next(snapshot, lambda camera, pending: False) == 0
        assert scheduler.pending_snapshot is snapshot
        scheduler.clear()
        assert scheduler.pending_snapshot is None
        submitted: list[str] = []
        assert (
            scheduler.next(snapshot, lambda camera, pending: submitted.append(camera) or True) == 2
        )
        assert submitted == ["wrist", "overhead"]  # tick advanced past the aborted pair
    finally:
        backend.close()


def test_busy_render_is_skipped_and_counted():
    backend = create_backend("fake", "configs/sim")
    try:
        backend.render_delay_s = 0.2
        coordinator = RenderCoordinator(backend)
        snapshot = backend.capture_snapshot()
        results: list[object] = []

        def render() -> None:
            results.append(coordinator.render("overhead", snapshot))

        first = threading.Thread(target=render)
        first.start()
        time.sleep(0.05)
        second = threading.Thread(target=render)
        second.start()
        first.join(timeout=2.0)
        second.join(timeout=2.0)

        assert any(frame is None for frame in results)
        assert coordinator.stats()["skipped_busy"] >= 1
    finally:
        backend.close()
