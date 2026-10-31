"""RenderCoordinator: one render in flight, reset quiesce, skip counters."""

from __future__ import annotations

import threading
import time

from sim2real.sim import create_backend
from sim2real.sim.rendering import RenderCoordinator


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
        assert coordinator.render("wrist", snapshot) is not None
        thread.join(timeout=2.0)
        assert finished.is_set()
        assert coordinator.stats()["published"] >= 2
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
