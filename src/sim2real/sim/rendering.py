"""Thread-safe rendering gate for the ROS node.

MuJoCo's EGL context is thread-bound, so all rendering runs on one dedicated
worker thread. Camera callbacks submit a snapshot and return immediately; if a
render is already in flight the submission is rejected and counted as skipped.
There is no queue: at most one snapshot is ever pending.

``begin_reset`` blocks new submissions and waits for the worker to finish before
a reset mutates the scene.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import replace

from sim2real.sim.backend import CameraFrame, RenderSnapshot, SimBackend

FrameCallback = Callable[[CameraFrame], None]


def round_robin_order(cameras: tuple[str, ...], tick: int) -> tuple[str, ...]:
    """Return ``cameras`` starting at ``tick % len(cameras)``.

    The order is stable and deterministic, so a slow renderer that can only
    serve one camera per tick still alternates which camera gets the first slot.
    """
    if not cameras:
        return ()
    offset = int(tick) % len(cameras)
    return tuple(cameras[offset:]) + tuple(cameras[:offset])


class CameraPairScheduler:
    """Serve every camera from one snapshot before advancing to a newer one.

    The render coordinator has one slot. If the camera tick submits both cameras
    eagerly, the first in the list can permanently starve the rest. This
    scheduler keeps a pending snapshot and submits at most one camera per slot,
    retrying the same camera when the coordinator is busy. The pair is complete
    only after every camera has been submitted, so all cameras in a pair share
    the snapshot's stamp. The first camera alternates round-robin between pairs.
    """

    def __init__(self, cameras: tuple[str, ...]) -> None:
        if not cameras:
            raise ValueError("at least one camera is required")
        self._cameras = tuple(cameras)
        self._tick = 0
        self._snapshot: RenderSnapshot | None = None
        self._order: tuple[str, ...] = ()
        self._index = 0

    @property
    def pending_snapshot(self) -> RenderSnapshot | None:
        return self._snapshot

    def clear(self) -> None:
        """Drop the pending pair (used by the reset quiesce)."""
        self._snapshot = None
        self._order = ()
        self._index = 0

    def next(
        self, latest_snapshot: RenderSnapshot, submit: Callable[[str, RenderSnapshot], bool]
    ) -> int:
        """Try to submit pending cameras; return how many were accepted.

        ``submit`` returns True when the coordinator accepted the render. On
        False (busy, stale, or reset) the same camera is retried on the next
        call, so it cannot be skipped permanently.
        """
        if self._snapshot is None:
            self._snapshot = latest_snapshot
            self._order = round_robin_order(self._cameras, self._tick)
            self._tick += 1
            self._index = 0
        submitted = 0
        while self._index < len(self._order):
            camera = self._order[self._index]
            if not submit(camera, self._snapshot):
                break
            self._index += 1
            submitted += 1
        if self._index >= len(self._order):
            self.clear()
        return submitted


class RenderCoordinator:
    def __init__(self, backend: SimBackend) -> None:
        self._backend = backend
        self._condition = threading.Condition()
        self._request: tuple[str, RenderSnapshot, FrameCallback | None] | None = None
        self._result: CameraFrame | None = None
        self._error: BaseException | None = None
        self._busy = False
        self._reset_in_progress = False
        self._stop = False
        self._generation = 0
        self.published = 0
        self.skipped_busy = 0
        self.skipped_reset = 0
        self.skipped_stale = 0
        self.failed = 0
        self.published_by_camera: dict[str, int] = {}
        self.skipped_busy_by_camera: dict[str, int] = {}
        self._thread = threading.Thread(target=self._worker, name="sim2real-render", daemon=True)
        self._thread.start()

    @property
    def render_in_flight(self) -> bool:
        with self._condition:
            return self._busy

    @property
    def reset_in_progress(self) -> bool:
        with self._condition:
            return self._reset_in_progress

    @property
    def generation(self) -> int:
        """Scene generation; incremented by every ``begin_reset``."""
        with self._condition:
            return self._generation

    def tag(self, snapshot: RenderSnapshot) -> RenderSnapshot:
        """Stamp a freshly captured snapshot with the current generation."""
        with self._condition:
            return replace(snapshot, generation=self._generation)

    def stats(self) -> dict[str, int]:
        with self._condition:
            return {
                "published": self.published,
                "skipped_busy": self.skipped_busy,
                "skipped_reset": self.skipped_reset,
                "skipped_stale": self.skipped_stale,
                "failed": self.failed,
                "published_by_camera": dict(self.published_by_camera),
                "skipped_busy_by_camera": dict(self.skipped_busy_by_camera),
            }

    def submit(self, camera: str, snapshot: RenderSnapshot, on_frame: FrameCallback) -> bool:
        """Queue one render without blocking; return False if busy/resetting/stale."""
        with self._condition:
            if self._reset_in_progress:
                self.skipped_reset += 1
                return False
            if snapshot.generation != self._generation:
                self.skipped_stale += 1
                return False
            if self._busy:
                self.skipped_busy += 1
                self.skipped_busy_by_camera[camera] = self.skipped_busy_by_camera.get(camera, 0) + 1
                return False
            self._busy = True
            self._request = (camera, snapshot, on_frame)
            self._result = None
            self._error = None
            self._condition.notify_all()
            return True

    def render(self, camera: str, snapshot: RenderSnapshot) -> CameraFrame | None:
        """Blocking render for tests and direct callers (not the ROS timer)."""
        with self._condition:
            if self._reset_in_progress:
                self.skipped_reset += 1
                return None
            if snapshot.generation != self._generation:
                self.skipped_stale += 1
                return None
            if self._busy:
                self.skipped_busy += 1
                self.skipped_busy_by_camera[camera] = self.skipped_busy_by_camera.get(camera, 0) + 1
                return None
            self._busy = True
            self._request = (camera, snapshot, None)
            self._result = None
            self._error = None
            self._condition.notify_all()
            while self._result is None and self._error is None:
                self._condition.wait()
            frame = self._result
            error = self._error
            self._request = None
            self._result = None
            self._error = None
            self._busy = False
            self._condition.notify_all()
        if error is not None:
            with self._condition:
                self.failed += 1
            raise error
        with self._condition:
            self.published += 1
            self.published_by_camera[camera] = self.published_by_camera.get(camera, 0) + 1
        return frame

    def _worker(self) -> None:
        while True:
            with self._condition:
                while self._request is None and not self._stop:
                    self._condition.wait()
                if self._stop:
                    return
                camera, snapshot, callback = self._request
            try:
                frame = self._backend.render_from_snapshot(camera, snapshot)
            except BaseException as exc:  # callback path has no caller to raise to
                with self._condition:
                    self._error = exc
                    self._request = None
                    self._busy = False
                    if callback is not None:
                        self.failed += 1
                        self._error = None
                    self._condition.notify_all()
                continue

            if callback is None:
                with self._condition:
                    # Clear the request before waking the blocking caller;
                    # otherwise the worker loop sees it still set and renders
                    # the same snapshot a second time.
                    self._request = None
                    self._result = frame
                    self._condition.notify_all()
                continue

            try:
                callback(frame)
            except BaseException:
                with self._condition:
                    self.failed += 1
            else:
                with self._condition:
                    self.published += 1
                    self.published_by_camera[camera] = self.published_by_camera.get(camera, 0) + 1
            finally:
                with self._condition:
                    self._request = None
                    self._busy = False
                    self._condition.notify_all()

    def begin_reset(self) -> None:
        """Block new submissions, invalidate old snapshots, wait for a render."""
        with self._condition:
            self._generation += 1
            self._reset_in_progress = True
            while self._busy:
                self._condition.wait()

    def end_reset(self) -> None:
        with self._condition:
            self._reset_in_progress = False
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        self._thread.join(timeout=5.0)
