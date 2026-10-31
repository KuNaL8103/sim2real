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

from sim2real.sim.backend import CameraFrame, RenderSnapshot, SimBackend

FrameCallback = Callable[[CameraFrame], None]


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
        self.published = 0
        self.skipped_busy = 0
        self.skipped_reset = 0
        self.failed = 0
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

    def stats(self) -> dict[str, int]:
        with self._condition:
            return {
                "published": self.published,
                "skipped_busy": self.skipped_busy,
                "skipped_reset": self.skipped_reset,
                "failed": self.failed,
            }

    def submit(self, camera: str, snapshot: RenderSnapshot, on_frame: FrameCallback) -> bool:
        """Queue one render without blocking; return False if busy/resetting."""
        with self._condition:
            if self._reset_in_progress:
                self.skipped_reset += 1
                return False
            if self._busy:
                self.skipped_busy += 1
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
            if self._busy:
                self.skipped_busy += 1
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
            finally:
                with self._condition:
                    self._request = None
                    self._busy = False
                    self._condition.notify_all()

    def begin_reset(self) -> None:
        """Block new submissions and wait for any in-flight render to finish."""
        with self._condition:
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
