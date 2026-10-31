"""Thread-safe rendering gate for the ROS node.

MuJoCo's EGL context is thread-bound, so all rendering runs on one dedicated
worker thread. The camera callback submits a snapshot and waits for that render;
if a render is already in flight, the call returns immediately and is counted as
skipped. There is no queue: at most one snapshot is ever pending.

``begin_reset`` blocks new renders and waits for the worker to finish before a
reset mutates the scene.
"""

from __future__ import annotations

import threading

from sim2real.sim.backend import CameraFrame, RenderSnapshot, SimBackend


class RenderCoordinator:
    def __init__(self, backend: SimBackend) -> None:
        self._backend = backend
        self._condition = threading.Condition()
        self._request: tuple[str, RenderSnapshot] | None = None
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

    def render(self, camera: str, snapshot: RenderSnapshot) -> CameraFrame | None:
        """Render on the worker thread, or return ``None`` if busy/resetting."""
        with self._condition:
            if self._reset_in_progress:
                self.skipped_reset += 1
                return None
            if self._busy:
                self.skipped_busy += 1
                return None
            self._busy = True
            self._request = (camera, snapshot)
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
                camera, snapshot = self._request
            try:
                frame = self._backend.render_from_snapshot(camera, snapshot)
            except BaseException as exc:  # propagate to the waiting caller
                with self._condition:
                    self._error = exc
                    self._condition.notify_all()
            else:
                with self._condition:
                    self._result = frame
                    self._condition.notify_all()

    def begin_reset(self) -> None:
        """Block new renders and wait for any in-flight render to finish."""
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
