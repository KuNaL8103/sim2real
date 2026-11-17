"""One-thread asynchronous inference worker with a latest-observation mailbox.

The measured real policy call on the throttled local GPU is about 1.3 s, far
longer than the 0.1 s policy period. A live ROS node must therefore never call
the policy from an executor callback. Callbacks copy data into the mailbox and
return; the single worker thread owns all server I/O.

There is no unbounded queue: the mailbox holds at most one *waiting*
observation. A newer submission replaces it. The worker reports measured
latency, monotonically increasing request ids, and a bounded restart policy.
"""

from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from sim2real.inference.chunk import select_actions

STARTING = "STARTING"
READY = "READY"
RUNNING = "RUNNING"
MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
FAILED = "FAILED"
STOPPED = "STOPPED"


@dataclass(frozen=True)
class InferenceRequest:
    request_id: int
    observation: Any
    stamp_ns: int
    observation_received_ns: int
    enqueued_ns: int


@dataclass(frozen=True)
class InferenceResult:
    request_id: int
    observation_stamp_ns: int
    observation_received_ns: int
    sent_ns: int
    received_ns: int
    latency_s: float
    actions: np.ndarray
    step_period_s: float

    def elapsed_since_observation_s(self) -> float:
        """Wall time from observation assembly to result receipt."""
        return (self.received_ns - self.observation_received_ns) / 1_000_000_000.0

    def equivalent_now_ns(self) -> int:
        """Observation-clock time corresponding to ``received_ns``.

        ROS header stamps may use a simulator clock that starts at zero, while
        ``time.monotonic_ns`` is unrelated. Adding the measured elapsed time to
        the observation stamp gives a same-clock "now" without assuming the
        message clock is the wall clock.
        """
        return int(self.observation_stamp_ns + (self.received_ns - self.observation_received_ns))


class AsyncInferenceWorker:
    """Latest-wins single-slot worker around a policy client."""

    def __init__(
        self,
        policy_factory: Any,
        *,
        call_timeout_s: float,
        max_restarts: int = 3,
        restart_backoff_s: float = 0.5,
        sample_seed: int = -1,
        stale_after_s: float | None = None,
        max_chunk_length: int = 100,
        on_result: Any | None = None,
        on_state: Any | None = None,
        logger: Any | None = None,
    ) -> None:
        if call_timeout_s <= 0:
            raise ValueError("call_timeout_s must be > 0")
        if max_restarts < 0:
            raise ValueError("max_restarts must be >= 0")
        if restart_backoff_s < 0:
            raise ValueError("restart_backoff_s must be >= 0")
        self._policy_factory = policy_factory
        self._call_timeout_s = float(call_timeout_s)
        self._max_restarts = int(max_restarts)
        self._restart_backoff_s = float(restart_backoff_s)
        self._sample_seed = int(sample_seed)
        self._stale_after_s = None if stale_after_s is None else float(stale_after_s)
        self._max_chunk_length = int(max_chunk_length)
        self._on_result = on_result
        self._on_state = on_state
        self._logger = logger

        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._mailbox: InferenceRequest | None = None
        self._policy: Any | None = None
        self._needs_reset = True
        self._state = STARTING
        self._detail: str | None = "worker not started"
        self._last_error: str | None = None
        self._request_counter = 0
        self._in_flight: int | None = None
        self._failures_since_success = 0
        self._terminal = False
        self._started_ns: int | None = None
        self._time_to_ready_s: float | None = None
        self._counters = {
            "submitted": 0,
            "replaced": 0,
            "completed": 0,
            "failed_calls": 0,
            "timed_out_calls": 0,
            "start_failures": 0,
            "restarts": 0,
            "skipped_stale": 0,
            "rejected_after_stop": 0,
            "max_waiting": 0,
        }

    # ------------------------------------------------------------------ state
    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def detail(self) -> str | None:
        with self._lock:
            return self._detail

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    @property
    def in_flight(self) -> int | None:
        with self._lock:
            return self._in_flight

    @property
    def request_id(self) -> int:
        with self._lock:
            return self._request_counter

    def stats(self) -> dict[str, Any]:
        with self._lock:
            data = dict(self._counters)
            data.update(
                {
                    "state": self._state,
                    "detail": self._detail,
                    "last_error": self._last_error,
                    "request_id": self._request_counter,
                    "in_flight": self._in_flight,
                    "mailbox_waiting": 0 if self._mailbox is None else 1,
                    "time_to_ready_s": self._time_to_ready_s,
                }
            )
            return data

    def _set_state(self, state: str, detail: str | None = None) -> None:
        with self._lock:
            self._state = state
            self._detail = detail
        if self._on_state is not None:
            try:
                self._on_state(state, detail)
            except Exception as exc:  # noqa: BLE001 - telemetry must not kill the worker
                self._log(f"on_state callback failed: {exc}")

    def _log(self, message: str) -> None:
        if self._logger is not None:
            with contextlib.suppress(Exception):
                self._logger(message)

    # --------------------------------------------------------------- lifecycle
    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._started_ns = time.monotonic_ns()
            self._thread = threading.Thread(
                target=self._run, name="vla_inference_worker", daemon=True
            )
            self._thread.start()

    def shutdown(self, timeout_s: float = 10.0) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        policy = self._policy
        if policy is not None:
            abort = getattr(policy, "abort", None)
            if callable(abort):
                with contextlib.suppress(Exception):
                    abort()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout_s)
            if thread.is_alive():
                with self._lock:
                    self._last_error = "worker thread did not stop within the shutdown timeout"
                self._log(self._last_error)

    # ---------------------------------------------------------------- submission
    def submit(
        self,
        observation: Any,
        *,
        stamp_ns: int | None = None,
        received_ns: int | None = None,
    ) -> int | None:
        with self._condition:
            if self._stop_event.is_set() or self._state == FAILED:
                self._counters["rejected_after_stop"] += 1
                return None
            self._request_counter += 1
            request_id = self._request_counter
            now = time.monotonic_ns() if received_ns is None else int(received_ns)
            request = InferenceRequest(
                request_id=request_id,
                observation=observation,
                stamp_ns=int(observation.stamp_ns if stamp_ns is None else stamp_ns),
                observation_received_ns=now,
                enqueued_ns=time.monotonic_ns(),
            )
            if self._mailbox is not None:
                self._counters["replaced"] += 1
            self._mailbox = request
            self._counters["submitted"] += 1
            self._counters["max_waiting"] = max(self._counters["max_waiting"], 1)
            self._condition.notify()
            return request_id

    def _next_request(self) -> InferenceRequest | None:
        with self._condition:
            while self._mailbox is None and not self._stop_event.is_set():
                self._condition.wait(timeout=0.5)
            if self._stop_event.is_set():
                return None
            request = self._mailbox
            self._mailbox = None
            return request

    # -------------------------------------------------------------------- thread
    def _run(self) -> None:
        self._set_state(STARTING, "starting policy backend")
        try:
            while not self._stop_event.is_set() and not self._terminal:
                if self._policy is None and not self._ensure_policy():
                    break
                request = self._next_request()
                if request is None:
                    break
                if self._is_stale(request):
                    with self._lock:
                        self._counters["skipped_stale"] += 1
                    continue
                with self._lock:
                    self._in_flight = request.request_id
                self._set_state(RUNNING, f"request {request.request_id}")
                try:
                    result = self._invoke(request)
                except Exception as exc:  # noqa: BLE001 - policy failure is a state
                    self._handle_call_failure(exc)
                    continue
                finally:
                    with self._lock:
                        self._in_flight = None
                with self._lock:
                    self._counters["completed"] += 1
                    self._failures_since_success = 0
                self._set_state(READY, "waiting for the next observation")
                if self._on_result is not None:
                    try:
                        self._on_result(result)
                    except Exception as exc:  # noqa: BLE001 - node telemetry is not fatal
                        self._log(f"on_result callback failed: {exc}")
        finally:
            self._close_policy()
            if self._last_error is None:
                self._set_state(STOPPED, "worker stopped")
            elif self._state not in (FAILED,):
                self._set_state(STOPPED, self._last_error)

    def _is_stale(self, request: InferenceRequest) -> bool:
        if self._stale_after_s is None:
            return False
        age_s = (time.monotonic_ns() - request.enqueued_ns) / 1_000_000_000.0
        return age_s > self._stale_after_s

    def _ensure_policy(self) -> bool:
        while not self._stop_event.is_set() and not self._terminal:
            try:
                policy = self._policy_factory()
                self._policy = policy
                start = getattr(policy, "start", None)
                if callable(start):
                    start()
                self._needs_reset = True
                # Do not clear the failure budget here: a policy that starts but
                # fails on every act must still reach FAILED. Only a successful
                # act resets it.
                if self._time_to_ready_s is None and self._started_ns is not None:
                    self._time_to_ready_s = (
                        time.monotonic_ns() - self._started_ns
                    ) / 1_000_000_000.0
                self._set_state(READY, "policy backend ready")
                return True
            except Exception as exc:  # noqa: BLE001 - start failure is reported, not fatal
                with self._lock:
                    self._counters["start_failures"] += 1
                self._close_policy()
                if not self._schedule_restart(exc, MODEL_UNAVAILABLE):
                    return False
        return False

    def _invoke(self, request: InferenceRequest) -> InferenceResult:
        policy = self._policy
        if policy is None:
            raise RuntimeError("policy backend is not available")
        observation = request.observation
        sent_ns = time.monotonic_ns()
        if self._needs_reset:
            policy.reset(observation.task_instruction, max(self._sample_seed, 0))
            self._needs_reset = False
        chunk = policy.act(observation)
        received_ns = time.monotonic_ns()
        latency_s = (received_ns - sent_ns) / 1_000_000_000.0
        if latency_s > self._call_timeout_s:
            raise TimeoutError(f"policy call took {latency_s:.3f} s > {self._call_timeout_s:.3f} s")
        actions = np.asarray(chunk.actions, dtype=np.float64)
        # Validate shape/finiteness without dropping rows the node may select.
        select_actions(actions, 1, max_chunk_length=self._max_chunk_length)
        period = float(chunk.step_period_s)
        if not np.isfinite(period) or period <= 0:
            raise ValueError(f"policy returned an invalid step period {period!r}")
        return InferenceResult(
            request_id=request.request_id,
            observation_stamp_ns=request.stamp_ns,
            observation_received_ns=request.observation_received_ns,
            sent_ns=sent_ns,
            received_ns=received_ns,
            latency_s=latency_s,
            actions=actions,
            step_period_s=period,
        )

    def _handle_call_failure(self, exc: Exception) -> None:
        timed_out = isinstance(exc, TimeoutError) or "timed out" in str(exc).lower()
        with self._lock:
            if timed_out:
                self._counters["timed_out_calls"] += 1
            else:
                self._counters["failed_calls"] += 1
        self._close_policy()
        if self._stop_event.is_set():
            # Shutdown aborts the server to unblock a call; that is not a
            # restart-worthy failure.
            return
        self._schedule_restart(exc, MODEL_UNAVAILABLE)

    def _schedule_restart(self, exc: Exception, state: str) -> bool:
        with self._lock:
            self._failures_since_success += 1
            self._last_error = f"{type(exc).__name__}: {exc}"
            failures = self._failures_since_success
        if failures > self._max_restarts:
            self._terminal = True
            with self._condition:
                self._mailbox = None
            self._set_state(FAILED, self._last_error)
            return False
        with self._lock:
            self._counters["restarts"] += 1
        delay_s = min(self._restart_backoff_s * (2 ** (failures - 1)), 60.0)
        self._set_state(state, self._last_error)
        if delay_s > 0 and self._stop_event.wait(delay_s):
            return False
        return not self._stop_event.is_set()

    def _close_policy(self) -> None:
        policy = self._policy
        self._policy = None
        self._needs_reset = True
        if policy is None:
            return
        abort = getattr(policy, "abort", None)
        close = getattr(policy, "close", None)
        try:
            if callable(abort):
                abort()
            elif callable(close):
                close()
        except Exception as exc:  # noqa: BLE001 - do not let cleanup mask the failure
            self._log(f"policy cleanup failed: {exc}")


__all__ = [
    "AsyncInferenceWorker",
    "FAILED",
    "InferenceRequest",
    "InferenceResult",
    "MODEL_UNAVAILABLE",
    "READY",
    "RUNNING",
    "STARTING",
    "STOPPED",
]
