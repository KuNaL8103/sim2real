"""CPU-only fake policies for inference tests and the default launch backend.

These are deliberately not a model: they exercise the transport, worker, and
chunk paths without torch. The default fake backend holds the current state, so
a published chunk is a valid but motionless trajectory.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from sim2real.evaluation.policy_interface import ActionChunk, PolicyError, PolicyObservation

FAILURE_MODES = (None, "raise", "nan", "wrong_shape", "empty", "hang", "wrong_dim")
POLICY_MODES = ("hold", "zeros", "ramp", "fixed")


@dataclass(frozen=True)
class RawChunk:
    """An intentionally unvalidated chunk for failure-injection tests."""

    actions: Any
    step_period_s: float = 0.1


class FakeChunkPolicy:
    """Configurable chunk policy with deterministic output and failure injection."""

    name = "fake_chunk"
    needs_images = True

    def __init__(
        self,
        *,
        chunk_size: int = 50,
        step_period_s: float = 0.1,
        delay_s: float = 0.0,
        failure: str | None = None,
        hang_s: float = 2.0,
        mode: str = "hold",
        fixed_actions: Any | None = None,
        ramp_step: float = 1.0e-3,
    ) -> None:
        if chunk_size < 1:
            raise PolicyError("chunk_size must be >= 1")
        if step_period_s <= 0 or not np.isfinite(step_period_s):
            raise PolicyError("step_period_s must be finite and > 0")
        if delay_s < 0 or not np.isfinite(delay_s):
            raise PolicyError("delay_s must be finite and >= 0")
        if hang_s < 0 or not np.isfinite(hang_s):
            raise PolicyError("hang_s must be finite and >= 0")
        if failure not in FAILURE_MODES:
            raise PolicyError(f"failure must be one of {FAILURE_MODES}")
        if mode not in POLICY_MODES:
            raise PolicyError(f"mode must be one of {POLICY_MODES}")
        if mode == "fixed" and fixed_actions is None:
            raise PolicyError("mode='fixed' requires fixed_actions")
        self.chunk_size = int(chunk_size)
        self.step_period_s = float(step_period_s)
        self.delay_s = float(delay_s)
        self.failure = failure
        self.hang_s = float(hang_s)
        self.mode = mode
        self.ramp_step = float(ramp_step)
        self.fixed_actions = None if fixed_actions is None else np.asarray(fixed_actions)
        self.calls = 0
        self.last_observation: PolicyObservation | None = None
        self.info: dict[str, Any] = {
            "name": self.name,
            "chunk_size": self.chunk_size,
            "step_period_s": self.step_period_s,
            "delay_s": self.delay_s,
            "failure": self.failure,
            "mode": self.mode,
        }

    def reset(self, task_instruction: str, seed: int) -> None:
        del task_instruction, seed
        self.calls = 0

    def _actions(self, observation: PolicyObservation) -> np.ndarray:
        state = np.asarray(observation.state, dtype=np.float64)
        if state.shape != (8,):
            raise PolicyError(f"fake policy needs an 8-D state, got {state.shape}")
        if self.mode == "fixed":
            assert self.fixed_actions is not None  # checked in __init__
            return np.array(self.fixed_actions, dtype=np.float64, copy=True)
        if self.mode == "zeros":
            return np.zeros((self.chunk_size, 8), dtype=np.float64)
        if self.mode == "ramp":
            steps = np.arange(self.chunk_size, dtype=np.float64)[:, None]
            return np.tile(state, (self.chunk_size, 1)) + steps * self.ramp_step
        return np.tile(state, (self.chunk_size, 1))

    def act(self, observation: PolicyObservation) -> ActionChunk | RawChunk:
        self.last_observation = observation
        self.calls += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.failure == "hang":
            time.sleep(self.hang_s)
        if self.failure == "raise":
            raise PolicyError("injected policy failure")
        actions = self._actions(observation)
        if self.failure == "nan":
            actions = np.array(actions, dtype=np.float64)
            actions[0, 0] = np.nan
            return RawChunk(actions, self.step_period_s)
        if self.failure == "wrong_shape":
            return RawChunk(np.zeros((self.chunk_size, 7), dtype=np.float64), self.step_period_s)
        if self.failure == "wrong_dim":
            return RawChunk(np.zeros((self.chunk_size, 8, 1), dtype=np.float64), self.step_period_s)
        if self.failure == "empty":
            return RawChunk(np.zeros((0, 8), dtype=np.float64), self.step_period_s)
        return ActionChunk(actions, self.step_period_s)

    def close(self) -> None:
        return None


def fake_hold_factory(
    *,
    chunk_size: int = 50,
    step_period_s: float = 0.1,
    delay_s: float = 0.0,
) -> FakeChunkPolicy:
    return FakeChunkPolicy(
        chunk_size=chunk_size,
        step_period_s=step_period_s,
        delay_s=delay_s,
        mode="hold",
    )


__all__ = [
    "FAILURE_MODES",
    "FakeChunkPolicy",
    "POLICY_MODES",
    "RawChunk",
    "fake_hold_factory",
]
