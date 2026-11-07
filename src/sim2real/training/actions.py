"""Action-chunk encoding with dimension padding and end-of-episode masks.

The real SmolVLA path uses LeRobot's own dataset batching.  This module is the
NumPy-only reference used by the fake adapter, offline validation, and tests.
It mirrors the verified LeRobot semantics:

* state/action vectors are zero-padded to ``max_state_dim``/``max_action_dim``;
* a chunk past the episode end repeats the last action and sets ``is_pad``;
* a chunk time span is tracked in **seconds** (``chunk_size / fps``), so sources
  at 5 Hz and 10 Hz are never compared as if they had the same horizon.

Horizon is a property of the chunk, not of the episode.  ``decode_chunk`` strips
the padding back to the native action dimension exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class ActionChunkError(ValueError):
    """Raised for an invalid episode, chunk, or padding request."""


def _series(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ActionChunkError(f"{name}: expected a non-empty (T, D) array")
    if not np.all(np.isfinite(array)):
        raise ActionChunkError(f"{name}: contains non-finite values (NaN or infinity)")
    return array


@dataclass(frozen=True)
class ActionChunk:
    """One padded chunk plus the mask that says which steps are real."""

    actions: np.ndarray  # (chunk_size, max_action_dim), float32
    state: np.ndarray  # (max_state_dim,), float32
    action_is_pad: np.ndarray  # (chunk_size,), bool
    action_dim: int
    state_dim: int
    start_index: int
    fps: float
    horizon_seconds: float

    def __post_init__(self) -> None:
        if self.actions.ndim != 2:
            raise ActionChunkError("actions: expected a 2-D array")
        if self.actions.shape[0] != self.action_is_pad.shape[0]:
            raise ActionChunkError("action_is_pad: length must equal chunk_size")
        if self.action_dim > self.actions.shape[1]:
            raise ActionChunkError("action_dim: larger than the padded width")
        if self.state_dim > self.state.shape[0]:
            raise ActionChunkError("state_dim: larger than the padded state")
        if not math.isfinite(self.horizon_seconds) or self.horizon_seconds <= 0:
            raise ActionChunkError("horizon_seconds: expected > 0")

    @property
    def chunk_size(self) -> int:
        return int(self.actions.shape[0])

    @property
    def valid_steps(self) -> int:
        return int((~self.action_is_pad).sum())

    def decode(self) -> np.ndarray:
        """Return ``(chunk_size, action_dim)`` with padding removed."""
        return self.actions[:, : self.action_dim].copy()


def encode_chunk(
    actions: np.ndarray,
    state: np.ndarray,
    *,
    start: int,
    chunk_size: int,
    fps: float,
    max_action_dim: int,
    max_state_dim: int,
) -> ActionChunk:
    """Build one K-step chunk starting at ``start``.

    Steps past the episode end repeat the last action and are marked
    ``action_is_pad=True``.  This includes an episode shorter than the chunk.
    """
    action_array = _series(actions, "actions")
    state_array = np.asarray(state, dtype=np.float64)
    if state_array.ndim == 2:
        state_array = state_array[-1]
    if state_array.ndim != 1 or state_array.size == 0:
        raise ActionChunkError("state: expected a non-empty (D,) or (T, D) array")
    if not np.all(np.isfinite(state_array)):
        raise ActionChunkError("state: contains non-finite values (NaN or infinity)")
    if chunk_size < 1:
        raise ActionChunkError("chunk_size: expected >= 1")
    if not math.isfinite(fps) or fps <= 0:
        raise ActionChunkError("fps: expected > 0")
    if max_action_dim < action_array.shape[1]:
        raise ActionChunkError("max_action_dim: smaller than the native action dimension")
    if max_state_dim < state_array.size:
        raise ActionChunkError("max_state_dim: smaller than the native state dimension")
    if not 0 <= start < action_array.shape[0]:
        raise ActionChunkError("start: expected in [0, number of steps)")

    indices = np.arange(start, start + chunk_size)
    valid = indices < action_array.shape[0]
    safe = np.minimum(indices, action_array.shape[0] - 1)

    padded_actions = np.zeros((chunk_size, max_action_dim), dtype=np.float32)
    padded_actions[:, : action_array.shape[1]] = action_array[safe].astype(np.float32)
    padded_state = np.zeros((max_state_dim,), dtype=np.float32)
    padded_state[: state_array.size] = state_array.astype(np.float32)

    return ActionChunk(
        actions=padded_actions,
        state=padded_state,
        action_is_pad=~valid,
        action_dim=int(action_array.shape[1]),
        state_dim=int(state_array.size),
        start_index=int(start),
        fps=float(fps),
        horizon_seconds=float(chunk_size) / float(fps),
    )


def decode_chunk(chunk: ActionChunk) -> np.ndarray:
    """Decode a chunk back to native dimensions (identical to ``chunk.decode``)."""
    return chunk.decode()


def chunk_offsets(chunk: ActionChunk) -> np.ndarray:
    """Time offsets in seconds for each chunk position (``k / fps``)."""
    return np.arange(chunk.chunk_size, dtype=np.float64) / float(chunk.fps)


def horizon_seconds(chunk_size: int, fps: float) -> float:
    """Time span of a chunk in seconds, with explicit argument validation."""
    if chunk_size < 1:
        raise ActionChunkError("chunk_size: expected >= 1")
    if not math.isfinite(fps) or fps <= 0:
        raise ActionChunkError("fps: expected > 0")
    return float(chunk_size) / float(fps)


__all__ = [
    "ActionChunk",
    "ActionChunkError",
    "chunk_offsets",
    "decode_chunk",
    "encode_chunk",
    "horizon_seconds",
]
