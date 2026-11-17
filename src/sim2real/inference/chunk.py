"""Action-chunk validation, selection, timing, and latency trimming.

The policy server returns de-normalized absolute joint targets from LeRobot's
own postprocessor. This module never normalizes or de-normalizes again; it only
checks the array, selects the first ``n_action_steps``, and maps them to a
ROS-independent trajectory description.

Time convention (Task 10 decision): action ``k`` is applied at
``observation_stamp + k * policy_period_s``, with ``k = 0`` the first action.
This matches the evaluation runner's default ``linear`` execution, where the
chunk's first action is used immediately after the observation and consecutive
actions are linearly interpolated. A ``JointTrajectory`` controller normally
places point ``k`` at ``(k + 1) * dt``; Task 11 owns that retiming decision. The
raw chunk keeps the observation stamp and per-point offsets, so the information
needed to retime is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim2real import contracts

MAX_ACTION_DIM = 8
FINGER_MIN_M = 0.0
FINGER_MAX_M = 0.04
DEFAULT_FRAME_ID = ""  # joint trajectories are frame-agnostic


class ChunkError(ValueError):
    """A malformed action chunk."""


@dataclass(frozen=True)
class TrajectoryChunk:
    """ROS-independent description of one raw action chunk."""

    stamp_ns: int
    step_period_s: float
    joint_names: tuple[str, ...]
    positions: np.ndarray  # (n, 8) float64
    time_from_start_ns: tuple[int, ...]
    frame_id: str = DEFAULT_FRAME_ID

    @property
    def point_count(self) -> int:
        return int(self.positions.shape[0])


def select_actions(
    actions: object,
    n_action_steps: int,
    *,
    max_action_dim: int = MAX_ACTION_DIM,
    max_chunk_length: int = 100,
) -> np.ndarray:
    """Validate a server chunk and return the first ``n_action_steps`` rows."""
    if isinstance(n_action_steps, bool) or not isinstance(n_action_steps, int):
        raise ChunkError(f"n_action_steps must be an integer, got {n_action_steps!r}")
    if n_action_steps < 1:
        raise ChunkError(f"n_action_steps must be >= 1, got {n_action_steps}")
    if n_action_steps > max_chunk_length:
        raise ChunkError(
            f"n_action_steps {n_action_steps} exceeds max_chunk_length {max_chunk_length}"
        )
    try:
        array = np.asarray(actions, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ChunkError(f"actions are not numeric: {exc}") from exc
    if array.ndim != 2:
        raise ChunkError(f"actions must be 2-D, got shape {array.shape}")
    if array.shape[1] != max_action_dim:
        raise ChunkError(f"actions must have {max_action_dim} columns, got {array.shape[1]}")
    if array.shape[0] < 1:
        raise ChunkError("actions must contain at least one row")
    if array.shape[0] > max_chunk_length:
        raise ChunkError(
            f"actions have {array.shape[0]} rows > max_chunk_length {max_chunk_length}"
        )
    if not np.all(np.isfinite(array)):
        raise ChunkError("actions contain NaN or infinity")
    if array.shape[0] < n_action_steps:
        raise ChunkError(f"chunk has {array.shape[0]} rows but n_action_steps={n_action_steps}")
    return np.array(array[:n_action_steps], dtype=np.float64, copy=True)


def make_chunk(
    actions: object,
    *,
    stamp_ns: int,
    step_period_s: float,
    n_action_steps: int,
    max_chunk_length: int = 100,
    frame_id: str = DEFAULT_FRAME_ID,
) -> TrajectoryChunk:
    if isinstance(stamp_ns, bool) or not isinstance(stamp_ns, int):
        raise ChunkError(f"stamp_ns must be an integer, got {stamp_ns!r}")
    if stamp_ns < 0:
        raise ChunkError(f"stamp_ns must be >= 0, got {stamp_ns}")
    period = float(step_period_s)
    if not np.isfinite(period) or period <= 0:
        raise ChunkError(f"step_period_s must be finite and > 0, got {step_period_s!r}")
    selected = select_actions(actions, n_action_steps, max_chunk_length=max_chunk_length)
    offsets = tuple(int(round(k * period * 1_000_000_000)) for k in range(len(selected)))
    return TrajectoryChunk(
        stamp_ns=int(stamp_ns),
        step_period_s=period,
        joint_names=tuple(contracts.CONTROLLED_JOINT_NAMES),
        positions=selected,
        time_from_start_ns=offsets,
        frame_id=frame_id,
    )


def trim_for_latency(chunk: TrajectoryChunk, now_ns: int) -> TrajectoryChunk | None:
    """Drop points whose target time has already passed.

    The chunk's ``stamp_ns`` is deliberately not rebased: a consumer must see the
    true age. A point exactly at ``now_ns`` is kept (it has not passed yet).
    Returns ``None`` when no useful point remains.
    """
    if isinstance(now_ns, bool) or not isinstance(now_ns, int):
        raise ChunkError(f"now_ns must be an integer, got {now_ns!r}")
    elapsed_ns = int(now_ns) - int(chunk.stamp_ns)
    if elapsed_ns <= 0:
        return chunk
    keep = [index for index, offset in enumerate(chunk.time_from_start_ns) if offset >= elapsed_ns]
    if not keep:
        return None
    first = keep[0]
    last = keep[-1] + 1
    positions = chunk.positions[first:last]
    offsets = chunk.time_from_start_ns[first:last]
    return TrajectoryChunk(
        stamp_ns=chunk.stamp_ns,
        step_period_s=chunk.step_period_s,
        joint_names=chunk.joint_names,
        positions=np.array(positions, dtype=np.float64, copy=True),
        time_from_start_ns=tuple(offsets),
        frame_id=chunk.frame_id,
    )


def finger_out_of_range(chunk: TrajectoryChunk) -> tuple[int, ...]:
    """Rows whose gripper target is outside the dataset's ``[0, 0.04]`` m.

    The raw node passes these through; the Task 11 safety layer owns clipping.
    """
    gripper = chunk.positions[:, 7]
    outside = (gripper < FINGER_MIN_M) | (gripper > FINGER_MAX_M)
    return tuple(int(index) for index in np.flatnonzero(outside))


__all__ = [
    "ChunkError",
    "DEFAULT_FRAME_ID",
    "FINGER_MAX_M",
    "FINGER_MIN_M",
    "MAX_ACTION_DIM",
    "TrajectoryChunk",
    "finger_out_of_range",
    "make_chunk",
    "select_actions",
    "trim_for_latency",
]
