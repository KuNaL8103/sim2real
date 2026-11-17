"""Stale-chunk policies and the controller command time convention.

The controller documentation (Jazzy, fetched 2026-11-17) says a zero
``header.stamp`` means "start now", and the trajectory's ``time_from_start`` is
relative to that start.  The safety node therefore emits a zero header stamp
and point ``j`` at ``(j+1) * command_period_s``.  This differs from the raw
chunk's ``k*dt`` from the observation stamp and from the evaluation's linear
execution at ``observation + k*dt``: after inference latency the original
schedule is partly in the past, so the remaining targets are replayed from
"now" rather than pretending the observation just happened.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SafetyConfig
from .types import STALE_CHUNK


@dataclass(frozen=True)
class StaleSelection:
    ok: bool
    reason: str | None
    positions: np.ndarray | None
    times_s: tuple[float, ...]
    dropped_points: int
    detail: str = ""


def select_stale_points(
    positions: np.ndarray,
    offsets_s: tuple[float, ...],
    age_s: float,
    config: SafetyConfig,
) -> StaleSelection:
    """Apply the configured stale policy to raw chunk points.

    Points at or before the current age are dropped.  This always drops the
    raw ``k=0`` point once any measurable time has passed, matching the Task
    10b observation that a real chunk's first point is usually already gone.

    ``reject`` rejects the whole chunk when every point is past or the age
    exceeds ``max_chunk_age_s``; otherwise it forwards the surviving points at
    their **original** offsets (a small delay, never a silent speed-up).
    ``retime_if_continuous`` instead asks for the surviving points to be
    replayed from now as ``(j+1)*dt``; the caller still applies the continuity
    gate and all limit checks to that final trajectory.
    """
    values = np.asarray(positions, dtype=np.float64)
    offsets = tuple(float(value) for value in offsets_s)
    if values.shape[0] != len(offsets):
        raise ValueError("positions and offsets must have the same length")
    age = 0.0 if not np.isfinite(age_s) or age_s < 0.0 else float(age_s)

    if age > config.max_chunk_age_s:
        return StaleSelection(
            False,
            STALE_CHUNK,
            None,
            (),
            0,
            f"chunk age {age:.4f} s > {config.max_chunk_age_s} s",
        )

    keep = [index for index, offset in enumerate(offsets) if offset > age]
    dropped = len(offsets) - len(keep)
    if not keep:
        return StaleSelection(False, STALE_CHUNK, None, (), dropped, "all points are past")

    first, last = keep[0], keep[-1] + 1
    if config.stale_chunk_policy == "reject":
        selected = values[first:last]
        times = tuple(offsets[first:last])
    else:  # retime_if_continuous
        selected = values[first:last]
        times = tuple((index + 1) * config.command_period_s for index in range(selected.shape[0]))
    return StaleSelection(True, None, np.array(selected, copy=True), times, dropped)


__all__ = ["StaleSelection", "select_stale_points"]
