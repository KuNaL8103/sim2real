"""Small, pure summaries for live inference status.

These helpers exist so the latency budget is measured from per-call samples, not
from 1 Hz status snapshots (which are biased toward long calls because a long
call occupies more wall time).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def summarize(values: Sequence[float]) -> dict[str, float | int | None]:
    """Return count/min/median/p95/max for finite values."""
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return {
            "samples": 0,
            "min": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    count = len(finite)
    median = (
        finite[count // 2] if count % 2 else 0.5 * (finite[count // 2 - 1] + finite[count // 2])
    )
    rank = max(0, min(count - 1, int(math.ceil(0.95 * count)) - 1))
    return {
        "samples": count,
        "min": finite[0],
        "median": median,
        "p95": finite[rank],
        "max": finite[-1],
    }


class LatencyTracker:
    """Bounded per-call latency samples."""

    def __init__(self, max_samples: int = 500) -> None:
        if max_samples < 1:
            raise ValueError("max_samples must be >= 1")
        self._values: list[float] = []
        self._max_samples = int(max_samples)

    def add(self, latency_s: float) -> None:
        value = float(latency_s)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"latency must be finite and >= 0, got {latency_s!r}")
        self._values.append(value)
        if len(self._values) > self._max_samples:
            del self._values[: len(self._values) - self._max_samples]

    def summary(self) -> dict[str, float | int | None]:
        return summarize(self._values)

    def __len__(self) -> int:
        return len(self._values)


def future_fraction(time_from_start_ns: Iterable[int], elapsed_ns: int) -> tuple[int, int, float]:
    """Fraction of chunk points strictly after ``elapsed_ns``.

    A point exactly at ``elapsed_ns`` is current (the trim boundary keeps it),
    not future, so it counts in the denominator but not the numerator.
    """
    offsets = [int(offset) for offset in time_from_start_ns]
    total = len(offsets)
    if total == 0:
        return 0, 0, 0.0
    future = sum(1 for offset in offsets if offset > int(elapsed_ns))
    return future, total, future / total


__all__ = ["LatencyTracker", "future_fraction", "summarize"]
