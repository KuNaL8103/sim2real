"""Latency/future-fraction summaries used by /vla/status."""

from __future__ import annotations

import pytest

from sim2real.inference.metrics import LatencyTracker, future_fraction, summarize


def test_summarize_percentiles_and_empty():
    summary = summarize([1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary["samples"] == 5
    assert summary["min"] == 1.0
    assert summary["median"] == 3.0
    assert summary["p95"] == 5.0
    assert summary["max"] == 5.0
    empty = summarize([])
    assert empty == {"samples": 0, "min": None, "median": None, "p95": None, "max": None}


def test_latency_tracker_is_bounded_and_rejects_invalid():
    tracker = LatencyTracker(max_samples=3)
    for value in (0.1, 0.2, 0.3, 0.4):
        tracker.add(value)
    summary = tracker.summary()
    assert summary["samples"] == 3
    assert summary["min"] == pytest.approx(0.2)
    assert summary["max"] == pytest.approx(0.4)
    with pytest.raises(ValueError):
        tracker.add(float("nan"))
    with pytest.raises(ValueError):
        tracker.add(-1.0)


def test_future_fraction_boundary_and_counts():
    offsets = (0, 100_000_000, 200_000_000)
    future, total, fraction = future_fraction(offsets, 0)
    assert (future, total) == (2, 3)  # offset 0 is current, not future
    assert fraction == pytest.approx(2 / 3)
    future, total, fraction = future_fraction(offsets, 100_000_000)
    assert (future, total) == (1, 3)
    assert fraction == pytest.approx(1 / 3)
    assert future_fraction(offsets, 500_000_000) == (0, 3, 0.0)
    assert future_fraction((), 0) == (0, 0, 0.0)
