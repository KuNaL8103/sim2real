"""Trajectory filtering: validation, metrics, decisions, and curated output."""

from __future__ import annotations

from sim2real.filtering.config import FilterConfig, FilterConfigError, load_filter_config
from sim2real.filtering.decision import FilterResult, decide
from sim2real.filtering.metrics import (
    DegenerateTrajectoryError,
    TrajectoryMetrics,
    compute_metrics,
)
from sim2real.filtering.pipeline import FilterPipeline
from sim2real.filtering.validation import ValidationIssue, validate_episode

__all__ = [
    "DegenerateTrajectoryError",
    "FilterConfig",
    "FilterConfigError",
    "FilterPipeline",
    "FilterResult",
    "TrajectoryMetrics",
    "ValidationIssue",
    "compute_metrics",
    "decide",
    "load_filter_config",
    "validate_episode",
]
