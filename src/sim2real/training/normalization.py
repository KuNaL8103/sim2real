"""Per-feature normalization that mirrors LeRobot's own arithmetic.

The real SmolVLA path uses LeRobot's preprocessor/postprocessor and the
dataset's authoritative ``meta/stats.json``.  This module exists for the fake
adapter, offline validation, and tests.  It is never applied on top of LeRobot's
normalization for the same tensor.

The formulas are copied from LeRobot 0.6.1
``lerobot/processor/normalize_processor.py``:

* ``MEAN_STD`` forward ``(x - mean) / (std + eps)``, inverse ``y * std + mean``
  (the inverse deliberately uses ``std``, matching LeRobot);
* ``QUANTILES`` forward ``2 * (x - q01) / (q99 - q01) - 1``, with a zero range
  replaced by ``eps``; inverse ``(y + 1) * range / 2 + q01``;
* ``IDENTITY`` returns the input.

No clipping is applied by default because LeRobot does not clip.  Callers may
opt in to a symmetric clip range; the policy is recorded in the returned object.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_EPS = 1e-8
MODES = ("identity", "mean_std", "quantile")


class NormalizationError(ValueError):
    """Raised for an unknown mode, missing stats, or an embodiment mismatch."""


@dataclass(frozen=True)
class FeatureStats:
    """The statistics needed by one normalization mode."""

    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    minimum: np.ndarray | None = None
    maximum: np.ndarray | None = None
    q01: np.ndarray | None = None
    q99: np.ndarray | None = None
    count: int | None = None

    @classmethod
    def from_mapping(cls, entry: dict[str, Any]) -> FeatureStats:
        if not isinstance(entry, dict):
            raise NormalizationError("stats: expected a mapping per feature")

        def array(name: str) -> np.ndarray | None:
            value = entry.get(name)
            if value is None:
                return None
            result = np.asarray(value, dtype=np.float64)
            if not np.all(np.isfinite(result)):
                raise NormalizationError(f"stats.{name}: contains NaN or infinity")
            return result

        count_raw = entry.get("count")
        count = None
        if count_raw is not None:
            values = np.asarray(count_raw).reshape(-1)
            count = int(values[0]) if values.size else None
        return cls(
            mean=array("mean"),
            std=array("std"),
            minimum=array("min"),
            maximum=array("max"),
            q01=array("q01"),
            q99=array("q99"),
            count=count,
        )

    def dimension(self, mode: str) -> int:
        if mode == "mean_std":
            if self.mean is None or self.std is None:
                raise NormalizationError("mean_std: mean and std are required")
            return int(self.mean.size)
        if mode == "quantile":
            if self.q01 is None or self.q99 is None:
                raise NormalizationError("quantile: q01 and q99 are required")
            return int(self.q01.size)
        return 0


@dataclass(frozen=True)
class Normalizer:
    """Apply or invert one feature's normalization."""

    mode: str
    stats: FeatureStats
    eps: float = DEFAULT_EPS
    clip: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise NormalizationError(f"mode: expected one of {list(MODES)}")
        if self.eps <= 0:
            raise NormalizationError("eps: expected > 0")
        if self.clip is not None and self.clip <= 0:
            raise NormalizationError("clip: expected > 0 or None")

    def _check_dimension(self, values: np.ndarray) -> None:
        dimension = self.stats.dimension(self.mode)
        if self.mode != "identity" and values.shape[-1] != dimension:
            raise NormalizationError(
                f"stats dimension {dimension} does not match values shape {values.shape}"
            )

    def apply(self, values: Any) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(array)):
            raise NormalizationError("values: contains NaN or infinity")
        self._check_dimension(array)
        if self.mode == "identity":
            return array.copy()
        if self.mode == "mean_std":
            assert self.stats.mean is not None and self.stats.std is not None
            result = (array - self.stats.mean) / (self.stats.std + self.eps)
        else:
            assert self.stats.q01 is not None and self.stats.q99 is not None
            span = self.stats.q99 - self.stats.q01
            span = np.where(span == 0.0, self.eps, span)
            result = 2.0 * (array - self.stats.q01) / span - 1.0
        if self.clip is not None:
            result = np.clip(result, -self.clip, self.clip)
        return result

    def invert(self, values: Any) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(array)):
            raise NormalizationError("values: contains NaN or infinity")
        self._check_dimension(array)
        if self.mode == "identity":
            return array.copy()
        if self.mode == "mean_std":
            assert self.stats.mean is not None and self.stats.std is not None
            return array * self.stats.std + self.stats.mean
        assert self.stats.q01 is not None and self.stats.q99 is not None
        span = self.stats.q99 - self.stats.q01
        span = np.where(span == 0.0, self.eps, span)
        return (array + 1.0) * span / 2.0 + self.stats.q01


class NormalizationBundle:
    """Normalizers keyed by ``(embodiment_id, feature)``.

    Keeping the embodiment id in the key makes mixing statistics from different
    embodiments impossible by construction: a lookup for another embodiment
    raises instead of silently reusing the wrong mean/std.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], Normalizer] = {}

    def add(self, embodiment_id: str, feature: str, normalizer: Normalizer) -> None:
        if not embodiment_id or not feature:
            raise NormalizationError("embodiment_id and feature must be non-empty")
        key = (embodiment_id, feature)
        if key in self._entries:
            raise NormalizationError(f"duplicate normalizer for {key}")
        self._entries[key] = normalizer

    def require(self, embodiment_id: str, feature: str) -> Normalizer:
        key = (embodiment_id, feature)
        if key not in self._entries:
            available = sorted({emb for emb, _feature in self._entries})
            raise NormalizationError(
                f"no normalization for embodiment {embodiment_id!r} / feature "
                f"{feature!r}; known embodiments: {available}"
            )
        return self._entries[key]

    def apply(self, embodiment_id: str, feature: str, values: Any) -> np.ndarray:
        return self.require(embodiment_id, feature).apply(values)

    def invert(self, embodiment_id: str, feature: str, values: Any) -> np.ndarray:
        return self.require(embodiment_id, feature).invert(values)

    def features(self, embodiment_id: str) -> list[str]:
        return sorted(feature for emb, feature in self._entries if emb == embodiment_id)


def bundle_from_stats(
    stats: dict[str, dict[str, Any]],
    *,
    embodiment_id: str,
    modes: dict[str, str],
    eps: float = DEFAULT_EPS,
    clip: float | None = None,
) -> NormalizationBundle:
    """Build a bundle from a LeRobot-style ``stats`` mapping.

    ``modes`` maps a feature name to ``identity``/``mean_std``/``quantile``.
    """
    if not embodiment_id:
        raise NormalizationError("embodiment_id must be non-empty")
    bundle = NormalizationBundle()
    for feature, mode in modes.items():
        if mode == "identity":
            feature_stats = FeatureStats()
        else:
            if feature not in stats:
                raise NormalizationError(f"stats: missing feature {feature!r}")
            feature_stats = FeatureStats.from_mapping(stats[feature])
        bundle.add(embodiment_id, feature, Normalizer(mode, feature_stats, eps=eps, clip=clip))
    return bundle


def constant_dimension_safe(values: Any, eps: float = DEFAULT_EPS) -> np.ndarray:
    """Return a denominator with zero/constant dimensions replaced by ``eps``."""
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise NormalizationError("values: contains NaN or infinity")
    return np.where(np.abs(array) < eps, eps, array)


def seconds_horizon(chunk_size: int, fps: float) -> float:
    """Chunk horizon in seconds; kept here so callers do not divide by a null fps."""
    if chunk_size < 1 or not math.isfinite(fps) or fps <= 0:
        raise NormalizationError("chunk_size must be >= 1 and fps > 0")
    return float(chunk_size) / float(fps)


__all__ = [
    "DEFAULT_EPS",
    "MODES",
    "FeatureStats",
    "NormalizationBundle",
    "NormalizationError",
    "Normalizer",
    "bundle_from_stats",
    "constant_dimension_safe",
    "seconds_horizon",
]
