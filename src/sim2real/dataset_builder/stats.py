"""Normalization statistics with streaming merge and bounded quantiles.

The output layout follows LeRobot v3: one dictionary per feature with
``mean``, ``std``, ``min``, ``max``, ``count``, and quantiles named ``q01``,
``q10``, ``q50``, ``q90``, ``q99``.  Vector features have per-dimension values;
images have per-channel values shaped ``(3, 1, 1)``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .schema import ImageFrame, UnifiedEpisode

DEFAULT_QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)
QUANTILE_KEYS = tuple(f"q{int(q * 100):02d}" for q in DEFAULT_QUANTILES)
STD_FLOOR = 1e-8


class StatsError(ValueError):
    """Raised for non-finite input or an invalid quantile configuration."""


def _values_array(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise StatsError(f"expected a 1-D or 2-D array, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise StatsError("statistics input contains NaN or infinity")
    return array


@dataclass
class RunningStats:
    """Per-dimension count, mean, std, min, max, and quantiles."""

    dim: int
    quantile_method: str = "exact"
    reservoir_size: int = 10_000
    seed: int = 0
    epsilon: float = STD_FLOOR
    count: int = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None
    minimum: np.ndarray | None = None
    maximum: np.ndarray | None = None
    _exact: list[np.ndarray] = field(default_factory=list)
    _reservoir: np.ndarray | None = None
    _seen: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise StatsError("dim must be > 0")
        if self.quantile_method not in {"exact", "reservoir"}:
            raise StatsError("quantile_method must be 'exact' or 'reservoir'")
        if self.reservoir_size <= 1:
            raise StatsError("reservoir_size must be > 1")
        self.mean = np.zeros(self.dim, dtype=np.float64)
        self.m2 = np.zeros(self.dim, dtype=np.float64)
        self.minimum = np.full(self.dim, np.inf, dtype=np.float64)
        self.maximum = np.full(self.dim, -np.inf, dtype=np.float64)
        self._rng = np.random.default_rng(self.seed)
        if self.quantile_method == "reservoir":
            self._reservoir = np.empty((0, self.dim), dtype=np.float64)

    def update(self, values: Any) -> None:
        array = _values_array(values)
        if array.shape[1] != self.dim:
            raise StatsError(f"expected {self.dim} dimensions, got {array.shape[1]}")
        if array.shape[0] == 0:
            raise StatsError("cannot update statistics with zero samples")
        n_new = array.shape[0]
        batch_mean = array.mean(axis=0)
        batch_m2 = ((array - batch_mean) ** 2).sum(axis=0)
        if self.count == 0:
            self.mean = batch_mean
            self.m2 = batch_m2
        else:
            total = self.count + n_new
            delta = batch_mean - self.mean
            self.mean = self.mean + delta * (n_new / total)
            self.m2 = self.m2 + batch_m2 + delta**2 * (self.count * n_new / total)
        self.count += n_new
        self.minimum = np.minimum(self.minimum, array.min(axis=0))
        self.maximum = np.maximum(self.maximum, array.max(axis=0))
        if self.quantile_method == "exact":
            self._exact.append(array.copy())
        else:
            self._update_reservoir(array)

    def _update_reservoir(self, array: np.ndarray) -> None:
        assert self._reservoir is not None
        for row in array:
            self._seen += 1
            if self._reservoir.shape[0] < self.reservoir_size:
                self._reservoir = np.vstack([self._reservoir, row])
            else:
                index = int(self._rng.integers(0, self._seen))
                if index < self.reservoir_size:
                    self._reservoir[index] = row

    def merge(self, other: RunningStats) -> None:
        if other.dim != self.dim:
            raise StatsError("cannot merge statistics with different dimensions")
        if other.count == 0:
            return
        if self.count == 0:
            self.count = other.count
            self.mean = other.mean.copy()
            self.m2 = other.m2.copy()
            self.minimum = other.minimum.copy()
            self.maximum = other.maximum.copy()
            if self.quantile_method == "exact":
                self._exact = [value.copy() for value in other._exact]
            else:
                assert other._reservoir is not None
                self._reservoir = other._reservoir.copy()
                self._seen = other._seen
        else:
            total = self.count + other.count
            delta = other.mean - self.mean
            self.mean = self.mean + delta * (other.count / total)
            self.m2 = self.m2 + other.m2 + delta**2 * (self.count * other.count / total)
            self.count = total
            self.minimum = np.minimum(self.minimum, other.minimum)
            self.maximum = np.maximum(self.maximum, other.maximum)
        if self.quantile_method == "exact":
            self._exact.extend(value.copy() for value in other._exact)
        else:
            if self._reservoir is None:
                self._reservoir = np.empty((0, self.dim), dtype=np.float64)
            assert other._reservoir is not None
            for row in other._reservoir:
                self._update_reservoir(row[None, :])

    def finalize(self) -> dict[str, np.ndarray]:
        if self.count == 0:
            raise StatsError("cannot finalize statistics with no samples")
        std = np.sqrt(np.maximum(self.m2 / self.count, 0.0))
        std = np.where(std < self.epsilon, self.epsilon, std)
        if self.quantile_method == "exact":
            assert self._exact
            values = np.concatenate(self._exact, axis=0)
        else:
            assert self._reservoir is not None
            values = self._reservoir
        quantiles = np.quantile(values, DEFAULT_QUANTILES, axis=0)
        result: dict[str, np.ndarray] = {
            "min": self.minimum.copy(),
            "max": self.maximum.copy(),
            "mean": self.mean.copy(),
            "std": std,
            "count": np.array([self.count], dtype=np.int64),
        }
        for key, value in zip(QUANTILE_KEYS, quantiles, strict=True):
            result[key] = np.asarray(value, dtype=np.float64)
        return result


def compute_feature_stats(
    values: Any,
    *,
    quantile_method: str = "exact",
    reservoir_size: int = 10_000,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    array = _values_array(values)
    stats = RunningStats(
        dim=array.shape[1],
        quantile_method=quantile_method,
        reservoir_size=reservoir_size,
        seed=seed,
    )
    stats.update(array)
    return stats.finalize()


def _image_array(frame: ImageFrame) -> np.ndarray:
    array = frame.load().astype(np.float64) / 255.0
    if array.shape[2] != 3:
        raise StatsError("image statistics require three channels")
    return array


def compute_dataset_stats(
    episodes: Iterable[UnifiedEpisode],
    *,
    include_images: bool = False,
    quantile_method: str = "exact",
    reservoir_size: int = 10_000,
    seed: int = 0,
) -> dict[str, dict[str, np.ndarray]]:
    """Stream over episodes and return LeRobot-compatible statistics."""
    state_stats: RunningStats | None = None
    action_stats: RunningStats | None = None
    image_stats: dict[str, RunningStats] = {}
    for episode in episodes:
        state = _values_array(episode.observation_state)
        action = _values_array(episode.action)
        if state_stats is None:
            state_stats = RunningStats(state.shape[1], quantile_method, reservoir_size, seed)
        if action_stats is None:
            action_stats = RunningStats(action.shape[1], quantile_method, reservoir_size, seed + 1)
        state_stats.update(state)
        action_stats.update(action)
        if include_images:
            for camera, frames in episode.images.items():
                per_channel = np.stack(
                    [_image_array(frame).transpose(2, 0, 1).mean(axis=(1, 2)) for frame in frames]
                )
                # Quantiles for images are per channel; use the mean over pixels
                # as the scalar sample for each channel.  This is the same
                # conservative summary LeRobot exposes for images.
                if camera not in image_stats:
                    image_stats[camera] = RunningStats(3, quantile_method, reservoir_size, seed + 2)
                image_stats[camera].update(per_channel)
    if state_stats is None or action_stats is None:
        raise StatsError("dataset contains no episodes")
    result = {
        "observation.state": state_stats.finalize(),
        "action": action_stats.finalize(),
    }
    for camera, stats in image_stats.items():
        raw = stats.finalize()
        result[f"observation.images.{camera}"] = {
            key: value.reshape(3, 1, 1) if key != "count" else value for key, value in raw.items()
        }
    return result


def stats_to_json(stats: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, Any]]:
    return {
        feature: {key: np.asarray(value).tolist() for key, value in values.items()}
        for feature, values in stats.items()
    }


__all__ = [
    "DEFAULT_QUANTILES",
    "QUANTILE_KEYS",
    "STD_FLOOR",
    "RunningStats",
    "StatsError",
    "compute_dataset_stats",
    "compute_feature_stats",
    "stats_to_json",
]
