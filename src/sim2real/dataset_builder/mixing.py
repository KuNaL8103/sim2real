"""Deterministic, resumable source mixing for training batches.

A mixture is an explicit manifest: each source keeps its own embodiment and
normalization.  The sampler only returns references; it never concatenates
actions from different embodiments.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .validation import validate_mixture_compatibility


class MixingError(ValueError):
    """Raised for an invalid mixture or sampler configuration."""


@dataclass(frozen=True)
class SourceIndex:
    source_id: str
    episode_lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.source_id:
            raise MixingError("source_id must be non-empty")
        if not self.episode_lengths or any(length <= 0 for length in self.episode_lengths):
            raise MixingError(f"{self.source_id}: episode_lengths must be positive")

    @property
    def num_episodes(self) -> int:
        return len(self.episode_lengths)

    @property
    def num_frames(self) -> int:
        return sum(self.episode_lengths)

    def frame_to_episode(self, frame_index: int) -> tuple[int, int]:
        if not 0 <= frame_index < self.num_frames:
            raise MixingError(f"{self.source_id}: frame_index {frame_index} out of range")
        remaining = frame_index
        for episode_index, length in enumerate(self.episode_lengths):
            if remaining < length:
                return episode_index, remaining
            remaining -= length
        raise MixingError("internal frame index error")


@dataclass(frozen=True)
class MixtureSource:
    source_id: str
    weight: float
    unit: str = "frames"
    max_fraction: float | None = None
    action_representation: str | None = None

    def __post_init__(self) -> None:
        if not self.source_id:
            raise MixingError("source id must be non-empty")
        if not np.isfinite(self.weight) or self.weight <= 0:
            raise MixingError(f"{self.source_id}: weight must be > 0")
        if self.unit not in {"frames", "episodes"}:
            raise MixingError(f"{self.source_id}: unit must be 'frames' or 'episodes'")
        if self.max_fraction is not None and not 0 < self.max_fraction <= 1:
            raise MixingError(f"{self.source_id}: max_fraction must be in (0, 1]")


@dataclass(frozen=True)
class MixtureConfig:
    name: str
    sources: tuple[MixtureSource, ...]
    seed: int
    normalize: bool = False
    replacement: bool = True
    source_lengths: dict[str, tuple[int, int]] = field(default_factory=dict)
    allow_mixed_action_semantics: bool = False

    def __post_init__(self) -> None:
        if not self.sources:
            raise MixingError("mixture must contain at least one source")
        ids = [source.source_id for source in self.sources]
        if len(set(ids)) != len(ids):
            raise MixingError(f"duplicate source ids {ids}")
        total = sum(source.weight for source in self.sources)
        if not self.normalize and abs(total - 1.0) > 1e-9:
            raise MixingError(f"weights sum to {total}; set normalize=true or make them sum to 1")
        if total <= 0:
            raise MixingError("weights must sum to a positive value")

    @property
    def weights(self) -> tuple[float, ...]:
        total = sum(source.weight for source in self.sources)
        return tuple(source.weight / total for source in self.sources)

    def source(self, source_id: str) -> MixtureSource:
        for source in self.sources:
            if source.source_id == source_id:
                return source
        raise MixingError(f"unknown source id {source_id!r}")


@dataclass(frozen=True)
class SampleRef:
    source_id: str
    episode_index: int
    frame_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "episode_index": self.episode_index,
            "frame_index": self.frame_index,
        }


def load_mixture_config(path: str | Path) -> MixtureConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise MixingError(f"missing mixture config: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise MixingError(f"invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise MixingError(f"{config_path}: expected a mapping")
    unknown = sorted(
        set(raw)
        - {
            "schema_version",
            "name",
            "seed",
            "normalize",
            "replacement",
            "allow_mixed_action_semantics",
            "sources",
            "source_lengths",
        }
    )
    if unknown:
        raise MixingError(f"{config_path}: unknown keys {unknown}")
    if raw.get("schema_version") != 1:
        raise MixingError(f"{config_path}: schema_version must be 1")
    raw_sources = raw.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise MixingError(f"{config_path}: sources must be a non-empty list")
    sources: list[MixtureSource] = []
    for index, raw_source in enumerate(raw_sources):
        if not isinstance(raw_source, dict):
            raise MixingError(f"{config_path}: sources[{index}] must be a mapping")
        extra = sorted(
            set(raw_source) - {"id", "weight", "unit", "max_fraction", "action_representation"}
        )
        if extra:
            raise MixingError(f"{config_path}: sources[{index}] has unknown keys {extra}")
        sources.append(
            MixtureSource(
                source_id=str(raw_source.get("id", "")),
                weight=float(raw_source.get("weight", 0.0)),
                unit=str(raw_source.get("unit", "frames")),
                max_fraction=(
                    None
                    if raw_source.get("max_fraction") is None
                    else float(raw_source["max_fraction"])
                ),
                action_representation=raw_source.get("action_representation"),
            )
        )
    source_lengths_raw = raw.get("source_lengths") or {}
    if not isinstance(source_lengths_raw, dict):
        raise MixingError(f"{config_path}: source_lengths must be a mapping")
    source_lengths: dict[str, tuple[int, int]] = {}
    for source_id, value in source_lengths_raw.items():
        if not isinstance(value, dict):
            raise MixingError(f"{config_path}: source_lengths.{source_id} must be a mapping")
        try:
            source_lengths[str(source_id)] = (
                int(value["num_episodes"]),
                int(value["episode_length"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MixingError(
                f"{config_path}: source_lengths.{source_id} needs num_episodes and episode_length"
            ) from exc
    config = MixtureConfig(
        name=str(raw.get("name", config_path.stem)),
        sources=tuple(sources),
        seed=int(raw.get("seed", 0)),
        normalize=bool(raw.get("normalize", False)),
        replacement=bool(raw.get("replacement", True)),
        source_lengths=source_lengths,
        allow_mixed_action_semantics=bool(raw.get("allow_mixed_action_semantics", False)),
    )
    if len(set(config.source_lengths) - {source.source_id for source in config.sources}):
        raise MixingError("source_lengths contains an unknown source id")
    return config


class MixingSampler:
    """Deterministic sampler with explicit save/restore of its stream state."""

    def __init__(self, config: MixtureConfig, indexes: dict[str, SourceIndex]) -> None:
        if set(indexes) != {source.source_id for source in config.sources}:
            raise MixingError(f"indexes must match sources exactly; got {sorted(indexes)}")
        self.config = config
        self.indexes = indexes
        for source in config.sources:
            if indexes[source.source_id].num_episodes == 0:
                raise MixingError(f"source {source.source_id!r} is empty")
        if not config.allow_mixed_action_semantics:
            reps = {
                source.source_id: source.action_representation
                for source in config.sources
                if source.action_representation
            }
            if reps:
                validate_mixture_compatibility(reps, allow_cross_embodiment=False)
        self._weights = np.asarray(config.weights, dtype=np.float64)
        self._rng = np.random.default_rng(config.seed)
        self._position = 0
        self._counts = {source.source_id: 0 for source in config.sources}
        self._seen: set[SampleRef] = set()

    def draw(self) -> SampleRef:
        if not self.config.replacement and len(self._seen) >= self._capacity():
            raise MixingError("sampling without replacement exhausted the source pool")
        for _attempt in range(10_000):
            source = self._choose_source()
            index = self.indexes[source.source_id]
            if source.unit == "frames":
                frame = int(self._rng.integers(0, index.num_frames))
                episode_index, frame_index = index.frame_to_episode(frame)
            else:
                episode_index = int(self._rng.integers(0, index.num_episodes))
                frame_index = int(self._rng.integers(0, index.episode_lengths[episode_index]))
            ref = SampleRef(source.source_id, episode_index, frame_index)
            if self.config.replacement or ref not in self._seen:
                self._seen.add(ref)
                self._position += 1
                self._counts[source.source_id] += 1
                return ref
        raise MixingError("could not draw a unique sample")

    def _capacity(self) -> int:
        return sum(
            index.num_frames
            if self.config.source(source_id).unit == "frames"
            else index.num_episodes
            for source_id, index in self.indexes.items()
        )

    def _choose_source(self) -> MixtureSource:
        allowed = []
        weights = []
        total = max(1, self._position)
        for source, weight in zip(self.config.sources, self._weights, strict=True):
            if (
                source.max_fraction is not None
                and self._counts[source.source_id] / total >= source.max_fraction
            ):
                continue
            allowed.append(source)
            weights.append(weight)
        if not allowed:
            raise MixingError("all sources reached max_fraction before the draw completed")
        normalized = np.asarray(weights, dtype=np.float64)
        normalized /= normalized.sum()
        return allowed[int(self._rng.choice(len(allowed), p=normalized))]

    def draw_n(self, count: int) -> list[SampleRef]:
        if count < 0:
            raise MixingError("count must be >= 0")
        return [self.draw() for _ in range(count)]

    def __iter__(self) -> Iterator[SampleRef]:
        while True:
            yield self.draw()

    def state_dict(self) -> dict[str, Any]:
        return {
            "seed": self.config.seed,
            "position": self._position,
            "counts": dict(self._counts),
            "rng_state": self._rng.bit_generator.state,
            "seen": [
                ref.to_dict()
                for ref in sorted(
                    self._seen,
                    key=lambda item: (item.source_id, item.episode_index, item.frame_index),
                )
            ],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["seed"]) != self.config.seed:
            raise MixingError("save state belongs to a different seed")
        self._position = int(state["position"])
        self._counts = {key: int(value) for key, value in state["counts"].items()}
        self._rng.bit_generator.state = state["rng_state"]
        self._seen = {
            SampleRef(
                source_id=str(item["source_id"]),
                episode_index=int(item["episode_index"]),
                frame_index=int(item["frame_index"]),
            )
            for item in state.get("seen", [])
        }

    def empirical_proportions(self, count: int) -> dict[str, float]:
        sampler = MixingSampler(self.config, self.indexes)
        counts = {source.source_id: 0 for source in self.config.sources}
        for _ in range(count):
            counts[sampler.draw().source_id] += 1
        return {key: value / count for key, value in counts.items()} if count else counts


def indexes_from_config(config: MixtureConfig) -> dict[str, SourceIndex]:
    if not config.source_lengths:
        raise MixingError("mixture config has no source_lengths for preview")
    missing = {source.source_id for source in config.sources} - set(config.source_lengths)
    if missing:
        raise MixingError(f"source_lengths missing {sorted(missing)}")
    return {
        source.source_id: SourceIndex(
            source.source_id,
            tuple(
                [config.source_lengths[source.source_id][1]]
                * config.source_lengths[source.source_id][0]
            ),
        )
        for source in config.sources
    }


try:  # optional adapter; importing torch is never required for the core
    import torch
except Exception:  # pragma: no cover - exercised only when torch is absent
    torch = None  # type: ignore[assignment]


if torch is not None:

    class TorchMixingSampler(torch.utils.data.Sampler):  # type: ignore[misc]
        def __init__(self, sampler: MixingSampler, num_samples: int) -> None:
            self._sampler = sampler
            self._num_samples = num_samples

        def __iter__(self):
            for _ in range(self._num_samples):
                yield self._sampler.draw()

        def __len__(self) -> int:
            return self._num_samples


else:

    class TorchMixingSampler:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise MixingError("torch is not installed; TorchMixingSampler is unavailable")


__all__ = [
    "MixingError",
    "MixingSampler",
    "MixtureConfig",
    "MixtureSource",
    "SampleRef",
    "SourceIndex",
    "TorchMixingSampler",
    "indexes_from_config",
    "load_mixture_config",
]
