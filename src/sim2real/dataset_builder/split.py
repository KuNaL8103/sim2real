"""Deterministic, episode-level train/val splitting."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .schema import UnifiedEpisode


class SplitError(ValueError):
    """Raised for an invalid fraction or an empty episode set."""


@dataclass(frozen=True)
class SplitResult:
    train: tuple[int, ...]
    val: tuple[int, ...]
    by_source: dict[str, dict[str, int]]
    seed: int
    val_fraction: float

    def manifest(self, episodes: Iterable[UnifiedEpisode]) -> dict[str, Any]:
        rows = list(episodes)
        return {
            "seed": self.seed,
            "val_fraction": self.val_fraction,
            "train": [
                {
                    "index": index,
                    "source": rows[index].source,
                    "episode_id": rows[index].original_episode_id,
                }
                for index in self.train
            ],
            "val": [
                {
                    "index": index,
                    "source": rows[index].source,
                    "episode_id": rows[index].original_episode_id,
                }
                for index in self.val
            ],
            "by_source": self.by_source,
        }


def _stable_score(seed: int, source: str, episode_id: str) -> int:
    payload = f"{seed}\0{source}\0{episode_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def split_episodes(
    episodes: list[UnifiedEpisode],
    *,
    val_fraction: float,
    seed: int,
) -> SplitResult:
    """Split per source by a stable hash; no episode is in both sets.

    The hash is computed from ``(seed, source, original_episode_id)``.  For a
    fixed episode list the split is deterministic.  Adding episodes can move the
    per-source count boundary, but an episode is always in exactly one side and
    frames from one episode never cross the boundary.  This is episode-level
    splitting, not frame-level.
    """
    if not episodes:
        raise SplitError("cannot split an empty episode list")
    if not 0.0 <= val_fraction < 1.0:
        raise SplitError("val_fraction must be in [0, 1)")
    train: list[int] = []
    val: list[int] = []
    by_source: dict[str, dict[str, int]] = {}
    grouped: dict[str, list[tuple[int, int, str]]] = {}
    for index, episode in enumerate(episodes):
        grouped.setdefault(episode.source, []).append(
            (
                index,
                _stable_score(seed, episode.source, episode.original_episode_id),
                episode.original_episode_id,
            )
        )
    for source in sorted(grouped):
        rows = sorted(grouped[source], key=lambda item: (item[1], item[2]))
        if val_fraction == 0.0:
            selected: set[int] = set()
        else:
            count = max(1, int(round(len(rows) * val_fraction))) if len(rows) >= 2 else 0
            selected = {item[0] for item in rows[:count]}
        for index, _score, _episode_id in rows:
            if index in selected:
                val.append(index)
            else:
                train.append(index)
        by_source[source] = {"train": len(rows) - len(selected), "val": len(selected)}
    train.sort()
    val.sort()
    if set(train) & set(val):
        raise SplitError("internal error: train/val overlap")
    return SplitResult(
        train=tuple(train),
        val=tuple(val),
        by_source=by_source,
        seed=seed,
        val_fraction=val_fraction,
    )


__all__ = ["SplitError", "SplitResult", "split_episodes"]
