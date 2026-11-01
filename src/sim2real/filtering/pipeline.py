"""Batch/session wrapper around the filter decision engine.

The pipeline owns the set of episode ids already seen in a run so a duplicate is
reported as ``DUPLICATE_EPISODE`` instead of silently overwriting curated data.
It has no ROS, filesystem, or simulator dependency.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from sim2real.filtering.config import FilterConfig
from sim2real.filtering.decision import FilterResult, decide


class FilterPipeline:
    def __init__(self, config: FilterConfig) -> None:
        self.config = config
        self.seen_ids: set[str] = set()

    def process(self, record: Any) -> FilterResult:
        result = decide(record, self.config, seen_ids=self.seen_ids)
        episode_id = str(getattr(record, "episode_id", "") or "")
        if episode_id:
            self.seen_ids.add(episode_id)
        return result

    def process_all(self, records: Iterable[Any]) -> Iterator[tuple[Any, FilterResult]]:
        for record in records:
            yield record, self.process(record)


__all__ = ["FilterPipeline"]
