"""Fast, vectorized obstacle-distance queries over a mapped surface.

The query is built from an occupied-point set (the TSDF surface after the
robot self-filter).  Points are rasterized into a dense 3-D grid over the
workspace at ``query_resolution_m``; a Euclidean distance transform
(``scipy.ndimage.distance_transform_edt``) gives the distance from every cell
centre to the nearest occupied cell.  ``distance`` and ``is_occupied`` are
vectorized NumPy lookups with no per-point Python loop.

Limitations (also in ``docs/ARCHITECTURE.md`` and ``docs/SAFETY.md``)
--------------------------------------------------------------------
* This is a **surface** map.  There is no free-space carving beyond the TSDF
  truncation band, so unknown space is not represented as occupied.
* A moved object leaves a ghost until the map is reset.
* "Unknown" is expressed only through :class:`MapStatus` (coverage and
  staleness).  An invalid status means **unsafe**; the Task 11 safety layer
  must not fall back to "no obstacle found".
* Points outside the grid return ``max_query_distance_m`` and are flagged by
  :meth:`outside_workspace`; they are not silently treated as free.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from .config import WorkspaceBounds

NO_DATA = "NO_DATA"
STALE = "STALE"
LOW_COVERAGE = "LOW_COVERAGE"
TF_FAILURES = "TF_FAILURES"

_BLOCKING_REASONS = (NO_DATA, STALE, LOW_COVERAGE)


@dataclass(frozen=True)
class MapStatus:
    """Map validity for consumers.  ``valid=False`` must be treated as unsafe."""

    valid: bool
    age_s: float
    num_integrations: int
    coverage_ratio: float
    reasons: tuple[str, ...] = ()

    def has(self, reason: str) -> bool:
        return reason in self.reasons

    @property
    def blocking_reasons(self) -> tuple[str, ...]:
        return tuple(reason for reason in self.reasons if reason in _BLOCKING_REASONS)


class ObstacleQuery:
    """Occupied-point set + 3-D Euclidean distance transform."""

    def __init__(
        self,
        occupied_points: np.ndarray,
        bounds: WorkspaceBounds,
        query_resolution_m: float,
        max_query_distance_m: float,
        *,
        status: MapStatus | None = None,
    ) -> None:
        if query_resolution_m <= 0.0:
            raise ValueError("query_resolution_m must be > 0")
        if max_query_distance_m <= 0.0:
            raise ValueError("max_query_distance_m must be > 0")
        self.bounds = bounds
        self.resolution = float(query_resolution_m)
        self.max_distance = float(max_query_distance_m)
        self.origin = np.asarray(bounds.minimum, dtype=np.float64)
        self.shape = tuple(
            max(1, int(np.ceil(extent / self.resolution))) for extent in bounds.extent
        )
        self._occupied = np.zeros(self.shape, dtype=bool)
        points = np.asarray(occupied_points, dtype=np.float64).reshape(-1, 3)
        if len(points):
            indices, outside = self._index(points)
            inside = ~outside
            if inside.any():
                idx = indices[inside]
                self._occupied[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        self._occupied_count = int(self._occupied.sum())
        if self._occupied_count:
            self._distance = distance_transform_edt(
                ~self._occupied, sampling=(self.resolution,) * 3
            )
        else:
            self._distance = np.full(self.shape, self.max_distance, dtype=np.float64)
        self._status = status

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------
    def _index(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scaled = (points - self.origin) / self.resolution
        raw = np.floor(scaled).astype(np.int64)
        shape = np.asarray(self.shape, dtype=np.int64)
        outside = np.any(raw < 0, axis=1) | np.any(raw >= shape, axis=1)
        clipped = np.clip(raw, 0, shape - 1)
        return clipped, outside

    def distance(self, points: np.ndarray) -> np.ndarray:
        """Vectorized distance to the nearest occupied cell, clamped."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(pts) == 0:
            return np.zeros(0, dtype=np.float64)
        indices, outside = self._index(pts)
        values = self._distance[indices[:, 0], indices[:, 1], indices[:, 2]]
        values = np.where(outside, self.max_distance, values)
        return np.minimum(values, self.max_distance)

    def is_occupied(self, points: np.ndarray) -> np.ndarray:
        """True where the query point falls in an occupied cell (inside grid)."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(pts) == 0:
            return np.zeros(0, dtype=bool)
        indices, outside = self._index(pts)
        occupied = self._occupied[indices[:, 0], indices[:, 1], indices[:, 2]]
        return occupied & ~outside

    def outside_workspace(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(pts) == 0:
            return np.zeros(0, dtype=bool)
        _, outside = self._index(pts)
        return outside

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def status(self) -> MapStatus:
        if self._status is not None:
            return self._status
        valid = self._occupied_count > 0
        return MapStatus(
            valid=valid,
            age_s=0.0,
            num_integrations=0,
            coverage_ratio=1.0 if valid else 0.0,
            reasons=() if valid else (NO_DATA,),
        )

    @property
    def occupied_cells(self) -> int:
        return self._occupied_count

    @property
    def num_cells(self) -> int:
        return int(np.prod(self.shape))

    @property
    def max_distance_m(self) -> float:
        return self.max_distance


def build_query_from_points(
    points: Iterable[Iterable[float]],
    bounds: WorkspaceBounds,
    query_resolution_m: float,
    max_query_distance_m: float,
    *,
    status: MapStatus | None = None,
) -> ObstacleQuery:
    """Convenience builder used by the ROS adapter and by tests."""
    return ObstacleQuery(
        np.asarray(list(points), dtype=np.float64).reshape(-1, 3),
        bounds,
        query_resolution_m,
        max_query_distance_m,
        status=status,
    )
