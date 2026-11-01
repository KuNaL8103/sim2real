"""Coarse robot self-filter based on one sphere per robot link frame.

This is deliberately simple and **not** a collision model.  It removes points
within a configured radius of each link-frame origin so the robot's own body is
not treated as an obstacle.

Known failure modes
-------------------
* A grasped object near ``panda_hand`` is removed together with the hand.
* Geometry between link frames (forearm, fingers, cable) is not covered.
* A radius large enough to cover a link also erases nearby scene points.
* Link positions come from TF; a missing transform means that link is not
  filtered at all (and is reported in ``missing_links``).

The Task 11 safety layer must not treat this filter as a guarantee that the
remaining points are obstacle-free.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .config import SelfFilterConfig


@dataclass(frozen=True)
class SelfFilterResult:
    kept: int
    removed: int
    missing_links: tuple[str, ...]


class RobotSelfFilter:
    """Remove points inside link-centred spheres."""

    def __init__(self, config: SelfFilterConfig) -> None:
        self.config = config

    def keep_mask(
        self,
        points: np.ndarray,
        link_positions: Mapping[str, np.ndarray],
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        keep = np.ones(len(pts), dtype=bool)
        missing: list[str] = []
        for frame in self.config.link_frames:
            position = link_positions.get(frame)
            if position is None:
                missing.append(frame)
                continue
            centre = np.asarray(position, dtype=np.float64).reshape(3)
            radius = self.config.radius(frame)
            distance_sq = np.sum((pts - centre) ** 2, axis=1)
            keep &= distance_sq >= radius * radius
        return keep, tuple(missing)

    def filter(
        self,
        points: np.ndarray,
        link_positions: Mapping[str, np.ndarray],
    ) -> tuple[np.ndarray, SelfFilterResult]:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        keep, missing = self.keep_mask(pts, link_positions)
        result = SelfFilterResult(
            kept=int(keep.sum()),
            removed=int(len(pts) - keep.sum()),
            missing_links=missing,
        )
        return pts[keep], result
