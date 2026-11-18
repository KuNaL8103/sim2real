"""Obstacle-distance gating from a mapped surface (ROS-free).

Only the end-effector point is checked.  This is deliberately **not** a
whole-arm, self-collision, or swept-volume check.  The table and the
manipulated object are in the map, so a naive threshold can reject a legitimate
grasp; the configured 1 cm threshold is measured against clean expert
trajectories in the safety report rather than assumed safe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim2real.mapping.obstacle_query import LOW_COVERAGE, NO_DATA, STALE, MapStatus, ObstacleQuery

from .config import SafetyConfig
from .types import (
    MAP_LOW_COVERAGE,
    MAP_NO_DATA,
    MAP_STALE,
    MAP_UNAVAILABLE,
    OBSTACLE_TOO_CLOSE,
    OUTSIDE_MAP,
)


@dataclass(frozen=True)
class ObstacleCheck:
    ok: bool
    reason: str | None
    min_clearance_m: float
    map_valid: bool
    detail: str = ""


def _reason_for_status(status: MapStatus) -> str:
    if status.has(NO_DATA):
        return MAP_NO_DATA
    if status.has(STALE):
        return MAP_STALE
    if status.has(LOW_COVERAGE):
        return MAP_LOW_COVERAGE
    return MAP_UNAVAILABLE


def _masked_distance(
    ee: np.ndarray,
    query: ObstacleQuery,
    config: SafetyConfig,
    target_center_m: np.ndarray | None,
    target_half_extent_m: float | None,
) -> float:
    points = query.points
    if points.size == 0:
        return float("nan")
    keep = np.abs(points[:, 2] - config.table_surface_z_m) > config.table_mask_band_m
    masked = points[keep]
    if (
        target_center_m is not None
        and target_half_extent_m is not None
        and np.linalg.norm(ee - target_center_m) <= config.target_allowance_radius_m
    ):
        half = float(target_half_extent_m) + config.target_allowance_margin_m
        inside = np.all(np.abs(masked - target_center_m) <= half, axis=1)
        masked = masked[~inside]
    if masked.size == 0:
        return float("nan")
    return float(np.min(np.linalg.norm(masked - ee, axis=1)))


def check_obstacle_distance(
    ee_positions: np.ndarray,
    query: ObstacleQuery | None,
    status: MapStatus | None,
    config: SafetyConfig,
    *,
    target_center_m: np.ndarray | None = None,
    target_half_extent_m: float | None = None,
) -> ObstacleCheck:
    """Return the minimum end-effector clearance and whether it is safe.

    ``map_required`` defaults to true.  When it is false the map is still used
    when valid, but an unavailable map is reported as unknown rather than
    blocking motion.  That is an explicit test/commissioning mode, not a safety
    claim; ``min_obstacle_distance_m`` is then ``nan`` (unknown, never safe).
    """
    points = np.asarray(ee_positions, dtype=np.float64).reshape(-1, 3)
    if points.size == 0:
        return ObstacleCheck(False, MAP_UNAVAILABLE, float("nan"), False, "no end-effector points")

    if query is None or status is None:
        if config.map_required:
            return ObstacleCheck(False, MAP_UNAVAILABLE, float("nan"), False, "no map")
        return ObstacleCheck(True, None, float("nan"), False, "map disabled by configuration")

    if not status.valid:
        if config.map_required:
            return ObstacleCheck(
                False,
                _reason_for_status(status),
                float("nan"),
                False,
                f"invalid map: {list(status.reasons)}",
            )
        return ObstacleCheck(True, None, float("nan"), False, "map invalid but not required")

    if not np.isfinite(status.age_s) or status.age_s > config.max_map_age_s:
        if config.map_required:
            return ObstacleCheck(
                False,
                MAP_STALE,
                float("nan"),
                False,
                f"map age {status.age_s!r} s > {config.max_map_age_s} s",
            )
        return ObstacleCheck(True, None, float("nan"), False, "map stale but not required")

    if query.occupied_cells == 0:
        if config.map_required:
            return ObstacleCheck(False, MAP_NO_DATA, float("nan"), False, "empty map")
        return ObstacleCheck(True, None, float("nan"), False, "empty map but not required")

    if status.coverage_ratio < config.min_map_coverage_ratio:
        if config.map_required:
            return ObstacleCheck(
                False,
                MAP_LOW_COVERAGE,
                float("nan"),
                False,
                f"coverage {status.coverage_ratio:.4f} < {config.min_map_coverage_ratio}",
            )
        return ObstacleCheck(True, None, float("nan"), False, "low coverage but not required")

    if config.obstacle_mode == "masked":
        masked_distances = np.array(
            [
                _masked_distance(ee, query, config, target_center_m, target_half_extent_m)
                for ee in points
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(masked_distances)):
            return ObstacleCheck(
                False,
                MAP_NO_DATA,
                float("nan"),
                False,
                "masked mode left no map points to check",
            )
        if np.any(query.outside_workspace(points)):
            return ObstacleCheck(
                False,
                OUTSIDE_MAP,
                float(np.min(masked_distances)),
                False,
                "end effector outside the mapped workspace",
            )
        clearance = masked_distances - config.obstacle_safety_radius_m
        minimum = float(np.min(clearance))
    else:
        distances = query.distance(points)
        if np.any(query.outside_workspace(points)):
            # A query outside the mapped grid is a clamp, not proof of clearance.
            return ObstacleCheck(
                False,
                OUTSIDE_MAP,
                float(np.min(distances)),
                False,
                "end effector outside the mapped workspace",
            )
        clearance = distances - config.obstacle_safety_radius_m
        minimum = float(np.min(clearance))

    if minimum < config.min_obstacle_distance_m:
        return ObstacleCheck(
            False,
            OBSTACLE_TOO_CLOSE,
            minimum,
            True,
            f"clearance {minimum:.4f} m < {config.min_obstacle_distance_m} m",
        )
    return ObstacleCheck(True, None, minimum, True)
