"""End-to-end validation of one raw VLA chunk.

Order matters: malformed/empty data is rejected before any arithmetic; the
finger is clipped and counted; stale points are handled by policy; EMA is
initialised from the measured state; then the **final** smoothed/retimed
trajectory is checked for continuity, workspace, joint limits, and obstacle
clearance.  Nothing is forwarded if any step fails.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sim2real import contracts
from sim2real.mapping.obstacle_query import MapStatus, ObstacleQuery
from sim2real.sim.config import PandaConfig

from .config import SafetyConfig
from .limits import check_joint_limits
from .obstacle import check_obstacle_distance
from .retime import select_stale_points
from .smoothing import ema_targets
from .types import (
    CONTINUITY,
    EMPTY_CHUNK,
    FUTURE_STAMP,
    INVALID_TIME,
    JOINT_NAMES_MISMATCH,
    JOINT_STATE_UNAVAILABLE,
    NON_FINITE,
    NON_INCREASING_TIME,
    TOO_MANY_POINTS,
    ChunkDecision,
    LimitRatios,
    RawChunk,
    SafetyCommand,
)
from .workspace import check_workspace


@dataclass(frozen=True)
class ValidationContext:
    """Clock and map inputs for one validation call."""

    now_stamp_ns: int | None
    now_monotonic_s: float
    received_monotonic_s: float
    query: ObstacleQuery | None = None
    map_status: MapStatus | None = None


def _reject(
    reason: str,
    *,
    detail: str = "",
    ratios: LimitRatios | None = None,
    age_s: float = float("nan"),
    finger_clipped_points: int = 0,
    map_valid: bool = False,
    min_obstacle_distance_m: float = float("nan"),
    retimed: bool = False,
    dropped_points: int = 0,
) -> ChunkDecision:
    return ChunkDecision(
        accepted=False,
        reason=reason,
        command=None,
        ratios=ratios or LimitRatios(),
        min_obstacle_distance_m=min_obstacle_distance_m,
        map_valid=map_valid,
        finger_clipped_points=finger_clipped_points,
        finger_clipped=finger_clipped_points > 0,
        age_s=age_s,
        retimed=retimed,
        dropped_points=dropped_points,
        detail=detail,
    )


def validate_chunk(
    chunk: RawChunk,
    measured: np.ndarray,
    config: SafetyConfig,
    panda: PandaConfig,
    context: ValidationContext,
) -> ChunkDecision:
    if context.now_stamp_ns is None:
        return _reject(JOINT_STATE_UNAVAILABLE, detail="no measured joint state stamp")

    names = tuple(chunk.joint_names)
    if names != contracts.CONTROLLED_JOINT_NAMES:
        return _reject(
            JOINT_NAMES_MISMATCH,
            detail=f"expected {list(contracts.CONTROLLED_JOINT_NAMES)}, got {list(names)}",
        )

    try:
        positions = np.asarray(chunk.positions, dtype=np.float64)
    except (TypeError, ValueError):
        return _reject(NON_FINITE, detail="positions are not numeric")
    if positions.ndim != 2 or positions.shape[1] != 8:
        return _reject(NON_FINITE, detail=f"positions must be (n, 8), got {positions.shape}")
    if positions.shape[0] == 0:
        return _reject(EMPTY_CHUNK, detail="chunk has no points")
    if positions.shape[0] > config.max_chunk_points:
        return _reject(
            TOO_MANY_POINTS,
            detail=f"{positions.shape[0]} points > {config.max_chunk_points}",
        )
    if not np.all(np.isfinite(positions)):
        return _reject(NON_FINITE, detail="positions contain NaN or infinity")

    offsets = tuple(float(value) for value in chunk.time_from_start_s)
    if len(offsets) != positions.shape[0]:
        return _reject(INVALID_TIME, detail="time_from_start length does not match points")
    if not all(math.isfinite(value) and value >= 0.0 for value in offsets):
        return _reject(INVALID_TIME, detail="time_from_start must be finite and >= 0")
    if any(b <= a for a, b in zip(offsets, offsets[1:], strict=False)):
        return _reject(NON_INCREASING_TIME, detail=f"time_from_start is not increasing: {offsets}")

    if chunk.stamp_ns < 0:
        return _reject(INVALID_TIME, detail="header stamp must be >= 0")
    future_ns = int(context.now_stamp_ns) - int(chunk.stamp_ns)
    if future_ns < 0 and -future_ns > int(config.future_stamp_tolerance_s * 1e9):
        return _reject(
            FUTURE_STAMP,
            detail=f"header stamp is {-future_ns / 1e9:.3f} s in the future",
        )

    clipped = np.array(positions, copy=True)
    finger = clipped[:, 7]
    outside = (finger < config.finger_min_m) | (finger > config.finger_max_m)
    finger_clipped = int(np.count_nonzero(outside))
    clipped[:, 7] = np.clip(finger, config.finger_min_m, config.finger_max_m)

    stamp_age_s = max(0.0, (int(context.now_stamp_ns) - int(chunk.stamp_ns)) / 1e9)
    wall_age_s = max(0.0, context.now_monotonic_s - context.received_monotonic_s)
    age_s = max(stamp_age_s, wall_age_s)

    stale = select_stale_points(clipped, offsets, age_s, config)
    if not stale.ok or stale.positions is None:
        return _reject(
            stale.reason or "STALE_CHUNK",
            detail=stale.detail,
            age_s=age_s,
            finger_clipped_points=finger_clipped,
            retimed=config.stale_chunk_policy == "retime_if_continuous",
            dropped_points=stale.dropped_points,
        )

    measured_array = np.asarray(measured, dtype=np.float64)
    if measured_array.shape != (8,) or not np.all(np.isfinite(measured_array)):
        return _reject(
            JOINT_STATE_UNAVAILABLE,
            detail="measured joint state is malformed",
            age_s=age_s,
            finger_clipped_points=finger_clipped,
        )

    smoothed = ema_targets(stale.positions, measured_array, config.ema_alpha)
    first_distance = float(np.max(np.abs(smoothed[0, :7] - measured_array[:7])))
    if first_distance > config.max_first_point_distance_rad:
        return _reject(
            CONTINUITY,
            detail=(
                f"first command point is {first_distance:.4f} rad from measured; "
                f"limit {config.max_first_point_distance_rad} rad"
            ),
            age_s=age_s,
            finger_clipped_points=finger_clipped,
            retimed=True,
            dropped_points=stale.dropped_points,
        )

    workspace = check_workspace(smoothed, panda, config)
    obstacle = check_obstacle_distance(
        workspace.ee_positions, context.query, context.map_status, config
    )
    limits = check_joint_limits(smoothed, stale.times_s, measured_array, config)

    if not workspace.ok:
        return _reject(
            workspace.reason or "WORKSPACE_VIOLATION",
            detail=workspace.detail,
            ratios=limits.ratios,
            age_s=age_s,
            finger_clipped_points=finger_clipped,
            map_valid=obstacle.map_valid,
            min_obstacle_distance_m=obstacle.min_clearance_m,
            retimed=True,
            dropped_points=stale.dropped_points,
        )
    if not limits.ok:
        return _reject(
            limits.reason or "LIMIT",
            detail=limits.detail,
            ratios=limits.ratios,
            age_s=age_s,
            finger_clipped_points=finger_clipped,
            map_valid=obstacle.map_valid,
            min_obstacle_distance_m=obstacle.min_clearance_m,
            retimed=True,
            dropped_points=stale.dropped_points,
        )
    if not obstacle.ok:
        return _reject(
            obstacle.reason or "OBSTACLE",
            detail=obstacle.detail,
            ratios=limits.ratios,
            age_s=age_s,
            finger_clipped_points=finger_clipped,
            map_valid=obstacle.map_valid,
            min_obstacle_distance_m=obstacle.min_clearance_m,
            retimed=True,
            dropped_points=stale.dropped_points,
        )

    command = SafetyCommand(
        joint_names=contracts.CONTROLLED_JOINT_NAMES,
        positions=smoothed,
        time_from_start_s=stale.times_s,
        header_stamp_ns=0,
        source_stamp_ns=int(chunk.stamp_ns),
        retimed=True,
        dropped_points=stale.dropped_points,
    )
    return ChunkDecision(
        accepted=True,
        reason=None,
        command=command,
        ratios=limits.ratios,
        min_obstacle_distance_m=obstacle.min_clearance_m,
        map_valid=obstacle.map_valid,
        finger_clipped_points=finger_clipped,
        finger_clipped=finger_clipped > 0,
        age_s=age_s,
        retimed=True,
        dropped_points=stale.dropped_points,
    )


__all__ = ["ValidationContext", "validate_chunk"]
