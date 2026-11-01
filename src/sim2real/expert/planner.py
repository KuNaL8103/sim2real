"""Bounded planner abstraction: Cartesian waypoints plus joint quintic timing.

``CartesianInterpolationPlanner`` is the default. It solves each Cartesian
waypoint with the configured IK solver and then time-scales the joint-space
segments with a minimum-jerk (quintic) profile bounded by the Panda velocity and
acceleration limits.

``MoveItPlanner`` is a guarded optional backend. MoveIt 2 is **not installed**
on this machine and was **not exercised**; if the import is missing it raises
:class:`PlannerUnavailable` with a clear message. Even when MoveIt is importable
this adapter deliberately refuses to plan until it has been implemented and
verified, so it cannot silently claim success.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from sim2real.expert.config import PlannerConfig
from sim2real.expert.ik import IKResult, IKSolver
from sim2real.sim.config import PandaConfig
from sim2real.sim.kinematics import tcp_pose


class PlannerUnavailable(RuntimeError):
    """Raised when a planner backend cannot run in this environment."""


@dataclass(frozen=True)
class JointPlan:
    times: np.ndarray
    positions: np.ndarray
    duration_s: float
    ik_results: tuple[IKResult, ...]
    success: bool = True
    reason: str = ""

    def __post_init__(self) -> None:
        if self.positions.ndim != 2 or self.positions.shape[1] != 7:
            raise ValueError("positions must have shape (N, 7)")
        if self.times.shape != (self.positions.shape[0],):
            raise ValueError("times must have one entry per position")


class Planner(Protocol):
    def plan_to_pose(
        self,
        q_start: np.ndarray,
        target_pose: np.ndarray,
        panda: PandaConfig,
        ik_solver: IKSolver,
        config: PlannerConfig,
        *,
        duration_scale: float = 1.0,
    ) -> JointPlan: ...


def quintic_scale(u: np.ndarray) -> np.ndarray:
    """Minimum-jerk position profile ``10u^3 - 15u^4 + 6u^5`` on ``[0, 1]``."""
    u = np.asarray(u, dtype=np.float64)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def _segment_duration(
    delta: np.ndarray,
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
    min_duration_s: float,
) -> float:
    """Duration so the quintic respects per-joint velocity/acceleration bounds."""
    magnitude = np.abs(delta)
    velocity_time = 1.875 * magnitude / velocity_limits
    acceleration_time = np.sqrt(5.7735 * magnitude / acceleration_limits)
    required = float(max(velocity_time.max(), acceleration_time.max(), min_duration_s))
    return max(required, min_duration_s)


class CartesianInterpolationPlanner:
    """Default planner: IK each waypoint, then a bounded joint quintic."""

    def plan_to_pose(
        self,
        q_start: np.ndarray,
        target_pose: np.ndarray,
        panda: PandaConfig,
        ik_solver: IKSolver,
        config: PlannerConfig,
        *,
        duration_scale: float = 1.0,
    ) -> JointPlan:
        return self.plan_waypoints(
            q_start,
            (np.asarray(target_pose, dtype=np.float64),),
            panda,
            ik_solver,
            config,
            duration_scale=duration_scale,
        )

    def plan_waypoints(
        self,
        q_start: np.ndarray,
        poses: tuple[np.ndarray, ...] | list[np.ndarray],
        panda: PandaConfig,
        ik_solver: IKSolver,
        config: PlannerConfig,
        *,
        duration_scale: float = 1.0,
    ) -> JointPlan:
        if not 0.0 < duration_scale <= 1.0:
            raise ValueError("duration_scale must be in (0, 1]")
        if len(poses) > config.max_waypoints:
            raise ValueError(f"too many waypoints: {len(poses)} > {config.max_waypoints}")
        velocity = (
            np.asarray(config.velocity_limits, dtype=np.float64)
            if config.velocity_limits is not None
            else np.asarray(panda.velocity_limits, dtype=np.float64)
        )
        acceleration = (
            np.asarray(config.acceleration_limits, dtype=np.float64)
            if config.acceleration_limits is not None
            else np.asarray(panda.acceleration_limits, dtype=np.float64)
        )
        dt = 1.0 / float(config.control_rate_hz)
        start = np.asarray(q_start, dtype=np.float64)
        if start.shape != (7,):
            raise ValueError("q_start must have 7 elements")

        waypoints = [start]
        ik_results: list[IKResult] = []
        q = start.copy()
        for pose in poses:
            result = ik_solver.solve(pose, q, panda)
            ik_results.append(result)
            if not result.success:
                return JointPlan(
                    times=np.array([0.0]),
                    positions=start[None, :],
                    duration_s=0.0,
                    ik_results=tuple(ik_results),
                    success=False,
                    reason=(
                        "IK failed: "
                        f"position_error={result.position_error:.6f} m "
                        f"orientation_error={result.orientation_error:.6f} rad"
                    ),
                )
            q = result.q
            waypoints.append(q)

        times: list[float] = [0.0]
        positions: list[np.ndarray] = [waypoints[0]]
        now = 0.0
        for index in range(1, len(waypoints)):
            segment = waypoints[index] - waypoints[index - 1]
            duration = (
                _segment_duration(segment, velocity, acceleration, config.min_duration_s)
                * duration_scale
            )
            duration = max(duration, dt)
            count = max(1, int(np.ceil(duration / dt)))
            u = np.linspace(0.0, 1.0, count + 1)[1:]
            scale = quintic_scale(u)
            for step, fraction in enumerate(scale, start=1):
                positions.append(waypoints[index - 1] + segment * fraction)
                times.append(now + duration * (step / count))
            now += duration

        times_array = np.asarray(times, dtype=np.float64)
        times_array[0] = 0.0
        positions_array = np.asarray(positions, dtype=np.float64)
        return JointPlan(
            times=times_array,
            positions=positions_array,
            duration_s=float(now),
            ik_results=tuple(ik_results),
            success=True,
        )


class MoveItPlanner:
    """Guarded MoveIt 2 adapter. UNVERIFIED: MoveIt is not installed here."""

    def __init__(self) -> None:
        if importlib.util.find_spec("moveit") is None:
            raise PlannerUnavailable(
                "MoveIt 2 is not importable in this environment; "
                "MoveItPlanner is an unverified guarded stub. "
                "Use CartesianInterpolationPlanner."
            )
        raise PlannerUnavailable(
            "MoveIt 2 is importable but this adapter has not been implemented or "
            "exercised; refusing to claim a plan. Use CartesianInterpolationPlanner."
        )

    def plan_to_pose(self, *args: object, **kwargs: object) -> JointPlan:  # pragma: no cover
        raise PlannerUnavailable("MoveItPlanner.plan_to_pose is not implemented")


def tcp_pose_for(q: np.ndarray, panda: PandaConfig) -> np.ndarray:
    """Re-export the shared TCP function for planner diagnostics."""
    return tcp_pose(q, panda)


__all__ = [
    "CartesianInterpolationPlanner",
    "JointPlan",
    "MoveItPlanner",
    "Planner",
    "PlannerUnavailable",
    "quintic_scale",
]
