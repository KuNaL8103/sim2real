"""Damped-least-squares inverse kinematics for the Panda.

The solver uses the shared finite-difference TCP Jacobian from
``sim2real.sim.kinematics``. It never raises for an unreachable target and never
loops past ``max_iterations``: failure is reported as ``success=False`` with the
measured residual.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from sim2real.expert.config import IKConfig
from sim2real.sim.config import PandaConfig
from sim2real.sim.kinematics import (
    KinematicsError,
    clamp_joints,
    numeric_jacobian,
    pose_error,
    rotation_vector,
    tcp_pose,
)


class IKSolver(Protocol):
    def solve(
        self,
        target_pose: np.ndarray,
        q_init: np.ndarray,
        panda: PandaConfig,
        config: IKConfig | None = None,
    ) -> IKResult: ...


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray
    success: bool
    position_error: float
    orientation_error: float
    iterations: int

    def to_dict(self) -> dict[str, object]:
        return {
            "success": bool(self.success),
            "position_error": float(self.position_error),
            "orientation_error": float(self.orientation_error),
            "iterations": int(self.iterations),
            "q": [float(value) for value in self.q],
        }


class DampedLeastSquaresIK:
    """Iterative DLS with joint-limit clamping and a bounded step size."""

    def __init__(self, config: IKConfig | None = None) -> None:
        self.config = config or IKConfig()

    def solve(
        self,
        target_pose: np.ndarray,
        q_init: np.ndarray,
        panda: PandaConfig,
        config: IKConfig | None = None,
    ) -> IKResult:
        settings = config or self.config
        target = np.asarray(target_pose, dtype=np.float64)
        if target.shape != (4, 4):
            raise ValueError(f"target_pose must be 4x4, got {target.shape}")
        q = clamp_joints(q_init, panda)
        iterations = 0
        position_error = np.full(3, np.inf)
        orientation_error = np.full(3, np.inf)

        try:
            for iterations in range(settings.max_iterations + 1):
                current = tcp_pose(q, panda)
                position_error, orientation_error = pose_error(current, target)
                position_norm = float(np.linalg.norm(position_error))
                orientation_norm = float(np.linalg.norm(orientation_error))
                if (
                    position_norm <= settings.position_tolerance_m
                    and orientation_norm <= settings.orientation_tolerance_rad
                ):
                    return IKResult(
                        q=q,
                        success=True,
                        position_error=position_norm,
                        orientation_error=orientation_norm,
                        iterations=iterations,
                    )
                if iterations == settings.max_iterations:
                    break
                jacobian = numeric_jacobian(q, panda, settings.jacobian_step_rad)
                error = np.concatenate((position_error, orientation_error))
                damping = settings.damping**2
                system = jacobian @ jacobian.T + damping * np.eye(6)
                try:
                    delta = jacobian.T @ np.linalg.solve(system, error)
                except np.linalg.LinAlgError:
                    delta = np.zeros(7, dtype=np.float64)
                largest = float(np.max(np.abs(delta)))
                if largest > settings.max_step_rad:
                    delta *= settings.max_step_rad / largest
                q = clamp_joints(q + delta, panda)
        except (KinematicsError, FloatingPointError):
            return IKResult(
                q=q,
                success=False,
                position_error=float(np.linalg.norm(position_error)),
                orientation_error=float(np.linalg.norm(orientation_error)),
                iterations=iterations,
            )

        return IKResult(
            q=q,
            success=False,
            position_error=float(np.linalg.norm(position_error)),
            orientation_error=float(np.linalg.norm(orientation_error)),
            iterations=iterations,
        )


__all__ = ["DampedLeastSquaresIK", "IKResult", "IKSolver", "rotation_vector"]
