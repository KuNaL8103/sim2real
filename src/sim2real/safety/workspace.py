"""End-effector workspace check using the single-source Panda FK."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim2real.sim.config import PandaConfig
from sim2real.sim.kinematics import tcp_pose

from .config import SafetyConfig
from .types import WORKSPACE_VIOLATION


@dataclass(frozen=True)
class WorkspaceCheck:
    ok: bool
    reason: str | None
    ee_positions: np.ndarray
    worst_margin_m: float
    outside_indices: tuple[int, ...]
    detail: str = ""


def end_effector_positions(command_positions: np.ndarray, panda: PandaConfig) -> np.ndarray:
    """TCP position for every command point, shape ``(n, 3)``."""
    positions = np.asarray(command_positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 8:
        raise ValueError(f"command_positions must have shape (n, 8), got {positions.shape}")
    output = np.empty((positions.shape[0], 3), dtype=np.float64)
    for index, row in enumerate(positions):
        output[index] = tcp_pose(row[:7], panda)[:3, 3]
    return output


def check_workspace(
    command_positions: np.ndarray,
    panda: PandaConfig,
    config: SafetyConfig,
) -> WorkspaceCheck:
    ee = end_effector_positions(command_positions, panda)
    lower = np.asarray(config.workspace_min_m, dtype=np.float64)
    upper = np.asarray(config.workspace_max_m, dtype=np.float64)
    # Signed margin: positive inside, negative outside. The minimum over
    # dimensions and points is the closest approach to any boundary.
    margin = np.minimum(ee - lower, upper - ee)
    worst = float(np.min(margin))
    outside = np.flatnonzero(np.any(margin < 0.0, axis=1))
    if len(outside):
        return WorkspaceCheck(
            False,
            WORKSPACE_VIOLATION,
            ee,
            worst,
            tuple(int(i) for i in outside),
            f"end effector outside [{lower.tolist()}, {upper.tolist()}]",
        )
    return WorkspaceCheck(True, None, ee, worst, ())
