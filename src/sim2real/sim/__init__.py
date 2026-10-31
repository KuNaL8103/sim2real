"""Simulator backends and the shared ``SimBackend`` contract."""

from __future__ import annotations

from pathlib import Path

from sim2real.sim.backend import (
    BackendCapabilities,
    BackendUnavailableError,
    CameraFrame,
    CameraIntrinsics,
    CollisionInfo,
    ConfigError,
    Observation,
    ResetConfig,
    RobotState,
    SimBackend,
    StepResult,
)
from sim2real.sim.camera import (
    MUJOCO_TO_OPTICAL,
    intrinsics_from_fovy,
    optical_pose_from_mujoco,
    pose_from_pos_quat,
    project_point,
)

__all__ = [
    "BackendCapabilities",
    "BackendUnavailableError",
    "CameraFrame",
    "CameraIntrinsics",
    "CollisionInfo",
    "ConfigError",
    "MUJOCO_TO_OPTICAL",
    "Observation",
    "ResetConfig",
    "RobotState",
    "SimBackend",
    "StepResult",
    "create_backend",
    "intrinsics_from_fovy",
    "optical_pose_from_mujoco",
    "pose_from_pos_quat",
    "project_point",
]


def create_backend(name: str, config_dir: str | Path) -> SimBackend:
    """Construct a backend by name without importing simulator packages early."""
    normalized = name.strip().lower()
    if normalized == "fake":
        from sim2real.sim.fake_backend import FakeBackend

        return FakeBackend(config_dir)
    if normalized == "mujoco":
        from sim2real.sim.mujoco_backend import MuJoCoBackend

        return MuJoCoBackend(config_dir)
    if normalized == "isaac":
        from sim2real.sim.isaac_backend import IsaacBackend

        return IsaacBackend(config_dir)
    raise ConfigError(f"unknown backend {name!r}; expected one of: fake, mujoco, isaac")
