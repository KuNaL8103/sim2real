"""Guarded Isaac Sim adapter.

Isaac Sim is **not installed** on this machine and its published requirements
exceed the measured hardware. Nothing in this module claims Isaac behaviour
works: ``is_available()`` reports why it cannot run, and construction raises
:class:`BackendUnavailableError`. The class exists so the ROS node can reject
``backend:=isaac`` with a clear message.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np

from sim2real.sim.backend import (
    BackendCapabilities,
    BackendUnavailableError,
    CameraFrame,
    CameraIntrinsics,
    CollisionInfo,
    Observation,
    ResetConfig,
    SimBackend,
    StepResult,
)

_UNAVAILABLE_REASON = (
    "Isaac Sim is not installed. The published Isaac Sim 6.2.0 minimums "
    "(RTX 4080-class GPU, 16 GB VRAM, 32 GB RAM, driver 595.58.03) exceed this "
    "machine (RTX 4050 Laptop 6 GiB, driver 551.76). This tier is UNAVAILABLE "
    "on this machine based on requirements, not tested."
)


class IsaacBackend(SimBackend):
    """Placeholder adapter. Every operation raises until Isaac is available."""

    name = "isaac"

    def __init__(self, config_dir: str | None = None) -> None:
        del config_dir
        available, reason = self.is_available()
        if not available:
            raise BackendUnavailableError(reason)
        raise BackendUnavailableError(
            "IsaacBackend is not implemented; the adapter is a guarded stub."
        )

    @staticmethod
    def is_available() -> tuple[bool, str]:
        """Return ``(False, reason)`` because no Isaac Sim module is present."""
        if importlib.util.find_spec("isaacsim") is None:
            return (False, _UNAVAILABLE_REASON)
        return (
            False,
            "isaacsim is importable but no behaviour is verified; this adapter "
            "remains a stub and must not be treated as working.",
        )

    def capabilities(self) -> BackendCapabilities:  # pragma: no cover - stub
        return BackendCapabilities(
            rgbd=False,
            lighting=False,
            materials=False,
            camera_extrinsics=False,
            physics_mass=False,
            physics_friction=False,
            joint_damping=False,
            distractors=False,
            reasons={name: "unverified: Isaac Sim is unavailable" for name in _FLAGS},
        )

    def reset(self, seed: int, config: ResetConfig | None = None) -> Observation:
        raise BackendUnavailableError(_UNAVAILABLE_REASON)

    def step(self, joint_target: np.ndarray, gripper_target: float) -> StepResult:
        raise BackendUnavailableError(_UNAVAILABLE_REASON)

    def get_observation(self) -> Observation:
        raise BackendUnavailableError(_UNAVAILABLE_REASON)

    def render(self, camera: str) -> CameraFrame:
        raise BackendUnavailableError(_UNAVAILABLE_REASON)

    def get_camera_info(self, camera: str) -> CameraIntrinsics:
        raise BackendUnavailableError(_UNAVAILABLE_REASON)

    def camera_pose_world(self, camera: str) -> np.ndarray:
        raise BackendUnavailableError(_UNAVAILABLE_REASON)

    def collision_report(self) -> CollisionInfo:
        raise BackendUnavailableError(_UNAVAILABLE_REASON)

    def close(self) -> None:
        return None

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - defensive
        raise BackendUnavailableError(_UNAVAILABLE_REASON)


_FLAGS = (
    "rgbd",
    "lighting",
    "materials",
    "camera_extrinsics",
    "physics_mass",
    "physics_friction",
    "joint_damping",
    "distractors",
)
