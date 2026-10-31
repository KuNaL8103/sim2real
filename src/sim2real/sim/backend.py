"""Simulator-agnostic backend interface.

Only absolute joint-position targets are supported in Task 2. Randomization is
Task 3 and joint-command subscription is deferred to Task 11.

Units: metres, radians, seconds. Depth images are float32 metres with ``NaN``
or ``0`` meaning invalid. RGB images are uint8 ``HxWx3``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


class BackendUnavailableError(RuntimeError):
    """Raised when a backend cannot be constructed in this environment."""


class ConfigError(ValueError):
    """Raised when a simulator config file is missing or malformed."""


@dataclass(frozen=True)
class RobotState:
    """Joint, gripper, and end-effector state at one simulation stamp."""

    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    gripper_position: float
    ee_pose: np.ndarray
    stamp_ns: int

    def __post_init__(self) -> None:
        if self.joint_positions.shape != self.joint_velocities.shape:
            raise ValueError("joint positions and velocities must have the same shape")
        if self.ee_pose.shape != (4, 4):
            raise ValueError("ee_pose must be a 4x4 homogeneous transform")


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics in pixels."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def as_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class CameraFrame:
    """One RGB-D frame. ``depth`` is metres; NaN/0 mean invalid."""

    rgb: np.ndarray
    depth: np.ndarray
    stamp_ns: int
    frame_id: str

    def __post_init__(self) -> None:
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError("rgb must have shape HxWx3")
        if self.rgb.dtype != np.uint8:
            raise ValueError("rgb must be uint8")
        if self.depth.shape != self.rgb.shape[:2]:
            raise ValueError("depth and rgb must share height and width")
        if self.depth.dtype != np.float32:
            raise ValueError("depth must be float32")


@dataclass(frozen=True)
class Observation:
    """State plus the latest rendered cameras and object poses."""

    robot_state: RobotState
    cameras: Mapping[str, CameraFrame] = field(default_factory=dict)
    object_poses: Mapping[str, np.ndarray] = field(default_factory=dict)

    @property
    def stamp_ns(self) -> int:
        return self.robot_state.stamp_ns


@dataclass(frozen=True)
class CollisionInfo:
    """Contact summary. Details are human-readable body/geom names."""

    num_unintended_contacts: int
    details: tuple[str, ...] = ()


@dataclass(frozen=True)
class StepResult:
    """Result of one control step."""

    observation: Observation
    collision: CollisionInfo
    done: bool = False
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderSnapshot:
    """Immutable-by-convention state used to render without blocking stepping.

    ``payload`` is backend-private (for MuJoCo, a private ``MjData`` copy). It
    must not be mutated after capture.
    """

    stamp_ns: int
    observation: Observation
    payload: Any = None
    generation: int = 0


@dataclass(frozen=True)
class BackendCapabilities:
    """Feature flags with an explicit reason for every flag."""

    rgbd: bool
    lighting: bool
    materials: bool
    camera_extrinsics: bool
    physics_mass: bool
    physics_friction: bool
    joint_damping: bool
    distractors: bool
    object_pose: bool
    texture: bool
    reasons: Mapping[str, str] = field(default_factory=dict)

    FLAG_NAMES: Sequence[str] = (
        "rgbd",
        "lighting",
        "materials",
        "camera_extrinsics",
        "physics_mass",
        "physics_friction",
        "joint_damping",
        "distractors",
        "object_pose",
        "texture",
    )

    def __post_init__(self) -> None:
        missing = [name for name in self.FLAG_NAMES if name not in self.reasons]
        if missing:
            raise ValueError(f"capabilities missing reasons for: {missing}")

    def as_dict(self) -> dict[str, bool]:
        return {name: bool(getattr(self, name)) for name in self.FLAG_NAMES}


@dataclass(frozen=True)
class ResetConfig:
    """Reset request passed to ``SimBackend.reset``.

    ``reset`` itself always restores nominal model parameters and state. Task 3
    randomization is applied afterwards by
    :func:`sim2real.randomization.applier.apply_randomization`, inside the same
    render-quiesce window.
    """

    seed: int = 0
    randomization_enabled: bool = False
    randomization_profile: str = "nominal"
    task_instruction: str = ""
    object_pose_override: np.ndarray | None = None

    def applied_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "seed": int(self.seed),
            "randomization_enabled": bool(self.randomization_enabled),
            "randomization_profile": self.randomization_profile or "nominal",
            "task_instruction": self.task_instruction,
            "object_pose_override_requested": self.object_pose_override is not None,
        }


class SimBackend(ABC):
    """Common interface for fake, MuJoCo, and (guarded) Isaac backends."""

    name: str = "abstract"
    control_rate_hz: float = 20.0
    joint_names: tuple[str, ...] = ()
    joint_limits: np.ndarray = np.empty((0, 2))
    # Directory the backend loaded its configs from; used to locate the
    # sibling ``randomization/`` profile directory. None means "unknown".
    config_dir: Path | None = None

    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Return feature flags; never claim unverified behaviour."""

    @abstractmethod
    def reset(self, seed: int, config: ResetConfig | None = None) -> Observation:
        """Reset to a nominal state and return the first observation."""

    @abstractmethod
    def step(self, joint_target: np.ndarray, gripper_target: float) -> StepResult:
        """Advance one control step toward absolute joint targets."""

    @abstractmethod
    def get_observation(self) -> Observation:
        """Return the latest observation without stepping."""

    @abstractmethod
    def render(self, camera: str) -> CameraFrame:
        """Render one camera on demand."""

    @abstractmethod
    def get_camera_info(self, camera: str) -> CameraIntrinsics:
        """Return intrinsics for a configured camera."""

    @abstractmethod
    def camera_pose_world(self, camera: str) -> np.ndarray:
        """Return the camera's optical pose in the world frame."""

    @abstractmethod
    def collision_report(self) -> CollisionInfo:
        """Return the latest contact summary."""

    @abstractmethod
    def close(self) -> None:
        """Release renderers and other resources."""

    def capture_snapshot(self) -> RenderSnapshot:
        """Copy the state needed to render later, without holding the step lock.

        Backends with mutable simulator state (MuJoCo) override this to make a
        private copy. The default implementation is safe for immutable
        observations such as the fake backend's.
        """
        observation = self.get_observation()
        return RenderSnapshot(stamp_ns=observation.stamp_ns, observation=observation)

    def render_from_snapshot(self, camera: str, snapshot: RenderSnapshot) -> CameraFrame:
        """Render from a previously captured snapshot.

        Backends that do not override this still work, but may render current
        state rather than the snapshot.
        """
        del snapshot
        return self.render(camera)

    def apply_randomization(self, sample: Any) -> Mapping[str, Any]:
        """Apply a randomization sample to model parameters.

        Backends that support randomization override this and return a mapping
        of parameter group -> ``GroupStatus``. The default returns an empty
        mapping, which the applier reports as ``SKIPPED``: a backend must never
        claim an application it did not perform.

        Model-parameter changes are only safe inside the reset-quiesce path; the
        ROS node guarantees that no render is in flight when this is called.
        """
        del sample
        return {}

    def link_poses(self) -> dict[str, tuple[str | None, np.ndarray]]:
        """Return ``frame -> (parent_frame, world_pose)`` for the TF chain.

        Backends that cannot expose individual links return only base and hand.
        """
        observation = self.get_observation()
        return {
            "panda_link0": (None, np.eye(4)),
            "panda_hand": ("panda_link0", observation.robot_state.ee_pose),
        }

    @property
    def sim_time_ns(self) -> int:
        return self.get_observation().stamp_ns
