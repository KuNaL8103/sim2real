"""Configuration-driven, seeded domain randomization.

Public surface:

* :func:`load_profile` / :func:`load_config_file` — validated YAML profiles.
* :func:`sample` — pure, deterministic ``(seed, episode_index)`` sampler.
* :func:`apply_randomization` — capability-gated application with an honest
  ``APPLIED`` / ``PARTIAL`` / ``SKIPPED`` report.
* :func:`kelvin_to_rgb` — approximate colour-temperature tint.

The package has no ``rclpy`` dependency. The ROS node only calls into it inside
the existing reset-quiesce path.
"""

from __future__ import annotations

from sim2real.randomization.applier import (
    APPLIED,
    PARTIAL,
    SKIPPED,
    AppliedRandomization,
    GroupStatus,
    RandomizationUnsupportedError,
    apply_randomization,
)
from sim2real.randomization.color import MATERIAL_POOL, kelvin_to_rgb
from sim2real.randomization.config import (
    MAX_CAMERA_ROTATION_JITTER_RAD,
    MAX_CAMERA_TRANSLATION_JITTER_M,
    MAX_LIGHT_DIRECTION_JITTER_RAD,
    SCHEMA_VERSION,
    CameraConfig,
    DistractorConfig,
    FrictionConfig,
    LightingConfig,
    ObjectPoseConfig,
    PhysicsConfig,
    RandomizationConfig,
    SceneDefaults,
    TextureConfig,
    load_config_file,
    load_profile,
    scene_defaults,
)
from sim2real.randomization.sampler import (
    GROUP_ORDER,
    CameraSample,
    DistractorSample,
    LightingSample,
    ObjectPoseSample,
    PhysicsSample,
    RandomizationError,
    RandomizationSample,
    TextureSample,
    sample,
)

__all__ = [
    "APPLIED",
    "GROUP_ORDER",
    "MATERIAL_POOL",
    "MAX_CAMERA_ROTATION_JITTER_RAD",
    "MAX_CAMERA_TRANSLATION_JITTER_M",
    "MAX_LIGHT_DIRECTION_JITTER_RAD",
    "PARTIAL",
    "SCHEMA_VERSION",
    "SKIPPED",
    "AppliedRandomization",
    "CameraConfig",
    "CameraSample",
    "DistractorConfig",
    "DistractorSample",
    "FrictionConfig",
    "GroupStatus",
    "LightingConfig",
    "LightingSample",
    "ObjectPoseConfig",
    "ObjectPoseSample",
    "PhysicsConfig",
    "PhysicsSample",
    "RandomizationConfig",
    "RandomizationError",
    "RandomizationSample",
    "RandomizationUnsupportedError",
    "SceneDefaults",
    "TextureConfig",
    "TextureSample",
    "apply_randomization",
    "kelvin_to_rgb",
    "load_config_file",
    "load_profile",
    "sample",
    "scene_defaults",
]
