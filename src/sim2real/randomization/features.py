"""Normalised failure-space feature vector for domain-randomization samples.

The feature space is the bridge between closed-loop failure analysis (Task 9)
and the existing seeded sampler.  Every feature is normalised to ``[0, 1]``
using the **configured profile bounds**, so a model fitted on failures lives in
a stable, backend-independent space.

Design rules:

* :func:`to_feature_vector` is a pure function of a sample and a profile.
* :func:`from_feature_vector` rebuilds a **complete valid** sample: groups that
  are outside the feature space are drawn from the normal deterministic sampler
  for ``(base_seed, episode_index)``.  The same inputs therefore always give the
  same sample.
* Out-of-range inputs are clipped and *reported* in the result; they are never
  silently accepted as if they were valid.
* The target cube is the first configured cube (``cube_red`` in the shipped
  scene).  Other cube positions are repaired deterministically if the targeted
  position violates ``min_separation_m``.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from sim2real.randomization.color import kelvin_to_rgb
from sim2real.randomization.config import RandomizationConfig
from sim2real.randomization.sampler import (
    CameraSample,
    DistractorSample,
    LightingSample,
    ObjectPoseSample,
    PhysicsSample,
    RandomizationSample,
    TextureSample,
    sample,
)
from sim2real.sim.camera import make_pose

FEATURE_NAMES: tuple[str, ...] = (
    "object_x",
    "object_y",
    "object_yaw",
    "friction_sliding",
    "object_mass",
    "light_intensity",
    "color_temperature",
    "camera_jitter_norm",
)
FEATURE_DIM = len(FEATURE_NAMES)
CAMERA_FEATURE = "overhead"
_REPAIR_SEED_SALT = 0x46454154  # "FEAT"
_MAX_REPAIR_ATTEMPTS = 1000


class FeatureSpaceError(ValueError):
    """Raised when a feature vector or profile cannot be mapped."""


def _norm(value: float, bounds: tuple[float, float]) -> float:
    low, high = bounds
    if not math.isfinite(value) or not math.isfinite(low) or not math.isfinite(high):
        raise FeatureSpaceError("feature bounds and values must be finite")
    if high <= low:
        return 0.5
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _denorm(value: float, bounds: tuple[float, float]) -> float:
    low, high = bounds
    if high <= low:
        return float(low)
    return float(low + float(np.clip(value, 0.0, 1.0)) * (high - low))


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def primary_cube(config: RandomizationConfig) -> str:
    return config.scene.cube_names[0]


def feature_bounds(config: RandomizationConfig) -> dict[str, tuple[float, float]]:
    """Return the un-normalised bounds of every feature from the profile."""
    object_pose = config.object_pose
    camera = config.camera
    jitter = np.asarray(camera.translation_jitter_m, dtype=np.float64)
    bounds = {
        "object_x": tuple(float(v) for v in object_pose.translation_bounds_m[0]),
        "object_y": tuple(float(v) for v in object_pose.translation_bounds_m[1]),
        "object_yaw": tuple(float(v) for v in object_pose.yaw_range_rad),
        "friction_sliding": tuple(float(v) for v in config.physics.friction.sliding),
        "object_mass": tuple(float(v) for v in config.physics.object_mass_kg),
        "light_intensity": tuple(float(v) for v in config.lighting.intensity_range),
        "color_temperature": tuple(float(v) for v in config.lighting.color_temperature_range_k),
        "camera_jitter_norm": (0.0, float(np.linalg.norm(jitter))),
    }
    return {name: bounds[name] for name in FEATURE_NAMES}


def _yaw_from_pose(pose: np.ndarray) -> float:
    return float(math.atan2(float(pose[1, 0]), float(pose[0, 0])))


def sample_from_dict(data: dict[str, object]) -> RandomizationSample:
    """Rebuild a :class:`RandomizationSample` from ``sample.to_dict()`` output."""
    object_pose = data.get("object_pose") or {}
    poses = {
        name: np.asarray(pose, dtype=np.float64)
        for name, pose in (object_pose.get("poses") or {}).items()
    }
    texture_data = data.get("texture") or {}
    texture = TextureSample(
        mode=str(texture_data.get("mode", "color_jitter")),
        rgba={
            name: tuple(float(v) for v in rgba)
            for name, rgba in (texture_data.get("rgba") or {}).items()
        },
        pool=dict(texture_data.get("pool") or {}),
    )
    lighting_data = data.get("lighting") or {}
    lighting = LightingSample(
        intensity_scale=float(lighting_data.get("intensity_scale", 1.0)),
        direction_jitter_rad=tuple(
            float(v) for v in (lighting_data.get("direction_jitter_rad") or (0.0, 0.0, 0.0))
        ),
        color_temperature_k=float(lighting_data.get("color_temperature_k", 6500.0)),
        rgb=tuple(float(v) for v in (lighting_data.get("rgb") or (1.0, 1.0, 1.0))),
    )
    camera_data = data.get("camera") or {}
    camera = CameraSample(
        translation_jitter={
            name: np.asarray(value, dtype=np.float64)
            for name, value in (camera_data.get("translation_jitter") or {}).items()
        },
        rotation_jitter={
            name: np.asarray(value, dtype=np.float64)
            for name, value in (camera_data.get("rotation_jitter") or {}).items()
        },
    )
    physics_data = data.get("physics") or {}
    mass_data = physics_data.get("object_mass_kg")
    friction_data = physics_data.get("friction")
    physics = PhysicsSample(
        object_mass_kg=(
            {name: float(value) for name, value in mass_data.items()}
            if mass_data is not None
            else None
        ),
        friction=(
            {name: tuple(float(v) for v in value) for name, value in friction_data.items()}
            if friction_data is not None
            else None
        ),
        joint_damping_scale=float(physics_data.get("joint_damping_scale", 1.0)),
    )
    distractor_data = data.get("distractors") or {}
    distractors = DistractorSample(
        enabled=bool(distractor_data.get("enabled", False)),
        count=int(distractor_data.get("count", 0)),
        names=tuple(str(name) for name in (distractor_data.get("names") or ())),
    )
    return RandomizationSample(
        nominal=bool(data.get("nominal", False)),
        object_pose=ObjectPoseSample(poses=poses),
        texture=texture,
        lighting=lighting,
        camera=camera,
        physics=physics,
        distractors=distractors,
        disabled_groups=tuple(str(name) for name in (data.get("disabled_groups") or ())),
    )


def _sample_cube_values(
    sample_obj: RandomizationSample, config: RandomizationConfig, name: str
) -> tuple[float, float, float]:
    pose = sample_obj.object_pose.poses.get(name)
    if pose is None:
        raise FeatureSpaceError(f"sample has no pose for {name}")
    return float(pose[0, 3]), float(pose[1, 3]), _yaw_from_pose(pose)


def to_feature_vector(sample_obj: RandomizationSample, config: RandomizationConfig) -> np.ndarray:
    """Map a full sample to the normalised feature vector."""
    bounds = feature_bounds(config)
    name = primary_cube(config)
    x, y, yaw = _sample_cube_values(sample_obj, config, name)
    physics = sample_obj.physics
    if physics.friction is not None and name in physics.friction:
        friction = float(physics.friction[name][0])
    else:
        friction = sum(bounds["friction_sliding"]) / 2.0
    if physics.object_mass_kg is not None and name in physics.object_mass_kg:
        mass = float(physics.object_mass_kg[name])
    else:
        mass = sum(bounds["object_mass"]) / 2.0
    jitter = sample_obj.camera.translation_jitter.get(CAMERA_FEATURE)
    jitter_norm = 0.0 if jitter is None else float(np.linalg.norm(jitter))
    raw = {
        "object_x": x,
        "object_y": y,
        "object_yaw": yaw,
        "friction_sliding": friction,
        "object_mass": mass,
        "light_intensity": float(sample_obj.lighting.intensity_scale),
        "color_temperature": float(sample_obj.lighting.color_temperature_k),
        "camera_jitter_norm": jitter_norm,
    }
    return np.asarray([_norm(raw[key], bounds[key]) for key in FEATURE_NAMES], dtype=np.float64)


def _repair_other_cubes(
    poses: dict[str, np.ndarray],
    *,
    target_name: str,
    target_xy: tuple[float, float],
    config: RandomizationConfig,
    base_seed: int,
    episode_index: int,
) -> list[str]:
    """Keep the target fixed and re-place other cubes away from it."""
    scene = config.scene
    separation = config.object_pose.min_separation_m
    (x_min, x_max), (y_min, y_max) = config.object_pose.translation_bounds_m
    rng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([base_seed, episode_index, _REPAIR_SEED_SALT]))
    )
    placed: list[tuple[float, float]] = [target_xy]
    repaired: list[str] = []
    for name in scene.cube_names:
        if name == target_name:
            continue
        pose = poses[name]
        current = (float(pose[0, 3]), float(pose[1, 3]))
        if math.hypot(current[0] - target_xy[0], current[1] - target_xy[1]) >= separation:
            placed.append(current)
            continue
        position: tuple[float, float] | None = None
        for _ in range(_MAX_REPAIR_ATTEMPTS):
            candidate = (float(rng.uniform(x_min, x_max)), float(rng.uniform(y_min, y_max)))
            if all(
                math.hypot(candidate[0] - px, candidate[1] - py) >= separation for px, py in placed
            ):
                position = candidate
                break
        if position is None:
            # Deterministic lattice fallback, same idea as the sampler's.
            position = (float(x_min), float(y_min))
            for ix in range(int((x_max - x_min) / separation) + 2):
                for iy in range(int((y_max - y_min) / separation) + 2):
                    candidate = (
                        min(x_max, x_min + ix * separation),
                        min(y_max, y_min + iy * separation),
                    )
                    if all(
                        math.hypot(candidate[0] - px, candidate[1] - py) >= separation
                        for px, py in placed
                    ):
                        position = candidate
                        break
                else:
                    continue
                break
        yaw = _yaw_from_pose(pose)
        poses[name] = make_pose(
            np.array([position[0], position[1], scene.cube_center_z_m[name]], dtype=np.float64),
            _rotation_z(yaw),
        )
        placed.append(position)
        repaired.append(name)
    return repaired


def from_feature_vector(
    features: np.ndarray | list[float],
    config: RandomizationConfig,
    base_seed: int,
    episode_index: int = 0,
) -> tuple[RandomizationSample, dict[str, object]]:
    """Rebuild a complete sample from a normalised feature vector.

    Returns ``(sample, report)``.  ``report`` records input clipping, separation
    repair, and camera-jitter clipping so the caller can log what changed.
    """
    values = np.asarray(features, dtype=np.float64).reshape(-1)
    if values.shape != (FEATURE_DIM,):
        raise FeatureSpaceError(
            f"feature vector must have {FEATURE_DIM} values, got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise FeatureSpaceError("feature vector must be finite")
    clipped = [FEATURE_NAMES[i] for i in range(FEATURE_DIM) if not (0.0 <= values[i] <= 1.0)]
    values = np.clip(values, 0.0, 1.0)
    bounds = feature_bounds(config)
    raw = {name: _denorm(float(values[i]), bounds[name]) for i, name in enumerate(FEATURE_NAMES)}

    base = sample(replace(config, seed=int(base_seed)), int(episode_index))
    poses = {name: pose.copy() for name, pose in base.object_pose.poses.items()}
    target = primary_cube(config)
    target_xy = (raw["object_x"], raw["object_y"])
    poses[target] = make_pose(
        np.array(
            [target_xy[0], target_xy[1], config.scene.cube_center_z_m[target]],
            dtype=np.float64,
        ),
        _rotation_z(raw["object_yaw"]),
    )
    repaired = _repair_other_cubes(
        poses,
        target_name=target,
        target_xy=target_xy,
        config=config,
        base_seed=int(base_seed),
        episode_index=int(episode_index),
    )

    cube_names = config.scene.cube_names
    base_friction = base.physics.friction
    friction = {}
    for name in cube_names:
        if base_friction is not None and name in base_friction:
            _, torsional, rolling = base_friction[name]
        else:
            torsional = sum(config.physics.friction.torsional) / 2.0
            rolling = sum(config.physics.friction.rolling) / 2.0
        friction[name] = (
            float(raw["friction_sliding"]),
            float(torsional),
            float(rolling),
        )
    mass = {name: float(raw["object_mass"]) for name in cube_names}

    camera_translation = {
        camera: vector.copy() for camera, vector in base.camera.translation_jitter.items()
    }
    camera_clipped = False
    if CAMERA_FEATURE in camera_translation:
        vector = camera_translation[CAMERA_FEATURE]
        norm = float(np.linalg.norm(vector))
        direction = vector / norm if norm > 0 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
        target_vector = direction * float(raw["camera_jitter_norm"])
        maximum = np.asarray(config.camera.translation_jitter_m, dtype=np.float64)
        clipped_vector = np.clip(target_vector, -maximum, maximum)
        if not np.allclose(clipped_vector, target_vector):
            camera_clipped = True
        camera_translation[CAMERA_FEATURE] = clipped_vector
    rotation = {camera: vector.copy() for camera, vector in base.camera.rotation_jitter.items()}

    kelvin = float(raw["color_temperature"])
    rebuilt = RandomizationSample(
        nominal=False,
        object_pose=type(base.object_pose)(poses=poses),
        texture=base.texture,
        lighting=type(base.lighting)(
            intensity_scale=float(raw["light_intensity"]),
            direction_jitter_rad=base.lighting.direction_jitter_rad,
            color_temperature_k=kelvin,
            rgb=kelvin_to_rgb(kelvin),
        ),
        camera=type(base.camera)(
            translation_jitter=camera_translation,
            rotation_jitter=rotation,
        ),
        physics=type(base.physics)(
            object_mass_kg=mass,
            friction=friction,
            joint_damping_scale=float(base.physics.joint_damping_scale),
        ),
        distractors=base.distractors,
        disabled_groups=(),
    )
    report: dict[str, object] = {
        "clipped_features": clipped,
        "separation_repaired": repaired,
        "camera_jitter_clipped": camera_clipped,
        "requested_features": [float(v) for v in values],
        "applied_features": [float(v) for v in to_feature_vector(rebuilt, config)],
    }
    return rebuilt, report


__all__ = [
    "FEATURE_DIM",
    "FEATURE_NAMES",
    "FeatureSpaceError",
    "feature_bounds",
    "from_feature_vector",
    "primary_cube",
    "sample_from_dict",
    "to_feature_vector",
]
