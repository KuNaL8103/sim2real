"""Deterministic domain-randomization sampler.

Determinism rules (the interview point):

* ``sample(config, episode_index)`` is a **pure function** of the profile and the
  episode index. It never reads or writes global NumPy/Python RNG state.
* The base seed is ``config.seed``. A ``numpy.random.SeedSequence`` is built as
  ``SeedSequence([config.seed, episode_index])`` and six independent child
  streams are spawned in a fixed order: object pose, texture, lighting, camera,
  physics, distractors.
* Each group draws only from its own child stream. Disabling or adding a group
  therefore never changes another group's values, and results do not depend on
  call order or on how many samples were drawn earlier.
* A disabled group is returned as nominal and consumes nothing from its stream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from sim2real.randomization.color import MATERIAL_POOL, kelvin_to_rgb
from sim2real.randomization.config import RandomizationConfig, SceneDefaults
from sim2real.sim.camera import make_pose

GROUP_ORDER = ("object_pose", "texture", "lighting", "camera", "physics", "distractors")
NOMINAL_KELVIN = 6500.0
_MAX_PLACEMENT_ATTEMPTS = 1000


class RandomizationError(RuntimeError):
    """Raised when a sample cannot be constructed from a valid config."""


@dataclass(frozen=True)
class ObjectPoseSample:
    poses: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class TextureSample:
    mode: str
    rgba: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)
    pool: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LightingSample:
    intensity_scale: float
    direction_jitter_rad: tuple[float, float, float]
    color_temperature_k: float
    rgb: tuple[float, float, float]


@dataclass(frozen=True)
class CameraSample:
    translation_jitter: dict[str, np.ndarray] = field(default_factory=dict)
    rotation_jitter: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class PhysicsSample:
    object_mass_kg: dict[str, float] | None = None
    friction: dict[str, tuple[float, float, float]] | None = None
    joint_damping_scale: float = 1.0


@dataclass(frozen=True)
class DistractorSample:
    enabled: bool = False
    count: int = 0
    names: tuple[str, ...] = ()


@dataclass(frozen=True)
class RandomizationSample:
    nominal: bool
    object_pose: ObjectPoseSample
    texture: TextureSample
    lighting: LightingSample
    camera: CameraSample
    physics: PhysicsSample
    distractors: DistractorSample
    disabled_groups: tuple[str, ...] = ()

    def group_enabled(self, group: str) -> bool:
        return not self.nominal and group not in self.disabled_groups

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation (plain floats, no NumPy scalars)."""
        return {
            "nominal": bool(self.nominal),
            "disabled_groups": list(self.disabled_groups),
            "object_pose": {
                "poses": {
                    name: [[float(value) for value in row] for row in pose]
                    for name, pose in self.object_pose.poses.items()
                }
            },
            "texture": {
                "mode": self.texture.mode,
                "rgba": {
                    name: [float(channel) for channel in rgba]
                    for name, rgba in self.texture.rgba.items()
                },
                "pool": dict(self.texture.pool),
            },
            "lighting": {
                "intensity_scale": float(self.lighting.intensity_scale),
                "direction_jitter_rad": [
                    float(value) for value in self.lighting.direction_jitter_rad
                ],
                "color_temperature_k": float(self.lighting.color_temperature_k),
                "rgb": [float(value) for value in self.lighting.rgb],
            },
            "camera": {
                "translation_jitter": {
                    name: [float(value) for value in jitter]
                    for name, jitter in self.camera.translation_jitter.items()
                },
                "rotation_jitter": {
                    name: [float(value) for value in jitter]
                    for name, jitter in self.camera.rotation_jitter.items()
                },
            },
            "physics": {
                "object_mass_kg": (
                    {name: float(mass) for name, mass in self.physics.object_mass_kg.items()}
                    if self.physics.object_mass_kg is not None
                    else None
                ),
                "friction": (
                    {
                        name: [float(value) for value in friction]
                        for name, friction in self.physics.friction.items()
                    }
                    if self.physics.friction is not None
                    else None
                ),
                "joint_damping_scale": float(self.physics.joint_damping_scale),
            },
            "distractors": {
                "enabled": bool(self.distractors.enabled),
                "count": int(self.distractors.count),
                "names": list(self.distractors.names),
            },
        }


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _nominal(scene: SceneDefaults, config: RandomizationConfig) -> RandomizationSample:
    poses = {
        name: make_pose(np.asarray(position, dtype=np.float64), np.eye(3))
        for name, position in scene.nominal_positions.items()
    }
    return RandomizationSample(
        nominal=True,
        object_pose=ObjectPoseSample(poses=poses),
        texture=TextureSample(mode="color_jitter", rgba=dict(scene.nominal_colors)),
        lighting=LightingSample(
            intensity_scale=1.0,
            direction_jitter_rad=(0.0, 0.0, 0.0),
            color_temperature_k=NOMINAL_KELVIN,
            rgb=kelvin_to_rgb(NOMINAL_KELVIN),
        ),
        camera=CameraSample(
            translation_jitter={
                camera: np.zeros(3, dtype=np.float64) for camera in config.camera.apply_to
            },
            rotation_jitter={
                camera: np.zeros(3, dtype=np.float64) for camera in config.camera.apply_to
            },
        ),
        physics=PhysicsSample(object_mass_kg=None, friction=None, joint_damping_scale=1.0),
        distractors=DistractorSample(enabled=False, count=0, names=()),
        disabled_groups=GROUP_ORDER,
    )


def _sample_object_poses(config: RandomizationConfig, rng: np.random.Generator) -> ObjectPoseSample:
    scene = config.scene
    group = config.object_pose
    (x_min, x_max), (y_min, y_max) = group.translation_bounds_m
    separation = group.min_separation_m
    placed: list[tuple[float, float]] = []
    poses: dict[str, np.ndarray] = {}

    for name in scene.cube_names:
        position: tuple[float, float] | None = None
        for _ in range(_MAX_PLACEMENT_ATTEMPTS):
            candidate = (float(rng.uniform(x_min, x_max)), float(rng.uniform(y_min, y_max)))
            if all(math.hypot(candidate[0] - x, candidate[1] - y) >= separation for x, y in placed):
                position = candidate
                break
        if position is None:
            position = _lattice_position(placed, (x_min, x_max), (y_min, y_max), separation)
        placed.append(position)
        yaw = float(rng.uniform(*group.yaw_range_rad))
        poses[name] = make_pose(
            np.array([position[0], position[1], scene.cube_center_z_m[name]], dtype=np.float64),
            _rotation_z(yaw),
        )
    return ObjectPoseSample(poses=poses)


def _lattice_position(
    placed: list[tuple[float, float]],
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    separation: float,
) -> tuple[float, float]:
    """Deterministic fallback if rejection sampling cannot place a cube."""
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    steps_x = max(1, int((x_max - x_min) / separation) + 1)
    steps_y = max(1, int((y_max - y_min) / separation) + 1)
    for ix in range(steps_x + 1):
        for iy in range(steps_y + 1):
            x = min(x_max, x_min + ix * separation)
            y = min(y_max, y_min + iy * separation)
            if all(math.hypot(x - px, y - py) >= separation for px, py in placed):
                return (x, y)
    raise RandomizationError(
        "could not place cubes with the configured min_separation_m inside the bounds"
    )


def _sample_texture(config: RandomizationConfig, rng: np.random.Generator) -> TextureSample:
    group = config.texture
    names = config.scene.cube_names
    if group.mode == "color_jitter":
        assert group.rgba_range is not None
        rgba = {
            name: tuple(float(rng.uniform(low, high)) for low, high in group.rgba_range)
            for name in names
        }
        return TextureSample(mode=group.mode, rgba=rgba)

    assert group.pool is not None
    indices = rng.integers(0, len(group.pool), size=len(names))
    pool = {name: group.pool[int(index)] for name, index in zip(names, indices, strict=True)}
    rgba = {name: MATERIAL_POOL[pool[name]] for name in names}
    return TextureSample(mode=group.mode, rgba=rgba, pool=pool)


def _sample_lighting(config: RandomizationConfig, rng: np.random.Generator) -> LightingSample:
    group = config.lighting
    intensity = float(rng.uniform(*group.intensity_range))
    magnitude = group.direction_jitter_rad
    direction = tuple(float(rng.uniform(-magnitude, magnitude)) for _ in range(3))
    kelvin = float(rng.uniform(*group.color_temperature_range_k))
    return LightingSample(
        intensity_scale=intensity,
        direction_jitter_rad=direction,
        color_temperature_k=kelvin,
        rgb=kelvin_to_rgb(kelvin),
    )


def _sample_camera(config: RandomizationConfig, rng: np.random.Generator) -> CameraSample:
    group = config.camera
    translation = {
        camera: np.array(
            [
                rng.uniform(-group.translation_jitter_m[0], group.translation_jitter_m[0]),
                rng.uniform(-group.translation_jitter_m[1], group.translation_jitter_m[1]),
                rng.uniform(-group.translation_jitter_m[2], group.translation_jitter_m[2]),
            ],
            dtype=np.float64,
        )
        for camera in group.apply_to
    }
    rotation = {
        camera: np.array(
            [
                rng.uniform(-group.rotation_jitter_rad[0], group.rotation_jitter_rad[0]),
                rng.uniform(-group.rotation_jitter_rad[1], group.rotation_jitter_rad[1]),
                rng.uniform(-group.rotation_jitter_rad[2], group.rotation_jitter_rad[2]),
            ],
            dtype=np.float64,
        )
        for camera in group.apply_to
    }
    return CameraSample(translation_jitter=translation, rotation_jitter=rotation)


def _sample_physics(config: RandomizationConfig, rng: np.random.Generator) -> PhysicsSample:
    group = config.physics
    names = config.scene.cube_names
    mass = {name: float(rng.uniform(*group.object_mass_kg)) for name in names}
    friction = {
        name: (
            float(rng.uniform(*group.friction.sliding)),
            float(rng.uniform(*group.friction.torsional)),
            float(rng.uniform(*group.friction.rolling)),
        )
        for name in names
    }
    return PhysicsSample(
        object_mass_kg=mass,
        friction=friction,
        joint_damping_scale=float(rng.uniform(*group.joint_damping_scale)),
    )


def _sample_distractors(config: RandomizationConfig, rng: np.random.Generator) -> DistractorSample:
    group = config.distractors
    count = int(rng.integers(group.count_range[0], group.count_range[1] + 1))
    if count == 0:
        return DistractorSample(enabled=True, count=0, names=())
    names = tuple(
        str(group.pool[int(index)]) for index in rng.integers(0, len(group.pool), size=count)
    )
    return DistractorSample(enabled=True, count=count, names=names)


def sample(config: RandomizationConfig, episode_index: int) -> RandomizationSample:
    """Return a deterministic sample for ``(config.seed, episode_index)``.

    The function is pure: the same inputs always produce the same values, and no
    global RNG state is touched.
    """
    if isinstance(episode_index, bool) or not isinstance(episode_index, int):
        raise ValueError("episode_index must be an integer")
    if episode_index < 0:
        raise ValueError("episode_index must be non-negative")
    if not config.enabled:
        return _nominal(config.scene, config)

    sequence = np.random.SeedSequence([config.seed, episode_index])
    children = sequence.spawn(len(GROUP_ORDER))
    generators = [np.random.Generator(np.random.PCG64(child)) for child in children]
    (
        object_rng,
        texture_rng,
        lighting_rng,
        camera_rng,
        physics_rng,
        distractor_rng,
    ) = generators

    disabled = tuple(name for name in GROUP_ORDER if not getattr(config, name).enabled)
    nominal = _nominal(config.scene, config)

    object_pose = (
        _sample_object_poses(config, object_rng)
        if config.object_pose.enabled
        else nominal.object_pose
    )
    texture = _sample_texture(config, texture_rng) if config.texture.enabled else nominal.texture
    lighting = (
        _sample_lighting(config, lighting_rng) if config.lighting.enabled else nominal.lighting
    )
    camera = _sample_camera(config, camera_rng) if config.camera.enabled else nominal.camera
    physics = _sample_physics(config, physics_rng) if config.physics.enabled else nominal.physics
    distractors = (
        _sample_distractors(config, distractor_rng)
        if config.distractors.enabled
        else nominal.distractors
    )

    return RandomizationSample(
        nominal=False,
        object_pose=object_pose,
        texture=texture,
        lighting=lighting,
        camera=camera,
        physics=physics,
        distractors=distractors,
        disabled_groups=disabled,
    )
