"""Task 3 domain randomization: config, sampler, applier, and quiesce."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from sim2real.metadata import validate_metadata_json
from sim2real.randomization import (
    APPLIED,
    GROUP_ORDER,
    PARTIAL,
    SKIPPED,
    apply_randomization,
    kelvin_to_rgb,
    load_config_file,
    load_profile,
    sample,
)
from sim2real.randomization.sampler import RandomizationSample
from sim2real.sim import create_backend
from sim2real.sim.backend import BackendCapabilities, ConfigError
from sim2real.sim.camera import (
    MUJOCO_TO_OPTICAL,
    euler_xyz_to_matrix,
    invert_pose,
    make_pose,
)
from sim2real.sim.config import load_sim_configs
from sim2real.sim.fake_backend import FakeBackend
from sim2real.sim.rendering import RenderCoordinator
from sim2real.sim.reset_handler import reset_environment

CONFIG_DIR = Path("configs/sim")
PROFILE_DIR = Path("configs/randomization")
REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS = REPO_ROOT / "assets" / "mujoco" / "franka_emika_panda"


@pytest.fixture(scope="module")
def configs():
    return load_sim_configs(CONFIG_DIR)


def profile(name: str, configs) -> object:
    return load_profile(name, scene=configs.scene, profiles_dir=PROFILE_DIR)


def _base_yaml() -> dict:
    return yaml.safe_load((PROFILE_DIR / "default.yaml").read_text(encoding="utf-8"))


# -- determinism -------------------------------------------------------------


def test_same_seed_and_index_are_identical_and_seeds_differ(configs):
    config = profile("default", configs)
    first = sample(config, 5)
    second = sample(config, 5)
    assert first.to_dict() == second.to_dict()

    other_index = sample(config, 6)
    assert first.to_dict() != other_index.to_dict()

    other_seed = sample(replace(config, seed=config.seed + 1), 5)
    assert first.to_dict() != other_seed.to_dict()


def test_order_independence_and_no_global_rng_state(configs):
    config = profile("default", configs)
    fresh = sample(config, 5).to_dict()
    for index in range(5):
        sample(config, index)
    after = sample(config, 5).to_dict()
    assert after == fresh

    np.random.seed(1234)
    expected = np.random.random(4)
    np.random.seed(1234)
    sample(config, 0)
    np.testing.assert_array_equal(np.random.random(4), expected)


def test_disabling_lighting_does_not_change_object_pose(configs):
    config = profile("default", configs)
    base = sample(config, 5)
    without_lighting = sample(replace(config, lighting=replace(config.lighting, enabled=False)), 5)
    assert "lighting" in without_lighting.disabled_groups
    for name, pose in base.object_pose.poses.items():
        np.testing.assert_array_equal(without_lighting.object_pose.poses[name], pose)


# -- bounds ------------------------------------------------------------------


def test_1000_samples_stay_within_every_configured_range(configs):
    config = profile("default", configs)
    (x_bounds, y_bounds) = config.object_pose.translation_bounds_m
    for index in range(1000):
        result = sample(config, index)
        positions: list[tuple[float, float]] = []
        for name, pose in result.object_pose.poses.items():
            x, y, z = (float(value) for value in pose[:3, 3])
            assert x_bounds[0] <= x <= x_bounds[1]
            assert y_bounds[0] <= y <= y_bounds[1]
            assert z == pytest.approx(config.scene.cube_center_z_m[name])
            yaw = math.atan2(pose[1, 0], pose[0, 0])
            assert config.object_pose.yaw_range_rad[0] - 1e-9 <= yaw
            assert yaw <= config.object_pose.yaw_range_rad[1] + 1e-9
            positions.append((x, y))
        for first in range(len(positions)):
            for second in range(first + 1, len(positions)):
                distance = math.dist(positions[first], positions[second])
                assert distance >= config.object_pose.min_separation_m - 1e-12

        for rgba in result.texture.rgba.values():
            for channel, (low, high) in zip(rgba, config.texture.rgba_range, strict=True):
                assert low <= channel <= high
        assert (
            config.lighting.intensity_range[0]
            <= result.lighting.intensity_scale
            <= config.lighting.intensity_range[1]
        )
        assert (
            config.lighting.color_temperature_range_k[0]
            <= result.lighting.color_temperature_k
            <= config.lighting.color_temperature_range_k[1]
        )
        for camera, jitter in result.camera.translation_jitter.items():
            for value, magnitude in zip(jitter, config.camera.translation_jitter_m, strict=True):
                assert abs(value) <= magnitude + 1e-12
            assert camera in config.camera.apply_to
        for _camera, jitter in result.camera.rotation_jitter.items():
            for value, magnitude in zip(jitter, config.camera.rotation_jitter_rad, strict=True):
                assert abs(value) <= magnitude + 1e-12
        assert result.physics.object_mass_kg is not None
        for mass in result.physics.object_mass_kg.values():
            assert config.physics.object_mass_kg[0] <= mass <= config.physics.object_mass_kg[1]
        assert result.physics.friction is not None
        for friction in result.physics.friction.values():
            assert config.physics.friction.sliding[0] <= friction[0]
            assert friction[0] <= config.physics.friction.sliding[1]
            assert config.physics.friction.torsional[0] <= friction[1]
            assert friction[1] <= config.physics.friction.torsional[1]
            assert config.physics.friction.rolling[0] <= friction[2]
            assert friction[2] <= config.physics.friction.rolling[1]
        assert (
            config.physics.joint_damping_scale[0]
            <= result.physics.joint_damping_scale
            <= config.physics.joint_damping_scale[1]
        )


# -- validation --------------------------------------------------------------


def _set_yaw_inverted(body: dict) -> None:
    body["object_pose"]["yaw_range_rad"] = [1.0, -1.0]


def _set_negative_mass(body: dict) -> None:
    body["physics"]["object_mass_kg"] = [-1.0, 0.1]


def _set_nan_mass(body: dict) -> None:
    body["physics"]["object_mass_kg"] = [float("nan"), 0.1]


def _set_wrong_type(body: dict) -> None:
    body["physics"]["object_mass_kg"] = "heavy"


def _set_unknown_key(body: dict) -> None:
    body["bogus"] = 1


def _set_outside_workspace(body: dict) -> None:
    body["object_pose"]["translation_bounds_m"] = [[0.0, 2.0], [-0.1, 0.1]]


def _set_camera_jitter_too_large(body: dict) -> None:
    body["camera"]["translation_jitter_m"] = [0.5, 0.0, 0.0]


def _set_bad_texture_mode(body: dict) -> None:
    body["texture"]["mode"] = "sparkle"


def _set_enabled_empty_distractor_pool(body: dict) -> None:
    body["distractors"]["enabled"] = True
    body["distractors"]["pool"] = []


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (_set_yaw_inverted, "randomization.object_pose.yaw_range_rad"),
        (_set_negative_mass, "randomization.physics.object_mass_kg"),
        (_set_nan_mass, "randomization.physics.object_mass_kg"),
        (_set_wrong_type, "randomization.physics.object_mass_kg"),
        (_set_unknown_key, "randomization: unknown key"),
        (_set_outside_workspace, "translation_bounds_m"),
        (_set_camera_jitter_too_large, "randomization.camera.translation_jitter_m"),
        (_set_bad_texture_mode, "randomization.texture.mode"),
        (_set_enabled_empty_distractor_pool, "randomization.distractors.pool"),
    ],
)
def test_invalid_configs_raise_path_qualified_errors(tmp_path, configs, mutate, expected):
    data = _base_yaml()
    mutate(data["randomization"])
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match=expected):
        load_config_file(path, configs.scene)


def test_profiles_load_and_hash_tracks_content(configs):
    loaded = {name: profile(name, configs) for name in ("none", "light", "default", "heavy")}
    assert len({config.config_hash for config in loaded.values()}) == 4
    assert loaded["none"].enabled is False
    assert loaded["default"].enabled is True


# -- nominal -----------------------------------------------------------------


def test_none_profile_and_disabled_config_are_nominal(configs):
    config = profile("none", configs)
    result = sample(config, 0)
    assert result.nominal is True
    assert result.disabled_groups == GROUP_ORDER
    for name, cube in configs.scene.cubes.items():
        np.testing.assert_allclose(result.object_pose.poses[name][:3, 3], cube.position)
        np.testing.assert_allclose(result.object_pose.poses[name][:3, :3], np.eye(3))
        np.testing.assert_allclose(result.texture.rgba[name][:3], cube.color)
        assert result.texture.rgba[name][3] == pytest.approx(1.0)
    assert result.lighting.intensity_scale == 1.0
    assert result.lighting.direction_jitter_rad == (0.0, 0.0, 0.0)
    assert result.lighting.color_temperature_k == pytest.approx(6500.0)
    for camera in config.camera.apply_to:
        np.testing.assert_array_equal(result.camera.translation_jitter[camera], np.zeros(3))
        np.testing.assert_array_equal(result.camera.rotation_jitter[camera], np.zeros(3))
    assert result.physics.object_mass_kg is None
    assert result.physics.friction is None
    assert result.physics.joint_damping_scale == 1.0
    assert result.distractors.enabled is False

    disabled = sample(replace(profile("default", configs), enabled=False), 0)
    assert disabled.nominal is True


# -- capability gating and fake backend -------------------------------------


class LimitedFakeBackend(FakeBackend):
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            rgbd=True,
            lighting=False,
            materials=False,
            camera_extrinsics=False,
            physics_mass=False,
            physics_friction=False,
            joint_damping=False,
            distractors=False,
            object_pose=False,
            texture=False,
            reasons={
                name: f"limited test backend: {name} disabled"
                for name in BackendCapabilities.FLAG_NAMES
            },
        )


def test_capability_gating_reports_skips_and_applies_nothing(configs):
    backend = LimitedFakeBackend(CONFIG_DIR)
    try:
        result = apply_randomization(backend, sample(profile("default", configs), 3))
        assert set(result.groups) == set(GROUP_ORDER)
        for name, status in result.groups.items():
            assert status.status == SKIPPED, name
            assert status.reason
        assert backend.applied_randomization == {}
    finally:
        backend.close()


def test_fake_backend_applies_and_reset_restores_nominal(configs):
    backend = create_backend("fake", CONFIG_DIR)
    try:
        nominal_camera = backend.camera_pose_world("overhead")
        result = sample(profile("default", configs), 3)
        report = apply_randomization(backend, result)

        assert report.groups["object_pose"].status == APPLIED
        assert report.groups["texture"].status == APPLIED
        assert report.groups["camera"].status == APPLIED
        assert report.groups["lighting"].status == SKIPPED
        assert report.groups["physics"].status == SKIPPED
        assert report.groups["distractors"].status == SKIPPED
        assert set(backend.applied_randomization) == {"object_pose", "texture", "camera"}

        observation = backend.get_observation()
        for name, pose in result.object_pose.poses.items():
            np.testing.assert_allclose(observation.object_poses[name], pose)
        assert not np.allclose(backend.camera_pose_world("overhead"), nominal_camera)

        backend.reset(seed=7)
        assert backend.applied_randomization == {}
        reset_observation = backend.get_observation()
        for name, cube in configs.scene.cubes.items():
            np.testing.assert_allclose(reset_observation.object_poses[name][:3, 3], cube.position)
            np.testing.assert_allclose(backend._cube_colors[name], cube.color)
        np.testing.assert_allclose(backend.camera_pose_world("overhead"), nominal_camera)
    finally:
        backend.close()


@pytest.mark.sim
def test_mujoco_applies_mass_friction_damping_camera_and_resets(configs):
    pytest.importorskip("mujoco")
    if not (ASSETS / "panda.xml").is_file() or not (ASSETS / "assets").is_dir():
        pytest.skip("Menagerie assets missing; run scripts/fetch_assets.sh")

    backend = create_backend("mujoco", CONFIG_DIR)
    try:
        result = sample(profile("default", configs), 3)
        backend.reset(seed=7)
        model = backend.model
        nominal_mass = {
            name: float(model.body_mass[model.body(name).id]) for name in configs.scene.cubes
        }
        nominal_inertia = {
            name: model.body_inertia[model.body(name).id].copy() for name in configs.scene.cubes
        }
        nominal_friction = {
            name: model.geom_friction[model.geom(f"{name}_geom").id].copy()
            for name in configs.scene.cubes
        }
        nominal_damping = model.dof_damping[backend._arm_dof_adr].copy()
        overhead = backend._camera_ids["overhead"]
        nominal_camera_pos = model.cam_pos[overhead].copy()

        report = apply_randomization(backend, result)
        for group in ("object_pose", "texture", "lighting", "camera", "physics"):
            assert report.groups[group].status == APPLIED, group
        assert report.groups["distractors"].status == SKIPPED

        # Regression: the physics group calls mj_setConst, which resets MjData to
        # qpos0. The arm must stay at home and the sampled object poses must
        # survive; Task 3 did not assert this and the reset silently zeroed both.
        np.testing.assert_allclose(backend.data.qpos[backend._arm_qpos_adr], configs.panda.home_arm)
        for name, pose in result.object_pose.poses.items():
            address = int(model.jnt_qposadr[model.joint(f"{name}_joint").id])
            np.testing.assert_allclose(backend.data.qpos[address : address + 3], pose[:3, 3])

        assert result.physics.object_mass_kg is not None
        assert result.physics.friction is not None
        for name in configs.scene.cubes:
            body = model.body(name)
            geom = model.geom(f"{name}_geom")
            mass = result.physics.object_mass_kg[name]
            assert model.body_mass[body.id] == pytest.approx(mass)
            np.testing.assert_allclose(
                model.body_inertia[body.id], nominal_inertia[name] * (mass / nominal_mass[name])
            )
            np.testing.assert_allclose(model.geom_friction[geom.id], result.physics.friction[name])
        np.testing.assert_allclose(
            model.dof_damping[backend._arm_dof_adr],
            nominal_damping * result.physics.joint_damping_scale,
        )
        np.testing.assert_allclose(
            model.cam_pos[overhead],
            nominal_camera_pos + result.camera.translation_jitter["overhead"],
        )
        jittered_camera = backend.camera_pose_world("overhead")

        backend.reset(seed=7)
        for name in configs.scene.cubes:
            body = model.body(name)
            geom = model.geom(f"{name}_geom")
            assert model.body_mass[body.id] == pytest.approx(nominal_mass[name])
            np.testing.assert_allclose(model.body_inertia[body.id], nominal_inertia[name])
            np.testing.assert_allclose(model.geom_friction[geom.id], nominal_friction[name])
        np.testing.assert_allclose(model.dof_damping[backend._arm_dof_adr], nominal_damping)
        np.testing.assert_allclose(model.cam_pos[overhead], nominal_camera_pos)
        assert not np.allclose(backend.camera_pose_world("overhead"), jittered_camera)
    finally:
        backend.close()


# -- colour ------------------------------------------------------------------


def test_kelvin_to_rgb_qualitative_properties():
    warm = kelvin_to_rgb(3000.0)
    cool = kelvin_to_rgb(9000.0)
    assert warm[0] >= warm[2]
    assert cool[2] >= cool[0]
    for kelvin in (1000.0, 3000.0, 6500.0, 9000.0, 40000.0):
        assert all(0.0 <= channel <= 1.0 for channel in kelvin_to_rgb(kelvin))
    ratios = [
        kelvin_to_rgb(kelvin)[2] / kelvin_to_rgb(kelvin)[0]
        for kelvin in (2000.0, 4000.0, 6000.0, 8000.0, 10000.0)
    ]
    assert all(first <= second for first, second in zip(ratios[:-1], ratios[1:], strict=True))
    for invalid in (999.0, 40001.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            kelvin_to_rgb(invalid)


# -- quiesce and camera consistency -----------------------------------------


class GuardedFakeBackend(FakeBackend):
    def __init__(self, config_dir):
        self.render_active = False
        self.render_violations = 0
        super().__init__(config_dir)

    def render_from_snapshot(self, camera, snapshot):
        self.render_active = True
        try:
            return super().render_from_snapshot(camera, snapshot)
        finally:
            self.render_active = False

    def apply_randomization(self, result: RandomizationSample):
        if self.render_active:
            self.render_violations += 1
        return super().apply_randomization(result)


def test_reset_quiesce_blocks_apply_and_rejects_stale_snapshots(configs):
    backend = GuardedFakeBackend(CONFIG_DIR)
    backend.render_delay_s = 0.3
    coordinator = RenderCoordinator(backend)
    try:
        stale = coordinator.tag(backend.capture_snapshot())
        finished = threading.Event()

        def slow_render() -> None:
            coordinator.render("overhead", stale)
            finished.set()

        thread = threading.Thread(target=slow_render)
        thread.start()
        time.sleep(0.05)

        coordinator.begin_reset()
        assert coordinator.render_in_flight is False
        outcome = reset_environment(
            backend,
            7,
            randomization_enabled=True,
            randomization_profile="default",
            randomization_dir=PROFILE_DIR,
        )
        coordinator.end_reset()
        thread.join(timeout=2.0)

        assert outcome.success is True
        assert finished.is_set()
        assert backend.render_violations == 0
        assert coordinator.render("overhead", stale) is None
        assert coordinator.stats()["skipped_stale"] == 1

        fresh = coordinator.tag(backend.capture_snapshot())
        assert coordinator.render("overhead", fresh) is not None
    finally:
        coordinator.close()
        backend.close()


def test_camera_jitter_is_reflected_in_camera_pose_world(configs):
    backend = create_backend("fake", CONFIG_DIR)
    try:
        nominal_overhead = backend.camera_pose_world("overhead")
        hand = backend.get_observation().robot_state.ee_pose
        nominal_wrist_relative = invert_pose(hand) @ backend.camera_pose_world("wrist")

        result = sample(profile("default", configs), 3)
        report = apply_randomization(backend, result)
        assert report.groups["camera"].status == APPLIED

        # Jitter is applied in the camera's MuJoCo local frame, before the
        # MuJoCo->optical conversion, so the optical pose is
        # ``nominal @ M @ jitter @ M`` (M is its own inverse).
        conversion = make_pose(np.zeros(3), MUJOCO_TO_OPTICAL)
        expected_overhead = (
            nominal_overhead
            @ conversion
            @ make_pose(
                result.camera.translation_jitter["overhead"],
                euler_xyz_to_matrix(result.camera.rotation_jitter["overhead"]),
            )
            @ conversion
        )
        np.testing.assert_allclose(backend.camera_pose_world("overhead"), expected_overhead)

        hand_after = backend.get_observation().robot_state.ee_pose
        expected_wrist = (
            nominal_wrist_relative
            @ conversion
            @ make_pose(
                result.camera.translation_jitter["wrist"],
                euler_xyz_to_matrix(result.camera.rotation_jitter["wrist"]),
            )
            @ conversion
        )
        np.testing.assert_allclose(
            invert_pose(hand_after) @ backend.camera_pose_world("wrist"), expected_wrist
        )
    finally:
        backend.close()


# -- service-level validation ------------------------------------------------


def test_reset_outcome_json_is_valid_and_records_profile(configs):
    backend = create_backend("fake", CONFIG_DIR)
    try:
        outcome = reset_environment(
            backend,
            7,
            randomization_enabled=True,
            randomization_profile="default",
            task_instruction="pick the blue cube",
            randomization_dir=PROFILE_DIR,
        )
        assert outcome.success is True
        payload = validate_metadata_json(outcome.applied_config_json)
        assert payload["profile"] == "default"
        assert payload["config_hash"] == profile("default", configs).config_hash
        assert payload["sample"]["nominal"] is False
        assert payload["report"]["groups"]["object_pose"]["status"] == APPLIED
        assert payload["report"]["groups"]["lighting"]["status"] == SKIPPED
    finally:
        backend.close()


def test_material_pool_profile_is_partial_on_mujoco_capabilities(configs):
    # The applier reports PARTIAL only when the backend says it executed with a
    # limitation; the fake backend supports flat colours, so verify its result.
    backend = create_backend("fake", CONFIG_DIR)
    try:
        result = sample(profile("heavy", configs), 0)
        report = apply_randomization(backend, result)
        assert result.texture.mode == "material_pool"
        assert report.groups["texture"].status == PARTIAL
        assert "approximated" in report.groups["texture"].reason
    finally:
        backend.close()
