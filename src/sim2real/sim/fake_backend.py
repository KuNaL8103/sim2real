"""Deterministic, NumPy-only kinematic backend used by default in CI.

There is no physics engine here: joint targets are clamped and applied
instantly, the arm uses a deliberately simple analytic forward kinematics, and
RGB-D frames are ray-cast against the table, cubes, and target zones. Identical
seeds and inputs produce byte-identical observations.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from sim2real import contracts
from sim2real.sim.backend import (
    BackendCapabilities,
    CameraFrame,
    CameraIntrinsics,
    CollisionInfo,
    ConfigError,
    Observation,
    RenderSnapshot,
    ResetConfig,
    RobotState,
    SimBackend,
    StepResult,
)
from sim2real.sim.camera import (
    MUJOCO_TO_OPTICAL,
    euler_xyz_to_matrix,
    intrinsics_from_fovy,
    make_pose,
    pose_from_pos_quat,
)
from sim2real.sim.config import SimConfigs, load_sim_configs

if TYPE_CHECKING:
    from sim2real.randomization.sampler import RandomizationSample


def _rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotation_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


class FakeBackend(SimBackend):
    """Seeded kinematic backend with synthetic RGB-D rendering."""

    name = "fake"

    def __init__(self, config_dir: str | Path) -> None:
        self.configs: SimConfigs = load_sim_configs(config_dir)
        self.config_dir = Path(config_dir).expanduser().resolve()
        self.control_rate_hz = float(self.configs.scene.control_rate_hz)
        self.joint_names = tuple(self.configs.panda.joint_names)
        self.joint_limits = np.asarray(self.configs.panda.joint_limits, dtype=np.float64)
        self._intrinsics = {
            name: intrinsics_from_fovy(camera.fovy_deg, camera.width, camera.height)
            for name, camera in self.configs.cameras.items()
        }
        self._arm = np.asarray(self.configs.panda.home_arm, dtype=np.float64)
        self._gripper = float(self.configs.panda.home_gripper)
        self._sim_time_ns = 0
        self._object_poses: dict[str, np.ndarray] = {}
        self._cube_colors: dict[str, np.ndarray] = {}
        self._camera_jitter: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.applied_randomization: dict[str, object] = {}
        self._last_collision = CollisionInfo(0, ())
        self._closed = False
        # Test hooks: simulate a slow renderer without touching MuJoCo timing.
        # delay_only skips the raycast so the test measures scheduling, not
        # NumPy CPU contention.
        self.render_delay_s = 0.0
        self.render_delay_only = False
        self.reset(seed=0)

    # -- backend API -------------------------------------------------------
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            rgbd=True,
            lighting=False,
            materials=False,
            camera_extrinsics=True,
            physics_mass=False,
            physics_friction=False,
            joint_damping=False,
            distractors=False,
            object_pose=True,
            texture=True,
            reasons={
                "rgbd": "analytic ray-cast depth and colour, metres",
                "lighting": "unlit constant shading; no light model",
                "materials": "flat RGBA colours only; no texture assets",
                "camera_extrinsics": "configured poses plus simple analytic FK",
                "physics_mass": "kinematic backend has no mass or contact dynamics",
                "physics_friction": "kinematic backend has no friction",
                "joint_damping": "targets are applied instantly",
                "distractors": "no pooled distractors in the kinematic backend",
                "object_pose": "cube poses are stored in the observation and rendered",
                "texture": "flat RGBA colour jitter; material_pool is approximated",
            },
        )

    def reset(self, seed: int, config: ResetConfig | None = None) -> Observation:
        del config  # reset is always nominal; Task 3 randomization is applied after
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self._arm = np.asarray(self.configs.panda.home_arm, dtype=np.float64).copy()
        self._gripper = float(self.configs.panda.home_gripper)
        self._sim_time_ns = 0
        self._object_poses = {
            name: make_pose(np.asarray(cube.position, dtype=np.float64), np.eye(3))
            for name, cube in self.configs.scene.cubes.items()
        }
        self._cube_colors = {
            name: np.asarray(cube.color, dtype=np.float64).copy()
            for name, cube in self.configs.scene.cubes.items()
        }
        self._camera_jitter = {}
        self.applied_randomization = {}
        self._last_collision = CollisionInfo(0, ())
        return self.get_observation()

    def step(self, joint_target: np.ndarray, gripper_target: float) -> StepResult:
        target = np.asarray(joint_target, dtype=np.float64)
        if target.shape != (7,):
            raise ValueError(f"joint_target must have 7 elements, got {target.shape}")
        if not np.all(np.isfinite(target)):
            raise ValueError("joint_target must be finite")
        self._arm = np.clip(target, self.joint_limits[:, 0], self.joint_limits[:, 1])
        low, high = self.configs.panda.gripper_limits
        self._gripper = float(np.clip(gripper_target, low, high))
        self._sim_time_ns += int(round(1e9 / self.control_rate_hz))
        return StepResult(
            observation=self.get_observation(),
            collision=self._last_collision,
            done=False,
            info={},
        )

    def get_observation(self) -> Observation:
        ee_pose = self._ee_pose()
        state = RobotState(
            joint_positions=self._arm.copy(),
            joint_velocities=np.zeros(7, dtype=np.float64),
            gripper_position=self._gripper,
            ee_pose=ee_pose,
            stamp_ns=self._sim_time_ns,
        )
        return Observation(
            robot_state=state,
            object_poses={name: pose.copy() for name, pose in self._object_poses.items()},
        )

    def render(self, camera: str) -> CameraFrame:
        return self.render_from_snapshot(camera, self.capture_snapshot())

    def capture_snapshot(self) -> RenderSnapshot:
        observation = self.get_observation()
        return RenderSnapshot(stamp_ns=observation.stamp_ns, observation=observation)

    def render_from_snapshot(self, camera: str, snapshot: RenderSnapshot) -> CameraFrame:
        if self.render_delay_s > 0.0:
            time.sleep(self.render_delay_s)
        config = self.configs.camera(camera)
        if self.render_delay_only:
            return CameraFrame(
                rgb=np.zeros((config.height, config.width, 3), dtype=np.uint8),
                depth=np.full((config.height, config.width), np.nan, dtype=np.float32),
                stamp_ns=snapshot.stamp_ns,
                frame_id=config.frame_id,
            )
        intrinsics = self._intrinsics[camera]
        pose = self._camera_pose_for(camera, snapshot.observation.robot_state.ee_pose)
        object_poses = snapshot.observation.object_poses
        stamp_ns = snapshot.stamp_ns
        height, width = config.height, config.width

        us = np.arange(width, dtype=np.float64) + 0.5
        vs = np.arange(height, dtype=np.float64) + 0.5
        u, v = np.meshgrid(us, vs)
        directions = np.stack(
            (
                (u - intrinsics.cx) / intrinsics.fx,
                (v - intrinsics.cy) / intrinsics.fy,
                np.ones_like(u),
            ),
            axis=-1,
        )
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        world_directions = directions @ pose[:3, :3].T
        origin = pose[:3, 3]
        optical_z = directions[..., 2]

        depth = np.full((height, width), np.nan, dtype=np.float64)
        rgb = np.zeros((height, width, 3), dtype=np.float64)
        # Sky gradient: brighter towards the top of the image.
        sky = np.linspace(0.75, 0.35, height)[:, None]
        rgb[..., 0] = sky
        rgb[..., 1] = sky
        rgb[..., 2] = 0.9

        table = self.configs.scene
        table_top = table.table_position[2] + table.table_size[2] / 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            t_plane = (table_top - origin[2]) / world_directions[..., 2]
        hit_plane = (t_plane > 0.0) & np.isfinite(t_plane)
        hit_points = origin + world_directions * t_plane[..., None]
        half_table = np.asarray(table.table_size) / 2.0
        inside_table = (np.abs(hit_points[..., 0] - table.table_position[0]) <= half_table[0]) & (
            np.abs(hit_points[..., 1] - table.table_position[1]) <= half_table[1]
        )
        hit_plane &= inside_table
        plane_depth = t_plane * optical_z
        depth[hit_plane] = plane_depth[hit_plane]
        rgb[hit_plane] = np.array([0.58, 0.48, 0.38])

        # Target zones are painted on the table.
        for zone in table.zones.values():
            dx = hit_points[..., 0] - zone.position[0]
            dy = hit_points[..., 1] - zone.position[1]
            in_zone = hit_plane & (dx * dx + dy * dy <= zone.radius**2)
            rgb[in_zone] = 0.55 * rgb[in_zone] + 0.45 * np.asarray(zone.color)

        for name, pose in object_poses.items():
            cube = table.cubes[name]
            half = np.full(3, cube.size / 2.0)
            lower = pose[:3, 3] - half
            upper = pose[:3, 3] + half
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = (lower - origin) / world_directions
                t2 = (upper - origin) / world_directions
            t_near = np.max(np.minimum(t1, t2), axis=-1)
            t_far = np.min(np.maximum(t1, t2), axis=-1)
            hit_cube = (t_near > 0.0) & (t_near <= t_far)
            cube_depth = t_near * optical_z
            closer = hit_cube & (cube_depth < depth)
            depth[closer] = cube_depth[closer]
            rgb[closer] = self._cube_colors.get(name, np.asarray(cube.color))[:3]

        rgb8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        return CameraFrame(
            rgb=rgb8,
            depth=depth.astype(np.float32),
            stamp_ns=stamp_ns,
            frame_id=config.frame_id,
        )

    def apply_randomization(self, sample: RandomizationSample) -> dict[str, object]:
        """Apply the groups this kinematic backend supports.

        Honours: object pose, flat-RGBA colour jitter, and camera local-pose
        jitter. Lighting, physics, and distractors are unsupported and are
        reported as SKIPPED by the applier. Applied values are recorded in
        ``self.applied_randomization`` for tests.
        """
        from sim2real.randomization.applier import APPLIED, PARTIAL, GroupStatus
        from sim2real.randomization.color import MATERIAL_POOL

        capabilities = self.capabilities()
        results: dict[str, object] = {}
        recorded: dict[str, object] = dict(self.applied_randomization)

        if (
            capabilities.object_pose
            and sample.group_enabled("object_pose")
            and sample.object_pose.poses
        ):
            self._object_poses = {
                name: pose.copy() for name, pose in sample.object_pose.poses.items()
            }
            recorded["object_pose"] = {
                name: pose[:3, 3].tolist() for name, pose in self._object_poses.items()
            }
            results["object_pose"] = GroupStatus(APPLIED, "cube poses stored")

        if capabilities.texture and sample.group_enabled("texture"):
            if sample.texture.mode == "color_jitter" and sample.texture.rgba:
                self._cube_colors = {
                    name: np.asarray(rgba, dtype=np.float64)
                    for name, rgba in sample.texture.rgba.items()
                }
                recorded["texture"] = {
                    name: color.tolist() for name, color in self._cube_colors.items()
                }
                results["texture"] = GroupStatus(APPLIED, "flat RGBA colour jitter")
            elif sample.texture.mode == "material_pool" and sample.texture.pool:
                self._cube_colors = {
                    name: np.asarray(MATERIAL_POOL[pool_name], dtype=np.float64)
                    for name, pool_name in sample.texture.pool.items()
                }
                recorded["texture"] = {
                    name: color.tolist() for name, color in self._cube_colors.items()
                }
                results["texture"] = GroupStatus(
                    PARTIAL,
                    "material_pool approximated as flat RGBA; no texture image swap",
                )

        if capabilities.camera_extrinsics and sample.group_enabled("camera"):
            for camera, translation in sample.camera.translation_jitter.items():
                rotation = sample.camera.rotation_jitter.get(camera, np.zeros(3, dtype=np.float64))
                self._camera_jitter[camera] = (
                    np.asarray(translation, dtype=np.float64).copy(),
                    euler_xyz_to_matrix(rotation),
                )
            recorded["camera"] = {
                camera: {
                    "translation": self._camera_jitter[camera][0].tolist(),
                    "rotation": sample.camera.rotation_jitter[camera].tolist(),
                }
                for camera in self._camera_jitter
            }
            results["camera"] = GroupStatus(APPLIED, "camera local pose jitter")

        self.applied_randomization = recorded
        return results

    def get_camera_info(self, camera: str) -> CameraIntrinsics:
        if camera not in self._intrinsics:
            raise ConfigError(f"unknown camera {camera!r}")
        return self._intrinsics[camera]

    def camera_pose_world(self, camera: str) -> np.ndarray:
        return self._camera_pose_for(camera, self._ee_pose())

    def _camera_pose_for(self, camera: str, ee_pose: np.ndarray) -> np.ndarray:
        config = self.configs.camera(camera)
        local = pose_from_pos_quat(config.position, config.quat_wxyz)
        jitter = self._camera_jitter.get(camera)
        if jitter is not None:
            translation, rotation = jitter
            local = local @ make_pose(translation, rotation)
        if config.parent_frame == contracts.WORLD_FRAME:
            parent = np.eye(4)
        elif config.parent_frame == contracts.PANDA_HAND_FRAME:
            parent = ee_pose
        else:
            raise ConfigError(
                f"camera {camera!r} has unsupported parent_frame {config.parent_frame!r}"
            )
        return parent @ local @ make_pose(np.zeros(3), MUJOCO_TO_OPTICAL)

    def collision_report(self) -> CollisionInfo:
        return self._last_collision

    def link_poses(self) -> dict[str, tuple[str | None, np.ndarray]]:
        ee_pose = self._ee_pose()
        positions = np.linspace(np.zeros(3), ee_pose[:3, 3], 8)
        poses: dict[str, tuple[str | None, np.ndarray]] = {}
        previous: str | None = None
        # panda_link0 is the fixed base; link1..link7 interpolate to the hand.
        base = make_pose(np.zeros(3), np.eye(3))
        poses["panda_link0"] = (None, base)
        previous = "panda_link0"
        for index in range(1, 8):
            pose = make_pose(positions[index], ee_pose[:3, :3])
            poses[f"panda_link{index}"] = (previous, pose)
            previous = f"panda_link{index}"
        poses[contracts.PANDA_HAND_FRAME] = (previous, ee_pose)
        return poses

    def close(self) -> None:
        self._closed = True

    # -- internals ---------------------------------------------------------
    def _ee_pose(self) -> np.ndarray:
        q = self._arm
        position = np.array(
            [
                0.35 + 0.12 * np.sin(q[0]) + 0.06 * np.sin(q[1] + q[2]),
                0.12 * np.sin(q[0]),
                0.60 + 0.08 * np.cos(q[1]) + 0.04 * np.cos(q[3]),
            ],
            dtype=np.float64,
        )
        rotation = _rotation_z(q[0]) @ _rotation_y(np.pi + 0.3 * q[1]) @ _rotation_z(0.5 * q[3])
        return make_pose(position, rotation)
