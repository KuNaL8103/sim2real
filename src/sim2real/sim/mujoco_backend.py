"""Headless MuJoCo backend for the Franka Panda tabletop scene.

MuJoCo 3.x is imported lazily and only when this backend is requested, so the
core library and the fake backend stay dependency-free. Rendering uses EGL by
default; on this WSL2 machine EGL resolves to Mesa llvmpipe (software), which is
slow but correct.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

# Must be set before importing mujoco. Callers can override it explicitly.
os.environ.setdefault("MUJOCO_GL", "egl")

from sim2real import contracts  # noqa: E402
from sim2real.sim.backend import (  # noqa: E402
    BackendCapabilities,
    BackendUnavailableError,
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
from sim2real.sim.camera import (  # noqa: E402
    euler_xyz_to_matrix,
    intrinsics_from_fovy,
    mat_to_quat,
    optical_pose_from_mujoco,
    pose_from_pos_quat,
    quat_wxyz_to_matrix,
)
from sim2real.sim.config import SimConfigs, load_sim_configs  # noqa: E402

if TYPE_CHECKING:
    from sim2real.randomization.sampler import RandomizationSample

CAMERA_MODEL_NAMES = {"overhead": "overhead_cam", "wrist": "wrist_cam"}


class MuJoCoBackend(SimBackend):
    """Absolute joint-position control of a Panda with an RGB-D tabletop."""

    name = "mujoco"

    def __init__(self, config_dir: str | Path) -> None:
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover - exercised by skip logic
            raise BackendUnavailableError(
                "mujoco is not installed; run `uv sync --group sim`"
            ) from exc

        self._mujoco = mujoco
        self.configs: SimConfigs = load_sim_configs(config_dir)
        self.config_dir = Path(config_dir).expanduser().resolve()
        self.control_rate_hz = float(self.configs.scene.control_rate_hz)
        self.joint_names = tuple(self.configs.panda.joint_names)
        self.joint_limits = np.asarray(self.configs.panda.joint_limits, dtype=np.float64)

        model_path = Path(self.configs.panda.model)
        if not model_path.is_file():
            raise ConfigError(
                f"MuJoCo model not found at {model_path}; run scripts/fetch_assets.sh"
            )

        spec = mujoco.MjSpec.from_file(str(model_path))
        self._configure_cameras(spec)
        self.model = spec.compile()
        self.model.opt.timestep = float(self.configs.scene.physics_dt)
        self.data = mujoco.MjData(self.model)

        self._arm_qpos_adr = self._addresses(self.configs.panda.mujoco_joint_names)
        self._arm_dof_adr = self._dof_addresses(self.configs.panda.mujoco_joint_names)
        self._gripper_qpos_adr = int(
            self.model.jnt_qposadr[self._joint_id(self.configs.panda.mujoco_gripper_joint)]
        )
        self._actuator_ids = self._resolve_actuators()
        self._camera_ids = {
            name: self.model.camera(CAMERA_MODEL_NAMES[name]).id for name in contracts.CAMERAS
        }
        self._robot_bodies = {mujoco_name for _, mujoco_name in self.configs.panda.body_frames}
        self._robot_bodies.update({"left_finger", "right_finger"})
        self._cube_bodies = set(self.configs.scene.cubes)
        self._color_renderers: dict[str, object] = {}
        self._depth_renderers: dict[str, object] = {}
        self._intrinsics = {
            name: intrinsics_from_fovy(camera.fovy_deg, camera.width, camera.height)
            for name, camera in self.configs.cameras.items()
        }
        # Nominal model parameters, captured once so reset can always restore
        # them. Task 3 only mutates these inside the reset-quiesce window.
        self._nominal_cube_mass = {
            name: float(self.model.body_mass[self.model.body(name).id])
            for name in self.configs.scene.cubes
        }
        self._nominal_cube_inertia = {
            name: self.model.body_inertia[self.model.body(name).id].copy()
            for name in self.configs.scene.cubes
        }
        self._nominal_cube_friction = {
            name: self.model.geom_friction[self.model.geom(f"{name}_geom").id].copy()
            for name in self.configs.scene.cubes
        }
        self._nominal_cube_rgba = {
            name: self.model.geom_rgba[self.model.geom(f"{name}_geom").id].copy()
            for name in self.configs.scene.cubes
        }
        self._nominal_arm_damping = self.model.dof_damping[self._arm_dof_adr].copy()
        self._nominal_light_diffuse = self.model.light_diffuse.copy()
        self._nominal_light_specular = self.model.light_specular.copy()
        self._nominal_light_dir = self.model.light_dir.copy()
        self._nominal_light_dir0 = self.model.light_dir0.copy()
        self._nominal_light_intensity = self.model.light_intensity.copy()
        self._nominal_cam_pos = self.model.cam_pos.copy()
        self._nominal_cam_quat = self.model.cam_quat.copy()
        self.applied_randomization: dict[str, object] = {}
        self._sim_time_ns = 0
        self._last_collision = CollisionInfo(0, ())
        self._closed = False
        self.reset(seed=0)

    # -- setup helpers -----------------------------------------------------
    def _configure_cameras(self, spec: object) -> None:
        mujoco = self._mujoco
        overhead = self.configs.camera("overhead")
        spec_overhead = spec.camera(CAMERA_MODEL_NAMES["overhead"])
        spec_overhead.pos = overhead.position.tolist()
        spec_overhead.quat = overhead.quat_wxyz.tolist()
        spec_overhead.fovy = float(overhead.fovy_deg)

        wrist = self.configs.camera("wrist")
        hand = spec.body(self.configs.panda.body_frames[-1][1])
        spec_wrist = hand.add_camera()
        spec_wrist.name = CAMERA_MODEL_NAMES["wrist"]
        spec_wrist.pos = wrist.position.tolist()
        spec_wrist.quat = wrist.quat_wxyz.tolist()
        spec_wrist.fovy = float(wrist.fovy_deg)
        # Touch the enum so a MuJoCo API change fails loudly at construction.
        assert mujoco.mjtObj.mjOBJ_CAMERA is not None

    def _joint_id(self, name: str) -> int:
        joint = self.model.joint(name)
        if joint is None:
            raise ConfigError(f"MuJoCo model has no joint named {name!r}")
        return int(joint.id)

    def _addresses(self, names: tuple[str, ...]) -> np.ndarray:
        return np.asarray(
            [self.model.jnt_qposadr[self._joint_id(name)] for name in names], dtype=np.int64
        )

    def _dof_addresses(self, names: tuple[str, ...]) -> np.ndarray:
        return np.asarray(
            [self.model.jnt_dofadr[self._joint_id(name)] for name in names], dtype=np.int64
        )

    def _resolve_actuators(self) -> np.ndarray:
        """Return actuator ids in canonical joint order, then the gripper."""
        ids: list[int] = []
        for index, joint_name in enumerate(self.configs.panda.mujoco_joint_names):
            joint_id = self._joint_id(joint_name)
            actuator_id = int(np.flatnonzero(self.model.actuator_trnid[:, 0] == joint_id)[0])
            if index < 7 and actuator_id != index:
                raise ConfigError(
                    f"actuator order mismatch: {joint_name} maps to actuator {actuator_id}"
                )
            ids.append(actuator_id)
        # The Menagerie gripper actuator drives a tendon, so match it by name.
        gripper = self.model.actuator("actuator8")
        if gripper is None:
            raise ConfigError("MuJoCo model has no gripper actuator named 'actuator8'")
        ids.append(int(gripper.id))
        return np.asarray(ids, dtype=np.int64)

    # -- backend API -------------------------------------------------------
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            rgbd=True,
            lighting=True,
            materials=True,
            camera_extrinsics=True,
            physics_mass=True,
            physics_friction=True,
            joint_damping=True,
            distractors=False,
            object_pose=True,
            texture=True,
            reasons={
                "rgbd": "MuJoCo Renderer with metric depth",
                "lighting": "three explicit lights in scene_tabletop.xml",
                "materials": "checker table material and per-cube RGBA",
                "camera_extrinsics": "camera poses read from the compiled model",
                "physics_mass": "MJCF body inertials",
                "physics_friction": "MJCF geom friction",
                "joint_damping": "Menagerie joints declare damping=1",
                "distractors": (
                    "no pooled parked bodies in the scene; runtime body creation is not used"
                ),
                "object_pose": "free-joint qpos for the three cubes",
                "texture": "geom_rgba colour jitter; material_pool is a flat approximation",
            },
        )

    def reset(self, seed: int, config: ResetConfig | None = None) -> Observation:
        del config  # reset is always nominal; Task 3 randomization is applied after
        if seed < 0:
            raise ValueError("seed must be non-negative")
        data = self.data
        self._restore_nominal_model()
        self._mujoco.mj_resetData(self.model, data)
        data.qpos[self._arm_qpos_adr] = self.configs.panda.home_arm
        data.qpos[self._gripper_qpos_adr] = self.configs.panda.home_gripper
        for name, cube in self.configs.scene.cubes.items():
            joint = self.model.joint(f"{name}_joint")
            address = int(self.model.jnt_qposadr[joint.id])
            data.qpos[address : address + 7] = [*cube.position, 1.0, 0.0, 0.0, 0.0]
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        data.ctrl[self._actuator_ids[:7]] = self.configs.panda.home_arm
        data.ctrl[self._actuator_ids[7]] = self._gripper_ctrl(self.configs.panda.home_gripper)
        self._sim_time_ns = 0
        self._mujoco.mj_forward(self.model, data)
        self._last_collision = self._contacts()
        return self._observation()

    def step(self, joint_target: np.ndarray, gripper_target: float) -> StepResult:
        target = np.asarray(joint_target, dtype=np.float64)
        if target.shape != (7,):
            raise ValueError(f"joint_target must have 7 elements, got {target.shape}")
        target = np.clip(target, self.joint_limits[:, 0], self.joint_limits[:, 1])
        self.data.ctrl[self._actuator_ids[:7]] = target
        self.data.ctrl[self._actuator_ids[7]] = self._gripper_ctrl(gripper_target)

        substeps = max(1, round(1.0 / (self.control_rate_hz * self.model.opt.timestep)))
        for _ in range(substeps):
            self._mujoco.mj_step(self.model, self.data)
        self._sim_time_ns += int(round(1e9 / self.control_rate_hz))
        self._last_collision = self._contacts()
        return StepResult(
            observation=self._observation(),
            collision=self._last_collision,
            done=False,
            info={"substeps": substeps},
        )

    def get_observation(self) -> Observation:
        return self._observation()

    def render(self, camera: str) -> CameraFrame:
        return self.render_from_snapshot(camera, self.capture_snapshot())

    def capture_snapshot(self) -> RenderSnapshot:
        """Copy MjData so rendering never touches the stepping data."""
        data = self._mujoco.MjData(self.model)
        self._mujoco.mj_copyData(data, self.model, self.data)
        observation = self._observation()
        return RenderSnapshot(stamp_ns=observation.stamp_ns, observation=observation, payload=data)

    def render_from_snapshot(self, camera: str, snapshot: RenderSnapshot) -> CameraFrame:
        if camera not in self._camera_ids:
            raise ConfigError(f"unknown camera {camera!r}")
        camera_config = self.configs.camera(camera)
        mujoco_name = CAMERA_MODEL_NAMES[camera]
        data = snapshot.payload if snapshot.payload is not None else self.data

        color_renderer = self._color_renderers.get(camera)
        if color_renderer is None:
            color_renderer = self._mujoco.Renderer(
                self.model, camera_config.height, camera_config.width
            )
            self._color_renderers[camera] = color_renderer
        color_renderer.update_scene(data, camera=mujoco_name)
        rgb = np.asarray(color_renderer.render(), dtype=np.uint8)

        depth_renderer = self._depth_renderers.get(camera)
        if depth_renderer is None:
            depth_renderer = self._mujoco.Renderer(
                self.model, camera_config.height, camera_config.width
            )
            depth_renderer.enable_depth_rendering()
            self._depth_renderers[camera] = depth_renderer
        depth_renderer.update_scene(data, camera=mujoco_name)
        depth = np.asarray(depth_renderer.render(), dtype=np.float32)
        depth[~np.isfinite(depth)] = np.nan
        # MuJoCo reports the scene extent (tens of metres here) for sky pixels;
        # treat those as invalid rather than as a real measurement.
        depth[depth > 20.0] = np.nan

        return CameraFrame(
            rgb=rgb,
            depth=depth,
            stamp_ns=snapshot.stamp_ns,
            frame_id=camera_config.frame_id,
        )

    def apply_randomization(self, sample: RandomizationSample) -> dict[str, object]:
        """Apply a sample to MuJoCo model parameters and object free joints.

        Mapping:

        * object pose -> free-joint ``qpos`` (position + yaw quaternion);
        * texture -> ``geom_rgba`` (``material_pool`` is a flat approximation,
          so it is reported PARTIAL because texture images cannot be swapped);
        * lighting -> ``light_intensity``, ``light_diffuse``/``light_specular``
          tint, and a small rotation of ``light_dir``/``light_dir0``;
        * camera -> ``cam_pos``/``cam_quat`` local jitter (overhead is a world
          camera, wrist is a hand child camera);
        * physics -> ``body_mass`` with proportional ``body_inertia``,
          ``geom_friction``, and ``dof_damping`` for the seven arm joints;
        * distractors -> SKIPPED: the scene has no pooled parked bodies.
        """
        from sim2real.randomization.applier import APPLIED, PARTIAL, SKIPPED, GroupStatus
        from sim2real.randomization.color import MATERIAL_POOL

        capabilities = self.capabilities()
        results: dict[str, object] = {}
        recorded: dict[str, object] = dict(self.applied_randomization)
        if sample.nominal:
            return results

        if (
            capabilities.object_pose
            and sample.group_enabled("object_pose")
            and sample.object_pose.poses
        ):
            for name, pose in sample.object_pose.poses.items():
                joint = self.model.joint(f"{name}_joint")
                address = int(self.model.jnt_qposadr[joint.id])
                quaternion = mat_to_quat(pose[:3, :3])
                self.data.qpos[address : address + 7] = [*pose[:3, 3], *quaternion]
            recorded["object_pose"] = {
                name: pose[:3, 3].tolist() for name, pose in sample.object_pose.poses.items()
            }
            results["object_pose"] = GroupStatus(APPLIED, "free-joint qpos")

        if capabilities.texture and sample.group_enabled("texture"):
            if sample.texture.mode == "color_jitter" and sample.texture.rgba:
                for name, rgba in sample.texture.rgba.items():
                    geom = self.model.geom(f"{name}_geom")
                    self.model.geom_rgba[geom.id] = rgba
                recorded["texture"] = {
                    name: list(rgba) for name, rgba in sample.texture.rgba.items()
                }
                results["texture"] = GroupStatus(APPLIED, "geom_rgba colour jitter")
            elif sample.texture.mode == "material_pool" and sample.texture.pool:
                for name, pool_name in sample.texture.pool.items():
                    geom = self.model.geom(f"{name}_geom")
                    self.model.geom_rgba[geom.id] = MATERIAL_POOL[pool_name]
                recorded["texture"] = dict(sample.texture.pool)
                results["texture"] = GroupStatus(
                    PARTIAL,
                    "material_pool approximated as flat geom_rgba; runtime texture "
                    "image swaps are not supported",
                )

        if capabilities.lighting and sample.group_enabled("lighting"):
            lighting = sample.lighting
            tint = np.asarray(lighting.rgb, dtype=np.float64)
            rotation = euler_xyz_to_matrix(lighting.direction_jitter_rad)
            for index in range(self.model.nlight):
                self.model.light_intensity[index] = (
                    self._nominal_light_intensity[index] * lighting.intensity_scale
                )
                self.model.light_diffuse[index] = self._nominal_light_diffuse[index] * tint
                self.model.light_specular[index] = self._nominal_light_specular[index] * tint
                direction = rotation @ self._nominal_light_dir[index]
                self.model.light_dir[index] = direction
                self.model.light_dir0[index] = direction
            recorded["lighting"] = {
                "intensity_scale": float(lighting.intensity_scale),
                "color_temperature_k": float(lighting.color_temperature_k),
                "rgb": list(lighting.rgb),
            }
            results["lighting"] = GroupStatus(
                APPLIED, "light intensity, direction, and colour-temperature tint"
            )

        if capabilities.camera_extrinsics and sample.group_enabled("camera"):
            for camera, translation in sample.camera.translation_jitter.items():
                camera_id = self._camera_ids[camera]
                self.model.cam_pos[camera_id] = self._nominal_cam_pos[camera_id] + np.asarray(
                    translation
                )
                rotation = euler_xyz_to_matrix(sample.camera.rotation_jitter[camera])
                nominal = quat_wxyz_to_matrix(self._nominal_cam_quat[camera_id])
                self.model.cam_quat[camera_id] = mat_to_quat(rotation @ nominal)
            recorded["camera"] = {
                camera: {
                    "translation": np.asarray(translation).tolist(),
                    "rotation": sample.camera.rotation_jitter[camera].tolist(),
                }
                for camera, translation in sample.camera.translation_jitter.items()
            }
            results["camera"] = GroupStatus(APPLIED, "cam_pos/cam_quat local jitter")

        if capabilities.physics_mass and sample.group_enabled("physics"):
            physics = sample.physics
            if physics.object_mass_kg:
                for name, mass in physics.object_mass_kg.items():
                    body = self.model.body(name)
                    nominal_mass = self._nominal_cube_mass[name]
                    self.model.body_mass[body.id] = mass
                    self.model.body_inertia[body.id] = self._nominal_cube_inertia[name] * (
                        mass / nominal_mass
                    )
            if physics.friction:
                for name, friction in physics.friction.items():
                    geom = self.model.geom(f"{name}_geom")
                    self.model.geom_friction[geom.id] = friction
            if capabilities.joint_damping:
                self.model.dof_damping[self._arm_dof_adr] = (
                    self._nominal_arm_damping * physics.joint_damping_scale
                )
            self._mujoco.mj_setConst(self.model, self.data)
            recorded["physics"] = {
                "object_mass_kg": (
                    {name: float(mass) for name, mass in physics.object_mass_kg.items()}
                    if physics.object_mass_kg
                    else None
                ),
                "friction": (
                    {name: list(value) for name, value in physics.friction.items()}
                    if physics.friction
                    else None
                ),
                "joint_damping_scale": float(physics.joint_damping_scale),
            }
            results["physics"] = GroupStatus(
                APPLIED, "body_mass/inertia, geom_friction, dof_damping"
            )

        if sample.group_enabled("distractors") and sample.distractors.enabled:
            results["distractors"] = GroupStatus(
                SKIPPED,
                "no pooled parked bodies in the scene; runtime body creation is not used",
            )

        self.applied_randomization = recorded
        self._mujoco.mj_forward(self.model, self.data)
        return results

    def get_camera_info(self, camera: str) -> CameraIntrinsics:
        if camera not in self._intrinsics:
            raise ConfigError(f"unknown camera {camera!r}")
        return self._intrinsics[camera]

    def camera_pose_world(self, camera: str) -> np.ndarray:
        if camera not in self._camera_ids:
            raise ConfigError(f"unknown camera {camera!r}")
        camera_id = self._camera_ids[camera]
        return optical_pose_from_mujoco(
            self.data.cam_xpos[camera_id], self.data.cam_xmat[camera_id].reshape(3, 3)
        )

    def collision_report(self) -> CollisionInfo:
        return self._last_collision

    def link_poses(self) -> dict[str, tuple[str | None, np.ndarray]]:
        poses: dict[str, tuple[str | None, np.ndarray]] = {}
        previous: str | None = None
        for ros_name, mujoco_name in self.configs.panda.body_frames:
            body = self.model.body(mujoco_name)
            pose = pose_from_pos_quat(self.data.xpos[body.id], self._mat_to_quat(body.id))
            poses[ros_name] = (previous, pose)
            previous = ros_name
        return poses

    def close(self) -> None:
        if self._closed:
            return
        for renderer in (*self._color_renderers.values(), *self._depth_renderers.values()):
            renderer.close()
        self._color_renderers.clear()
        self._depth_renderers.clear()
        self._closed = True

    # -- internals ---------------------------------------------------------
    def _restore_nominal_model(self) -> None:
        """Restore every model parameter Task 3 may change."""
        for name in self.configs.scene.cubes:
            body = self.model.body(name)
            geom = self.model.geom(f"{name}_geom")
            self.model.body_mass[body.id] = self._nominal_cube_mass[name]
            self.model.body_inertia[body.id] = self._nominal_cube_inertia[name]
            self.model.geom_friction[geom.id] = self._nominal_cube_friction[name]
            self.model.geom_rgba[geom.id] = self._nominal_cube_rgba[name]
        self.model.dof_damping[self._arm_dof_adr] = self._nominal_arm_damping
        self.model.light_diffuse[:] = self._nominal_light_diffuse
        self.model.light_specular[:] = self._nominal_light_specular
        self.model.light_dir[:] = self._nominal_light_dir
        self.model.light_dir0[:] = self._nominal_light_dir0
        self.model.light_intensity[:] = self._nominal_light_intensity
        self.model.cam_pos[:] = self._nominal_cam_pos
        self.model.cam_quat[:] = self._nominal_cam_quat
        self.applied_randomization = {}
        self._mujoco.mj_setConst(self.model, self.data)

    def _gripper_ctrl(self, target: float) -> float:
        low, high = self.configs.panda.gripper_limits
        clamped = float(np.clip(target, low, high))
        # Menagerie remaps 0..0.04 m to 0..255.
        return (clamped - low) / (high - low) * 255.0

    def _mat_to_quat(self, body_id: int) -> np.ndarray:
        from sim2real.sim.camera import mat_to_quat

        return mat_to_quat(self.data.xmat[body_id].reshape(3, 3))

    def _observation(self) -> Observation:
        state = RobotState(
            joint_positions=np.array(self.data.qpos[self._arm_qpos_adr], dtype=np.float64),
            joint_velocities=np.array(self.data.qvel[self._arm_dof_adr], dtype=np.float64),
            gripper_position=float(self.data.qpos[self._gripper_qpos_adr]),
            ee_pose=pose_from_pos_quat(
                self.data.xpos[self.model.body("hand").id],
                self._mat_to_quat(self.model.body("hand").id),
            ),
            stamp_ns=self._sim_time_ns,
        )
        object_poses = {}
        for name in self.configs.scene.cubes:
            body = self.model.body(name)
            object_poses[name] = pose_from_pos_quat(
                self.data.xpos[body.id], self._mat_to_quat(body.id)
            )
        return Observation(robot_state=state, object_poses=object_poses)

    def _contacts(self) -> CollisionInfo:
        details: list[str] = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            body1 = self._body_name(int(self.model.geom_bodyid[contact.geom1]))
            body2 = self._body_name(int(self.model.geom_bodyid[contact.geom2]))
            if body1 in self._robot_bodies and body2 in self._robot_bodies:
                continue
            if body1 == "world" or body2 == "world":
                continue
            if body1 == "table" or body2 == "table":
                continue
            if body1 in self._cube_bodies or body2 in self._cube_bodies:
                details.append(f"{body1}<->{body2}")
        return CollisionInfo(num_unintended_contacts=len(details), details=tuple(details))

    def _body_name(self, body_id: int) -> str:
        name = self._mujoco.mj_id2name(self.model, self._mujoco.mjtObj.mjOBJ_BODY, body_id)
        return name or f"body{body_id}"
