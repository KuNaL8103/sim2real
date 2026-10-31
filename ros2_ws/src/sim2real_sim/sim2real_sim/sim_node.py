"""ROS 2 adapter that publishes simulator state, TF, and RGB-D frames.

The node is deliberately thin: all state lives in ``sim2real.sim`` backends and
all reset logic lives in ``sim2real.sim.reset_handler``.

Stamps: the backend holds an explicit simulation clock that starts at zero on
reset and advances by ``1 / control_rate_hz`` per step. Images and camera info
published in the same callback share that stamp exactly. ``/clock`` is **not**
published; these are sim-relative stamps, not wall time. Downstream consumers
should use ``message_filters.ApproximateTimeSynchronizer(queue=10, slop=0.05)``
for real data; in simulation the stamps are exact.

Joint-command subscription is NOT implemented in Task 2 (deferred to Task 11).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rclpy

# The ROS node runs under system Python, while the core library may be editable
# only in .venv and MuJoCo only exists in .venv-ros. Add those paths when needed;
# .venv-ros deliberately contains no NumPy, so system NumPy stays authoritative.
_bootstrap_root = None
for _parent in Path(__file__).resolve().parents:
    if (_parent / "src" / "sim2real" / "__init__.py").is_file():
        _bootstrap_root = _parent
        break
if importlib.util.find_spec("sim2real") is None and _bootstrap_root is not None:
    sys.path.insert(0, str(_bootstrap_root / "src"))
if importlib.util.find_spec("mujoco") is None and _bootstrap_root is not None:
    _venv_site = _bootstrap_root / ".venv-ros" / "lib" / "python3.12" / "site-packages"
    if _venv_site.is_dir():
        sys.path.append(str(_venv_site))
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image
from sensor_msgs.msg import JointState as JointStateMsg
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from sim2real import contracts
from sim2real.ros_utils.image_conversion import camera_info_fields, image_fields
from sim2real.sim import BackendUnavailableError, create_backend
from sim2real.sim.camera import invert_pose, mat_to_quat, pose_from_pos_quat
from sim2real.sim.reset_handler import reset_environment
from sim2real.sim.ros_names import camera_topics

try:  # The interface package may not be built when this module is imported.
    from sim2real_msgs.srv import ResetEnv
except ImportError:  # pragma: no cover - ROS build provides it
    ResetEnv = None  # type: ignore[assignment,misc]

JOINT_STATE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
)
# RViz's Camera display requests RELIABLE. A reliable publisher is also
# compatible with best-effort (sensor-data) subscribers, so this is the most
# interoperable choice for recorded and live consumers.
CAMERA_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    reliability=ReliabilityPolicy.RELIABLE,
)
STATIC_TF_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


def _stamp(stamp_ns: int) -> Time:
    stamp = Time()
    stamp.sec = int(stamp_ns // 1_000_000_000)
    stamp.nanosec = int(stamp_ns % 1_000_000_000)
    return stamp


def _transform(parent: str, child: str, pose: np.ndarray, stamp_ns: int) -> TransformStamped:
    message = TransformStamped()
    message.header.stamp = _stamp(stamp_ns)
    message.header.frame_id = parent
    message.child_frame_id = child
    message.transform.translation.x = float(pose[0, 3])
    message.transform.translation.y = float(pose[1, 3])
    message.transform.translation.z = float(pose[2, 3])
    w, x, y, z = mat_to_quat(pose[:3, :3])
    message.transform.rotation.x = float(x)
    message.transform.rotation.y = float(y)
    message.transform.rotation.z = float(z)
    message.transform.rotation.w = float(w)
    return message


class SimNode(Node):
    """Publish one backend's state and cameras."""

    def __init__(self) -> None:
        super().__init__("sim_node")
        self.declare_parameter("backend", "fake")
        self.declare_parameter("config_dir", "")
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("camera_rate_hz", 10.0)
        self.declare_parameter("publish_images", True)
        self.declare_parameter("tf_source", "backend_fk")

        backend_name = str(self.get_parameter("backend").value)
        config_dir = str(self.get_parameter("config_dir").value)
        if not config_dir:
            from ament_index_python.packages import get_package_share_directory

            config_dir = os.path.join(get_package_share_directory("sim2real_sim"), "config")

        tf_source = str(self.get_parameter("tf_source").value)
        if tf_source != "backend_fk":
            self.get_logger().warning(
                f"tf_source={tf_source!r} is not implemented; using backend_fk. "
                "A URDF/robot_state_publisher is optional, but RViz RobotModel needs a URDF."
            )
        self._tf_source = "backend_fk"
        self._publish_images = bool(self.get_parameter("publish_images").value)
        self._control_rate = float(self.get_parameter("control_rate_hz").value)
        self._camera_rate = float(self.get_parameter("camera_rate_hz").value)
        if self._control_rate <= 0.0 or self._camera_rate <= 0.0:
            raise ValueError("control_rate_hz and camera_rate_hz must be positive")

        self.get_logger().info(f"starting backend={backend_name!r} config_dir={config_dir!r}")
        self.backend = create_backend(backend_name, config_dir)
        self.get_logger().info(
            f"backend {self.backend.name} ready; control={self._control_rate} Hz "
            f"cameras={self._camera_rate} Hz images={self._publish_images}"
        )
        self._target = np.asarray(self.backend.configs.panda.home_arm, dtype=np.float64)
        self._gripper_target = float(self.backend.configs.panda.home_gripper)

        self._joint_publisher = self.create_publisher(
            JointStateMsg, contracts.JOINT_STATES_TOPIC, JOINT_STATE_QOS
        )
        self._tf_broadcaster = TransformBroadcaster(self)
        self._static_tf_broadcaster = StaticTransformBroadcaster(self, qos=STATIC_TF_QOS)
        self._image_publishers: dict[str, dict[str, Any]] = {}
        for camera in contracts.CAMERAS:
            color_topic, depth_topic, info_topic = camera_topics(camera)
            self._image_publishers[camera] = {
                "color": self.create_publisher(Image, color_topic, CAMERA_QOS),
                "depth": self.create_publisher(Image, depth_topic, CAMERA_QOS),
                "info": self.create_publisher(CameraInfo, info_topic, CAMERA_QOS),
            }

        self._publish_static_transforms()
        if ResetEnv is None:
            raise BackendUnavailableError(
                "sim2real_msgs is not importable; build and source ros2_ws first"
            )
        self._reset_service = self.create_service(
            ResetEnv, contracts.RESET_ENV_SERVICE, self._handle_reset
        )
        self.create_timer(1.0 / self._control_rate, self._control_tick)
        if self._publish_images:
            self.create_timer(1.0 / self._camera_rate, self._camera_tick)

    # -- publications ------------------------------------------------------
    def _publish_static_transforms(self) -> None:
        stamp_ns = self.backend.sim_time_ns
        transforms = [
            _transform(contracts.WORLD_FRAME, contracts.PANDA_BASE_FRAME, np.eye(4), stamp_ns)
        ]
        overhead_pose = self.backend.camera_pose_world(contracts.CAMERA_OVERHEAD)
        transforms.append(
            _transform(
                contracts.WORLD_FRAME,
                contracts.CAMERA_OVERHEAD_OPTICAL_FRAME,
                overhead_pose,
                stamp_ns,
            )
        )
        hand_pose = self.backend.get_observation().robot_state.ee_pose
        wrist_world = self.backend.camera_pose_world(contracts.CAMERA_WRIST)
        wrist_relative = invert_pose(hand_pose) @ wrist_world
        transforms.append(
            _transform(
                contracts.PANDA_HAND_FRAME,
                contracts.CAMERA_WRIST_OPTICAL_FRAME,
                wrist_relative,
                stamp_ns,
            )
        )
        self._static_tf_broadcaster.sendTransform(transforms)

    def _control_tick(self) -> None:
        if not rclpy.ok():
            return
        result = self.backend.step(self._target, self._gripper_target)
        observation = result.observation
        self._publish_joint_state(observation)
        self._publish_dynamic_transforms(observation)

    def _publish_joint_state(self, observation: Any) -> None:
        message = JointStateMsg()
        message.header.stamp = _stamp(observation.stamp_ns)
        message.name = list(contracts.CONTROLLED_JOINT_NAMES)
        message.position = [
            *observation.robot_state.joint_positions.tolist(),
            float(observation.robot_state.gripper_position),
        ]
        message.velocity = [*observation.robot_state.joint_velocities.tolist(), 0.0]
        self._joint_publisher.publish(message)

    def _publish_dynamic_transforms(self, observation: Any) -> None:
        poses = self.backend.link_poses()
        transforms = []
        for frame, (parent, pose) in poses.items():
            if parent is None:
                continue
            parent_pose = poses.get(parent, (None, np.eye(4)))[1]
            relative = invert_pose(parent_pose) @ pose
            transforms.append(_transform(parent, frame, relative, observation.stamp_ns))
        if transforms:
            self._tf_broadcaster.sendTransform(transforms)

    def _camera_tick(self) -> None:
        if not rclpy.ok():
            return
        for camera in contracts.CAMERAS:
            frame = self.backend.render(camera)
            info = self.backend.get_camera_info(camera)
            stamp = _stamp(frame.stamp_ns)
            color_fields = image_fields(
                frame.rgb, contracts.COLOR_ENCODING, frame.frame_id, frame.stamp_ns
            )
            depth_fields = image_fields(
                frame.depth, contracts.DEPTH_ENCODING, frame.frame_id, frame.stamp_ns
            )
            info_fields = camera_info_fields(info, frame.frame_id, frame.stamp_ns)

            color = Image()
            color.header.stamp = stamp
            color.header.frame_id = frame.frame_id
            color.height = int(color_fields["height"])
            color.width = int(color_fields["width"])
            color.encoding = str(color_fields["encoding"])
            color.is_bigendian = int(color_fields["is_bigendian"])
            color.step = int(color_fields["step"])
            color.data = bytes(color_fields["data"])
            self._image_publishers[camera]["color"].publish(color)

            depth = Image()
            depth.header.stamp = stamp
            depth.header.frame_id = frame.frame_id
            depth.height = int(depth_fields["height"])
            depth.width = int(depth_fields["width"])
            depth.encoding = str(depth_fields["encoding"])
            depth.is_bigendian = int(depth_fields["is_bigendian"])
            depth.step = int(depth_fields["step"])
            # sensor_msgs/Image.data is uint8[] even for 32FC1: raw little-endian
            # float32 bytes, exactly as packed by image_fields().
            depth.data = bytes(depth_fields["data"])
            self._image_publishers[camera]["depth"].publish(depth)

            camera_info = CameraInfo()
            camera_info.header.stamp = stamp
            camera_info.header.frame_id = frame.frame_id
            camera_info.height = int(info_fields["height"])
            camera_info.width = int(info_fields["width"])
            camera_info.distortion_model = str(info_fields["distortion_model"])
            camera_info.d = list(info_fields["d"])
            camera_info.k = list(info_fields["k"])
            camera_info.r = list(info_fields["r"])
            camera_info.p = list(info_fields["p"])
            camera_info.binning_x = int(info_fields["binning_x"])
            camera_info.binning_y = int(info_fields["binning_y"])
            self._image_publishers[camera]["info"].publish(camera_info)

    # -- reset service -----------------------------------------------------
    def _handle_reset(self, request: Any, response: Any) -> Any:
        override = None
        if bool(request.has_object_pose_override):
            pose = request.object_pose_override
            override = pose_from_pos_quat(
                np.array([pose.position.x, pose.position.y, pose.position.z]),
                np.array(
                    [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]
                ),
            )
        outcome = reset_environment(
            self.backend,
            int(request.seed),
            randomization_enabled=bool(request.randomization_enabled),
            randomization_profile=str(request.randomization_profile),
            task_instruction=str(request.task_instruction),
            object_pose_override=override,
        )
        response.success = outcome.success
        response.message = outcome.message
        response.applied_seed = outcome.applied_seed
        response.applied_config_json = outcome.applied_config_json
        if outcome.success:
            self._target = np.asarray(self.backend.configs.panda.home_arm, dtype=np.float64)
            self._gripper_target = float(self.backend.configs.panda.home_gripper)
            self.get_logger().info(f"reset applied; seed={outcome.applied_seed}")
        else:
            self.get_logger().warning(f"reset failed: {outcome.message}")
        return response

    def destroy_node(self) -> None:
        try:
            self.backend.close()
        finally:
            super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SimNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        # A timer may lose the race with SIGINT and publish into a closed
        # context. Suppress only that shutdown case; real errors still raise.
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
