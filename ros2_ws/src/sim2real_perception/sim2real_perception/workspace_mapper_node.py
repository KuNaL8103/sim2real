"""ROS 2 adapter for RGB-D workspace mapping.

The node is thin plumbing: it decodes ``rgb8``/``32FC1`` images with NumPy,
looks camera poses up with tf2 at each image stamp, and calls the pure
``sim2real.mapping.WorkspaceMapper``.  Both cameras fuse into the same volume;
cross-camera time synchronization is **not** required because each frame uses
its own TF lookup and stamp.

Topics:
  /camera/<name>/color/image_raw  in   sensor_msgs/Image (rgb8)
  /camera/<name>/depth/image_raw  in   sensor_msgs/Image (32FC1, metres)
  /camera/<name>/camera_info      in   sensor_msgs/CameraInfo
  /tf_static                      in   tf2_msgs/TFMessage (camera jitter reset)
  /workspace/pointcloud           out  sensor_msgs/PointCloud2 (xyz+rgb)
  /workspace/tsdf_mesh            out  visualization_msgs/Marker (TRIANGLE_LIST)
Service:
  /workspace/reset_map            std_srvs/Trigger
"""

from __future__ import annotations

import importlib.util
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import rclpy

_bootstrap_root = None
for _parent in Path(__file__).resolve().parents:
    if (_parent / "src" / "sim2real" / "__init__.py").is_file():
        _bootstrap_root = _parent
        break
if importlib.util.find_spec("sim2real") is None and _bootstrap_root is not None:
    sys.path.insert(0, str(_bootstrap_root / "src"))
# .venv-ros has open3d/scipy installed --no-deps; append so system NumPy stays
# the visible version.
_venv_site = (
    _bootstrap_root / ".venv-ros" / "lib" / "python3.12" / "site-packages"
    if _bootstrap_root is not None
    else None
)
if (
    _venv_site is not None
    and _venv_site.is_dir()
    and any(importlib.util.find_spec(name) is None for name in ("open3d", "scipy"))
):
    sys.path.append(str(_venv_site))

import tf2_ros  # noqa: E402
from builtin_interfaces.msg import Time  # noqa: E402
from geometry_msgs.msg import Point as MarkerPoint  # noqa: E402
from message_filters import ApproximateTimeSynchronizer, Subscriber  # noqa: E402
from rclpy.executors import ExternalShutdownException  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import (  # noqa: E402
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Duration  # noqa: E402
from rclpy.time import Time as RclpyTime  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField  # noqa: E402
from std_msgs.msg import ColorRGBA as MarkerColor  # noqa: E402
from std_srvs.srv import Trigger  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402
from visualization_msgs.msg import Marker  # noqa: E402

from sim2real import contracts  # noqa: E402
from sim2real.mapping import (  # noqa: E402
    RobotSelfFilter,
    WorkspaceMapper,
    load_mapping_config,
)
from sim2real.ros_utils.pointcloud import (  # noqa: E402
    build_pointcloud2,
    mesh_to_marker,
)
from sim2real.sim.qos import (  # noqa: E402
    MAP_CAMERA_SUB_QOS,
    MAP_QOS,
    MAP_STATIC_TF_QOS,
)
from sim2real.sim.ros_names import camera_topics  # noqa: E402


def _qos_profile(spec) -> QoSProfile:
    reliability = (
        ReliabilityPolicy.RELIABLE
        if spec.reliability == "reliable"
        else ReliabilityPolicy.BEST_EFFORT
    )
    durability = (
        DurabilityPolicy.TRANSIENT_LOCAL
        if spec.durability == "transient_local"
        else DurabilityPolicy.VOLATILE
    )
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=spec.depth,
        reliability=reliability,
        durability=durability,
    )


def _stamp_message(stamp_ns: int) -> Time:
    stamp_ns = int(stamp_ns)
    return Time(sec=stamp_ns // 1_000_000_000, nanosec=stamp_ns % 1_000_000_000)


def quaternion_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_to_matrix(transform: Any) -> np.ndarray:
    translation = transform.translation
    rotation = transform.rotation
    matrix = np.eye(4)
    matrix[:3, :3] = quaternion_matrix(rotation.x, rotation.y, rotation.z, rotation.w)
    matrix[:3, 3] = [translation.x, translation.y, translation.z]
    return matrix


def _decode_color(message: Image) -> np.ndarray:
    if message.encoding != contracts.COLOR_ENCODING:
        raise ValueError(f"expected {contracts.COLOR_ENCODING}, got {message.encoding}")
    row = np.frombuffer(message.data, dtype=np.uint8)
    expected = message.height * message.step
    if row.size < expected:
        raise ValueError("color data is shorter than height * step")
    row = row[:expected].reshape(message.height, message.step)
    return np.ascontiguousarray(row[:, : message.width * 3]).reshape(
        message.height, message.width, 3
    )


def _decode_depth(message: Image) -> np.ndarray:
    if message.encoding != contracts.DEPTH_ENCODING:
        raise ValueError(f"expected {contracts.DEPTH_ENCODING}, got {message.encoding}")
    if message.step % 4 != 0:
        raise ValueError("32FC1 step must be a multiple of 4")
    row = np.frombuffer(message.data, dtype=np.float32)
    expected = message.height * (message.step // 4)
    if row.size < expected:
        raise ValueError("depth data is shorter than height * step")
    row = row[:expected].reshape(message.height, message.step // 4)
    return np.ascontiguousarray(row[:, : message.width]).reshape(message.height, message.width)


def _decode_intrinsics(message: CameraInfo) -> np.ndarray:
    if len(message.k) != 9:
        raise ValueError("camera_info.k must have 9 entries")
    return np.asarray(message.k, dtype=np.float64).reshape(3, 3)


class _TfProvider:
    def __init__(self, node: Node, buffer: tf2_ros.Buffer, timeout_s: float) -> None:
        self._node = node
        self._buffer = buffer
        self._timeout = float(timeout_s)

    def lookup(
        self, target_frame: str, source_frame: str, stamp: float | None = None
    ) -> np.ndarray | None:
        when = RclpyTime() if stamp is None else RclpyTime(nanoseconds=int(stamp))
        try:
            transform = self._buffer.lookup_transform(
                target_frame,
                source_frame,
                when,
                timeout=Duration(seconds=self._timeout),
            )
        except tf2_ros.TransformException:
            return None
        except Exception:  # pragma: no cover - tf2 raises several subclasses
            return None
        return transform_to_matrix(transform.transform)


class WorkspaceMapperNode(Node):
    def __init__(self) -> None:
        super().__init__("workspace_mapper_node")
        self.declare_parameter("config_file", "")
        self.declare_parameter("publish_rate_hz", 0.0)
        self.declare_parameter("tf_timeout_s", 0.0)

        config_file = str(self.get_parameter("config_file").value or "")
        self.config = load_mapping_config(config_file or None)
        publish_rate = float(self.get_parameter("publish_rate_hz").value or 0.0)
        self.publish_rate_hz = publish_rate or self.config.publish_rate_hz
        tf_timeout = float(self.get_parameter("tf_timeout_s").value or 0.0)
        self.tf_timeout_s = tf_timeout or self.config.tf_timeout_s

        self._wall_clock = time.monotonic
        self._tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._tf_provider = _TfProvider(self, self._tf_buffer, self.tf_timeout_s)
        # The sim clock restarts at 0 on reset. Keep the latest static transforms
        # so a map reset can clear the tf2 buffer (which otherwise ignores the
        # new, older-stamped dynamic TF as TF_OLD_DATA) and re-add them.
        self._last_static_transforms: list = []
        self.mapper = WorkspaceMapper(self.config, self._tf_provider, clock=self._wall_clock)
        self.self_filter = RobotSelfFilter(self.config.self_filter)
        self._integration_ms_total = 0.0
        self._integration_count = 0
        self._integration_ms_ema = 0.0

        camera_qos = _qos_profile(MAP_CAMERA_SUB_QOS)
        self._synchronizers = []
        for camera in contracts.CAMERAS:
            color_topic, depth_topic, info_topic = camera_topics(camera)
            color_sub = Subscriber(self, Image, color_topic, qos_profile=camera_qos)
            depth_sub = Subscriber(self, Image, depth_topic, qos_profile=camera_qos)
            info_sub = Subscriber(self, CameraInfo, info_topic, qos_profile=camera_qos)
            sync = ApproximateTimeSynchronizer(
                [color_sub, depth_sub, info_sub],
                queue_size=10,
                slop=self.config.sync_slop_s,
            )
            sync.registerCallback(
                lambda color, depth, info, camera=camera: self._on_images(
                    camera, color, depth, info
                )
            )
            self._synchronizers.append(sync)

        self.create_subscription(
            TFMessage,
            contracts.TF_STATIC_TOPIC,
            self._on_static_tf,
            _qos_profile(MAP_STATIC_TF_QOS),
        )
        map_qos = _qos_profile(MAP_QOS)
        self._cloud_publisher = self.create_publisher(
            PointCloud2, contracts.WORKSPACE_POINTCLOUD_TOPIC, map_qos
        )
        self._mesh_publisher = self.create_publisher(
            Marker, contracts.WORKSPACE_TSDF_MESH_TOPIC, map_qos
        )
        self.create_service(Trigger, contracts.WORKSPACE_RESET_MAP_SERVICE, self._on_reset_map)
        self._publish_timer = self.create_timer(1.0 / max(self.publish_rate_hz, 0.1), self._publish)
        self.get_logger().info(
            "workspace_mapper_node started: "
            f"voxel={self.config.voxel_size_m} m, "
            f"bounds={self.config.workspace_bounds.minimum}.."
            f"{self.config.workspace_bounds.maximum}, "
            f"publish={self.publish_rate_hz:.1f} Hz"
        )

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    def _on_images(self, camera, color: Image, depth: Image, info: CameraInfo) -> None:
        try:
            rgb = _decode_color(color)
            depth_m = _decode_depth(depth)
            intrinsics = _decode_intrinsics(info)
            stamp_ns = int(color.header.stamp.sec) * 1_000_000_000 + int(color.header.stamp.nanosec)
            started = time.perf_counter()
            result = self.mapper.integrate_frame(
                camera,
                rgb,
                depth_m,
                intrinsics,
                stamp_ns,
                color.header.frame_id,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._integration_ms_total += elapsed_ms
            self._integration_count += 1
            self._integration_ms_ema = (
                elapsed_ms
                if self._integration_count == 1
                else 0.9 * self._integration_ms_ema + 0.1 * elapsed_ms
            )
            if self._integration_count % 20 == 0:
                self.get_logger().info(
                    "integration mean="
                    f"{self._integration_ms_total / self._integration_count:.1f} ms "
                    f"(n={self._integration_count}), last={elapsed_ms:.1f} ms"
                )
            if (
                not result.accepted
                and result.reason == "NO_TF"
                and self.mapper.tf_failure_log_due()
            ):
                self.get_logger().warning(
                    f"skipping {camera} frame: no TF ({self.mapper.last_tf_failure})"
                )
        except Exception as exc:  # never let a malformed frame kill the node
            self.get_logger().error(f"failed to integrate {camera} frame: {exc}")

    def _on_static_tf(self, message: TFMessage) -> None:
        try:
            transforms = {}
            for stamped in message.transforms:
                key = (stamped.header.frame_id, stamped.child_frame_id)
                transforms[key] = transform_to_matrix(stamped.transform)
            self._last_static_transforms = list(message.transforms)
            if self.mapper.on_static_transforms(transforms):
                # The sim clock restarts at 0; drop old dynamic TF so the new
                # samples are not rejected as TF_OLD_DATA, then restore static.
                self._reset_tf_buffer()
                self.get_logger().warning(
                    "camera static transform changed beyond epsilon; map reset"
                )
        except Exception as exc:
            self.get_logger().error(f"failed to process /tf_static: {exc}")

    def _reset_tf_buffer(self) -> None:
        self._tf_buffer.clear()
        for stamped in self._last_static_transforms:
            self._tf_buffer.set_transform_static(stamped, "workspace_mapper")

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    def _link_positions(self) -> dict[str, np.ndarray]:
        positions: dict[str, np.ndarray] = {}
        for frame in self.config.self_filter.link_frames:
            try:
                transform = self._tf_buffer.lookup_transform(
                    contracts.WORLD_FRAME, frame, RclpyTime()
                )
            except Exception:
                continue
            positions[frame] = transform_to_matrix(transform.transform)[:3, 3]
        return positions

    def _publish(self) -> None:
        try:
            if not self.mapper.map.has_data:
                return
            cloud = self.mapper.map.extract_point_cloud()
            if len(cloud.points) == 0:
                return
            link_positions = self._link_positions()
            keep, _ = self.self_filter.keep_mask(cloud.points, link_positions)
            points = cloud.points[keep]
            colors = cloud.colors[keep] if cloud.colors is not None else None
            stamp_ns = self.mapper.map.stats.last_stamp or self.get_clock().now().nanoseconds
            cloud_data = build_pointcloud2(
                points, colors, frame_id=contracts.WORLD_FRAME, stamp_ns=stamp_ns
            )
            self._cloud_publisher.publish(self._cloud_message(cloud_data))

            mesh = self.mapper.map.extract_mesh()
            if len(mesh.triangles):
                keep_vertices, _ = self.self_filter.keep_mask(mesh.vertices, link_positions)
                triangles = mesh.triangles[np.all(keep_vertices[mesh.triangles], axis=1)]
                if len(triangles):
                    marker_data = mesh_to_marker(
                        mesh.vertices,
                        triangles,
                        mesh.colors,
                        max_triangles=self.config.max_triangles,
                        frame_id=contracts.WORLD_FRAME,
                        stamp_ns=stamp_ns,
                    )
                    self._mesh_publisher.publish(self._marker_message(marker_data))
        except Exception as exc:
            self.get_logger().error(f"map publication failed: {exc}")

    @staticmethod
    def _cloud_message(data) -> PointCloud2:
        message = PointCloud2()
        message.header.stamp = _stamp_message(data.stamp_ns)
        message.header.frame_id = data.frame_id
        message.height = data.height
        message.width = data.width
        message.fields = [
            PointField(
                name=field.name,
                offset=field.offset,
                datatype=field.datatype,
                count=field.count,
            )
            for field in data.fields
        ]
        message.is_bigendian = data.is_bigendian
        message.point_step = data.point_step
        message.row_step = data.row_step
        message.data = data.data
        message.is_dense = data.is_dense
        return message

    @staticmethod
    def _marker_message(data) -> Marker:
        marker = Marker()
        marker.header.stamp = _stamp_message(data.stamp_ns)
        marker.header.frame_id = data.frame_id
        marker.ns = data.ns
        marker.id = data.marker_id
        marker.type = data.type
        marker.action = data.action
        marker.pose.orientation.w = 1.0
        marker.scale.x, marker.scale.y, marker.scale.z = data.scale
        marker.points = [
            MarkerPoint(x=float(point[0]), y=float(point[1]), z=float(point[2]))
            for point in data.points
        ]
        marker.colors = [
            MarkerColor(
                r=float(color[0]),
                g=float(color[1]),
                b=float(color[2]),
                a=float(color[3]),
            )
            for color in data.colors
        ]
        return marker

    # ------------------------------------------------------------------
    # Service
    # ------------------------------------------------------------------
    def _on_reset_map(self, request, response):
        try:
            self.mapper.reset()
            self._reset_tf_buffer()
            response.success = True
            response.message = f"map reset; counters={self.mapper.counters.as_dict()}"
        except Exception as exc:
            response.success = False
            response.message = f"reset failed: {exc}"
        return response

    def destroy_node(self) -> bool:
        mean_ms = (
            self._integration_ms_total / self._integration_count if self._integration_count else 0.0
        )
        self.get_logger().info(
            f"integration mean={mean_ms:.1f} ms over {self._integration_count} frames; "
            f"counters={self.mapper.counters.as_dict()}"
        )
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = WorkspaceMapperNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
