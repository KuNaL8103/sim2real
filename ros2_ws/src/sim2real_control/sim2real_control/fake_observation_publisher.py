"""Test-only fake observation publisher for the mock-hardware pipeline.

This node exists so ``pipeline_mock_hw.launch.py`` can exercise the full
``cameras -> inference -> safety -> controller`` graph without a simulator.
It publishes synthetic images, a synthetic table-top point cloud, and a task
instruction.

It deliberately does **not** publish ``/joint_states``: the ros2_control
``joint_state_broadcaster`` is the only owner of that topic.  The node is a test
double, not a perception system and not a simulator.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import rclpy

# The core library may be editable only in .venv; add the checkout's src/ when
# the module is not already importable (same bootstrap as sim_node).
_bootstrap_root: Path | None = None
for _parent in Path(__file__).resolve().parents:
    if (_parent / "src" / "sim2real" / "__init__.py").is_file():
        _bootstrap_root = _parent
        break
if importlib.util.find_spec("sim2real") is None and _bootstrap_root is not None:
    sys.path.insert(0, str(_bootstrap_root / "src"))

from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import (  # noqa: E402
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from sim2real import contracts  # noqa: E402
from sim2real.ros_utils.pointcloud import build_pointcloud2  # noqa: E402
from sim2real.sim.qos import CAMERA_QOS, MAP_QOS, TASK_INSTRUCTION_QOS, QosSpec  # noqa: E402


def _profile(spec: QosSpec) -> QoSProfile:
    reliability = {
        "reliable": ReliabilityPolicy.RELIABLE,
        "best_effort": ReliabilityPolicy.BEST_EFFORT,
    }[spec.reliability]
    durability = {
        "volatile": DurabilityPolicy.VOLATILE,
        "transient_local": DurabilityPolicy.TRANSIENT_LOCAL,
    }[spec.durability]
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=spec.depth,
        reliability=reliability,
        durability=durability,
    )


def _table_points() -> np.ndarray:
    xs = np.arange(0.26, 0.65, 0.02)
    ys = np.arange(-0.28, 0.29, 0.02)
    return np.array([[x, y, 0.43] for x in xs for y in ys], dtype=np.float64)


def _pointcloud(points: np.ndarray, stamp_ns: int) -> PointCloud2:
    data = build_pointcloud2(points, frame_id=contracts.WORLD_FRAME, stamp_ns=stamp_ns)
    message = PointCloud2()
    message.header.stamp.sec = data.stamp_ns // 1_000_000_000
    message.header.stamp.nanosec = data.stamp_ns % 1_000_000_000
    message.header.frame_id = data.frame_id
    message.height = data.height
    message.width = data.width
    message.point_step = data.point_step
    message.row_step = data.row_step
    message.is_dense = data.is_dense
    message.is_bigendian = data.is_bigendian
    message.fields = [
        PointField(name=field.name, offset=field.offset, datatype=field.datatype, count=field.count)
        for field in data.fields
    ]
    message.data = data.data
    return message


class FakeObservationPublisher(Node):
    """Publish fake cameras, a fake map, and a task instruction."""

    def __init__(self) -> None:
        super().__init__("fake_observation_publisher")
        self.declare_parameter("instruction", "pick up the red cube and place it in the left zone")
        self.declare_parameter("map_rate_hz", 2.0)
        self.declare_parameter("camera_rate_hz", 5.0)
        self.declare_parameter("image_size", 8)
        self._instruction = str(self.get_parameter("instruction").value)
        self._image_size = int(self.get_parameter("image_size").value)

        self._map_publisher = self.create_publisher(
            PointCloud2, contracts.WORKSPACE_POINTCLOUD_TOPIC, _profile(MAP_QOS)
        )
        self._instruction_publisher = self.create_publisher(
            String, contracts.TASK_INSTRUCTION_TOPIC, _profile(TASK_INSTRUCTION_QOS)
        )
        self._camera_publishers = []
        for camera in contracts.CAMERAS:
            self._camera_publishers.append(
                (
                    camera,
                    self.create_publisher(
                        Image, f"/camera/{camera}/color/image_raw", _profile(CAMERA_QOS)
                    ),
                    self.create_publisher(
                        Image, f"/camera/{camera}/depth/image_raw", _profile(CAMERA_QOS)
                    ),
                    self.create_publisher(
                        CameraInfo, f"/camera/{camera}/camera_info", _profile(CAMERA_QOS)
                    ),
                )
            )
        map_period = 1.0 / max(0.1, float(self.get_parameter("map_rate_hz").value))
        camera_period = 1.0 / max(0.1, float(self.get_parameter("camera_rate_hz").value))
        self.create_timer(map_period, self._publish_map)
        self.create_timer(camera_period, self._publish_cameras)
        self.create_timer(1.0, self._publish_instruction)
        self.get_logger().warning(
            "fake_observation_publisher is a TEST DOUBLE: synthetic images, a synthetic "
            "table cloud, and a fixed instruction. It never publishes /joint_states."
        )

    def _publish_map(self) -> None:
        stamp_ns = self.get_clock().now().nanoseconds
        self._map_publisher.publish(_pointcloud(_table_points(), stamp_ns))

    def _publish_instruction(self) -> None:
        message = String()
        message.data = self._instruction
        self._instruction_publisher.publish(message)

    def _publish_cameras(self) -> None:
        size = self._image_size
        for camera, color_pub, depth_pub, info_pub in self._camera_publishers:
            stamp = self.get_clock().now().to_msg()
            color = Image()
            color.header.stamp = stamp
            color.header.frame_id = f"camera_{camera}_optical_frame"
            color.height = size
            color.width = size
            color.encoding = "rgb8"
            color.step = size * 3
            color.data = bytes(size * size * 3)
            depth = Image()
            depth.header.stamp = stamp
            depth.header.frame_id = color.header.frame_id
            depth.height = size
            depth.width = size
            depth.encoding = "32FC1"
            depth.step = size * 4
            depth.data = np.full((size, size), 0.5, dtype=np.float32).tobytes()
            info = CameraInfo()
            info.header.stamp = stamp
            info.header.frame_id = color.header.frame_id
            info.height = size
            info.width = size
            info.k = [10.0, 0.0, size / 2, 0.0, 10.0, size / 2, 0.0, 0.0, 1.0]
            info.p = [
                10.0,
                0.0,
                size / 2,
                0.0,
                0.0,
                10.0,
                size / 2,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ]
            color_pub.publish(color)
            depth_pub.publish(depth)
            info_pub.publish(info)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = FakeObservationPublisher()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
