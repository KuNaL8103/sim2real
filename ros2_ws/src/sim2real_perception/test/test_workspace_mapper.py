"""ROS integration: mapper publishes a non-empty /workspace/pointcloud.

Skipped with an explicit reason when rclpy or the built interfaces are absent;
the core mapping logic is covered by hardware-free tests under ``tests/``.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

# colcon test imports this module from the package directory; make the repo
# fixtures and the editable core library importable there as well as under
# `pytest` from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[4]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tests.fixtures.mapping import write_config  # noqa: E402
from tests.fixtures.synthetic_rgbd import (  # noqa: E402
    SyntheticIntrinsics,
    SyntheticScene,
    overhead_camera,
    render,
)

pytestmark = pytest.mark.ros

ROS_AVAILABLE = False
try:
    import rclpy
    from geometry_msgs.msg import TransformStamped
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image, PointCloud2
    from sim2real_perception.workspace_mapper_node import WorkspaceMapperNode
    from std_srvs.srv import Trigger
    from tf2_ros import StaticTransformBroadcaster

    ROS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by the core suite
    pass


def matrix_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    m = rotation
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return x, y, z, w


@pytest.mark.skipif(not ROS_AVAILABLE, reason="rclpy / sim2real_msgs not available")
def test_mapper_publishes_pointcloud_and_resets(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path,
        workspace_bounds={"min": [-0.2, -0.2, -0.1], "max": [0.2, 0.2, 0.2]},
        publish_rate_hz=10.0,
        min_coverage_ratio=0.01,
    )
    frame = render(SyntheticScene(), SyntheticIntrinsics(), overhead_camera())
    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            f"config_file:={config_path}",
            "-p",
            "publish_rate_hz:=10.0",
        ]
    )
    mapper = None
    publisher_node = None
    executor = None
    try:
        mapper = WorkspaceMapperNode()
        publisher_node = Node("mapper_test_publisher")

        broadcaster = StaticTransformBroadcaster(publisher_node)
        transform = TransformStamped()
        transform.header.stamp = publisher_node.get_clock().now().to_msg()
        transform.header.frame_id = "world"
        transform.child_frame_id = "camera_overhead_optical_frame"
        transform.transform.translation.x = float(frame.T_world_camera[0, 3])
        transform.transform.translation.y = float(frame.T_world_camera[1, 3])
        transform.transform.translation.z = float(frame.T_world_camera[2, 3])
        qx, qy, qz, qw = matrix_to_quaternion(frame.T_world_camera[:3, :3])
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        broadcaster.sendTransform(transform)

        color_pub = publisher_node.create_publisher(Image, "/camera/overhead/color/image_raw", 10)
        depth_pub = publisher_node.create_publisher(Image, "/camera/overhead/depth/image_raw", 10)
        info_pub = publisher_node.create_publisher(CameraInfo, "/camera/overhead/camera_info", 10)

        received: list[PointCloud2] = []
        sub_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        def on_cloud(message: PointCloud2) -> None:
            received.append(message)

        publisher_node.create_subscription(PointCloud2, "/workspace/pointcloud", on_cloud, sub_qos)

        def publish() -> None:
            now = publisher_node.get_clock().now()
            stamp = now.to_msg()
            color = Image()
            color.header.stamp = stamp
            color.header.frame_id = "camera_overhead_optical_frame"
            color.height, color.width = frame.rgb.shape[:2]
            color.encoding = "rgb8"
            color.step = frame.rgb.shape[1] * 3
            color.data = np.ascontiguousarray(frame.rgb).tobytes()

            depth = Image()
            depth.header.stamp = stamp
            depth.header.frame_id = "camera_overhead_optical_frame"
            depth.height, depth.width = frame.depth.shape
            depth.encoding = "32FC1"
            depth.step = frame.depth.shape[1] * 4
            depth.data = np.ascontiguousarray(frame.depth, dtype=np.float32).tobytes()

            info = CameraInfo()
            info.header.stamp = stamp
            info.header.frame_id = "camera_overhead_optical_frame"
            info.height, info.width = frame.depth.shape
            info.k = [float(v) for v in frame.K.reshape(-1)]
            info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            info.p = [
                frame.K[0, 0],
                0.0,
                frame.K[0, 2],
                0.0,
                0.0,
                frame.K[1, 1],
                frame.K[1, 2],
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ]
            color_pub.publish(color)
            depth_pub.publish(depth)
            info_pub.publish(info)

        publisher_node.create_timer(0.1, publish)
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(mapper)
        executor.add_node(publisher_node)

        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline and not received:
            executor.spin_once(timeout_sec=0.1)

        assert received, "mapper did not publish /workspace/pointcloud"
        message = received[-1]
        assert message.header.frame_id == "world"
        assert message.width > 0
        assert message.point_step == 16
        assert len(message.data) == message.width * message.point_step

        client = publisher_node.create_client(Trigger, "/workspace/reset_map")
        assert client.wait_for_service(timeout_sec=5.0)
        future = client.call_async(Trigger.Request())
        service_deadline = time.monotonic() + 5.0
        while not future.done() and time.monotonic() < service_deadline:
            executor.spin_once(timeout_sec=0.1)
        assert future.done()
        assert future.result().success
    finally:
        if executor is not None:
            executor.shutdown()
        if publisher_node is not None:
            publisher_node.destroy_node()
        if mapper is not None:
            mapper.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
