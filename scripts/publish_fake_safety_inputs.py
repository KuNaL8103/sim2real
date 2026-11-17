#!/usr/bin/env python3
"""Publish fake safety inputs for the manual Task 11 checkpoint.

This is a test utility, not a controller and not a simulator.  It publishes a
synthetic tabletop PointCloud2 on `/workspace/pointcloud` and periodic raw
chunks on `/vla/raw_action` built from the current measured joint state.  It
never publishes `/joint_states` (the single owner is the running broadcaster).

Run it from a shell with ROS 2 sourced and with `.venv-ros` on PATH, for
example:

    .venv-ros/bin/python scripts/publish_fake_safety_inputs.py --duration-s 120
"""

from __future__ import annotations

import argparse
import time

import numpy as np

try:
    import rclpy
    from builtin_interfaces.msg import Duration, Time
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
    from sensor_msgs.msg import JointState as JointStateMsg
    from std_msgs.msg import String
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    from sim2real import contracts
    from sim2real.ros_utils.pointcloud import build_pointcloud2
    from sim2real_msgs.msg import SafetyStatus
except ImportError as exc:  # pragma: no cover - manual utility
    raise SystemExit(
        "ROS 2 is not importable. Source /opt/ros/jazzy/setup.bash and "
        "ros2_ws/install/setup.bash, and use .venv-ros/bin/python."
    ) from exc


def _qos(depth: int = 10) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def _stamp(ns: int) -> Time:
    return Time(sec=int(ns // 10**9), nanosec=int(ns % 10**9))


def _duration(seconds: float) -> Duration:
    ns = int(round(seconds * 1e9))
    return Duration(sec=ns // 10**9, nanosec=ns % 10**9)


def _table_cloud() -> PointCloud2:
    xs = np.arange(0.26, 0.65, 0.02)
    ys = np.arange(-0.28, 0.29, 0.02)
    points = np.array([[x, y, 0.43] for x in xs for y in ys], dtype=np.float64)
    data = build_pointcloud2(points, frame_id=contracts.WORLD_FRAME, stamp_ns=time.time_ns())
    message = PointCloud2()
    message.header.stamp = _stamp(data.stamp_ns)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=120.0)
    parser.add_argument("--chunk-period-s", type=float, default=1.0)
    parser.add_argument("--delta-rad", type=float, default=0.02)
    parser.add_argument(
        "--cameras",
        action="store_true",
        help="publish fake camera streams for the fake inference backend; do not publish chunks",
    )
    parser.add_argument("--task-instruction", type=str, default="")
    args = parser.parse_args()

    rclpy.init()
    node = Node("fake_safety_inputs")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    latest = {"joint": None, "status": None}
    node.create_subscription(
        JointStateMsg,
        contracts.JOINT_STATES_TOPIC,
        lambda message: latest.__setitem__("joint", message),
        _qos(),
    )
    node.create_subscription(
        SafetyStatus,
        contracts.SAFETY_STATUS_TOPIC,
        lambda message: latest.__setitem__("status", message),
        _qos(),
    )
    map_pub = node.create_publisher(PointCloud2, contracts.WORKSPACE_POINTCLOUD_TOPIC, _qos(1))
    chunk_pub = node.create_publisher(JointTrajectory, contracts.VLA_RAW_ACTION_TOPIC, _qos())
    instruction_pub = node.create_publisher(String, contracts.TASK_INSTRUCTION_TOPIC, _qos(1))
    camera_pubs = []
    if args.cameras:
        for camera in contracts.CAMERAS:
            camera_pubs.append(
                (
                    camera,
                    node.create_publisher(Image, f"/camera/{camera}/color/image_raw", _qos(5)),
                    node.create_publisher(Image, f"/camera/{camera}/depth/image_raw", _qos(5)),
                    node.create_publisher(CameraInfo, f"/camera/{camera}/camera_info", _qos(5)),
                )
            )
    if args.task_instruction:
        instruction = String()
        instruction.data = args.task_instruction
        instruction_pub.publish(instruction)
    next_camera = time.monotonic()
    start = time.monotonic()
    next_chunk = start + 1.0
    try:
        while time.monotonic() - start < args.duration_s:
            map_pub.publish(_table_cloud())
            now = time.monotonic()
            if camera_pubs and now >= next_camera:
                stamp_ns = time.time_ns()
                for camera, color_pub, depth_pub, info_pub in camera_pubs:
                    color = Image()
                    color.header.stamp = _stamp(stamp_ns)
                    color.header.frame_id = f"camera_{camera}_optical_frame"
                    color.height = 8
                    color.width = 8
                    color.encoding = "rgb8"
                    color.step = 24
                    color.data = bytes(8 * 8 * 3)
                    depth = Image()
                    depth.header.stamp = _stamp(stamp_ns)
                    depth.header.frame_id = f"camera_{camera}_optical_frame"
                    depth.height = 8
                    depth.width = 8
                    depth.encoding = "32FC1"
                    depth.step = 32
                    depth.data = np.full((8, 8), 0.5, dtype=np.float32).tobytes()
                    info = CameraInfo()
                    info.header.stamp = _stamp(stamp_ns)
                    info.header.frame_id = f"camera_{camera}_optical_frame"
                    info.height = 8
                    info.width = 8
                    info.k = [10.0, 0.0, 4.0, 0.0, 10.0, 4.0, 0.0, 0.0, 1.0]
                    info.p = [10.0, 0.0, 4.0, 0.0, 0.0, 10.0, 4.0, 0.0, 0.0, 0.0, 1.0, 0.0]
                    color_pub.publish(color)
                    depth_pub.publish(depth)
                    info_pub.publish(info)
                next_camera = now + 0.1
            executor.spin_once(timeout_sec=0.02)
            if args.cameras or latest["joint"] is None or time.monotonic() < next_chunk:
                continue
            status = latest["status"]
            if status is None or not status.is_safe:
                continue
            joint = latest["joint"]
            by_name = dict(zip(joint.name, joint.position, strict=False))
            positions = np.array(
                [by_name[name] for name in contracts.CONTROLLED_JOINT_NAMES],
                dtype=np.float64,
            )
            positions[0] += args.delta_rad
            message = JointTrajectory()
            message.header.stamp = _stamp(
                int(joint.header.stamp.sec) * 10**9 + joint.header.stamp.nanosec
            )
            message.header.frame_id = ""
            message.joint_names = list(contracts.CONTROLLED_JOINT_NAMES)
            for index in range(3):
                point = JointTrajectoryPoint()
                point.positions = [float(value) for value in positions]
                point.time_from_start = _duration(0.1 * (index + 1))
                message.points.append(point)
            chunk_pub.publish(message)
            next_chunk = time.monotonic() + args.chunk_period_s
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
