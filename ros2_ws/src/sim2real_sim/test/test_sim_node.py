"""ROS integration: sim_node publishes joint states, TF, and one camera triplet.

Written as ``unittest`` because colcon runs ``python -m unittest`` for
``ament_python`` packages. Skipped with a reason when ROS is unavailable; a
skip is never a pass.
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
        qos_profile_sensor_data,
    )
    from sensor_msgs.msg import CameraInfo, Image
    from sensor_msgs.msg import JointState as JointStateMsg
    from sim2real_sim.sim_node import SimNode
    from tf2_msgs.msg import TFMessage

    from sim2real import contracts

    IMPORT_ERROR: str | None = None
except ImportError as exc:  # pragma: no cover - depends on the environment
    IMPORT_ERROR = f"ROS prerequisites missing: {exc}"

pytestmark = pytest.mark.ros


def _spin_until(executor, predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.05)
        if predicate():
            return True
    return False


@unittest.skipIf(IMPORT_ERROR is not None, IMPORT_ERROR or "")
class TestSimNode(unittest.TestCase):
    def test_publishes_state_tf_and_one_camera_triplet(self) -> None:
        rclpy.init()
        node = SimNode()
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        received: dict[str, object] = {"static_tf": []}

        node.create_subscription(
            JointStateMsg,
            contracts.JOINT_STATES_TOPIC,
            lambda message: received.setdefault("joint_states", message),
            10,
        )
        node.create_subscription(
            Image,
            "/camera/overhead/color/image_raw",
            lambda message: received.setdefault("color", message),
            qos_profile_sensor_data,
        )
        node.create_subscription(
            Image,
            "/camera/overhead/depth/image_raw",
            lambda message: received.setdefault("depth", message),
            qos_profile_sensor_data,
        )
        node.create_subscription(
            CameraInfo,
            "/camera/overhead/camera_info",
            lambda message: received.setdefault("info", message),
            qos_profile_sensor_data,
        )
        static_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        node.create_subscription(
            TFMessage,
            contracts.TF_STATIC_TOPIC,
            lambda message: received["static_tf"].append(message),
            static_qos,
        )

        try:
            self.assertTrue(
                _spin_until(executor, lambda: "joint_states" in received),
                "no /joint_states message",
            )
            self.assertTrue(
                _spin_until(
                    executor,
                    lambda: all(key in received for key in ("color", "depth", "info")),
                ),
                "no camera triplet",
            )
            self.assertTrue(
                _spin_until(executor, lambda: bool(received["static_tf"])),
                "no /tf_static message",
            )

            joint_state = received["joint_states"]
            self.assertEqual(list(joint_state.name), list(contracts.CONTROLLED_JOINT_NAMES))
            self.assertEqual(len(joint_state.position), 8)

            color = received["color"]
            depth = received["depth"]
            info = received["info"]
            self.assertEqual(color.encoding, contracts.COLOR_ENCODING)
            self.assertEqual(depth.encoding, contracts.DEPTH_ENCODING)
            self.assertEqual(color.header.frame_id, contracts.CAMERA_OVERHEAD_OPTICAL_FRAME)
            self.assertEqual(depth.header.frame_id, color.header.frame_id)
            self.assertEqual(info.header.frame_id, color.header.frame_id)
            color_stamp = (color.header.stamp.sec, color.header.stamp.nanosec)
            self.assertEqual((depth.header.stamp.sec, depth.header.stamp.nanosec), color_stamp)
            self.assertEqual((info.header.stamp.sec, info.header.stamp.nanosec), color_stamp)
            self.assertEqual(len(color.data), color.height * color.width * 3)
            # sensor_msgs/Image.data is uint8[]; 32FC1 packs four bytes per pixel.
            self.assertEqual(len(depth.data), depth.height * depth.width * 4)
            decoded = np.frombuffer(bytes(depth.data), dtype=np.float32)
            self.assertEqual(decoded.size, depth.height * depth.width)
            self.assertTrue(np.isfinite(decoded).any())
            self.assertEqual(info.distortion_model, "plumb_bob")

            static_frames = {
                transform.child_frame_id
                for message in received["static_tf"]
                for transform in message.transforms
            }
            self.assertIn(contracts.PANDA_BASE_FRAME, static_frames)
            self.assertIn(contracts.CAMERA_OVERHEAD_OPTICAL_FRAME, static_frames)
            self.assertIn(contracts.CAMERA_WRIST_OPTICAL_FRAME, static_frames)
        finally:
            executor.remove_node(node)
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
