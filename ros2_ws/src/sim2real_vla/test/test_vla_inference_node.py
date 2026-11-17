"""ROS-level tests for the VLA inference node (fake backend, no torch).

Written as ``unittest`` because colcon runs ``python -m unittest`` for
``ament_python`` packages. Skipped with a reason when ROS is unavailable; a
skip is never a pass.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

import numpy as np
import pytest

# Isolate this package's DDS graph from parallel colcon package tests (for
# example sim_node's camera/joint publishers use the same topic names).
os.environ.setdefault("ROS_DOMAIN_ID", "77")
os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    import rclpy
    from builtin_interfaces.msg import Time
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.parameter import Parameter
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image
    from sensor_msgs.msg import JointState as JointStateMsg
    from std_msgs.msg import String
    from trajectory_msgs.msg import JointTrajectory

    from sim2real import contracts
    from sim2real.sim.qos import (
        CAMERA_QOS,
        JOINT_STATES_QOS,
        TASK_INSTRUCTION_QOS,
        VLA_RAW_ACTION_QOS,
        VLA_STATUS_QOS,
        QosSpec,
    )
    from sim2real_vla.vla_inference_node import VlaInferenceNode

    IMPORT_ERROR: str | None = None
except ImportError as exc:  # pragma: no cover - depends on the environment
    IMPORT_ERROR = f"ROS prerequisites missing: {exc}"

pytestmark = pytest.mark.ros


def _profile(spec: QosSpec) -> QoSProfile:
    _RELIABILITY = {
        "reliable": ReliabilityPolicy.RELIABLE,
        "best_effort": ReliabilityPolicy.BEST_EFFORT,
    }
    _DURABILITY = {
        "volatile": DurabilityPolicy.VOLATILE,
        "transient_local": DurabilityPolicy.TRANSIENT_LOCAL,
    }
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=spec.depth,
        reliability=_RELIABILITY[spec.reliability],
        durability=_DURABILITY[spec.durability],
    )


def _spin_until(executor, predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.05)
        if predicate():
            return True
    return False


def _stamp(stamp_ns: int) -> Time:
    return Time(sec=int(stamp_ns // 1_000_000_000), nanosec=int(stamp_ns % 1_000_000_000))


def _image(stamp_ns: int) -> Image:
    message = Image()
    message.header.stamp = _stamp(stamp_ns)
    message.header.frame_id = "camera_overhead_optical_frame"
    message.height = 8
    message.width = 8
    message.encoding = "rgb8"
    message.is_bigendian = 0
    message.step = 24
    message.data = np.zeros((8, 8, 3), dtype=np.uint8).tobytes()
    return message


def _joint_state(stamp_ns: int) -> JointStateMsg:
    message = JointStateMsg()
    message.header.stamp = _stamp(stamp_ns)
    message.name = list(contracts.CONTROLLED_JOINT_NAMES)
    message.position = [0.0] * 7 + [0.02]
    return message


@unittest.skipIf(IMPORT_ERROR is not None, IMPORT_ERROR or "")
class TestVlaInferenceContracts(unittest.TestCase):
    def setUp(self) -> None:
        rclpy.init()

    def tearDown(self) -> None:
        if rclpy.ok():
            rclpy.shutdown()

    def test_publishers_match_contracts_and_qos_table(self) -> None:
        node = VlaInferenceNode(parameter_overrides=[Parameter("config_file", value="")])
        try:
            self.assertEqual(node._action_publisher.topic_name, contracts.VLA_RAW_ACTION_TOPIC)
            self.assertEqual(node._status_publisher.topic_name, contracts.VLA_STATUS_TOPIC)
            self.assertEqual(
                contracts.TOPICS[contracts.VLA_RAW_ACTION_TOPIC],
                "trajectory_msgs/msg/JointTrajectory",
            )
            self.assertEqual(contracts.TOPICS[contracts.VLA_STATUS_TOPIC], "std_msgs/msg/String")
            self.assertEqual(
                node._action_publisher.qos_profile.reliability,
                ReliabilityPolicy.RELIABLE,
            )
            self.assertEqual(node._action_publisher.qos_profile.depth, 10)
            self.assertEqual(
                node._status_publisher.qos_profile.reliability,
                ReliabilityPolicy.RELIABLE,
            )
            self.assertEqual(node._status_publisher.qos_profile.depth, 10)
            # The subscriber side is pinned to the same specs as the docs.
            self.assertEqual(VLA_RAW_ACTION_QOS.depth, 10)
            self.assertEqual(VLA_STATUS_QOS.depth, 10)
            self.assertEqual(TASK_INSTRUCTION_QOS.depth, 1)
        finally:
            node.destroy_node()

    def test_node_import_does_not_load_torch(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [
                str(REPO_ROOT / "src"),
                str(REPO_ROOT / "ros2_ws" / "src"),
                str(REPO_ROOT / "ros2_ws" / "src" / "sim2real_vla"),
                env.get("PYTHONPATH", ""),
            ]
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys\n"
                "import sim2real_vla.vla_inference_node  # noqa: F401\n"
                "assert 'torch' not in sys.modules\n",
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipIf(IMPORT_ERROR is not None, IMPORT_ERROR or "")
class TestVlaInferenceEndToEnd(unittest.TestCase):
    def setUp(self) -> None:
        rclpy.init()
        self.node = VlaInferenceNode(
            parameter_overrides=[
                Parameter("config_file", value=""),
                # Keep the raw 10-point chunk deterministic for this ROS-level
                # assertion; pure chunk tests cover stale trimming.
                Parameter("latency_compensation", value=False),
            ]
        )
        self.executor = MultiThreadedExecutor(num_threads=2)
        self.executor.add_node(self.node)
        self.actions: list[JointTrajectory] = []
        self.statuses: list[dict] = []
        self.node.create_subscription(
            JointTrajectory,
            contracts.VLA_RAW_ACTION_TOPIC,
            self.actions.append,
            _profile(VLA_RAW_ACTION_QOS),
        )
        self.node.create_subscription(
            String,
            contracts.VLA_STATUS_TOPIC,
            lambda message: self.statuses.append(json.loads(message.data)),
            _profile(VLA_STATUS_QOS),
        )
        self.overhead = self.node.create_publisher(
            Image,
            "/camera/overhead/color/image_raw",
            _profile(CAMERA_QOS),
        )
        self.wrist = self.node.create_publisher(
            Image,
            "/camera/wrist/color/image_raw",
            _profile(CAMERA_QOS),
        )
        self.joints = self.node.create_publisher(
            JointStateMsg,
            contracts.JOINT_STATES_TOPIC,
            _profile(JOINT_STATES_QOS),
        )
        self.instruction = self.node.create_publisher(
            String,
            contracts.TASK_INSTRUCTION_TOPIC,
            _profile(TASK_INSTRUCTION_QOS),
        )

    def tearDown(self) -> None:
        self.executor.shutdown()
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def _publish(self, stamp_ns: int, instruction: str | None) -> None:
        overhead = _image(stamp_ns)
        wrist = _image(stamp_ns)
        wrist.header.frame_id = "camera_wrist_optical_frame"
        self.overhead.publish(overhead)
        self.wrist.publish(wrist)
        self.joints.publish(_joint_state(stamp_ns))
        if instruction is not None:
            self.instruction.publish(String(data=instruction))

    def test_publishes_raw_chunk_from_synchronized_inputs(self) -> None:
        instruction = "pick up the red cube and place it in the left zone"
        stamp_ns = 5_000_000_000
        _spin_until(self.executor, lambda: False, timeout=0.3)
        deadline = time.monotonic() + 10.0
        attempt = 0
        while time.monotonic() < deadline and not self.actions:
            self._publish(stamp_ns + attempt * 100_000_000, instruction)
            self.executor.spin_once(timeout_sec=0.1)
            attempt += 1
        self.assertTrue(self.actions, "no JointTrajectory was published")
        message = self.actions[-1]
        self.assertEqual(message.header.frame_id, "", "joint trajectory should be frame-agnostic")
        self.assertEqual(list(message.joint_names), list(contracts.CONTROLLED_JOINT_NAMES))
        self.assertEqual(len(message.points), 10)
        published_stamps = {stamp_ns + index * 100_000_000 for index in range(attempt)}
        message_stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(
            message.header.stamp.nanosec
        )
        self.assertIn(message_stamp_ns, published_stamps)
        for index, point in enumerate(message.points):
            self.assertAlmostEqual(
                point.time_from_start.sec + point.time_from_start.nanosec / 1e9,
                index * 0.1,
                places=6,
            )
            np.testing.assert_allclose(point.positions, [0.0] * 7 + [0.02], rtol=0, atol=1e-6)
        self.assertTrue(
            _spin_until(
                self.executor,
                lambda: (
                    bool(self.statuses) and self.statuses[-1]["counters"]["chunks_published"] >= 1
                ),
                timeout=5.0,
            ),
            "no /vla/status sample with a published chunk",
        )
        latest = self.statuses[-1]
        self.assertGreaterEqual(latest["counters"]["chunks_published"], 1)
        self.assertFalse(latest["depth_used"])
        self.assertEqual(latest["policy_backend"], "fake")
        self.assertIsNotNone(latest["time_to_ready_s"])
        self.assertGreaterEqual(latest["latency"]["samples"], 1)
        self.assertGreaterEqual(latest["published_future_fraction"]["samples"], 1)
        self.assertGreaterEqual(latest["last_published_future_fraction"], 0.0)
        self.assertLessEqual(latest["last_published_future_fraction"], 1.0)

    def test_missing_instruction_publishes_nothing(self) -> None:
        stamp_ns = 7_000_000_000
        _spin_until(self.executor, lambda: False, timeout=0.3)
        deadline = time.monotonic() + 3.0
        attempt = 0
        while time.monotonic() < deadline:
            self._publish(stamp_ns + attempt * 100_000_000, None)
            self.executor.spin_once(timeout_sec=0.1)
            attempt += 1
        self.assertEqual(self.actions, [])
        self.assertTrue(self.statuses)
        latest = self.statuses[-1]
        self.assertEqual(latest["state"], "WAITING_FOR_INPUT")
        self.assertGreaterEqual(latest["rejected"].get("NO_INSTRUCTION", 0), 1)


@unittest.skipIf(IMPORT_ERROR is not None, IMPORT_ERROR or "")
class TestVlaInferenceFailure(unittest.TestCase):
    def setUp(self) -> None:
        rclpy.init()

    def tearDown(self) -> None:
        if rclpy.ok():
            rclpy.shutdown()

    def test_missing_checkpoint_reaches_failed_and_publishes_nothing(self) -> None:
        node = VlaInferenceNode(
            parameter_overrides=[
                Parameter("config_file", value=""),
                Parameter("policy_backend", value="server"),
                Parameter("checkpoint_path", value="/nonexistent/checkpoint"),
                Parameter("max_restarts", value=0),
                Parameter("startup_timeout_s", value=0.5),
            ]
        )
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        actions: list[JointTrajectory] = []
        statuses: list[dict] = []
        node.create_subscription(
            JointTrajectory,
            contracts.VLA_RAW_ACTION_TOPIC,
            actions.append,
            _profile(VLA_RAW_ACTION_QOS),
        )
        node.create_subscription(
            String,
            contracts.VLA_STATUS_TOPIC,
            lambda message: statuses.append(json.loads(message.data)),
            _profile(VLA_STATUS_QOS),
        )
        try:
            self.assertTrue(
                _spin_until(
                    executor,
                    lambda: bool(statuses) and statuses[-1]["state"] == "FAILED",
                    timeout=10.0,
                ),
                f"status never reached FAILED: {statuses[-1] if statuses else None}",
            )
            self.assertEqual(actions, [])
        finally:
            executor.shutdown()
            node.destroy_node()
