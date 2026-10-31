"""ROS integration: reset_env applies a profile and republishes static camera TF.

Written as ``unittest`` because colcon runs ``python -m unittest`` for
``ament_python`` packages. Skipped with a reason when ROS is unavailable; a
skip is never a pass.
"""

from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    import rclpy
    from geometry_msgs.msg import Pose
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sim2real_sim.sim_node import SimNode
    from tf2_msgs.msg import TFMessage

    from sim2real import contracts
    from sim2real.metadata import validate_metadata_json
    from sim2real.sim import create_backend
    from sim2real_msgs.srv import ResetEnv

    IMPORT_ERROR: str | None = None
except ImportError as exc:  # pragma: no cover - depends on the environment
    IMPORT_ERROR = f"ROS prerequisites missing: {exc}"

pytestmark = pytest.mark.ros


@unittest.skipIf(IMPORT_ERROR is not None, IMPORT_ERROR or "")
class TestSimNodeRandomization(unittest.TestCase):
    def test_reset_service_is_deterministic_and_republishes_static_tf(self) -> None:
        rclpy.init()
        backend = create_backend("fake", str(REPO_ROOT / "configs" / "sim"))
        node = SimNode(injected_backend=backend)
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()

        static_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        messages: list[TFMessage] = []
        node.create_subscription(
            TFMessage,
            contracts.TF_STATIC_TOPIC,
            lambda message: messages.append(message),
            static_qos,
        )
        client = node.create_client(ResetEnv, contracts.RESET_ENV_SERVICE)

        def overhead_transform(start: int):
            for message in messages[start:]:
                for transform in message.transforms:
                    if transform.child_frame_id == contracts.CAMERA_OVERHEAD_OPTICAL_FRAME:
                        return transform
            return None

        def wait_for_overhead(start: int, timeout: float = 10.0):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                transform = overhead_transform(start)
                if transform is not None:
                    return transform
                time.sleep(0.01)
            return None

        def call(seed: int, profile: str, enabled: bool = True, override: Pose | None = None):
            request = ResetEnv.Request()
            request.seed = seed
            request.randomization_enabled = enabled
            request.randomization_profile = profile
            if override is not None:
                request.has_object_pose_override = True
                request.object_pose_override = override
            future = client.call_async(request)
            deadline = time.monotonic() + 10.0
            while not future.done() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(future.done(), "reset service call timed out")
            return future.result()

        try:
            self.assertTrue(client.wait_for_service(timeout_sec=10.0), "no reset service")
            self.assertIsNotNone(wait_for_overhead(0), "no initial /tf_static")

            # Capture the list length *before* the call: the static TF publish
            # happens inside the service callback, so the subscriber may receive
            # it before the client future completes.
            start = len(messages)
            first = call(7, "default")
            self.assertTrue(first.success, first.message)
            self.assertEqual(first.applied_seed, 7)
            payload = validate_metadata_json(first.applied_config_json)
            self.assertEqual(payload["profile"], "default")
            self.assertEqual(payload["sample"]["nominal"], False)
            self.assertEqual(payload["report"]["groups"]["object_pose"]["status"], "APPLIED")

            transform_seed7 = wait_for_overhead(start)
            self.assertIsNotNone(transform_seed7, "no /tf_static after the first reset")

            second = call(7, "default")
            self.assertTrue(second.success, second.message)
            self.assertEqual(first.applied_config_json, second.applied_config_json)

            start = len(messages)
            call(8, "default")
            transform_seed8 = wait_for_overhead(start)
            self.assertIsNotNone(transform_seed8, "no /tf_static after the second seed")
            self.assertNotEqual(
                (
                    transform_seed7.transform.translation.x,
                    transform_seed7.transform.translation.y,
                    transform_seed7.transform.translation.z,
                ),
                (
                    transform_seed8.transform.translation.x,
                    transform_seed8.transform.translation.y,
                    transform_seed8.transform.translation.z,
                ),
                "camera jitter did not change the republished overhead static transform",
            )

            unknown = call(7, "does_not_exist")
            self.assertFalse(unknown.success)
            validate_metadata_json(unknown.applied_config_json)

            outside = Pose()
            outside.position.x = 2.0
            outside.orientation.w = 1.0
            invalid_override = call(7, "none", enabled=False, override=outside)
            self.assertFalse(invalid_override.success)
            validate_metadata_json(invalid_override.applied_config_json)
        finally:
            executor.shutdown()
            thread.join(timeout=2.0)
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
