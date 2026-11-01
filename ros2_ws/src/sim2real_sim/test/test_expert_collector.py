"""ROS integration: expert collector service round-trip with FakeBackend.

Written as ``unittest`` for colcon. Skipped with a reason when ROS is
unavailable; a skip is never a pass.
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sim2real_sim.expert_collector_node import ExpertCollectorNode

    from sim2real import contracts
    from sim2real.expert.staging import StagingBuffer
    from sim2real.metadata import validate_metadata_json
    from sim2real.sim import create_backend
    from sim2real_msgs.msg import RolloutEpisode
    from sim2real_msgs.srv import GenerateRollouts

    IMPORT_ERROR: str | None = None
except ImportError as exc:  # pragma: no cover - depends on the environment
    IMPORT_ERROR = f"ROS prerequisites missing: {exc}"

pytestmark = pytest.mark.ros

INSTRUCTION = "pick up the red cube and place it in the left zone"


@unittest.skipIf(IMPORT_ERROR is not None, IMPORT_ERROR or "")
class TestExpertCollector(unittest.TestCase):
    def test_service_round_trip_and_bounds(self) -> None:
        rclpy.init()
        tmp = tempfile.TemporaryDirectory()
        backend = create_backend("fake", str(REPO_ROOT / "configs" / "sim"))
        node = ExpertCollectorNode(
            injected_backend=backend,
            output_dir=tmp.name,
            config_dir=str(REPO_ROOT / "configs" / "sim"),
            randomization_dir=str(REPO_ROOT / "configs" / "randomization"),
            expert_config_dir=str(REPO_ROOT / "configs" / "expert"),
        )
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()

        messages: list[RolloutEpisode] = []
        node.create_subscription(
            RolloutEpisode,
            contracts.STAGING_EPISODE_TOPIC,
            lambda message: messages.append(message),
            QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        client = node.create_client(GenerateRollouts, contracts.GENERATE_ROLLOUTS_SERVICE)

        def call(request: GenerateRollouts.Request):
            future = client.call_async(request)
            deadline = time.monotonic() + 60.0
            while not future.done() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(future.done(), "generate_rollouts call timed out")
            return future.result()

        try:
            self.assertTrue(client.wait_for_service(timeout_sec=15.0), "no collector service")

            request = GenerateRollouts.Request()
            request.num_rollouts = 2
            request.seed = 1
            request.task_instruction = INSTRUCTION
            request.sampling_mode = contracts.SAMPLING_UNIFORM
            request.include_perturbed = False
            response = call(request)
            self.assertTrue(response.success, response.message)
            self.assertEqual(response.num_generated, 2)
            payload = json.loads(response.result_metadata_json)
            validate_metadata_json(response.result_metadata_json)
            self.assertEqual(payload["num_requested"], 2)
            self.assertEqual(len(payload["seeds"]), 2)
            self.assertEqual(sum(payload["counts"].values()), 2)
            self.assertEqual(payload["counts"]["success"] + payload["counts"]["failure"], 2)
            self.assertEqual(payload["num_perturbed"], 0)

            deadline = time.monotonic() + 10.0
            while len(messages) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(len(messages), 2)
            for message in messages:
                self.assertEqual(message.backend, "fake")
                self.assertEqual(message.task_instruction, INSTRUCTION)
                self.assertTrue(message.steps)
                self.assertIn(
                    message.outcome,
                    {
                        contracts.OUTCOME_SUCCESS,
                        contracts.OUTCOME_FAILURE,
                        contracts.OUTCOME_TIMEOUT,
                        contracts.OUTCOME_ABORTED,
                    },
                )
                metadata = validate_metadata_json(message.metadata_json)
                self.assertEqual(metadata["run_id"], payload["run_id"])

            buffer = StagingBuffer(tmp.name, payload["run_id"])
            verification = buffer.verify()
            self.assertEqual(verification["manifest_rows"], 2)
            self.assertEqual(verification["valid"], 2)
            self.assertEqual(verification["corrupt"], 0)

            zero = GenerateRollouts.Request()
            zero.num_rollouts = 0
            zero.task_instruction = INSTRUCTION
            zero.sampling_mode = contracts.SAMPLING_UNIFORM
            zero_response = call(zero)
            self.assertFalse(zero_response.success)
            self.assertIn("out of bounds", zero_response.message)

            too_many = GenerateRollouts.Request()
            too_many.num_rollouts = 65
            too_many.task_instruction = INSTRUCTION
            too_many.sampling_mode = contracts.SAMPLING_UNIFORM
            too_many_response = call(too_many)
            self.assertFalse(too_many_response.success)
            self.assertEqual(too_many_response.num_generated, 0)

            targeted = GenerateRollouts.Request()
            targeted.num_rollouts = 1
            targeted.task_instruction = INSTRUCTION
            targeted.sampling_mode = contracts.SAMPLING_FAILURE_TARGETED
            targeted_response = call(targeted)
            self.assertFalse(targeted_response.success)
            self.assertIn(
                "FAILURE_TARGETED not implemented until Task 9", targeted_response.message
            )
        finally:
            executor.shutdown()
            node.destroy_node()
            tmp.cleanup()
            if rclpy.ok():
                rclpy.shutdown()
