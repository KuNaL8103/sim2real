"""ROS integration: closed_loop_eval_node with the fake backend.

Written as ``unittest`` for colcon.  Module-level imports are guarded so the
file imports without ``rclpy`` and is skipped with a reason instead of failing
collection.
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
    from std_msgs.msg import String
    from std_srvs.srv import Trigger

    from sim2real import contracts
    from sim2real.sim import create_backend
    from sim2real_sim.closed_loop_eval_node import EVAL_SUMMARY_QOS, ClosedLoopEvalNode

    IMPORT_ERROR: str | None = None
except ImportError as exc:  # pragma: no cover - depends on the environment
    IMPORT_ERROR = f"ROS prerequisites missing: {exc}"

pytestmark = pytest.mark.ros


@unittest.skipIf(IMPORT_ERROR is not None, IMPORT_ERROR or "")
class TestClosedLoopEvalNode(unittest.TestCase):
    def test_run_eval_publishes_a_summary(self) -> None:
        rclpy.init()
        tmp = tempfile.TemporaryDirectory()
        node = None
        executor = None
        try:
            node = ClosedLoopEvalNode(
                injected_backend=create_backend("fake", str(REPO_ROOT / "configs" / "sim")),
                config_dir=str(REPO_ROOT / "configs" / "sim"),
                expert_config_dir=str(REPO_ROOT / "configs" / "expert"),
                randomization_dir=str(REPO_ROOT / "configs" / "randomization"),
                evaluation_config=str(REPO_ROOT / "configs" / "evaluation" / "default.yaml"),
                filter_config=str(REPO_ROOT / "configs" / "filtering" / "default.yaml"),
                output_dir=tmp.name,
                policy_type="noisy_expert",
                num_episodes=3,
                eval_seed=20000,
                n_action_steps=10,
                auto_generate=False,
            )
            executor = MultiThreadedExecutor(num_threads=4)
            executor.add_node(node)
            thread = threading.Thread(target=executor.spin, daemon=True)
            thread.start()

            summaries: list[str] = []
            node.create_subscription(
                String,
                contracts.EVAL_SUMMARY_TOPIC,
                lambda message: summaries.append(message.data),
                EVAL_SUMMARY_QOS,
            )
            client = node.create_client(Trigger, contracts.RUN_EVAL_SERVICE)
            self.assertTrue(client.wait_for_service(timeout_sec=5.0))
            future = client.call_async(Trigger.Request())
            deadline = time.monotonic() + 60.0
            while not future.done() and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue(future.done(), "run_eval did not finish")
            response = future.result()
            self.assertTrue(response.success, response.message)
            deadline = time.monotonic() + 10.0
            while not summaries and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue(summaries, "no summary was published")
            summary = json.loads(summaries[-1])
            self.assertEqual(summary["aggregates"]["num_episodes"], 3)
            self.assertIn("success_rate", summary["aggregates"])
        finally:
            if executor is not None:
                executor.shutdown()
            if node is not None:
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            tmp.cleanup()
