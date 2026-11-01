"""ROS integration: trajectory filter pub/sub round-trip and finalize service.

Skipped with a reason when ROS or the built interfaces are unavailable.
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
    from builtin_interfaces.msg import Time
    from geometry_msgs.msg import Pose
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sim2real_sim.trajectory_filter_node import TrajectoryFilterNode
    from std_srvs.srv import Trigger

    from sim2real import contracts
    from sim2real.expert.conversion import episode_to_rollout_fields
    from sim2real.filtering.synthetic import make_episode
    from sim2real_msgs.msg import FilterMetrics, RolloutEpisode, TrajectoryStep

    # Build the QoS profile only when the ROS imports succeeded. Defining it at
    # module scope unconditionally would raise NameError during collection in
    # the core CI job, where rclpy is intentionally absent.
    QOS = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=50,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    IMPORT_ERROR: str | None = None
except ImportError as exc:  # pragma: no cover - depends on the environment
    IMPORT_ERROR = f"ROS prerequisites missing: {exc}"
    QOS = None

pytestmark = pytest.mark.ros


def _stamp(stamp_ns: int) -> Time:
    return Time(sec=stamp_ns // 1_000_000_000, nanosec=stamp_ns % 1_000_000_000)


def _message(record, episode_id: str | None = None) -> RolloutEpisode:
    record.metadata = {"num_steps": len(record.steps)}
    fields = episode_to_rollout_fields(record)
    message = RolloutEpisode()
    message.episode_id = episode_id or fields["episode_id"]
    message.schema_version = fields["schema_version"]
    message.backend = fields["backend"]
    message.seed = fields["seed"]
    message.task_instruction = fields["task_instruction"]
    message.start_stamp = _stamp(fields["start_stamp_ns"])
    message.control_rate_hz = float(fields["control_rate_hz"])
    message.is_perturbed = bool(fields["is_perturbed"])
    message.outcome = int(fields["outcome"])
    message.failure_reason = fields["failure_reason"]
    message.metadata_json = fields["metadata_json"]
    steps = []
    for item in fields["steps"]:
        step = TrajectoryStep()
        step.stamp = _stamp(item["stamp_ns"])
        step.step_index = int(item["step_index"])
        step.task_instruction = item["task_instruction"]
        step.joint_names = list(item["joint_names"])
        step.joint_positions = [float(value) for value in item["joint_positions"]]
        step.joint_velocities = [float(value) for value in item["joint_velocities"]]
        pose = Pose()
        pose.position.x = float(item["ee_pose"]["position"][0])
        pose.position.y = float(item["ee_pose"]["position"][1])
        pose.position.z = float(item["ee_pose"]["position"][2])
        pose.orientation.w = 1.0
        step.ee_pose = pose
        step.ee_frame_id = item["ee_frame_id"]
        step.action = [float(value) for value in item["action"]]
        step.action_space = int(item["action_space"])
        step.gripper_position = float(item["gripper_position"])
        step.unintended_collision = bool(item["unintended_collision"])
        steps.append(step)
    message.steps = steps
    return message


@unittest.skipIf(IMPORT_ERROR is not None, IMPORT_ERROR or "")
class TestTrajectoryFilterNode(unittest.TestCase):
    def test_pub_sub_and_finalize(self) -> None:
        rclpy.init()
        tmp = tempfile.TemporaryDirectory()
        node = TrajectoryFilterNode(
            config_path=str(REPO_ROOT / "configs" / "filtering" / "default.yaml"),
            curated_root=str(Path(tmp.name) / "curated"),
            rejected_root=str(Path(tmp.name) / "rejected"),
            dataset_name="ros_test",
        )
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()

        metrics: list[FilterMetrics] = []
        filtered: list[RolloutEpisode] = []
        node.create_subscription(FilterMetrics, contracts.FILTER_METRICS_TOPIC, metrics.append, QOS)
        node.create_subscription(
            RolloutEpisode, contracts.FILTERED_EPISODE_TOPIC, filtered.append, QOS
        )
        publisher = node.create_publisher(RolloutEpisode, contracts.STAGING_EPISODE_TOPIC, QOS)
        client = node.create_client(Trigger, contracts.FINALIZE_DATASET_SERVICE)

        def wait_for(predicate, timeout=20.0):
            deadline = time.monotonic() + timeout
            while not predicate() and time.monotonic() < deadline:
                time.sleep(0.02)
            return predicate()

        try:
            self.assertTrue(client.wait_for_service(timeout_sec=15.0), "no finalize service")
            time.sleep(0.5)  # allow discovery of the metrics/filtered subscriptions

            good = make_episode(episode_id="ros-good")
            publisher.publish(_message(good))
            self.assertTrue(wait_for(lambda: len(metrics) >= 1), "no metrics message")
            self.assertTrue(wait_for(lambda: len(filtered) >= 1), "no filtered message")
            self.assertEqual(metrics[0].episode_id, "ros-good")
            self.assertEqual(metrics[0].decision, FilterMetrics.DECISION_ACCEPT)
            self.assertTrue(metrics[0].task_success)
            self.assertGreater(metrics[0].dimensionless_squared_jerk, 0.0)
            self.assertEqual(filtered[0].episode_id, "ros-good")

            malformed = RolloutEpisode()
            malformed.episode_id = "ros-malformed"
            malformed.steps = []
            publisher.publish(malformed)
            self.assertTrue(wait_for(lambda: len(metrics) >= 2), "no invalid metrics message")
            self.assertEqual(metrics[1].episode_id, "ros-malformed")
            self.assertEqual(metrics[1].decision, FilterMetrics.DECISION_INVALID)
            self.assertTrue(metrics[1].rejection_reasons)
            self.assertFalse(any(message.episode_id == "ros-malformed" for message in filtered))

            request = Trigger.Request()
            future = client.call_async(request)
            deadline = time.monotonic() + 20.0
            while not future.done() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(future.done(), "finalize call timed out")
            response = future.result()
            self.assertTrue(response.success, response.message)
            self.assertIn("v1.0.0", response.message)

            version = Path(tmp.name) / "curated" / "ros_test" / "v1.0.0"
            info = json.loads((version / "meta" / "info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["schema_version"], "curated_v1")
            self.assertEqual(info["num_episodes"], 1)
            self.assertTrue((version / "data" / "chunk-000").is_dir())
            rejected = Path(tmp.name) / "rejected"
            self.assertTrue(list(rejected.rglob("rejections.jsonl")))
        finally:
            executor.shutdown()
            node.destroy_node()
            tmp.cleanup()
            if rclpy.ok():
                rclpy.shutdown()
