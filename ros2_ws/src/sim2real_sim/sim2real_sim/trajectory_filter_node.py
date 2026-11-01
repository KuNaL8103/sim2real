"""ROS 2 adapter for trajectory filtering and curated dataset writing.

The node is thin plumbing: it converts a ``RolloutEpisode`` message to the core
``EpisodeRecord`` (via a pure conversion module), calls the deterministic filter,
publishes metrics for every episode, republishes accepted episodes, and buffers
them until ``/sim2real/finalize_dataset`` (or shutdown) writes a ``curated_v1``
version.

It never raises out of a callback: a malformed message produces an ``INVALID``
``FilterMetrics`` message and is written to the rejection log.

Topics (all reliable, keep-last depth 50):
  /sim2real/staging_episode   in   RolloutEpisode
  /sim2real/filter_metrics    out  FilterMetrics   (every episode)
  /sim2real/filtered_episode  out  RolloutEpisode  (accepted only)
Service:
  /sim2real/finalize_dataset  std_srvs/Trigger
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
from pathlib import Path
from typing import Any

import rclpy

# Same bootstrap as sim_node / expert_collector_node.
_bootstrap_root = None
for _parent in Path(__file__).resolve().parents:
    if (_parent / "src" / "sim2real" / "__init__.py").is_file():
        _bootstrap_root = _parent
        break
if importlib.util.find_spec("sim2real") is None and _bootstrap_root is not None:
    sys.path.insert(0, str(_bootstrap_root / "src"))
# `ros2 run` uses system Python, where scipy may exist but pyarrow does not.
# .venv-ros has scipy/pyarrow installed --no-deps; append (not prepend) so system
# NumPy 1.26.4 stays the visible version.
_venv_site = (
    _bootstrap_root / ".venv-ros" / "lib" / "python3.12" / "site-packages"
    if _bootstrap_root is not None
    else None
)
if (
    _venv_site is not None
    and _venv_site.is_dir()
    and any(importlib.util.find_spec(name) is None for name in ("scipy", "pyarrow"))
):
    sys.path.append(str(_venv_site))

from rclpy.callback_groups import ReentrantCallbackGroup  # noqa: E402
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import (  # noqa: E402
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_srvs.srv import Trigger  # noqa: E402

from sim2real import contracts  # noqa: E402
from sim2real.data import CuratedDatasetWriter, RejectionLog  # noqa: E402
from sim2real.expert.rollout import EpisodeRecord  # noqa: E402
from sim2real.filtering.config import load_filter_config  # noqa: E402
from sim2real.filtering.conversion import fields_to_episode  # noqa: E402
from sim2real.filtering.decision import FilterResult  # noqa: E402
from sim2real.filtering.pipeline import FilterPipeline  # noqa: E402

try:
    from sim2real_msgs.msg import FilterMetrics, RolloutEpisode
except ImportError:  # pragma: no cover - the ROS build provides these
    FilterMetrics = None  # type: ignore[assignment,misc]
    RolloutEpisode = None  # type: ignore[assignment,misc]

FILTER_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=50,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


def fields_from_message(message: Any) -> dict[str, Any]:
    """Extract plain fields from a ``RolloutEpisode`` message."""
    steps = []
    for step in message.steps:
        pose = step.ee_pose
        stamp = step.stamp
        steps.append(
            {
                "stamp_ns": int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec),
                "step_index": int(step.step_index),
                "joint_positions": [float(value) for value in step.joint_positions],
                "joint_velocities": [float(value) for value in step.joint_velocities],
                "ee_pose": {
                    "position": [
                        float(pose.position.x),
                        float(pose.position.y),
                        float(pose.position.z),
                    ],
                    "orientation": [
                        float(pose.orientation.x),
                        float(pose.orientation.y),
                        float(pose.orientation.z),
                        float(pose.orientation.w),
                    ],
                },
                "action": [float(value) for value in step.action],
                "action_space": int(step.action_space),
                "gripper_position": float(step.gripper_position),
                "unintended_collision": bool(step.unintended_collision),
            }
        )
    return {
        "episode_id": str(message.episode_id),
        "backend": str(message.backend),
        "seed": int(message.seed),
        "task_instruction": str(message.task_instruction),
        "control_rate_hz": float(message.control_rate_hz),
        "is_perturbed": bool(message.is_perturbed),
        "outcome": int(message.outcome),
        "failure_reason": str(message.failure_reason),
        "metadata_json": str(message.metadata_json),
        "steps": steps,
    }


def _metric_message(result: FilterResult) -> Any:
    message = FilterMetrics()
    message.episode_id = result.episode_id
    metrics = result.metrics
    message.dimensionless_squared_jerk = (
        float(metrics.dimensionless_squared_jerk) if metrics else float("nan")
    )
    message.sparc = float(metrics.sparc) if metrics else float("nan")
    message.path_length_m = float(metrics.path_length_m) if metrics else float("nan")
    message.min_joint_limit_margin = (
        float(metrics.min_joint_limit_margin) if metrics else float("nan")
    )
    message.min_velocity_limit_margin = (
        float(metrics.min_velocity_limit_margin) if metrics else float("nan")
    )
    message.collision_count = int(metrics.collision_count) if metrics else 0
    message.task_success = bool(metrics.task_success) if metrics else False
    message.acceptance_score = float(result.acceptance_score)
    message.decision = int(result.decision)
    message.rejection_reasons = [reason.text() for reason in result.reasons]
    try:
        message.metadata_json = result.metadata_json()
    except ValueError:
        message.metadata_json = "{}"
    return message


class TrajectoryFilterNode(Node):
    def __init__(
        self,
        config_path: str | None = None,
        curated_root: str | None = None,
        rejected_root: str | None = None,
        dataset_name: str | None = None,
    ) -> None:
        super().__init__("trajectory_filter_node")
        self.declare_parameter("config_path", "")
        self.declare_parameter("curated_root", "data/curated")
        self.declare_parameter("rejected_root", "data/rejected")
        self.declare_parameter("dataset_name", "staged_curated")

        config_path = config_path or str(self.get_parameter("config_path").value)
        if not config_path:
            from ament_index_python.packages import get_package_share_directory

            share = get_package_share_directory("sim2real_sim")
            config_path = os.path.join(share, "filtering", "default.yaml")
        self.config = load_filter_config(config_path)
        self.pipeline = FilterPipeline(self.config)
        self.curated_root = curated_root or str(self.get_parameter("curated_root").value)
        self.rejected_root = rejected_root or str(self.get_parameter("rejected_root").value)
        self.dataset_name = dataset_name or str(self.get_parameter("dataset_name").value)

        self._lock = threading.Lock()
        self._accepted: list[tuple[EpisodeRecord, FilterResult]] = []
        self._logs: dict[str, RejectionLog] = {}

        callback_group = ReentrantCallbackGroup()
        self.create_subscription(
            RolloutEpisode,
            contracts.STAGING_EPISODE_TOPIC,
            self._on_episode,
            FILTER_QOS,
            callback_group=callback_group,
        )
        self._metrics_publisher = self.create_publisher(
            FilterMetrics, contracts.FILTER_METRICS_TOPIC, FILTER_QOS
        )
        self._filtered_publisher = self.create_publisher(
            RolloutEpisode, contracts.FILTERED_EPISODE_TOPIC, FILTER_QOS
        )
        self.create_service(
            Trigger,
            contracts.FINALIZE_DATASET_SERVICE,
            self._on_finalize,
            callback_group=callback_group,
        )
        self.get_logger().info(
            f"filtering with {config_path}; curated={self.curated_root} "
            f"rejected={self.rejected_root} dataset={self.dataset_name}"
        )

    # -- callbacks ---------------------------------------------------------
    def _on_episode(self, message: Any) -> None:
        episode_id = str(getattr(message, "episode_id", "") or "")
        try:
            record = fields_to_episode(fields_from_message(message))
            result = self.pipeline.process(record)
        except Exception as exc:  # noqa: BLE001 - malformed input must not crash
            self.get_logger().error(f"invalid episode {episode_id!r}: {exc}")
            result = self._invalid_result(episode_id, exc)
            record = None
        self._metrics_publisher.publish(_metric_message(result))
        if result.accepted and record is not None:
            with self._lock:
                self._accepted.append((record, result))
            self._filtered_publisher.publish(message)
            self.get_logger().info(f"accepted {episode_id}")
            return
        self.get_logger().info(
            f"rejected {episode_id}: {[reason.code for reason in result.reasons]}"
        )
        if record is not None:
            self._log_rejection(record, result)

    def _invalid_result(self, episode_id: str, exc: Exception) -> FilterResult:
        from sim2real.filtering.decision import RejectionReason

        return FilterResult(
            episode_id=episode_id,
            decision=contracts.DECISION_INVALID,
            acceptance_score=0.0,
            reasons=[RejectionReason("MALFORMED_EPISODE", None, None, None, str(exc))],
            config_hash=self.config.config_hash(),
        )

    def _log_rejection(self, record: EpisodeRecord, result: FilterResult) -> None:
        run_id = record.run_id or "ros_filter"
        log = self._logs.get(run_id)
        if log is None:
            log = RejectionLog(self.rejected_root, run_id)
            self._logs[run_id] = log
        try:
            log.append(record, result, staging_reference=record.episode_id)
        except Exception as exc:  # noqa: BLE001 - logging must not stop filtering
            self.get_logger().error(f"could not write rejection log: {exc}")

    def _on_finalize(self, request: Any, response: Any) -> Any:
        del request
        try:
            written = self._finalize_buffer()
        except Exception as exc:  # noqa: BLE001 - service must return an error, not crash
            response.success = False
            response.message = f"finalize failed: {exc}"
            self.get_logger().error(response.message)
            return response
        if written is None:
            response.success = False
            response.message = "no accepted episodes are buffered"
            return response
        response.success = True
        response.message = (
            f"wrote {written.version} with {written.num_episodes} episodes "
            f"({written.num_new_episodes} new) to {written.path}"
        )
        return response

    def _finalize_buffer(self) -> Any | None:
        with self._lock:
            accepted = list(self._accepted)
            if not accepted:
                return None
            writer = CuratedDatasetWriter(self.curated_root, self.dataset_name)
            written = writer.write(accepted, config=self.config)
            self._accepted.clear()
        return written

    def destroy_node(self) -> None:
        try:
            written = self._finalize_buffer()
            if written is not None:
                self.get_logger().info(
                    f"shutdown: wrote {written.version} with {written.num_episodes} episodes"
                )
        except Exception as exc:  # noqa: BLE001 - shutdown must not raise
            self.get_logger().error(f"shutdown finalize failed: {exc}")
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = TrajectoryFilterNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
