"""ROS 2 VLA inference node.

Consumes synchronized overhead/wrist RGB images, ``/joint_states``, and
``/task_instruction``; sends the observation to the out-of-process policy
server on a worker thread; publishes raw, **unsafe** action chunks on
``/vla/raw_action``. No controller is started or connected here: Task 11 owns
the safety layer and the final time convention.

The node never imports torch. The ML environment is a child process.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

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

from builtin_interfaces.msg import Duration, Time  # noqa: E402
from message_filters import ApproximateTimeSynchronizer, Subscriber  # noqa: E402
from rclpy.executors import (
    ExternalShutdownException,  # noqa: E402
    MultiThreadedExecutor,  # noqa: E402
)
from rclpy.node import Node  # noqa: E402
from rclpy.qos import (  # noqa: E402
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image  # noqa: E402
from sensor_msgs.msg import JointState as JointStateMsg  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint  # noqa: E402

from sim2real import contracts  # noqa: E402
from sim2real.inference import (  # noqa: E402
    AsyncInferenceWorker,
    ChunkError,
    ImageInput,
    InferenceConfig,
    InferenceConfigError,
    JointStateInput,
    LatencyTracker,
    ObservationAssembler,
    ValidationError,
    finger_out_of_range,
    future_fraction,
    load_inference_config,
    make_chunk,
    make_policy_factory,
    summarize,
    trim_for_latency,
)
from sim2real.inference.async_worker import (  # noqa: E402
    FAILED,
    MODEL_UNAVAILABLE,
    READY,
    RUNNING,
    STARTING,
    STOPPED,
)
from sim2real.sim.qos import (  # noqa: E402
    CAMERA_QOS,
    JOINT_STATES_QOS,
    TASK_INSTRUCTION_QOS,
    VLA_RAW_ACTION_QOS,
    VLA_STATUS_QOS,
    QosSpec,
)

_RELIABILITY = {
    "reliable": ReliabilityPolicy.RELIABLE,
    "best_effort": ReliabilityPolicy.BEST_EFFORT,
}
_DURABILITY = {
    "volatile": DurabilityPolicy.VOLATILE,
    "transient_local": DurabilityPolicy.TRANSIENT_LOCAL,
}

WAITING_FOR_INPUT = "WAITING_FOR_INPUT"


def _profile(spec: QosSpec) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=spec.depth,
        reliability=_RELIABILITY[spec.reliability],
        durability=_DURABILITY[spec.durability],
    )


def _stamp_ns(stamp: Time) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _duration_ns(value: int) -> Duration:
    return Duration(sec=int(value // 1_000_000_000), nanosec=int(value % 1_000_000_000))


def _stamp_message(stamp_ns: int) -> Time:
    return Time(sec=int(stamp_ns // 1_000_000_000), nanosec=int(stamp_ns % 1_000_000_000))


def _find_repo_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "sim2real" / "__init__.py").is_file():
            return parent
    return None


def _installed_default_config() -> Path | None:
    try:
        from ament_index_python.packages import get_package_share_directory

        path = Path(get_package_share_directory("sim2real_vla")) / "config" / "default.yaml"
        return path if path.is_file() else None
    except Exception:  # noqa: BLE001 - ament index is optional in tests
        return None


class VlaInferenceNode(Node):
    """Async, policy-agnostic VLA inference adapter."""

    def __init__(self, **node_kwargs: Any) -> None:
        super().__init__("vla_inference_node", **node_kwargs)
        self.declare_parameter("config_file", "")
        config_file = str(self.get_parameter("config_file").value)
        config = self._load_config(config_file)

        # Declare every field so launch/CLI overrides win over the YAML default.
        for name, value in config.as_dict().items():
            self.declare_parameter(name, value)
        values = {name: self.get_parameter(name).value for name in config.as_dict()}
        try:
            self._config = InferenceConfig.from_values(values)
        except InferenceConfigError as exc:
            raise ValueError(f"invalid inference parameters: {exc}") from exc

        self._repo_root = _find_repo_root()
        self._lock = threading.RLock()
        self._assembler = ObservationAssembler(self._config)

        self._joint_state: JointStateInput | None = None
        self._overhead: ImageInput | None = None
        self._wrist: ImageInput | None = None
        self._instruction: str | None = None
        self._observation_sequence = 0
        self._last_attempted_sequence = -1
        self._last_accepted_sequence: int | None = None
        self._last_submit_ns = 0
        self._last_status_log_ns: dict[str, int] = {}

        self._state = STARTING
        self._detail = "starting"
        self._last_error: str | None = None
        self._last_latency_s: float | None = None
        self._last_chunk_points = 0
        self._last_publish_stamp_ns: int | None = None
        self._last_finger_out_of_range = False
        self._latency_tracker = LatencyTracker(max_samples=500)
        self._published_future_fractions: deque[float] = deque(maxlen=500)
        self._last_observation_elapsed_s: float | None = None
        self._last_selected_future_fraction: float | None = None
        self._last_published_future_fraction: float | None = None
        self._counters = {
            "observations_submitted": 0,
            "chunks_published": 0,
            "chunks_rejected": 0,
            "chunks_dropped_stale": 0,
            "instructions_received": 0,
            "joint_states_received": 0,
            "image_pairs_received": 0,
        }
        self._rejected: dict[str, int] = {}

        self._action_publisher = self.create_publisher(
            JointTrajectory,
            contracts.VLA_RAW_ACTION_TOPIC,
            _profile(VLA_RAW_ACTION_QOS),
        )
        self._status_publisher = self.create_publisher(
            String,
            contracts.VLA_STATUS_TOPIC,
            _profile(VLA_STATUS_QOS),
        )

        overhead_topic = f"/camera/{contracts.CAMERA_OVERHEAD}/color/image_raw"
        wrist_topic = f"/camera/{contracts.CAMERA_WRIST}/color/image_raw"
        self._overhead_subscriber = Subscriber(
            self, Image, overhead_topic, qos_profile=_profile(CAMERA_QOS)
        )
        self._wrist_subscriber = Subscriber(
            self, Image, wrist_topic, qos_profile=_profile(CAMERA_QOS)
        )
        self._synchronizer = ApproximateTimeSynchronizer(
            [self._overhead_subscriber, self._wrist_subscriber],
            queue_size=10,
            slop=float(self._config.max_camera_skew_s),
        )
        self._synchronizer.registerCallback(self._on_images)

        self.create_subscription(
            JointStateMsg,
            contracts.JOINT_STATES_TOPIC,
            self._on_joint_states,
            _profile(JOINT_STATES_QOS),
        )
        self.create_subscription(
            String,
            contracts.TASK_INSTRUCTION_TOPIC,
            self._on_instruction,
            _profile(TASK_INSTRUCTION_QOS),
        )

        self._policy_factory = make_policy_factory(self._config, self._repo_root)
        self._worker = AsyncInferenceWorker(
            self._policy_factory,
            call_timeout_s=self._config.call_timeout_s,
            max_restarts=self._config.max_restarts,
            restart_backoff_s=self._config.restart_backoff_s,
            sample_seed=self._config.sample_seed,
            stale_after_s=self._config.max_observation_age_s,
            max_chunk_length=self._config.max_chunk_length,
            on_result=self._on_result,
            on_state=self._on_worker_state,
            logger=self.get_logger().warning,
        )
        self._status_timer = self.create_timer(
            1.0 / self._config.status_rate_hz, self._on_status_timer
        )
        self._worker.start()

        self.get_logger().info(
            "vla_inference_node starting: "
            f"backend={self._config.policy_backend} "
            f"period={self._config.policy_period_s:.3f}s "
            f"n_action_steps={self._config.n_action_steps} "
            f"latency_compensation={self._config.latency_compensation} "
            f"sample_seed={self._config.sample_seed}"
        )
        if self._config.sample_seed < 0:
            self.get_logger().info("sample_seed=-1: policy sampling is unseeded")
        self.get_logger().info("depth topics are not used by the model; training used RGB only")

    # ------------------------------------------------------------------ config
    def _load_config(self, config_file: str) -> InferenceConfig:
        path: Path | None = None
        if config_file:
            path = Path(config_file).expanduser()
        else:
            root = _find_repo_root()
            candidate = None if root is None else root / "configs" / "inference" / "default.yaml"
            if candidate is not None and candidate.is_file():
                path = candidate
            else:
                path = _installed_default_config()
        if path is None:
            return InferenceConfig()
        return load_inference_config(path)

    # --------------------------------------------------------------- callbacks
    def _on_joint_states(self, message: JointStateMsg) -> None:
        now_ns = time.monotonic_ns()
        item = JointStateInput(
            names=tuple(message.name),
            positions=tuple(message.position),
            stamp_ns=_stamp_ns(message.header.stamp),
            received_ns=now_ns,
        )
        with self._lock:
            self._joint_state = item
            self._counters["joint_states_received"] += 1
        self._refresh_state()

    def _on_instruction(self, message: String) -> None:
        with self._lock:
            self._instruction = str(message.data)
            self._counters["instructions_received"] += 1
            self._observation_sequence += 1
        self._maybe_submit()

    def _on_images(self, overhead: Image, wrist: Image) -> None:
        now_ns = time.monotonic_ns()
        overhead_input = ImageInput(
            data=bytes(overhead.data),
            encoding=str(overhead.encoding),
            height=int(overhead.height),
            width=int(overhead.width),
            step=int(overhead.step),
            stamp_ns=_stamp_ns(overhead.header.stamp),
            received_ns=now_ns,
            is_bigendian=int(overhead.is_bigendian),
            frame_id=str(overhead.header.frame_id),
        )
        wrist_input = ImageInput(
            data=bytes(wrist.data),
            encoding=str(wrist.encoding),
            height=int(wrist.height),
            width=int(wrist.width),
            step=int(wrist.step),
            stamp_ns=_stamp_ns(wrist.header.stamp),
            received_ns=now_ns,
            is_bigendian=int(wrist.is_bigendian),
            frame_id=str(wrist.header.frame_id),
        )
        with self._lock:
            self._overhead = overhead_input
            self._wrist = wrist_input
            self._counters["image_pairs_received"] += 1
            self._observation_sequence += 1
        self._maybe_submit()

    # ------------------------------------------------------------------- submit
    def _maybe_submit(self) -> None:
        with self._lock:
            if self._state in (FAILED, STOPPED):
                return
            if self._instruction is None:
                self._reject_locked("NO_INSTRUCTION", "no /task_instruction received")
                return
            if self._joint_state is None or self._overhead is None or self._wrist is None:
                self._reject_locked("MISSING_INPUT", "waiting for state and both cameras")
                return
            now_ns = time.monotonic_ns()
            period_ns = int(1_000_000_000 / self._config.max_inference_rate_hz)
            if now_ns - self._last_submit_ns < period_ns:
                return
            sequence = self._observation_sequence
            if sequence == self._last_attempted_sequence:
                return
            self._last_attempted_sequence = sequence
            joint_state = self._joint_state
            overhead = self._overhead
            wrist = self._wrist
            instruction = self._instruction
        try:
            observation = self._assembler.assemble(
                joint_state=joint_state,
                overhead=overhead,
                wrist=wrist,
                instruction=instruction,
                now_ns=now_ns,
                sequence=sequence,
                last_sequence=self._last_accepted_sequence,
            )
        except ValidationError as exc:
            self._record_rejection(exc.reason, exc.detail)
            return
        request_id = self._worker.submit(
            observation.to_policy_observation(),
            stamp_ns=observation.stamp_ns,
            received_ns=observation.received_ns,
        )
        if request_id is None:
            self._record_rejection("WORKER_NOT_ACCEPTING", "worker is failed or stopping")
            return
        with self._lock:
            self._last_accepted_sequence = observation.sequence
            self._last_submit_ns = now_ns
            self._counters["observations_submitted"] += 1
        self._refresh_state()

    def _reject_locked(self, reason: str, detail: str) -> None:
        self._rejected[reason] = self._rejected.get(reason, 0) + 1
        self._last_error = f"{reason}: {detail}"
        self._log_throttled(reason, detail)
        self._refresh_state_locked()

    def _record_rejection(self, reason: str, detail: str) -> None:
        with self._lock:
            self._reject_locked(reason, detail)

    def _log_throttled(self, reason: str, detail: str) -> None:
        now_ns = time.monotonic_ns()
        last = self._last_status_log_ns.get(reason, 0)
        if now_ns - last < 1_000_000_000:
            return
        self._last_status_log_ns[reason] = now_ns
        self.get_logger().warning(f"observation rejected: {reason}: {detail}")

    # ------------------------------------------------------------------- result
    def _on_result(self, result: Any) -> None:
        try:
            chunk = make_chunk(
                result.actions,
                stamp_ns=result.observation_stamp_ns,
                step_period_s=self._config.policy_period_s,
                n_action_steps=self._config.n_action_steps,
                max_chunk_length=self._config.max_chunk_length,
            )
        except ChunkError as exc:
            with self._lock:
                self._counters["chunks_rejected"] += 1
                self._record_rejection("CHUNK_INVALID", str(exc))
            return

        elapsed_ns = int(result.received_ns) - int(result.observation_received_ns)
        selected_future, _selected_total, selected_fraction = future_fraction(
            chunk.time_from_start_ns, elapsed_ns
        )
        with self._lock:
            self._latency_tracker.add(float(result.latency_s))
            self._last_observation_elapsed_s = elapsed_ns / 1_000_000_000.0
            self._last_selected_future_fraction = selected_fraction

        if self._config.latency_compensation:
            trimmed = trim_for_latency(chunk, int(result.observation_stamp_ns) + elapsed_ns)
            if trimmed is None:
                with self._lock:
                    self._counters["chunks_dropped_stale"] += 1
                    self._last_error = (
                        "STALE_CHUNK: every selected action is older than the "
                        "observation stamp horizon"
                    )
                self._log_throttled("STALE_CHUNK", "chunk entirely in the past")
                return
            chunk = trimmed

        _future, _total, published_fraction = future_fraction(chunk.time_from_start_ns, elapsed_ns)
        outside = finger_out_of_range(chunk)
        message = self._to_message(chunk)
        with self._lock:
            if self._state in (FAILED, STOPPED):
                return
            self._counters["chunks_published"] += 1
            self._last_latency_s = float(result.latency_s)
            self._last_chunk_points = chunk.point_count
            self._last_publish_stamp_ns = int(chunk.stamp_ns)
            self._last_finger_out_of_range = bool(outside)
            self._last_published_future_fraction = published_fraction
            self._published_future_fractions.append(published_fraction)
            self._last_error = (
                f"GRIPPER_OUT_OF_RANGE: rows {list(outside)} passed through raw"
                if outside
                else None
            )
        self._action_publisher.publish(message)
        if outside:
            self._log_throttled(
                "GRIPPER_OUT_OF_RANGE",
                "gripper target outside [0, 0.04] m was passed through raw",
            )
        self._refresh_state()

    def _to_message(self, chunk: Any) -> JointTrajectory:
        message = JointTrajectory()
        message.header.stamp = _stamp_message(chunk.stamp_ns)
        message.header.frame_id = chunk.frame_id
        message.joint_names = list(chunk.joint_names)
        for offset_ns, positions in zip(chunk.time_from_start_ns, chunk.positions, strict=True):
            point = JointTrajectoryPoint()
            point.positions = [float(value) for value in positions]
            point.time_from_start = _duration_ns(int(offset_ns))
            message.points.append(point)
        return message

    # -------------------------------------------------------------------- state
    def _on_worker_state(self, state: str, detail: str | None) -> None:
        with self._lock:
            self._detail = detail or state
            if state == FAILED:
                self._state = FAILED
            elif state in (MODEL_UNAVAILABLE, STARTING, STOPPED):
                self._state = state
            else:
                self._refresh_state_locked()
        if state in (MODEL_UNAVAILABLE, FAILED):
            self.get_logger().warning(f"worker state {state}: {detail}")

    def _refresh_state(self) -> None:
        with self._lock:
            self._refresh_state_locked()

    def _refresh_state_locked(self) -> None:
        worker = self._worker if hasattr(self, "_worker") else None
        if worker is None:
            self._state = STARTING
            return
        stats = worker.stats()
        worker_state = stats["state"]
        if worker_state == FAILED:
            self._state = FAILED
            self._detail = stats.get("last_error") or "worker failed"
            return
        if worker_state in (MODEL_UNAVAILABLE, STARTING, STOPPED):
            self._state = worker_state
            self._detail = stats.get("detail") or worker_state
            return
        if stats.get("in_flight") is not None:
            self._state = RUNNING
            self._detail = f"request {stats['in_flight']} in flight"
            return
        now_ns = time.monotonic_ns()
        missing = []
        if self._instruction is None:
            missing.append("instruction")
        if self._joint_state is None:
            missing.append("joint_states")
        if self._overhead is None or self._wrist is None:
            missing.append("cameras")
        if missing:
            self._state = WAITING_FOR_INPUT
            self._detail = "waiting for " + ", ".join(missing)
            return
        assert self._joint_state is not None
        assert self._overhead is not None
        assert self._wrist is not None
        state_age_s = (now_ns - self._joint_state.received_ns) / 1_000_000_000.0
        camera_age_s = (
            now_ns - min(self._overhead.received_ns, self._wrist.received_ns)
        ) / 1_000_000_000.0
        if state_age_s > self._config.max_state_age_s:
            self._state = WAITING_FOR_INPUT
            self._detail = f"joint state is stale ({state_age_s:.3f} s)"
            return
        if camera_age_s > self._config.max_observation_age_s:
            self._state = WAITING_FOR_INPUT
            self._detail = f"cameras are stale ({camera_age_s:.3f} s)"
            return
        self._state = READY
        self._detail = "ready"

    # ------------------------------------------------------------------- status
    def _status_payload(self) -> dict[str, Any]:
        with self._lock:
            payload: dict[str, Any] = {
                "state": self._state,
                "detail": self._detail,
                "policy_backend": self._config.policy_backend,
                "checkpoint": (
                    self._config.checkpoint_path if self._config.policy_backend == "server" else ""
                ),
                "counters": dict(self._counters),
                "rejected": dict(self._rejected),
                "last_latency_s": self._last_latency_s,
                "latency": self._latency_tracker.summary(),
                "last_observation_elapsed_s": self._last_observation_elapsed_s,
                "last_selected_future_fraction": self._last_selected_future_fraction,
                "last_published_future_fraction": self._last_published_future_fraction,
                "published_future_fraction": summarize(list(self._published_future_fractions)),
                "last_error": self._last_error,
                "last_chunk_points": self._last_chunk_points,
                "last_publish_stamp_ns": self._last_publish_stamp_ns,
                "last_finger_out_of_range": self._last_finger_out_of_range,
                "policy_period_s": self._config.policy_period_s,
                "n_action_steps": self._config.n_action_steps,
                "latency_compensation": self._config.latency_compensation,
                "sample_seed": self._config.sample_seed,
                "depth_used": False,
            }
        worker = getattr(self, "_worker", None)
        if worker is not None:
            stats = worker.stats()
            payload["worker"] = stats
            payload["time_to_ready_s"] = stats.get("time_to_ready_s")
        else:
            payload["time_to_ready_s"] = None
        return payload

    def _on_status_timer(self) -> None:
        self._refresh_state()
        try:
            message = String(
                data=json.dumps(self._status_payload(), sort_keys=True, allow_nan=False)
            )
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            self.get_logger().warning(f"could not serialise status: {exc}")
            return
        self._status_publisher.publish(message)

    # ---------------------------------------------------------------- shutdown
    def destroy_node(self) -> None:
        try:
            self._worker.shutdown()
        finally:
            stats = self._worker.stats()
            print(
                "vla inference stats: "
                f"state={stats['state']} submitted={stats['submitted']} "
                f"replaced={stats['replaced']} completed={stats['completed']} "
                f"failed_calls={stats['failed_calls']} "
                f"timed_out_calls={stats['timed_out_calls']} "
                f"restarts={stats['restarts']} "
                f"published={self._counters['chunks_published']} "
                f"stale={self._counters['chunks_dropped_stale']} "
                f"rejected={self._counters['chunks_rejected']}",
                flush=True,
            )
            super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = VlaInferenceNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():
            raise
    finally:
        # A second SIGINT can arrive while rclpy tears down publishers. The
        # worker has already been stopped by destroy_node's own try/finally.
        try:
            executor.shutdown()
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
