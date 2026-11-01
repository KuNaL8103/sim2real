"""ROS 2 adapter for the scripted expert rollout collector.

Why the backend runs **in-process**
----------------------------------
The expert plans and executes at the simulator's control rate. Taking a ROS
round-trip per step would add scheduler latency and make trajectories depend on
the executor, not the seed. The collector therefore owns one backend instance in
its own process/thread and calls ``run_episode`` directly.

This node does **not** publish ``/joint_states`` and is not a simulator. When
``sim_node`` runs, it remains the only ``/joint_states`` publisher. The collector
publishes only completed ``RolloutEpisode`` messages on
``/sim2real/staging_episode`` and writes the same records to the staging buffer.
Images are not captured by default and are never embedded in the message.

Service: ``/sim2real/generate_rollouts`` (UNIFORM mode only). Batches run on a
worker thread so other callbacks keep spinning; a second concurrent call is
rejected instead of queueing.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import rclpy

# Same bootstrap as sim_node: add src/ and, when needed, .venv-ros site-packages.
_bootstrap_root = None
for _parent in Path(__file__).resolve().parents:
    if (_parent / "src" / "sim2real" / "__init__.py").is_file():
        _bootstrap_root = _parent
        break
if importlib.util.find_spec("sim2real") is None and _bootstrap_root is not None:
    sys.path.insert(0, str(_bootstrap_root / "src"))
if importlib.util.find_spec("mujoco") is None and _bootstrap_root is not None:
    _venv_site = _bootstrap_root / ".venv-ros" / "lib" / "python3.12" / "site-packages"
    if _venv_site.is_dir():
        sys.path.append(str(_venv_site))

from builtin_interfaces.msg import Time  # noqa: E402
from geometry_msgs.msg import Pose  # noqa: E402
from rclpy.callback_groups import ReentrantCallbackGroup  # noqa: E402
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402

from sim2real import contracts  # noqa: E402
from sim2real.expert import (  # noqa: E402
    UnparseableInstructionError,
    episode_to_rollout_fields,
    load_expert_config,
    load_perturbations,
    parse_instruction,
    run_episode,
    select_perturbation,
)
from sim2real.expert.staging import StagingBuffer, StagingError  # noqa: E402
from sim2real.metadata import validate_metadata_json  # noqa: E402
from sim2real.sim import create_backend  # noqa: E402
from sim2real.sim.camera import mat_to_quat  # noqa: E402

try:
    from sim2real_msgs.msg import RolloutEpisode, TrajectoryStep
    from sim2real_msgs.srv import GenerateRollouts
except ImportError:  # pragma: no cover - the ROS build provides these
    RolloutEpisode = None  # type: ignore[assignment,misc]
    TrajectoryStep = None  # type: ignore[assignment,misc]
    GenerateRollouts = None  # type: ignore[assignment,misc]

STAGING_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    # Transient-local so a filter/recorder that starts just after a batch still
    # receives the most recent completed episodes. Episodes carry no images.
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


def _stamp(stamp_ns: int) -> Time:
    stamp = Time()
    stamp.sec = int(stamp_ns // 1_000_000_000)
    stamp.nanosec = int(stamp_ns % 1_000_000_000)
    return stamp


class ExpertCollectorNode(Node):
    """Run bounded expert batches in-process and stage each episode."""

    def __init__(
        self,
        injected_backend: Any | None = None,
        output_dir: str | None = None,
        config_dir: str | None = None,
        randomization_dir: str | None = None,
        expert_config_dir: str | None = None,
    ) -> None:
        super().__init__("expert_collector_node")
        self.declare_parameter("backend", "fake")
        self.declare_parameter("config_dir", "")
        self.declare_parameter("randomization_dir", "")
        self.declare_parameter("expert_config_dir", "")
        self.declare_parameter("output_dir", "data/staging")
        self.declare_parameter("max_rollouts_per_call", 64)
        self.declare_parameter("perturbed_fraction", -1.0)

        config_dir = config_dir or str(self.get_parameter("config_dir").value)
        from ament_index_python.packages import get_package_share_directory

        share = get_package_share_directory("sim2real_sim")
        if not config_dir:
            config_dir = os.path.join(share, "config")
        randomization_dir = randomization_dir or str(self.get_parameter("randomization_dir").value)
        if not randomization_dir:
            randomization_dir = os.path.join(os.path.dirname(config_dir), "randomization")
        expert_config_dir = expert_config_dir or str(self.get_parameter("expert_config_dir").value)
        if not expert_config_dir:
            expert_config_dir = os.path.join(share, "expert")

        self._backend_name = str(self.get_parameter("backend").value)
        self._config_dir = config_dir
        self._randomization_dir = randomization_dir
        self._expert_config_dir = expert_config_dir
        self._output_dir = output_dir or str(self.get_parameter("output_dir").value)
        self._max_rollouts = int(self.get_parameter("max_rollouts_per_call").value)
        self._perturbed_fraction_override = float(self.get_parameter("perturbed_fraction").value)
        if self._max_rollouts < 1:
            raise ValueError("max_rollouts_per_call must be >= 1")

        self._backend = injected_backend
        self._backend_lock = threading.Lock()
        self._batch_lock = threading.Lock()
        self._callback_group = ReentrantCallbackGroup()

        if RolloutEpisode is None or GenerateRollouts is None:
            raise RuntimeError("sim2real_msgs is not importable; build and source ros2_ws")
        self._episode_publisher = self.create_publisher(
            RolloutEpisode, contracts.STAGING_EPISODE_TOPIC, STAGING_QOS
        )
        self._service = self.create_service(
            GenerateRollouts,
            contracts.GENERATE_ROLLOUTS_SERVICE,
            self._handle_generate,
            callback_group=self._callback_group,
        )
        self.get_logger().info(
            f"expert collector ready; backend={self._backend_name} "
            f"output_dir={self._output_dir} max_rollouts={self._max_rollouts}"
        )

    # -- backend -----------------------------------------------------------
    def _get_backend(self) -> Any:
        with self._backend_lock:
            if self._backend is None:
                self._backend = create_backend(self._backend_name, self._config_dir)
            return self._backend

    # -- service -----------------------------------------------------------
    def _handle_generate(self, request: Any, response: Any) -> Any:
        response.success = False
        response.num_generated = 0
        response.num_failed = 0
        response.message = ""
        response.result_metadata_json = ""

        count = int(request.num_rollouts)
        if count == 0 or count > self._max_rollouts:
            response.message = (
                f"num_rollouts={count} is out of bounds; expected 1..{self._max_rollouts}"
            )
            return response
        mode = int(request.sampling_mode)
        if mode == contracts.SAMPLING_FAILURE_TARGETED:
            response.message = "FAILURE_TARGETED not implemented until Task 9"
            return response
        if mode != contracts.SAMPLING_UNIFORM:
            response.message = f"unknown sampling_mode={mode}; only UNIFORM (0) is supported"
            return response
        if not self._batch_lock.acquire(blocking=False):
            response.message = "collector is busy with another batch"
            return response

        try:
            done = threading.Event()
            holder: dict[str, Any] = {}

            def worker() -> None:
                try:
                    holder["result"] = self._run_batch(request)
                except Exception as exc:  # noqa: BLE001 - reported by the callback
                    holder["error"] = exc
                finally:
                    done.set()

            thread = threading.Thread(target=worker, name="expert_batch", daemon=True)
            thread.start()
            timeout_s = max(60.0, float(request.num_rollouts) * 30.0)
            if not done.wait(timeout_s):
                response.message = (
                    f"batch did not finish within {timeout_s:.0f} s; "
                    "episodes written so far remain staged"
                )
                return response
            if "error" in holder:
                raise holder["error"]
            result = holder["result"]
            response.success = bool(result["success"])
            response.num_generated = int(result["num_generated"])
            response.num_failed = int(result["num_failed"])
            response.message = str(result["message"])
            response.result_metadata_json = str(result["result_metadata_json"])
        except Exception as exc:  # noqa: BLE001 - report, do not kill the node
            self.get_logger().error(f"generate_rollouts failed: {exc}")
            response.message = f"generate_rollouts failed: {type(exc).__name__}: {exc}"
            response.result_metadata_json = json.dumps(
                {"error": response.message}, sort_keys=True, allow_nan=False
            )
        finally:
            self._batch_lock.release()
        return response

    def _run_batch(self, request: Any) -> dict[str, Any]:
        config = load_expert_config(self._expert_config_dir)
        backend = self._get_backend()
        instruction = str(request.task_instruction).strip()
        if not instruction:
            return self._failure("task_instruction must not be empty")
        try:
            task = parse_instruction(instruction, backend.configs.scene, config.instructions_file)
        except UnparseableInstructionError as exc:
            return self._failure(f"unparseable instruction: {exc}")

        perturbations = load_perturbations(config)
        fraction = self._perturbed_fraction_override
        if fraction < 0.0:
            fraction = perturbations.perturbed_fraction
        if not bool(request.include_perturbed):
            fraction = 0.0

        now = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        run_id = f"expert_{now}_{int(request.seed)}"
        buffer = StagingBuffer(self._output_dir, run_id)
        count = int(request.num_rollouts)
        seeds = [int(request.seed) + index for index in range(count)]
        counts = {"success": 0, "failure": 0, "timeout": 0, "aborted": 0, "unknown": 0}
        perturbed = 0
        generated = 0

        outcome_names = {
            contracts.OUTCOME_SUCCESS: "success",
            contracts.OUTCOME_FAILURE: "failure",
            contracts.OUTCOME_TIMEOUT: "timeout",
            contracts.OUTCOME_ABORTED: "aborted",
            contracts.OUTCOME_UNKNOWN: "unknown",
        }

        for index, seed in enumerate(seeds):
            perturbation = select_perturbation(
                perturbations, seed, index, perturbed_fraction=fraction
            )
            record = run_episode(
                backend,
                task,
                seed,
                instruction=instruction,
                episode_index=index,
                run_id=run_id,
                randomization_profile=config.default_profile,
                randomization_dir=self._randomization_dir,
                perturbation=perturbation,
                config=config,
            )
            try:
                buffer.write(record)
            except StagingError as exc:
                self.get_logger().error(f"staging write failed for {record.episode_id}: {exc}")
                counts[outcome_names.get(record.outcome, "unknown")] += 1
                continue
            generated += 1
            counts[outcome_names.get(record.outcome, "unknown")] += 1
            if record.is_perturbed:
                perturbed += 1
            try:
                self._episode_publisher.publish(self._to_message(record))
            except Exception as exc:  # noqa: BLE001 - a bad record must not stop the batch
                self.get_logger().warning(
                    f"episode {record.episode_id} staged but not published: {exc}"
                )

        num_failed = counts["failure"] + counts["timeout"] + counts["aborted"]
        payload = {
            "run_id": run_id,
            "backend": self._backend_name,
            "seed": int(request.seed),
            "seeds": seeds,
            "num_requested": count,
            "num_generated": generated,
            "num_failed": num_failed,
            "counts": counts,
            "num_perturbed": perturbed,
            "perturbed_fraction": fraction,
            "profile": config.default_profile,
            "staging_dir": str(buffer.root),
        }
        text = json.dumps(payload, sort_keys=True, allow_nan=False)
        validate_metadata_json(text)
        message = (
            f"generated {generated}/{count} episodes; success={counts['success']} "
            f"failure={counts['failure']} timeout={counts['timeout']} "
            f"aborted={counts['aborted']} perturbed={perturbed}; run_id={run_id}"
        )
        self.get_logger().info(message)
        return {
            "success": True,
            "num_generated": generated,
            "num_failed": num_failed,
            "message": message,
            "result_metadata_json": text,
        }

    @staticmethod
    def _failure(message: str) -> dict[str, Any]:
        return {
            "success": False,
            "num_generated": 0,
            "num_failed": 0,
            "message": message,
            "result_metadata_json": json.dumps({"error": message}, sort_keys=True, allow_nan=False),
        }

    # -- message conversion -------------------------------------------------
    def _to_message(self, record: Any) -> Any:
        fields = episode_to_rollout_fields(record)
        message = RolloutEpisode()
        message.episode_id = fields["episode_id"]
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
            step.joint_positions = [float(v) for v in item["joint_positions"]]
            step.joint_velocities = [float(v) for v in item["joint_velocities"]]
            pose = Pose()
            pose.position.x = float(item["ee_pose"]["position"][0])
            pose.position.y = float(item["ee_pose"]["position"][1])
            pose.position.z = float(item["ee_pose"]["position"][2])
            matrix = np.asarray(item["ee_pose"]["orientation_matrix"], dtype=np.float64)
            w, x, y, z = mat_to_quat(matrix)
            pose.orientation.w = float(w)
            pose.orientation.x = float(x)
            pose.orientation.y = float(y)
            pose.orientation.z = float(z)
            step.ee_pose = pose
            step.ee_frame_id = item["ee_frame_id"]
            step.action = [float(v) for v in item["action"]]
            step.action_space = int(item["action_space"])
            step.gripper_position = float(item["gripper_position"])
            step.unintended_collision = bool(item["unintended_collision"])
            steps.append(step)
        message.steps = steps
        return message

    def destroy_node(self) -> None:
        with self._backend_lock:
            if self._backend is not None:
                self._backend.close()
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ExpertCollectorNode()
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
