"""Safety watchdog node: the only consumer allowed to forward /vla/raw_action.

Responsibilities
----------------
* Validate every raw chunk against workspace, joint position/velocity/
  acceleration/jerk limits, continuity, and obstacle clearance.
* Smooth with an EMA initialised from the measured state, retime the final
  command, and only then publish to the JointTrajectoryController.
* Watch /joint_states and /workspace/pointcloud freshness.  On stale/missing
  inputs, enter UNSAFE and publish exactly one hold command built from the
  current measured positions (never zeros, never the last chunk).
* Provide a latched /estop service and an explicit /estop_reset.

A software e-stop is **not** a safety-rated emergency stop.  Mock hardware
tests in this repository do not establish physical safety.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
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
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import (  # noqa: E402
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import JointState as JointStateMsg  # noqa: E402
from sensor_msgs.msg import PointCloud2  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from std_srvs.srv import Trigger  # noqa: E402
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint  # noqa: E402

from sim2real import contracts  # noqa: E402
from sim2real.expert import parse_instruction  # noqa: E402
from sim2real.mapping.config import WorkspaceBounds  # noqa: E402
from sim2real.mapping.obstacle_query import (  # noqa: E402
    LOW_COVERAGE,
    NO_DATA,
    MapStatus,
    ObstacleQuery,
)
from sim2real.safety import (  # noqa: E402
    ESTOP,
    OK,
    UNSAFE,
    ChunkDecision,
    RawChunk,
    SafetyCommand,
    SafetyGate,
    load_safety_config,
    validate_chunk,
)
from sim2real.safety.types import (  # noqa: E402
    ESTOP_LATCHED,
    JOINT_STATE_UNAVAILABLE,
    MAP_INVALID,
    MAP_TOO_LARGE,
    TRANSFORM_UNAVAILABLE,
)
from sim2real.sim.config import load_sim_configs  # noqa: E402
from sim2real.sim.qos import (  # noqa: E402
    CONTROLLER_TRAJECTORY_QOS,
    JOINT_STATES_QOS,
    MAP_QOS,
    SAFETY_DIAGNOSTICS_QOS,
    SAFETY_STATUS_QOS,
    TASK_INSTRUCTION_QOS,
    VLA_RAW_ACTION_QOS,
)
from sim2real_msgs.msg import SafetyStatus  # noqa: E402


def _qos_profile(spec) -> QoSProfile:
    reliability = (
        ReliabilityPolicy.RELIABLE
        if spec.reliability == "reliable"
        else ReliabilityPolicy.BEST_EFFORT
    )
    durability = (
        DurabilityPolicy.TRANSIENT_LOCAL
        if spec.durability == "transient_local"
        else DurabilityPolicy.VOLATILE
    )
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=spec.depth,
        reliability=reliability,
        durability=durability,
    )


def _stamp_message(stamp_ns: int) -> Time:
    stamp_ns = max(0, int(stamp_ns))
    return Time(sec=stamp_ns // 1_000_000_000, nanosec=stamp_ns % 1_000_000_000)


def _duration_message(seconds: float) -> Duration:
    nanoseconds = max(0, int(round(float(seconds) * 1e9)))
    return Duration(sec=nanoseconds // 1_000_000_000, nanosec=nanoseconds % 1_000_000_000)


def _stamp_ns(message: JointStateMsg) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)


def decode_pointcloud_xyz(
    message: PointCloud2,
    *,
    max_points: int,
    max_bytes: int,
) -> np.ndarray:
    """Bounded XYZ extraction from a PointCloud2.

    Supports float32/float64 fields at arbitrary offsets, row padding, and
    height > 1.  Rejects oversized payloads before allocating a full array.
    """
    if message.width < 0 or message.height < 0 or message.point_step <= 0:
        raise ValueError("malformed PointCloud2 dimensions")
    count = int(message.width) * int(message.height)
    if count > max_points:
        raise ValueError(f"point count {count} > {max_points}")
    if len(message.data) > max_bytes:
        raise ValueError(f"point cloud bytes {len(message.data)} > {max_bytes}")
    fields = {field.name: field for field in message.fields}
    missing = {"x", "y", "z"} - set(fields)
    if missing:
        raise ValueError(f"missing fields {sorted(missing)}")
    formats = []
    offsets = []
    for name in ("x", "y", "z"):
        field = fields[name]
        if field.datatype == 7:  # FLOAT32
            formats.append("<f4")
        elif field.datatype == 8:  # FLOAT64
            formats.append("<f8")
        else:
            raise ValueError(f"{name}: unsupported datatype {field.datatype}")
        offsets.append(int(field.offset))
    if max(offsets) + 8 > message.point_step:
        raise ValueError("point_step is smaller than the XYZ fields")
    dtype = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": formats,
            "offsets": offsets,
            "itemsize": int(message.point_step),
        }
    )
    raw = np.frombuffer(bytes(message.data), dtype=np.uint8)
    row_step = int(message.row_step) or int(message.point_step) * int(message.width)
    if row_step * int(message.height) > raw.size:
        raise ValueError("PointCloud2 data is shorter than row_step * height")
    rows = []
    for row in range(int(message.height)):
        start = row * row_step
        stop = start + int(message.width) * int(message.point_step)
        rows.append(np.frombuffer(raw[start:stop].tobytes(), dtype=dtype, count=int(message.width)))
    if not rows:
        return np.zeros((0, 3), dtype=np.float64)
    structured = np.concatenate(rows)
    points = np.column_stack([structured["x"], structured["y"], structured["z"]]).astype(np.float64)
    finite = np.all(np.isfinite(points), axis=1)
    return points[finite]


class _MapWorker:
    """Build ObstacleQuery off the executor thread; keep only the latest cloud."""

    def __init__(self, node: SafetyWatchdogNode) -> None:
        self.node = node
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._pending: PointCloud2 | None = None
        self._thread = threading.Thread(target=self._run, name="safety-map-worker", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=5.0)

    def submit(self, message: PointCloud2) -> None:
        with self._lock:
            self._pending = message
        self._wake.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            if self._stop.is_set():
                break
            with self._lock:
                message = self._pending
                self._pending = None
            if message is None:
                continue
            try:
                self.node._build_map(message)
            except Exception as exc:  # never kill the worker on malformed input
                self.node._set_map_error(f"{type(exc).__name__}: {exc}")


class SafetyWatchdogNode(Node):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("safety_watchdog_node", **kwargs)
        self.declare_parameter("config_file", "")
        self.declare_parameter("panda_config", "")
        self.declare_parameter("sim_config_dir", "")
        self.declare_parameter("instructions_config", "")

        config_file = str(self.get_parameter("config_file").value or "")
        panda_config = str(self.get_parameter("panda_config").value or "")
        self.config, self.panda = load_safety_config(
            config_file or self._default_config_path(),
            panda_config_path=panda_config or self._default_panda_path(),
        )
        self._scene = load_sim_configs(
            str(self.get_parameter("sim_config_dir").value or self._default_sim_config_dir())
        ).scene
        self._instructions_path = str(
            self.get_parameter("instructions_config").value or self._default_instructions_path()
        )
        self._target_object_id: str | None = None
        self._target_center_m: np.ndarray | None = None
        self._target_half_extent_m: float | None = None
        self.gate = SafetyGate(self.config)
        self._lock = threading.RLock()
        self._monotonic = time.monotonic

        self._joint_positions: np.ndarray | None = None
        self._joint_stamp_ns: int | None = None
        self._joint_received_s: float | None = None
        self._joint_valid = False

        self._map_query: ObstacleQuery | None = None
        self._map_base_status: MapStatus | None = None
        self._map_received_s: float | None = None
        self._map_error: str | None = None

        self._last_chunk_received_s: float | None = None
        self._last_decision: ChunkDecision | None = None
        self._last_command_points = 0
        self._last_hold_published = False
        self._hold_failures = 0
        self._fresh_since_s: float | None = None
        self._estop_fresh_since_s: float | None = None
        self._previous_state = self.gate.state

        self._status_pub = self.create_publisher(
            SafetyStatus,
            contracts.SAFETY_STATUS_TOPIC,
            _qos_profile(SAFETY_STATUS_QOS),
        )
        self._diagnostics_pub = self.create_publisher(
            String,
            contracts.SAFETY_DIAGNOSTICS_TOPIC,
            _qos_profile(SAFETY_DIAGNOSTICS_QOS),
        )
        self._command_pub = self.create_publisher(
            JointTrajectory,
            contracts.CONTROLLER_TRAJECTORY_TOPIC,
            _qos_profile(CONTROLLER_TRAJECTORY_QOS),
        )

        self.create_subscription(
            JointStateMsg,
            contracts.JOINT_STATES_TOPIC,
            self._on_joint_state,
            _qos_profile(JOINT_STATES_QOS),
        )
        self.create_subscription(
            JointTrajectory,
            contracts.VLA_RAW_ACTION_TOPIC,
            self._on_raw_action,
            _qos_profile(VLA_RAW_ACTION_QOS),
        )
        self.create_subscription(
            String,
            contracts.TASK_INSTRUCTION_TOPIC,
            self._on_task_instruction,
            _qos_profile(TASK_INSTRUCTION_QOS),
        )
        self.create_subscription(
            PointCloud2,
            contracts.WORKSPACE_POINTCLOUD_TOPIC,
            self._on_pointcloud,
            _qos_profile(MAP_QOS),
        )
        self.create_service(Trigger, contracts.ESTOP_SERVICE, self._on_estop)
        self.create_service(Trigger, contracts.ESTOP_RESET_SERVICE, self._on_estop_reset)

        self._map_worker = _MapWorker(self)
        self._map_worker.start()
        self.create_timer(0.05, self._on_watchdog)
        self.create_timer(1.0 / self.config.status_rate_hz, self._publish_status)
        self.create_timer(1.0 / self.config.diagnostics_rate_hz, self._publish_diagnostics)
        self.get_logger().info(
            "safety_watchdog_node started: "
            f"policy={self.config.stale_chunk_policy}, "
            f"ema_alpha={self.config.ema_alpha}, "
            f"max_chunk_age={self.config.max_chunk_age_s} s, "
            f"map_required={self.config.map_required}"
        )

    # ------------------------------------------------------------------ paths
    def _default_config_path(self) -> str:
        return str(Path(__file__).resolve().parents[4] / "configs" / "safety" / "default.yaml")

    def _default_panda_path(self) -> str:
        return str(Path(__file__).resolve().parents[4] / "configs" / "sim" / "panda.yaml")

    @staticmethod
    def _repo_root() -> Path:
        if _bootstrap_root is not None:
            return _bootstrap_root
        return Path(__file__).resolve().parents[4]

    def _default_sim_config_dir(self) -> str:
        return str(self._repo_root() / "configs" / "sim")

    def _default_instructions_path(self) -> str:
        return str(self._repo_root() / "configs" / "expert" / "instructions.yaml")

    # ------------------------------------------------------------------ inputs
    def _on_joint_state(self, message: JointStateMsg) -> None:
        try:
            index = {name: i for i, name in enumerate(message.name)}
            if any(name not in index for name in contracts.CONTROLLED_JOINT_NAMES):
                raise ValueError("missing controlled joints")
            positions = np.array(
                [float(message.position[index[name]]) for name in contracts.CONTROLLED_JOINT_NAMES],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(positions)):
                raise ValueError("non-finite joint position")
            stamp_ns = _stamp_ns(message)
        except Exception as exc:
            with self._lock:
                self._joint_valid = False
            self.get_logger().warning(f"ignoring malformed /joint_states: {exc}")
            return
        with self._lock:
            self._joint_positions = positions
            self._joint_stamp_ns = stamp_ns
            self._joint_received_s = self._monotonic()
            self._joint_valid = True

    def _on_task_instruction(self, message: String) -> None:
        try:
            task = parse_instruction(message.data, self._scene, self._instructions_path)
            cube = self._scene.cubes.get(task.object_id)
            if cube is None:
                raise ValueError(f"unknown scene object {task.object_id!r}")
            with self._lock:
                self._target_object_id = task.object_id
                self._target_center_m = np.asarray(cube.position, dtype=np.float64)
                self._target_half_extent_m = float(cube.size) / 2.0
            self.get_logger().info(
                f"target object for obstacle allowance: {task.object_id} "
                f"centre={np.asarray(cube.position).tolist()}"
            )
        except Exception as exc:
            with self._lock:
                self._target_object_id = None
                self._target_center_m = None
                self._target_half_extent_m = None
            self.get_logger().warning(f"instruction does not resolve a target object: {exc}")

    def _on_pointcloud(self, message: PointCloud2) -> None:
        # The callback only hands the latest message to the worker.
        self._map_worker.submit(message)

    def _build_map(self, message: PointCloud2) -> None:
        if message.header.frame_id != self.config.safety_frame:
            self._set_map_error(
                f"{TRANSFORM_UNAVAILABLE}: frame {message.header.frame_id!r} != "
                f"{self.config.safety_frame!r}"
            )
            return
        try:
            points = decode_pointcloud_xyz(
                message,
                max_points=self.config.pointcloud_max_points,
                max_bytes=self.config.pointcloud_max_bytes,
            )
        except ValueError as exc:
            self._set_map_error(f"{MAP_TOO_LARGE if '> ' in str(exc) else MAP_INVALID}: {exc}")
            return
        if len(points) == 0:
            self._set_map_error(f"{MAP_INVALID}: no finite points")
            return
        # Expand the *query grid* by one cell so float32 surface points exactly
        # on the workspace boundary (notably the table top at z=0.42) are still
        # indexed. The end-effector workspace check uses the unexpanded bounds.
        margin = self.config.map_query_resolution_m
        bounds = WorkspaceBounds(
            tuple(float(v) - margin for v in self.config.workspace_min_m),
            tuple(float(v) + margin for v in self.config.workspace_max_m),
        )
        query = ObstacleQuery(
            points,
            bounds,
            self.config.map_query_resolution_m,
            self.config.map_max_query_distance_m,
        )
        coverage = query.occupied_cells / max(1, query.num_cells)
        reasons: list[str] = []
        if query.occupied_cells == 0:
            reasons.append(NO_DATA)
        if coverage < self.config.min_map_coverage_ratio:
            reasons.append(LOW_COVERAGE)
        status = MapStatus(
            valid=not reasons,
            age_s=0.0,
            num_integrations=1,
            coverage_ratio=coverage,
            reasons=tuple(reasons),
        )
        with self._lock:
            self._map_query = query
            self._map_base_status = status
            self._map_received_s = self._monotonic()
            self._map_error = None

    def _set_map_error(self, error: str) -> None:
        with self._lock:
            self._map_query = None
            self._map_base_status = None
            self._map_received_s = None
            self._map_error = error

    # ------------------------------------------------------------------ chunk
    def _on_raw_action(self, message: JointTrajectory) -> None:
        received_s = self._monotonic()
        try:
            positions = np.asarray(
                [list(point.positions) for point in message.points], dtype=np.float64
            )
            offsets = tuple(
                float(point.time_from_start.sec) + float(point.time_from_start.nanosec) / 1e9
                for point in message.points
            )
            chunk = RawChunk(
                stamp_ns=_stamp_ns(message),
                joint_names=tuple(message.joint_names),
                positions=positions,
                time_from_start_s=offsets,
                frame_id=str(message.header.frame_id),
            )
        except Exception as exc:
            with self._lock:
                self._last_chunk_received_s = received_s
                self._last_decision = ChunkDecision(
                    False, MAP_INVALID, None, detail=f"cannot decode chunk: {exc}"
                )
                self.gate.on_chunk(False, MAP_INVALID)
            return

        with self._lock:
            self._last_chunk_received_s = received_s
            state = self.gate.state
            estop = self.gate.estop_latched
            measured = None if self._joint_positions is None else self._joint_positions.copy()
            now_stamp_ns = self._joint_stamp_ns
            query = self._map_query
            map_status = self._map_status_locked()
            if estop:
                reason = ESTOP_LATCHED
            elif state == UNSAFE:
                reason = self.gate.failure_reason or "UNSAFE"
            elif state != OK:
                reason = JOINT_STATE_UNAVAILABLE
            else:
                reason = None

            if reason is not None or measured is None:
                decision = ChunkDecision(
                    False,
                    reason or JOINT_STATE_UNAVAILABLE,
                    None,
                    detail="not forwarded because the safety state is not OK",
                )
            else:
                decision = validate_chunk(
                    chunk,
                    measured,
                    self.config,
                    self.panda,
                    self._validation_context(now_stamp_ns, received_s, query, map_status),
                )
            self.gate.on_chunk(decision.accepted, decision.reason)
            self._last_decision = decision
            if decision.accepted and decision.command is not None and self.gate.state == OK:
                self._publish_command(decision.command)
                self._last_command_points = decision.command.positions.shape[0]

    def _validation_context(self, now_stamp_ns, received_s, query, map_status):
        from sim2real.safety import ValidationContext

        return ValidationContext(
            now_stamp_ns=now_stamp_ns,
            now_monotonic_s=self._monotonic(),
            received_monotonic_s=received_s,
            query=query,
            map_status=map_status,
            target_center_m=(
                None if self._target_center_m is None else self._target_center_m.copy()
            ),
            target_half_extent_m=self._target_half_extent_m,
        )

    # ------------------------------------------------------------------ output
    def _publish_command(self, command) -> None:
        message = JointTrajectory()
        message.header.stamp = Time(sec=0, nanosec=0)
        message.header.frame_id = ""
        message.joint_names = list(command.joint_names)
        for offset_s, positions in zip(command.time_from_start_s, command.positions, strict=True):
            point = JointTrajectoryPoint()
            point.positions = [float(value) for value in positions]
            point.time_from_start = _duration_message(offset_s)
            message.points.append(point)
        self._command_pub.publish(message)

    def _measured_is_valid(self) -> bool:
        with self._lock:
            positions = None if self._joint_positions is None else self._joint_positions
        if positions is None or positions.shape != (8,) or not np.all(np.isfinite(positions)):
            return False
        arm = positions[:7]
        if np.any(arm < self.config.joint_lower_rad) or np.any(arm > self.config.joint_upper_rad):
            return False
        return bool(self.config.finger_min_m <= positions[7] <= self.config.finger_max_m)

    def _publish_hold_locked(self) -> bool:
        if not self._measured_is_valid():
            self._hold_failures += 1
            return False
        assert self._joint_positions is not None
        command = SafetyCommand(
            joint_names=contracts.CONTROLLED_JOINT_NAMES,
            positions=self._joint_positions.reshape(1, 8).copy(),
            time_from_start_s=(self.config.hold_duration_s,),
            header_stamp_ns=0,
            retimed=False,
        )
        self._publish_command(command)
        self._last_hold_published = True
        return True

    # ------------------------------------------------------------------ watchdog
    def _ages(self) -> tuple[float | None, float | None]:
        now = self._monotonic()
        with self._lock:
            joint = None if self._joint_received_s is None else now - self._joint_received_s
            map_age = None if self._map_received_s is None else now - self._map_received_s
        return joint, map_age

    def _map_status_locked(self) -> MapStatus | None:
        if self._map_base_status is None or self._map_received_s is None:
            return None
        age = self._monotonic() - self._map_received_s
        return replace(self._map_base_status, age_s=age)

    def _inputs_fresh_locked(self) -> bool:
        joint_age, map_age = self._ages()
        joint_ok = (
            joint_age is not None
            and self._joint_valid
            and joint_age <= self.config.max_joint_state_age_s
        )
        if not self.config.map_required:
            return joint_ok
        return bool(
            joint_ok
            and self._map_query is not None
            and map_age is not None
            and map_age <= self.config.max_map_age_s
        )

    def _on_watchdog(self) -> None:
        joint_age, map_age = self._ages()
        with self._lock:
            map_valid = self._map_query is not None and self._map_base_status is not None
            changed, state, reason = self.gate.update_inputs(
                joint_state_age_s=joint_age,
                map_age_s=map_age,
                map_valid=map_valid,
            )
            now = self._monotonic()
            if self._inputs_fresh_locked():
                if self._fresh_since_s is None:
                    self._fresh_since_s = now
            else:
                self._fresh_since_s = None
            if self.gate.estop_latched:
                if self._inputs_fresh_locked():
                    if self._estop_fresh_since_s is None:
                        self._estop_fresh_since_s = now
                else:
                    self._estop_fresh_since_s = None
            else:
                self._estop_fresh_since_s = None

            previous = self._previous_state
            self._previous_state = state
            if state in (UNSAFE, ESTOP) and previous != state:
                published = self._publish_hold_locked()
                if published:
                    self.get_logger().error(
                        f"safety state {state} ({reason}); hold command published once"
                    )
                else:
                    self.get_logger().error(
                        f"safety state {state} ({reason}); hold NOT published because "
                        "the measured state is invalid"
                    )

    # ------------------------------------------------------------------ services
    def _on_estop(self, _request, response):
        with self._lock:
            changed = self.gate.trigger_estop()
            published = self._publish_hold_locked() if changed else False
            self._previous_state = self.gate.state
            self._estop_fresh_since_s = None
        response.success = True
        if changed:
            response.message = (
                "e-stop latched; hold command published"
                if published
                else "e-stop latched; hold NOT published (measured state invalid)"
            )
        else:
            response.message = "e-stop already latched"
        return response

    def _on_estop_reset(self, _request, response):
        with self._lock:
            if not self.gate.estop_latched:
                response.success = False
                response.message = "e-stop is not latched"
                return response
            if not self._inputs_fresh_locked():
                response.success = False
                response.message = "inputs are not fresh and valid; reset refused"
                return response
            if self._estop_fresh_since_s is None:
                response.success = False
                response.message = (
                    f"inputs have not been fresh for {self.config.estop_reset_hold_s} s; "
                    "reset refused"
                )
                return response
            held = self._monotonic() - self._estop_fresh_since_s
            if held < self.config.estop_reset_hold_s:
                response.success = False
                response.message = (
                    f"inputs fresh for {held:.2f} s < {self.config.estop_reset_hold_s} s; "
                    "reset refused"
                )
                return response
            ok, message = self.gate.reset(inputs_fresh=True)
            self._previous_state = self.gate.state
        response.success = ok
        response.message = message
        return response

    # ------------------------------------------------------------------ status
    def _status_state(self) -> int:
        if self.gate.state == OK:
            return contracts.STATE_NOMINAL
        if self.gate.state == ESTOP:
            return contracts.STATE_ESTOP
        if self.gate.state == UNSAFE:
            return contracts.STATE_SAFE_STOP
        return contracts.STATE_DEGRADED

    def _publish_status(self) -> None:
        with self._lock:
            joint_age, _map_age = self._ages()
            now = self._monotonic()
            action_age = (
                None if self._last_chunk_received_s is None else now - self._last_chunk_received_s
            )
            decision = self._last_decision
            status = self._map_status_locked()
            message = SafetyStatus()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = contracts.WORLD_FRAME
            message.is_safe = self.gate.state == OK and not self.gate.estop_latched
            message.estop_latched = self.gate.estop_latched
            message.state = self._status_state()
            message.failure_reason = self.gate.failure_reason or ""
            message.obstacle_map_valid = bool(
                status is not None and status.valid and status.age_s <= self.config.max_map_age_s
            )
            message.min_obstacle_distance_m = (
                float("nan")
                if decision is None or not math.isfinite(decision.min_obstacle_distance_m)
                else float(decision.min_obstacle_distance_m)
            )
            ratios = decision.ratios if decision is not None else None
            message.max_joint_limit_ratio = (
                float("nan") if ratios is None else float(ratios.position)
            )
            message.max_velocity_ratio = float("nan") if ratios is None else float(ratios.velocity)
            message.max_acceleration_ratio = (
                float("nan") if ratios is None else float(ratios.acceleration)
            )
            message.max_jerk_ratio = float("nan") if ratios is None else float(ratios.jerk)
            message.observation_age_s = float("nan") if joint_age is None else float(joint_age)
            message.action_age_s = float("nan") if action_age is None else float(action_age)
        self._status_pub.publish(message)

    def _publish_diagnostics(self) -> None:
        with self._lock:
            decision = self._last_decision
            status = self._map_status_locked()
            start, _map_age = self._ages()
            payload: dict[str, Any] = {
                "gate": self.gate.as_dict(),
                "joint_state_age_s": None if start is None else float(start),
                "map": None
                if status is None
                else {
                    "valid": status.valid,
                    "age_s": status.age_s,
                    "coverage_ratio": status.coverage_ratio,
                    "reasons": list(status.reasons),
                    "error": self._map_error,
                },
                "last_chunk": None if decision is None else decision.as_dict(),
                "last_command_points": self._last_command_points,
                "last_hold_published": self._last_hold_published,
                "hold_failures": self._hold_failures,
                "config": {
                    "stale_chunk_policy": self.config.stale_chunk_policy,
                    "ema_alpha": self.config.ema_alpha,
                    "max_chunk_age_s": self.config.max_chunk_age_s,
                    "max_joint_state_age_s": self.config.max_joint_state_age_s,
                    "max_map_age_s": self.config.max_map_age_s,
                    "command_period_s": self.config.command_period_s,
                    "command_time_convention": self.config.command_time_convention,
                    "obstacle_mode": self.config.obstacle_mode,
                    "table_mask_band_m": self.config.table_mask_band_m,
                    "target_object_id": self._target_object_id,
                },
            }
        message = String()
        message.data = json.dumps(payload, sort_keys=True)
        self._diagnostics_pub.publish(message)

    def destroy_node(self) -> bool:
        self._map_worker.stop()
        return super().destroy_node()


def main(args: list[str] | None = None) -> int:
    rclpy.init(args=args)
    node = None
    try:
        node = SafetyWatchdogNode()
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


__all__ = ["SafetyWatchdogNode", "decode_pointcloud_xyz", "main"]
