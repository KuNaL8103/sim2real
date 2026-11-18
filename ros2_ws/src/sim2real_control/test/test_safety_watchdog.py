"""ROS-level tests for the safety watchdog (fake inputs, no torch, no GPU).

Unit tests live in ``tests/test_safety_core.py``; this file covers the rclpy
adapter: real publishers/subscribers/services, status fields, and the
single-owner launch guard.  Skipped with an explicit reason when ROS is
unavailable; a skip is never a pass.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

import numpy as np
import pytest
import yaml

# Isolate this package's DDS graph from parallel colcon package tests.
os.environ.setdefault("ROS_DOMAIN_ID", "78")
os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    import rclpy
    from builtin_interfaces.msg import Duration, Time
    from controller_manager_msgs.srv import ListControllers
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState as JointStateMsg
    from sensor_msgs.msg import PointCloud2, PointField
    from std_srvs.srv import Trigger
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    from sim2real import contracts
    from sim2real.ros_utils.pointcloud import build_pointcloud2
    from sim2real.sim.qos import (
        CONTROLLER_TRAJECTORY_QOS,
        JOINT_STATES_QOS,
        MAP_QOS,
        SAFETY_DIAGNOSTICS_QOS,
        SAFETY_STATUS_QOS,
        VLA_RAW_ACTION_QOS,
        QosSpec,
    )
    from sim2real_control.safety_watchdog_node import SafetyWatchdogNode
    from sim2real_control.single_owner import JointStateOwnerError, check_single_owner
    from sim2real_msgs.msg import SafetyStatus

    IMPORT_ERROR: str | None = None
except ImportError as exc:  # pragma: no cover - depends on the environment
    IMPORT_ERROR = f"ROS prerequisites missing: {exc}"

pytestmark = pytest.mark.ros


def _profile(spec: QosSpec) -> QoSProfile:
    reliability = {
        "reliable": ReliabilityPolicy.RELIABLE,
        "best_effort": ReliabilityPolicy.BEST_EFFORT,
    }[spec.reliability]
    durability = {
        "volatile": DurabilityPolicy.VOLATILE,
        "transient_local": DurabilityPolicy.TRANSIENT_LOCAL,
    }[spec.durability]
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=spec.depth,
        reliability=reliability,
        durability=durability,
    )


def _stamp(stamp_ns: int) -> Time:
    return Time(sec=int(stamp_ns // 1_000_000_000), nanosec=int(stamp_ns % 1_000_000_000))


def _home() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 0.04])


def _joint_state(stamp_ns: int) -> JointStateMsg:
    message = JointStateMsg()
    message.header.stamp = _stamp(stamp_ns)
    message.name = list(contracts.CONTROLLED_JOINT_NAMES)
    message.position = [float(value) for value in _home()]
    return message


def _chunk(
    stamp_ns: int,
    positions: np.ndarray,
    offsets: tuple[float, ...] | None = None,
) -> JointTrajectory:
    message = JointTrajectory()
    message.header.stamp = _stamp(stamp_ns)
    message.header.frame_id = ""
    message.joint_names = list(contracts.CONTROLLED_JOINT_NAMES)
    if offsets is None:
        offsets = tuple(0.1 * index for index in range(len(positions)))
    for offset, row in zip(offsets, positions, strict=True):
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in row]
        seconds = int(round(offset * 1e9))
        point.time_from_start = Duration(
            sec=seconds // 1_000_000_000, nanosec=seconds % 1_000_000_000
        )
        message.points.append(point)
    return message


def _pointcloud(points: np.ndarray, stamp_ns: int, frame_id: str = "world") -> PointCloud2:
    data = build_pointcloud2(points, frame_id=frame_id, stamp_ns=stamp_ns)
    message = PointCloud2()
    message.header.stamp = _stamp(stamp_ns)
    message.header.frame_id = data.frame_id
    message.height = data.height
    message.width = data.width
    message.point_step = data.point_step
    message.row_step = data.row_step
    message.is_dense = data.is_dense
    message.is_bigendian = data.is_bigendian
    message.fields = [
        PointField(name=field.name, offset=field.offset, datatype=field.datatype, count=field.count)
        for field in data.fields
    ]
    message.data = data.data
    return message


def _table_points() -> np.ndarray:
    xs = np.arange(0.26, 0.65, 0.02)
    ys = np.arange(-0.28, 0.29, 0.02)
    grid = np.array([[x, y, 0.43] for x in xs for y in ys], dtype=np.float64)
    return grid


def _write_config(tmp_path: Path, **overrides) -> Path:
    raw = yaml.safe_load((REPO_ROOT / "configs" / "safety" / "default.yaml").read_text())
    raw["safety"].update(overrides)
    path = tmp_path / "safety.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def _spin_until(executor, predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.05)
        if predicate():
            return True
    return False


@unittest.skipIf(IMPORT_ERROR is not None, IMPORT_ERROR or "")
class TestSafetyWatchdogNode(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path("/tmp/sim2real") / f"safety_{self._testMethodName}_{os.getpid()}"
        self.tmp.mkdir(parents=True, exist_ok=True)
        # Keep the in-process tests robust against Python scheduling while
        # exercising the same code path; the shipped defaults are covered by
        # tests/test_safety_core.py.
        self.config_path = _write_config(
            self.tmp,
            estop_reset_hold_s=0.2,
            max_joint_state_age_s=1.0,
            max_map_age_s=5.0,
            watchdog_timeout_s=5.0,
        )
        rclpy.init()
        self.node = SafetyWatchdogNode(
            parameter_overrides=[
                Parameter("config_file", value=str(self.config_path)),
                Parameter("panda_config", value=str(REPO_ROOT / "configs" / "sim" / "panda.yaml")),
                Parameter("sim_config_dir", value=str(REPO_ROOT / "configs" / "sim")),
                Parameter(
                    "instructions_config",
                    value=str(REPO_ROOT / "configs" / "expert" / "instructions.yaml"),
                ),
            ]
        )
        self.helper = Node("safety_test_helper")
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.executor.add_node(self.helper)
        self.commands: list[JointTrajectory] = []
        self.statuses: list[SafetyStatus] = []
        self.joint_pub = self.helper.create_publisher(
            JointStateMsg, contracts.JOINT_STATES_TOPIC, _profile(JOINT_STATES_QOS)
        )
        self.chunk_pub = self.helper.create_publisher(
            JointTrajectory, contracts.VLA_RAW_ACTION_TOPIC, _profile(VLA_RAW_ACTION_QOS)
        )
        self.map_pub = self.helper.create_publisher(
            PointCloud2, contracts.WORKSPACE_POINTCLOUD_TOPIC, _profile(MAP_QOS)
        )
        self.helper.create_subscription(
            JointTrajectory,
            contracts.CONTROLLER_TRAJECTORY_TOPIC,
            self.commands.append,
            _profile(CONTROLLER_TRAJECTORY_QOS),
        )
        self.helper.create_subscription(
            SafetyStatus,
            contracts.SAFETY_STATUS_TOPIC,
            self.statuses.append,
            _profile(SAFETY_STATUS_QOS),
        )
        self.estop_client = self.helper.create_client(Trigger, contracts.ESTOP_SERVICE)
        self.reset_client = self.helper.create_client(Trigger, contracts.ESTOP_RESET_SERVICE)
        time.sleep(0.25)
        self.executor.spin_once(timeout_sec=0.05)
        # Wait for DDS discovery before asserting on forwarded commands.
        assert _spin_until(
            self.executor,
            lambda: (
                self.node._command_pub.get_subscription_count() > 0
                and self.joint_pub.get_subscription_count() > 0
                and self.chunk_pub.get_subscription_count() > 0
                and self.map_pub.get_subscription_count() > 0
            ),
            timeout=10.0,
        )

    def tearDown(self) -> None:
        self.executor.shutdown()
        self.helper.destroy_node()
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def _publish_map(self, stamp_ns: int) -> None:
        self.map_pub.publish(_pointcloud(_table_points(), stamp_ns))
        assert _spin_until(self.executor, lambda: self.node._map_query is not None, timeout=5.0)

    def _publish_joint(self, stamp_ns: int) -> None:
        self.joint_pub.publish(_joint_state(stamp_ns))
        assert _spin_until(
            self.executor, lambda: self.node._joint_positions is not None, timeout=5.0
        )

    def _setup_ok(self) -> int:
        """Publish a fresh map, then a fresh joint state, and wait for OK."""
        map_stamp = time.time_ns()
        self._publish_map(map_stamp)
        joint_stamp = time.time_ns()
        self._publish_joint(joint_stamp)
        assert _spin_until(self.executor, lambda: self.node.gate.state == "OK", timeout=5.0)
        return joint_stamp

    def _latest_status(self, predicate, timeout: float = 5.0) -> SafetyStatus | None:
        _spin_until(
            self.executor, lambda: bool(self.statuses) and predicate(self.statuses[-1]), timeout
        )
        return self.statuses[-1] if self.statuses else None

    def _call(self, client, timeout: float = 3.0):
        assert client.wait_for_service(timeout_sec=timeout)
        future = client.call_async(Trigger.Request())
        assert _spin_until(self.executor, future.done, timeout=timeout)
        return future.result()

    def test_contract_topics_types_and_qos(self) -> None:
        assert self.node._command_pub.topic_name == contracts.CONTROLLER_TRAJECTORY_TOPIC
        assert self.node._status_pub.topic_name == contracts.SAFETY_STATUS_TOPIC
        assert self.node._diagnostics_pub.topic_name == contracts.SAFETY_DIAGNOSTICS_TOPIC
        assert self.node._command_pub.msg_type is JointTrajectory
        assert self.node._status_pub.msg_type is SafetyStatus
        for node, expected in (
            (self.node._command_pub, CONTROLLER_TRAJECTORY_QOS),
            (self.node._status_pub, SAFETY_STATUS_QOS),
            (self.node._diagnostics_pub, SAFETY_DIAGNOSTICS_QOS),
        ):
            assert node.qos_profile.depth == expected.depth
            assert node.qos_profile.reliability == ReliabilityPolicy.RELIABLE
        publishers = dict(
            self.node.get_publisher_names_and_types_by_node(
                self.node.get_name(), self.node.get_namespace()
            )
        )
        assert contracts.CONTROLLER_TRAJECTORY_TOPIC in publishers
        assert contracts.SAFETY_STATUS_TOPIC in publishers
        assert contracts.SAFETY_DIAGNOSTICS_TOPIC in publishers
        services = dict(
            self.node.get_service_names_and_types_by_node(
                self.node.get_name(), self.node.get_namespace()
            )
        )
        assert contracts.ESTOP_SERVICE in services
        assert contracts.ESTOP_RESET_SERVICE in services

    def test_valid_chunk_is_forwarded_and_status_is_filled(self) -> None:
        stamp = self._setup_ok()
        before = len(self.commands)
        self.chunk_pub.publish(_chunk(stamp, np.tile(_home(), (3, 1))))
        ok = _spin_until(
            self.executor,
            lambda: any(len(message.points) == 2 for message in self.commands[before:]),
            timeout=5.0,
        )
        assert ok
        forwarded = next(message for message in self.commands[before:] if len(message.points) == 2)
        assert forwarded.header.stamp.sec == 0
        assert forwarded.header.stamp.nanosec == 0
        assert forwarded.joint_names == list(contracts.CONTROLLED_JOINT_NAMES)
        assert [
            p.time_from_start.sec + p.time_from_start.nanosec / 1e9 for p in forwarded.points
        ] == (pytest.approx([0.1, 0.2]))
        assert np.allclose(forwarded.points[0].positions, _home())

        status = self._latest_status(
            lambda message: message.is_safe and np.isfinite(message.action_age_s), timeout=5.0
        )
        assert status is not None
        assert status.state == SafetyStatus.STATE_NOMINAL
        assert status.obstacle_map_valid
        assert status.max_joint_limit_ratio >= 0.0
        assert np.isfinite(status.max_velocity_ratio)
        assert np.isfinite(status.min_obstacle_distance_m)
        assert np.isfinite(status.observation_age_s)
        assert np.isfinite(status.action_age_s)

    def test_invalid_chunk_and_missing_map_publish_no_trajectory(self) -> None:
        stamp = time.time_ns()
        self._publish_joint(stamp)
        self.chunk_pub.publish(_chunk(stamp, np.tile(_home(), (3, 1))))
        time.sleep(0.5)
        self.executor.spin_once(timeout_sec=0.05)
        assert not [message for message in self.commands if len(message.points) > 1]

        stamp = self._setup_ok()
        before = len(self.commands)
        positions = np.tile(_home(), (3, 1))
        positions[1, 2] = np.nan
        self.chunk_pub.publish(_chunk(stamp, positions))
        time.sleep(0.5)
        self.executor.spin_once(timeout_sec=0.05)
        assert not [message for message in self.commands[before:] if len(message.points) > 1]

    def test_watchdog_publishes_one_hold_and_recovers(self) -> None:
        self._setup_ok()
        before = len(self.commands)
        assert _spin_until(
            self.executor,
            lambda: bool(self.statuses) and self.statuses[-1].state == SafetyStatus.STATE_SAFE_STOP,
            timeout=3.0,
        )
        holds = [message for message in self.commands[before:] if len(message.points) == 1]
        assert len(holds) == 1
        assert not [message for message in self.commands[before:] if len(message.points) > 1]

        self._publish_joint(time.time_ns())
        assert _spin_until(self.executor, lambda: self.node.gate.state == "OK", timeout=3.0)
        after = len(self.commands)
        time.sleep(0.5)
        self.executor.spin_once(timeout_sec=0.05)
        assert len(self.commands) == after

    def test_estop_latches_blocks_and_reset_requires_hold(self) -> None:
        self._setup_ok()

        first = self._call(self.estop_client)
        assert first.success
        assert self.node.gate.estop_latched
        status = self._latest_status(lambda message: message.estop_latched, timeout=3.0)
        assert status is not None and status.state == SafetyStatus.STATE_ESTOP
        second = self._call(self.estop_client)
        assert second.success and "already" in second.message

        before = len(self.commands)
        self.chunk_pub.publish(_chunk(time.time_ns(), np.tile(_home(), (3, 1))))
        time.sleep(0.5)
        self.executor.spin_once(timeout_sec=0.05)
        assert not [message for message in self.commands[before:] if len(message.points) > 1]

        refused = self._call(self.reset_client)
        assert not refused.success

        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            self._publish_joint(time.time_ns())
            self.executor.spin_once(timeout_sec=0.05)
        accepted = self._call(self.reset_client)
        assert accepted.success, accepted.message
        assert not self.node.gate.estop_latched

        before = len(self.commands)
        stamp = time.time_ns()
        self._publish_joint(stamp)
        assert _spin_until(self.executor, lambda: self.node.gate.state == "OK", timeout=3.0)
        self.chunk_pub.publish(_chunk(stamp, np.tile(_home(), (3, 1))))
        assert _spin_until(
            self.executor,
            lambda: any(len(message.points) > 1 for message in self.commands[before:]),
            timeout=5.0,
        )


@unittest.skipIf(IMPORT_ERROR is not None, IMPORT_ERROR or "")
class TestSingleOwnerAndTorchIsolation(unittest.TestCase):
    def test_single_owner_guard_and_launch_arguments(self) -> None:
        with pytest.raises(JointStateOwnerError):
            check_single_owner(sim_node=True, ros2_control=True)
        check_single_owner(sim_node=True, ros2_control=False)
        sim_launch = (
            REPO_ROOT / "ros2_ws" / "src" / "sim2real_sim" / "launch" / "sim.launch.py"
        ).read_text()
        control_launch = (
            REPO_ROOT / "ros2_ws" / "src" / "sim2real_control" / "launch" / "control_mock.launch.py"
        ).read_text()
        assert "start_ros2_control" in sim_launch
        assert "start_sim" in control_launch
        assert "check_single_owner" in control_launch

    def test_node_and_core_import_without_torch(self) -> None:
        script = (
            "import sys; "
            f"sys.path.insert(0, {str(REPO_ROOT / 'src')!r}); "
            "import sim2real.safety; "
            "import sim2real_control.safety_watchdog_node; "
            "assert 'torch' not in sys.modules, 'torch was imported'; print('ok')"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(REPO_ROOT),
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "ok"


@unittest.skipIf(IMPORT_ERROR is not None, IMPORT_ERROR or "")
class TestHilIntegration(unittest.TestCase):
    """Mock ros2_control + safety node + fake map/chunk.

    This is integration evidence for the plumbing only.  mock_components does
    not model dynamics, latency, torque, velocity enforcement, or collisions;
    it says nothing about physical safety.
    """

    @classmethod
    def setUpClass(cls) -> None:
        import shutil

        if shutil.which("ros2") is None:
            raise unittest.SkipTest("ros2 CLI is not available")
        probe = subprocess.run(
            ["ros2", "pkg", "prefix", "controller_manager"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if probe.returncode != 0:
            raise unittest.SkipTest("controller_manager is not installed")
        cls.launch_log = Path("/tmp/sim2real") / f"hil_{os.getpid()}.log"
        cls.launch_log.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["ROS_DOMAIN_ID"] = "79"
        env["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
        cls.environment = env
        cls.process = subprocess.Popen(
            ["ros2", "launch", "sim2real_control", "control_mock.launch.py"],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=cls.launch_log.open("w"),
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        import signal

        process = getattr(cls, "process", None)
        if process is not None and process.poll() is None:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait(timeout=5)

    def setUp(self) -> None:
        os.environ["ROS_DOMAIN_ID"] = "79"
        os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
        rclpy.init()
        self.node = Node("hil_safety_probe")
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.state = {"joint": None, "status": None}
        self.commands: list[JointTrajectory] = []
        self.node.create_subscription(
            JointStateMsg,
            contracts.JOINT_STATES_TOPIC,
            lambda message: self.state.__setitem__("joint", message),
            _profile(JOINT_STATES_QOS),
        )
        self.node.create_subscription(
            SafetyStatus,
            contracts.SAFETY_STATUS_TOPIC,
            lambda message: self.state.__setitem__("status", message),
            _profile(SAFETY_STATUS_QOS),
        )
        self.node.create_subscription(
            JointTrajectory,
            contracts.CONTROLLER_TRAJECTORY_TOPIC,
            self.commands.append,
            _profile(CONTROLLER_TRAJECTORY_QOS),
        )
        self.map_pub = self.node.create_publisher(
            PointCloud2, contracts.WORKSPACE_POINTCLOUD_TOPIC, _profile(MAP_QOS)
        )
        self.chunk_pub = self.node.create_publisher(
            JointTrajectory, contracts.VLA_RAW_ACTION_TOPIC, _profile(VLA_RAW_ACTION_QOS)
        )
        self.estop = self.node.create_client(Trigger, contracts.ESTOP_SERVICE)
        self.reset = self.node.create_client(Trigger, contracts.ESTOP_RESET_SERVICE)

    def tearDown(self) -> None:
        self.executor.shutdown()
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        os.environ["ROS_DOMAIN_ID"] = "78"

    def _spin(self, predicate, timeout=10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.map_pub.publish(_pointcloud(_table_points(), time.time_ns()))
            self.executor.spin_once(timeout_sec=0.05)
            if predicate():
                return True
        return False

    def _wait_controller_active(self, name: str, timeout: float = 20.0) -> bool:
        client = self.node.create_client(ListControllers, "/controller_manager/list_controllers")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.map_pub.publish(_pointcloud(_table_points(), time.time_ns()))
            self.executor.spin_once(timeout_sec=0.05)
            if not client.service_is_ready():
                continue
            future = client.call_async(ListControllers.Request())
            call_deadline = time.monotonic() + 2.0
            while not future.done() and time.monotonic() < call_deadline:
                self.executor.spin_once(timeout_sec=0.05)
            if future.done() and future.result() is not None:
                states = {c.name: c.state for c in future.result().controller}
                if states.get(name) == "active":
                    return True
        return False

    def _call(self, client):
        assert client.wait_for_service(timeout_sec=5.0)
        future = client.call_async(Trigger.Request())
        assert self._spin(future.done, timeout=5.0)
        return future.result()

    def _chunk_from_measured(self, *, stamp_ns=None, delta=0.05, offsets=(0.0, 0.1, 0.2, 0.3)):
        joint = self.state["joint"]
        assert joint is not None
        by_name = dict(zip(joint.name, joint.position, strict=True))
        positions = np.array(
            [by_name[name] for name in contracts.CONTROLLED_JOINT_NAMES], dtype=np.float64
        )
        positions[0] += delta
        if stamp_ns is None:
            stamp_ns = int(joint.header.stamp.sec) * 10**9 + joint.header.stamp.nanosec
        return _chunk(stamp_ns, np.tile(positions, (len(offsets), 1)), offsets)

    def test_mock_stack_follows_commands_and_stops_on_estop_and_stale(self) -> None:
        assert self._spin(lambda: self.state["joint"] is not None, timeout=20.0), "no joint states"
        assert self._spin(
            lambda: self.state["status"] is not None and self.state["status"].is_safe,
            timeout=20.0,
        ), "safety node never became safe"
        assert self._wait_controller_active("joint_trajectory_controller", timeout=20.0)
        assert self.state["status"].obstacle_map_valid

        before = len(self.commands)
        self.chunk_pub.publish(self._chunk_from_measured())
        assert self._spin(lambda: len(self.commands) > before, timeout=5.0), "no forwarded command"
        forwarded = self.commands[-1]
        assert len(forwarded.points) == 3
        assert forwarded.header.stamp.sec == 0 and forwarded.header.stamp.nanosec == 0
        target = forwarded.points[-1].positions[0]
        assert self._spin(
            lambda: (
                abs(
                    dict(
                        zip(
                            self.state["joint"].name,
                            self.state["joint"].position,
                            strict=False,
                        )
                    )["panda_joint1"]
                    - target
                )
                < 0.01
            ),
            timeout=5.0,
        ), "mock hardware did not follow the command"

        estop = self._call(self.estop)
        assert estop.success
        assert self._spin(
            lambda: self.state["status"] is not None and self.state["status"].estop_latched,
            timeout=5.0,
        )
        before = len(self.commands)
        self.chunk_pub.publish(self._chunk_from_measured())
        self._spin(lambda: False, timeout=0.6)
        assert len(self.commands) == before, "a command was forwarded after e-stop"

        refused = self._call(self.reset)
        assert not refused.success
        self._spin(lambda: False, timeout=2.3)
        accepted = self._call(self.reset)
        assert accepted.success, accepted.message
        assert self._spin(
            lambda: self.state["status"] is not None and self.state["status"].is_safe,
            timeout=5.0,
        )

        joint = self.state["joint"]
        old_stamp = int(joint.header.stamp.sec) * 10**9 + joint.header.stamp.nanosec - 600_000_000
        before = len(self.commands)
        self.chunk_pub.publish(self._chunk_from_measured(stamp_ns=old_stamp))
        self._spin(lambda: False, timeout=0.8)
        # A one-point hold is an allowed safety response; a multi-point chunk
        # would mean the stale chunk itself was forwarded.
        assert not [message for message in self.commands[before:] if len(message.points) > 1], (
            "a stale chunk was forwarded"
        )


if __name__ == "__main__":
    unittest.main()
