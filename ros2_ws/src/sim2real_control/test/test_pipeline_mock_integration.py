"""Integrated ROS pipeline tests: fake observations -> inference -> safety.

The graph is real (rclpy nodes, DDS, QoS, services); the simulator, the policy,
and the observations are test doubles.  ``mock_components``/``GenericSystem`` is
not involved here: the helper publishes ``/joint_states`` at 20 Hz so the safety
node has measured state.  In ``pipeline_mock_hw.launch.py`` the ros2_control
broadcaster is the single owner instead.

A skip due to missing ROS prerequisites is never a pass.
"""

from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path

import numpy as np
import pytest
import yaml

os.environ.setdefault("ROS_DOMAIN_ID", "82")
os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    import rclpy
    from builtin_interfaces.msg import Duration, Time
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState as JointStateMsg
    from sensor_msgs.msg import PointCloud2, PointField
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    from sim2real import contracts
    from sim2real.ros_utils.pointcloud import build_pointcloud2
    from sim2real.safety import load_safety_config
    from sim2real.sim.config import load_panda_config
    from sim2real.sim.kinematics import tcp_pose
    from sim2real.sim.qos import (
        CONTROLLER_TRAJECTORY_QOS,
        JOINT_STATES_QOS,
        MAP_QOS,
        SAFETY_STATUS_QOS,
        VLA_RAW_ACTION_QOS,
        QosSpec,
    )
    from sim2real_control.fake_observation_publisher import FakeObservationPublisher
    from sim2real_control.safety_watchdog_node import SafetyWatchdogNode
    from sim2real_control.single_owner import JointStateOwnerError, check_single_owner
    from sim2real_msgs.msg import SafetyStatus
    from sim2real_vla.vla_inference_node import VlaInferenceNode

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


def _chunk(stamp_ns: int, positions: np.ndarray) -> JointTrajectory:
    message = JointTrajectory()
    message.header.stamp = _stamp(stamp_ns)
    message.joint_names = list(contracts.CONTROLLED_JOINT_NAMES)
    for index, row in enumerate(positions):
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in row]
        ns = int(round(0.1 * index * 1e9))
        point.time_from_start = Duration(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)
        message.points.append(point)
    return message


def _pointcloud(points: np.ndarray, stamp_ns: int) -> PointCloud2:
    data = build_pointcloud2(points, frame_id=contracts.WORLD_FRAME, stamp_ns=stamp_ns)
    message = PointCloud2()
    message.header.stamp = _stamp(data.stamp_ns)
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
    return np.array([[x, y, 0.43] for x in xs for y in ys], dtype=np.float64)


def _block_points(center: np.ndarray, half: float = 0.01) -> np.ndarray:
    axis = np.linspace(-half, half, 3)
    points = []
    x, y, z = center
    for u in axis:
        for v in axis:
            points.extend(
                [
                    [x + u, y + v, z + half],
                    [x + u, y + v, z - half],
                    [x + u, y + half, z + v],
                    [x + u, y - half, z + v],
                    [x + half, y + u, z + v],
                    [x - half, y + u, z + v],
                ]
            )
    return np.asarray(points, dtype=np.float64)


def _write_safety_config(tmp_path: Path, source: Path) -> Path:
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["safety"].update(
        {
            "max_joint_state_age_s": 1.0,
            "max_map_age_s": 2.0,
            "watchdog_timeout_s": 5.0,
            "estop_reset_hold_s": 0.2,
        }
    )
    path = tmp_path / source.name
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def _spin_until(executor, predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.02)
        if predicate():
            return True
    return False


@unittest.skipIf(IMPORT_ERROR is not None, IMPORT_ERROR or "")
class TestPipelineGraph(unittest.TestCase):
    """fake observations + fake inference + safety, in-process."""

    def setUp(self) -> None:
        self.tmp = Path("/tmp/opencode") / f"pipeline_graph_{os.getpid()}"
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.safety_config = _write_safety_config(
            self.tmp, REPO_ROOT / "configs" / "safety" / "sim_mock.yaml"
        )
        rclpy.init()
        self.safety = SafetyWatchdogNode(
            parameter_overrides=[
                Parameter("config_file", value=str(self.safety_config)),
                Parameter("panda_config", value=str(REPO_ROOT / "configs/sim/panda.yaml")),
                Parameter("sim_config_dir", value=str(REPO_ROOT / "configs/sim")),
                Parameter(
                    "instructions_config",
                    value=str(REPO_ROOT / "configs/expert/instructions.yaml"),
                ),
            ]
        )
        self.vla = VlaInferenceNode(
            parameter_overrides=[
                Parameter("config_file", value=str(REPO_ROOT / "configs/inference/default.yaml")),
                Parameter("policy_backend", value="fake"),
                Parameter("latency_compensation", value=False),
            ]
        )
        self.observations = FakeObservationPublisher()
        self.helper = Node("pipeline_graph_helper")
        self.executor = MultiThreadedExecutor(num_threads=4)
        for node in (self.safety, self.vla, self.observations, self.helper):
            self.executor.add_node(node)

        self.commands: list[JointTrajectory] = []
        self.statuses: list[SafetyStatus] = []
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
        self.joint_pub = self.helper.create_publisher(
            JointStateMsg, contracts.JOINT_STATES_TOPIC, _profile(JOINT_STATES_QOS)
        )
        self.chunk_pub = self.helper.create_publisher(
            JointTrajectory, contracts.VLA_RAW_ACTION_TOPIC, _profile(VLA_RAW_ACTION_QOS)
        )
        self.helper.create_timer(0.05, lambda: self.joint_pub.publish(_joint_state(time.time_ns())))
        assert _spin_until(
            self.executor,
            lambda: (
                self.joint_pub.get_subscription_count() > 0
                and self.chunk_pub.get_subscription_count() > 0
            ),
            timeout=15.0,
        )

    def tearDown(self) -> None:
        self.executor.shutdown()
        for node in (self.observations, self.vla, self.safety, self.helper):
            if node is not None:
                node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def test_full_graph_forwards_a_validated_chunk_and_safety_is_the_only_publisher(self) -> None:
        assert _spin_until(self.executor, lambda: self.safety.gate.state == "OK", timeout=20.0), (
            "safety never reached OK"
        )
        assert _spin_until(
            self.executor,
            lambda: any(len(message.points) > 1 for message in self.commands),
            timeout=25.0,
        ), "no multi-point command was forwarded"
        forwarded = next(message for message in self.commands if len(message.points) > 1)
        assert forwarded.header.stamp.sec == 0 and forwarded.header.stamp.nanosec == 0
        assert forwarded.joint_names == list(contracts.CONTROLLED_JOINT_NAMES)
        assert forwarded.points[0].time_from_start.sec == 0
        first_offset = (
            forwarded.points[0].time_from_start.sec
            + forwarded.points[0].time_from_start.nanosec / 1e9
        )
        assert first_offset == pytest.approx(0.1, abs=0.01)

        publishers = self.helper.get_publishers_info_by_topic(contracts.CONTROLLER_TRAJECTORY_TOPIC)
        names = sorted(info.node_name for info in publishers)
        assert names == ["safety_watchdog_node"], names

    def test_invalid_chunk_and_missing_map_produce_no_command(self) -> None:
        assert _spin_until(self.executor, lambda: self.safety.gate.state == "OK", timeout=20.0)
        # Remove the fake inference and observation sources so only the helper
        # publishes raw chunks and no new map is produced.
        self.executor.remove_node(self.vla)
        self.executor.remove_node(self.observations)
        self.vla.destroy_node()
        self.observations.destroy_node()
        self.vla = None
        self.observations = None

        before = len(self.commands)
        positions = np.tile(_home(), (3, 1))
        positions[1, 2] = np.nan
        self.chunk_pub.publish(_chunk(time.time_ns(), positions))
        time.sleep(0.7)
        assert not [m for m in self.commands[before:] if len(m.points) > 1], "NaN chunk forwarded"

        # The map ages out; the watchdog enters UNSAFE and stops forwarding.
        assert _spin_until(
            self.executor,
            lambda: bool(self.statuses) and not self.statuses[-1].is_safe,
            timeout=10.0,
        ), "safety stayed safe with a stale/missing map"
        before = len(self.commands)
        for _ in range(5):
            self.joint_pub.publish(_joint_state(time.time_ns()))
            self.chunk_pub.publish(_chunk(time.time_ns(), np.tile(_home(), (3, 1))))
            self.executor.spin_once(timeout_sec=0.05)
        time.sleep(0.5)
        assert not [m for m in self.commands[before:] if len(m.points) > 1], (
            "chunk forwarded without a map"
        )


@unittest.skipIf(IMPORT_ERROR is not None, IMPORT_ERROR or "")
class TestStrictProfileStillGuards(unittest.TestCase):
    """The shipped default profile stays strict and still rejects a close obstacle."""

    def setUp(self) -> None:
        self.tmp = Path("/tmp/opencode") / f"pipeline_strict_{os.getpid()}"
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.safety_config = _write_safety_config(
            self.tmp, REPO_ROOT / "configs" / "safety" / "default.yaml"
        )
        rclpy.init()
        self.safety = SafetyWatchdogNode(
            parameter_overrides=[
                Parameter("config_file", value=str(self.safety_config)),
                Parameter("panda_config", value=str(REPO_ROOT / "configs/sim/panda.yaml")),
                Parameter("sim_config_dir", value=str(REPO_ROOT / "configs/sim")),
                Parameter(
                    "instructions_config",
                    value=str(REPO_ROOT / "configs/expert/instructions.yaml"),
                ),
            ]
        )
        self.helper = Node("pipeline_strict_helper")
        self.executor = MultiThreadedExecutor(num_threads=2)
        self.executor.add_node(self.safety)
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
        assert _spin_until(
            self.executor,
            lambda: (
                self.joint_pub.get_subscription_count() > 0
                and self.chunk_pub.get_subscription_count() > 0
                and self.map_pub.get_subscription_count() > 0
            ),
            timeout=15.0,
        )

    def tearDown(self) -> None:
        self.executor.shutdown()
        self.safety.destroy_node()
        self.helper.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def _cycle(self, points: np.ndarray, timeout: float = 1.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.map_pub.publish(_pointcloud(points, time.time_ns()))
            self.joint_pub.publish(_joint_state(time.time_ns()))
            self.executor.spin_once(timeout_sec=0.02)

    def _forwarded(self, timeout: float = 5.0) -> bool:
        before = len(self.commands)
        for _ in range(6):
            self.joint_pub.publish(_joint_state(time.time_ns()))
            self.chunk_pub.publish(_chunk(time.time_ns(), np.tile(_home(), (3, 1))))
            if _spin_until(
                self.executor,
                lambda: any(len(m.points) > 1 for m in self.commands[before:]),
                timeout=timeout / 6.0,
            ):
                return True
        return False

    def test_strict_profile_forwards_clean_chunk_and_rejects_a_near_obstacle(self) -> None:
        strict_config, _ = load_safety_config(
            self.safety_config,
            panda_config_path=REPO_ROOT / "configs" / "sim" / "panda.yaml",
        )
        assert strict_config.obstacle_mode == "strict"
        self._cycle(_table_points(), timeout=1.0)
        assert _spin_until(self.executor, lambda: self.safety.gate.state == "OK", timeout=10.0)
        assert self._forwarded(timeout=8.0), "strict profile did not forward a clean chunk"

        tcp = tcp_pose(_home()[:7], load_panda_config(REPO_ROOT / "configs/sim/panda.yaml"))
        obstacle = tcp[:3, 3] + np.array([0.0, 0.0, 0.004])
        self._cycle(np.vstack([_table_points(), _block_points(obstacle)]), timeout=2.0)
        before = len(self.commands)
        for _ in range(6):
            self.joint_pub.publish(_joint_state(time.time_ns()))
            self.chunk_pub.publish(_chunk(time.time_ns(), np.tile(_home(), (3, 1))))
            self.executor.spin_once(timeout_sec=0.02)
        time.sleep(0.6)
        assert not [m for m in self.commands[before:] if len(m.points) > 1], (
            "strict profile forwarded a chunk with a 4 mm obstacle at the TCP"
        )


@unittest.skipIf(IMPORT_ERROR is not None, IMPORT_ERROR or "")
class TestLaunchGuards(unittest.TestCase):
    def test_integrated_launches_refuse_a_second_joint_state_owner(self) -> None:
        with pytest.raises(JointStateOwnerError):
            check_single_owner(sim_node=True, ros2_control=True)
        for name in ("pipeline_sim.launch.py", "pipeline_mock_hw.launch.py"):
            text = (REPO_ROOT / "ros2_ws" / "src" / "sim2real_control" / "launch" / name).read_text(
                encoding="utf-8"
            )
            assert "check_single_owner" in text
        sim = (
            REPO_ROOT / "ros2_ws" / "src" / "sim2real_control" / "launch" / "pipeline_sim.launch.py"
        ).read_text(encoding="utf-8")
        mock = (
            REPO_ROOT
            / "ros2_ws"
            / "src"
            / "sim2real_control"
            / "launch"
            / "pipeline_mock_hw.launch.py"
        ).read_text(encoding="utf-8")
        assert "start_ros2_control" in sim
        assert "start_sim" in mock
        assert "fake_observation_publisher" in mock
        assert "sim_mock.yaml" in sim and "sim_mock.yaml" in mock

    def test_sim_mock_profile_relaxes_only_the_obstacle_mode(self) -> None:
        default = yaml.safe_load(
            (REPO_ROOT / "configs" / "safety" / "default.yaml").read_text(encoding="utf-8")
        )["safety"]
        mock = yaml.safe_load(
            (REPO_ROOT / "configs" / "safety" / "sim_mock.yaml").read_text(encoding="utf-8")
        )["safety"]
        assert default["obstacle_mode"] == "strict"
        assert mock["obstacle_mode"] == "masked"
        differences = {
            key for key in default.keys() | mock.keys() if default.get(key) != mock.get(key)
        }
        assert differences == {"obstacle_mode"}, differences


if __name__ == "__main__":
    unittest.main()
