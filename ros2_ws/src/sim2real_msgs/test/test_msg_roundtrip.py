"""Round-trip tests for every sim2real_msgs interface.

These tests require a sourced ROS 2 Jazzy environment and a built
``sim2real_msgs`` package. Without rclpy they are skipped with an explicit
reason, never reported as passed.
"""

from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.ros

rclpy = pytest.importorskip(
    "rclpy",
    reason="rclpy is not importable; source /opt/ros/jazzy/setup.bash and build ros2_ws",
)

from builtin_interfaces.msg import Time  # noqa: E402
from geometry_msgs.msg import Pose  # noqa: E402
from rclpy.serialization import deserialize_message, serialize_message  # noqa: E402
from std_msgs.msg import Header  # noqa: E402

from sim2real_msgs.msg import (  # noqa: E402
    FilterMetrics,
    RolloutEpisode,
    SafetyStatus,
    TrajectoryStep,
)
from sim2real_msgs.srv import GenerateRollouts, ResetEnv  # noqa: E402


def roundtrip(message, message_type):
    return deserialize_message(serialize_message(message), message_type)


def test_trajectory_step_roundtrip():
    message = TrajectoryStep()
    message.stamp = Time(sec=12, nanosec=345)
    message.step_index = 3
    message.task_instruction = "pick up the cube"
    message.joint_names = ["panda_joint1", "panda_finger_joint1"]
    message.joint_positions = [0.1, 0.02]
    message.joint_velocities = [0.0, 0.0]
    message.ee_pose = Pose()
    message.ee_pose.position.x = 0.4
    message.ee_frame_id = "panda_hand"
    message.action = [0.1, 0.02]
    message.action_space = TrajectoryStep.ACTION_JOINT_POSITION_TARGET

    result = roundtrip(message, TrajectoryStep)
    assert result.stamp.sec == 12
    assert result.stamp.nanosec == 345
    assert result.step_index == 3
    assert result.task_instruction == "pick up the cube"
    assert result.joint_names == ["panda_joint1", "panda_finger_joint1"]
    assert result.joint_positions == pytest.approx([0.1, 0.02])
    assert result.ee_frame_id == "panda_hand"
    assert result.action == pytest.approx([0.1, 0.02])
    assert result.action_space == 0


def test_rollout_episode_roundtrip():
    step = TrajectoryStep()
    step.step_index = 0
    step.joint_positions = [0.0] * 8

    message = RolloutEpisode()
    message.episode_id = "ep-0001"
    message.schema_version = "1"
    message.backend = "fake"
    message.seed = 42
    message.task_instruction = "open the drawer"
    message.start_stamp = Time(sec=100)
    message.control_rate_hz = 20.0
    message.is_perturbed = True
    message.outcome = RolloutEpisode.OUTCOME_SUCCESS
    message.failure_reason = ""
    message.steps = [step]
    message.metadata_json = '{"source": "test"}'

    result = roundtrip(message, RolloutEpisode)
    assert result.episode_id == "ep-0001"
    assert result.backend == "fake"
    assert result.seed == 42
    assert result.start_stamp.sec == 100
    assert result.control_rate_hz == pytest.approx(20.0)
    assert result.is_perturbed is True
    assert result.outcome == RolloutEpisode.OUTCOME_SUCCESS
    assert len(result.steps) == 1
    assert result.steps[0].joint_positions == pytest.approx([0.0] * 8)
    assert result.metadata_json == '{"source": "test"}'


def test_filter_metrics_roundtrip():
    message = FilterMetrics()
    message.episode_id = "ep-0001"
    message.dimensionless_squared_jerk = 1.5
    message.sparc = -2.0
    message.path_length_m = 0.25
    message.min_joint_limit_margin = 0.1
    message.min_velocity_limit_margin = 0.5
    message.collision_count = 0
    message.task_success = True
    message.acceptance_score = 0.9
    message.decision = FilterMetrics.DECISION_ACCEPT
    message.rejection_reasons = []
    message.metadata_json = ""

    result = roundtrip(message, FilterMetrics)
    assert result.episode_id == "ep-0001"
    assert result.sparc == pytest.approx(-2.0)
    assert result.task_success is True
    assert result.decision == FilterMetrics.DECISION_ACCEPT


def test_safety_status_roundtrip_preserves_nan():
    message = SafetyStatus()
    message.header = Header()
    message.header.frame_id = "world"
    message.header.stamp = Time(sec=7, nanosec=8)
    message.is_safe = False
    message.estop_latched = True
    message.state = SafetyStatus.STATE_ESTOP
    message.failure_reason = "operator e-stop"
    message.obstacle_map_valid = False
    message.min_obstacle_distance_m = math.nan
    message.max_joint_limit_ratio = 1.2
    message.max_velocity_ratio = 0.3
    message.max_acceleration_ratio = 0.4
    message.max_jerk_ratio = 0.5
    message.observation_age_s = 0.01
    message.action_age_s = 0.02

    result = roundtrip(message, SafetyStatus)
    assert result.header.frame_id == "world"
    assert result.header.stamp.sec == 7
    assert result.state == SafetyStatus.STATE_ESTOP
    assert result.estop_latched is True
    assert math.isnan(result.min_obstacle_distance_m)
    assert result.max_joint_limit_ratio == pytest.approx(1.2)


def test_reset_env_roundtrip():
    request = ResetEnv.Request()
    request.seed = 7
    request.randomization_enabled = True
    request.randomization_profile = "mild"
    request.has_object_pose_override = True
    request.object_pose_override.position.z = 0.05
    request.task_instruction = "stack the blocks"

    response = ResetEnv.Response()
    response.success = True
    response.message = "ok"
    response.applied_seed = 7
    response.applied_config_json = '{"profile": "mild"}'

    request_result = roundtrip(request, ResetEnv.Request)
    response_result = roundtrip(response, ResetEnv.Response)
    assert request_result.seed == 7
    assert request_result.randomization_enabled is True
    assert request_result.object_pose_override.position.z == pytest.approx(0.05)
    assert response_result.success is True
    assert response_result.applied_seed == 7


def test_generate_rollouts_roundtrip():
    request = GenerateRollouts.Request()
    request.num_rollouts = 10
    request.seed = 99
    request.task_instruction = "stack the blocks"
    request.sampling_mode = GenerateRollouts.Request.SAMPLING_FAILURE_TARGETED
    request.failure_distribution_ref = "hard_cases_v1"
    request.include_perturbed = True

    response = GenerateRollouts.Response()
    response.success = True
    response.num_generated = 8
    response.num_failed = 2
    response.message = "2 failed to stabilize"
    response.result_metadata_json = "{}"

    request_result = roundtrip(request, GenerateRollouts.Request)
    response_result = roundtrip(response, GenerateRollouts.Response)
    assert request_result.num_rollouts == 10
    assert request_result.sampling_mode == 1
    assert response_result.num_generated == 8
    assert response_result.num_failed == 2
