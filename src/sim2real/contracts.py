"""Canonical topic, service, frame, and type names for the project.

This module is intentionally pure Python: it must not import ``rclpy`` or any
ROS package. The strings below are the single source of truth for the core
library; ``docs/ARCHITECTURE.md`` mirrors them in its contract table and
``tests/test_contracts.py`` fails if the two drift apart.
"""

from __future__ import annotations

from typing import Final

SCHEMA_VERSION: Final = "1"

# ---------------------------------------------------------------------------
# Message and service type strings
# ---------------------------------------------------------------------------
JOINT_STATE_TYPE: Final = "sensor_msgs/msg/JointState"
TF_MESSAGE_TYPE: Final = "tf2_msgs/msg/TFMessage"
IMAGE_TYPE: Final = "sensor_msgs/msg/Image"
CAMERA_INFO_TYPE: Final = "sensor_msgs/msg/CameraInfo"
STRING_TYPE: Final = "std_msgs/msg/String"
ROLLOUT_EPISODE_TYPE: Final = "sim2real_msgs/msg/RolloutEpisode"
FILTER_METRICS_TYPE: Final = "sim2real_msgs/msg/FilterMetrics"
SAFETY_STATUS_TYPE: Final = "sim2real_msgs/msg/SafetyStatus"
POINT_CLOUD2_TYPE: Final = "sensor_msgs/msg/PointCloud2"
MARKER_TYPE: Final = "visualization_msgs/msg/Marker"
JOINT_TRAJECTORY_TYPE: Final = "trajectory_msgs/msg/JointTrajectory"
TRIGGER_TYPE: Final = "std_srvs/srv/Trigger"
RESET_ENV_TYPE: Final = "sim2real_msgs/srv/ResetEnv"
GENERATE_ROLLOUTS_TYPE: Final = "sim2real_msgs/srv/GenerateRollouts"

# ---------------------------------------------------------------------------
# Topics: name -> type string
# ---------------------------------------------------------------------------
JOINT_STATES_TOPIC: Final = "/joint_states"
TF_TOPIC: Final = "/tf"
TF_STATIC_TOPIC: Final = "/tf_static"
TASK_INSTRUCTION_TOPIC: Final = "/task_instruction"
STAGING_EPISODE_TOPIC: Final = "/sim2real/staging_episode"
FILTERED_EPISODE_TOPIC: Final = "/sim2real/filtered_episode"
FILTER_METRICS_TOPIC: Final = "/sim2real/filter_metrics"
WORKSPACE_POINTCLOUD_TOPIC: Final = "/workspace/pointcloud"
WORKSPACE_TSDF_MESH_TOPIC: Final = "/workspace/tsdf_mesh"
VLA_RAW_ACTION_TOPIC: Final = "/vla/raw_action"
CONTROLLER_TRAJECTORY_TOPIC: Final = "/joint_trajectory_controller/joint_trajectory"
SAFETY_STATUS_TOPIC: Final = "/safety/status"

TOPICS: Final[dict[str, str]] = {
    JOINT_STATES_TOPIC: JOINT_STATE_TYPE,
    TF_TOPIC: TF_MESSAGE_TYPE,
    TF_STATIC_TOPIC: TF_MESSAGE_TYPE,
    "/camera/overhead/color/image_raw": IMAGE_TYPE,
    "/camera/overhead/depth/image_raw": IMAGE_TYPE,
    "/camera/overhead/camera_info": CAMERA_INFO_TYPE,
    "/camera/wrist/color/image_raw": IMAGE_TYPE,
    "/camera/wrist/depth/image_raw": IMAGE_TYPE,
    "/camera/wrist/camera_info": CAMERA_INFO_TYPE,
    TASK_INSTRUCTION_TOPIC: STRING_TYPE,
    STAGING_EPISODE_TOPIC: ROLLOUT_EPISODE_TYPE,
    FILTERED_EPISODE_TOPIC: ROLLOUT_EPISODE_TYPE,
    FILTER_METRICS_TOPIC: FILTER_METRICS_TYPE,
    WORKSPACE_POINTCLOUD_TOPIC: POINT_CLOUD2_TYPE,
    WORKSPACE_TSDF_MESH_TOPIC: MARKER_TYPE,
    VLA_RAW_ACTION_TOPIC: JOINT_TRAJECTORY_TYPE,
    CONTROLLER_TRAJECTORY_TOPIC: JOINT_TRAJECTORY_TYPE,
    SAFETY_STATUS_TOPIC: SAFETY_STATUS_TYPE,
}

# ---------------------------------------------------------------------------
# Services: name -> type string
# ---------------------------------------------------------------------------
ESTOP_SERVICE: Final = "/estop"
ESTOP_RESET_SERVICE: Final = "/estop_reset"
RESET_ENV_SERVICE: Final = "/sim2real/reset_env"
GENERATE_ROLLOUTS_SERVICE: Final = "/sim2real/generate_rollouts"

SERVICES: Final[dict[str, str]] = {
    ESTOP_SERVICE: TRIGGER_TYPE,
    ESTOP_RESET_SERVICE: TRIGGER_TYPE,
    RESET_ENV_SERVICE: RESET_ENV_TYPE,
    GENERATE_ROLLOUTS_SERVICE: GENERATE_ROLLOUTS_TYPE,
}

# ---------------------------------------------------------------------------
# Frames (ROS frame IDs are relative: no leading slash)
# ---------------------------------------------------------------------------
WORLD_FRAME: Final = "world"
PANDA_BASE_FRAME: Final = "panda_link0"
PANDA_HAND_FRAME: Final = "panda_hand"
CAMERA_OVERHEAD_OPTICAL_FRAME: Final = "camera_overhead_optical_frame"
CAMERA_WRIST_OPTICAL_FRAME: Final = "camera_wrist_optical_frame"

FRAMES: Final[tuple[str, ...]] = (
    WORLD_FRAME,
    PANDA_BASE_FRAME,
    PANDA_HAND_FRAME,
    CAMERA_OVERHEAD_OPTICAL_FRAME,
    CAMERA_WRIST_OPTICAL_FRAME,
)

# ---------------------------------------------------------------------------
# Cameras and image encodings
# ---------------------------------------------------------------------------
CAMERA_OVERHEAD: Final = "overhead"
CAMERA_WRIST: Final = "wrist"
CAMERAS: Final[tuple[str, ...]] = (CAMERA_OVERHEAD, CAMERA_WRIST)
COLOR_ENCODING: Final = "rgb8"
DEPTH_ENCODING: Final = "32FC1"  # metres, float32 single channel

# ---------------------------------------------------------------------------
# Joint names
# ---------------------------------------------------------------------------
PANDA_ARM_JOINTS: Final[tuple[str, ...]] = tuple(f"panda_joint{i}" for i in range(1, 8))
GRIPPER_JOINT: Final = "panda_finger_joint1"
# /vla/raw_action carries seven arm joints plus one gripper finger joint.
CONTROLLED_JOINT_NAMES: Final[tuple[str, ...]] = (*PANDA_ARM_JOINTS, GRIPPER_JOINT)

# ---------------------------------------------------------------------------
# Enumerations mirrored from sim2real_msgs (wire-level source of truth).
# ---------------------------------------------------------------------------
ACTION_JOINT_POSITION_TARGET: Final = 0
ACTION_JOINT_DELTA: Final = 1
ACTION_EE_DELTA: Final = 2

OUTCOME_UNKNOWN: Final = 0
OUTCOME_SUCCESS: Final = 1
OUTCOME_FAILURE: Final = 2
OUTCOME_TIMEOUT: Final = 3
OUTCOME_ABORTED: Final = 4

DECISION_ACCEPT: Final = 0
DECISION_REJECT: Final = 1
DECISION_INVALID: Final = 2

STATE_NOMINAL: Final = 0
STATE_DEGRADED: Final = 1
STATE_SAFE_STOP: Final = 2
STATE_ESTOP: Final = 3

SAMPLING_UNIFORM: Final = 0
SAMPLING_FAILURE_TARGETED: Final = 1
