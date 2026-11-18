"""Integrated simulation pipeline: sim + mapper + fake VLA inference + safety.

What actually runs:

* ``sim_node`` (fake or headless MuJoCo) publishes ``/joint_states``, TF and
  both camera streams and owns ``/joint_states``;
* ``workspace_mapper_node`` builds ``/workspace/pointcloud``;
* ``vla_inference_node`` (``policy_backend:=fake`` by default) publishes
  ``/vla/raw_action``;
* ``safety_watchdog_node`` validates chunks and publishes the controller
  topic with the simulation/mock safety profile (``configs/safety/sim_mock.yaml``).

**Documented gap.** No controller and no ros2_control stack are started here,
because ``sim_node`` has no command subscription: it holds its internal joint
target and does not consume ``/joint_trajectory_controller/joint_trajectory``.
The live ROS chain therefore does not close a loop through MuJoCo; closed-loop
evaluation uses the in-process runner. See ``docs/ARCHITECTURE.md``.

This launch is evidence for the graph and the safety/validation path only. It
does not demonstrate a working policy and says nothing about physical safety.
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from sim2real_control.single_owner import check_single_owner


def _guard(context) -> list:
    requested = context.perform_substitution(LaunchConfiguration("start_ros2_control")).lower()
    check_single_owner(sim_node=True, ros2_control=requested in {"1", "true", "yes"})
    return []


def generate_launch_description() -> LaunchDescription:
    sim_share = get_package_share_directory("sim2real_sim")
    perception_share = get_package_share_directory("sim2real_perception")
    vla_share = get_package_share_directory("sim2real_vla")
    control_share = get_package_share_directory("sim2real_control")

    sim_config_dir = os.path.join(sim_share, "config")
    mapper_config = os.path.join(perception_share, "config", "default.yaml")
    vla_config = os.path.join(vla_share, "config", "default.yaml")
    safety_config = os.path.join(control_share, "config", "sim_mock.yaml")
    panda_config = os.path.join(sim_share, "config", "panda.yaml")
    instructions_config = os.path.join(sim_share, "expert", "instructions.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument("backend", default_value="fake"),
            DeclareLaunchArgument("sim_config_dir", default_value=sim_config_dir),
            DeclareLaunchArgument("mapper_config", default_value=mapper_config),
            DeclareLaunchArgument("vla_config", default_value=vla_config),
            DeclareLaunchArgument(
                "safety_config",
                default_value=safety_config,
                description=(
                    "simulation/mock profile (obstacle_mode masked); real hardware "
                    "must use configs/safety/default.yaml (strict)"
                ),
            ),
            DeclareLaunchArgument("policy_backend", default_value="fake"),
            DeclareLaunchArgument("checkpoint_path", default_value=""),
            DeclareLaunchArgument("latency_compensation", default_value="true"),
            DeclareLaunchArgument(
                "start_ros2_control",
                default_value="false",
                description="Must stay false; sim_node and ros2_control cannot share /joint_states",
            ),
            OpaqueFunction(function=_guard),
            Node(
                package="sim2real_sim",
                executable="sim_node",
                name="sim_node",
                output="screen",
                parameters=[
                    {
                        "backend": LaunchConfiguration("backend"),
                        "config_dir": LaunchConfiguration("sim_config_dir"),
                        "publish_images": True,
                        "control_rate_hz": 20.0,
                        "camera_rate_hz": 10.0,
                        "tf_source": "backend_fk",
                    }
                ],
            ),
            Node(
                package="sim2real_perception",
                executable="workspace_mapper_node",
                name="workspace_mapper_node",
                output="screen",
                parameters=[
                    {
                        "config_file": LaunchConfiguration("mapper_config"),
                        "publish_rate_hz": ParameterValue(0.0, value_type=float),
                    }
                ],
            ),
            Node(
                package="sim2real_vla",
                executable="vla_inference_node",
                name="vla_inference_node",
                output="screen",
                parameters=[
                    {
                        "config_file": LaunchConfiguration("vla_config"),
                        "policy_backend": LaunchConfiguration("policy_backend"),
                        "checkpoint_path": LaunchConfiguration("checkpoint_path"),
                        "server_python": "",
                        "latency_compensation": ParameterValue(
                            LaunchConfiguration("latency_compensation"), value_type=bool
                        ),
                    }
                ],
            ),
            Node(
                package="sim2real_control",
                executable="safety_watchdog_node",
                name="safety_watchdog_node",
                output="screen",
                parameters=[
                    {
                        "config_file": LaunchConfiguration("safety_config"),
                        "panda_config": panda_config,
                        "sim_config_dir": sim_config_dir,
                        "instructions_config": instructions_config,
                    }
                ],
            ),
        ]
    )
