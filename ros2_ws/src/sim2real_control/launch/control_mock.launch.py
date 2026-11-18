"""Mock-hardware control stack: ros2_control + safety watchdog.

This is a kinematic mock, not the official Franka description and not a
physical-safety test.  ``mock_components/GenericSystem`` mirrors commands to
states with no dynamics, latency, torque, velocity enforcement, or collisions.

The launch refuses ``start_sim:=true`` because ros2_control's
joint_state_broadcaster and sim_node would both publish ``/joint_states``.
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
    TimerAction,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from sim2real_control.single_owner import check_single_owner


def _launch(context):
    start_sim = context.perform_substitution(LaunchConfiguration("start_sim")).lower()
    check_single_owner(sim_node=start_sim in {"1", "true", "yes"}, ros2_control=True)

    control_share = get_package_share_directory("sim2real_control")
    sim_share = get_package_share_directory("sim2real_sim")
    urdf_path = os.path.join(control_share, "urdf", "panda_mock.urdf")
    controllers = os.path.join(control_share, "config", "ros2_control.yaml")
    safety_config = context.perform_substitution(
        LaunchConfiguration("safety_config")
    ) or os.path.join(control_share, "config", "default.yaml")
    panda_config = os.path.join(sim_share, "config", "panda.yaml")
    sim_config_dir = os.path.join(sim_share, "config")
    instructions_config = os.path.join(sim_share, "expert", "instructions.yaml")
    with open(urdf_path, encoding="utf-8") as handle:
        robot_description = handle.read()

    controller_manager = Node(
        package="controller_manager",
        executable="ros2_control_node",
        output="screen",
        parameters=[{"robot_description": robot_description}, controllers],
    )
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
    )
    broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
        output="screen",
    )
    trajectory = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_trajectory_controller"],
        output="screen",
    )
    safety = Node(
        package="sim2real_control",
        executable="safety_watchdog_node",
        name="safety_watchdog_node",
        output="screen",
        parameters=[
            {
                "config_file": safety_config,
                "panda_config": panda_config,
                "sim_config_dir": sim_config_dir,
                "instructions_config": instructions_config,
            }
        ],
    )
    return [
        controller_manager,
        robot_state_publisher,
        TimerAction(period=2.0, actions=[broadcaster]),
        TimerAction(period=3.0, actions=[trajectory]),
        safety,
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "start_sim",
                default_value="false",
                description="Must stay false; sim_node and ros2_control cannot share /joint_states",
            ),
            DeclareLaunchArgument(
                "safety_config",
                default_value="",
                description="Optional safety YAML; empty uses the installed default",
            ),
            OpaqueFunction(function=_launch),
        ]
    )
