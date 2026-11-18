"""Launch only the safety watchdog (no controller, no simulator).

Use this with an externally started ros2_control stack.  It refuses to start a
sim node; the mock stack launch owns the only ``/joint_states`` publisher.
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch(context):
    config_file = context.perform_substitution(LaunchConfiguration("config_file"))
    panda_config = context.perform_substitution(LaunchConfiguration("panda_config"))
    sim_config_dir = context.perform_substitution(LaunchConfiguration("sim_config_dir"))
    instructions_config = context.perform_substitution(LaunchConfiguration("instructions_config"))
    sim_share = get_package_share_directory("sim2real_sim")
    if not config_file:
        config_file = os.path.join(
            get_package_share_directory("sim2real_control"), "config", "default.yaml"
        )
    if not panda_config:
        panda_config = os.path.join(sim_share, "config", "panda.yaml")
    if not sim_config_dir:
        sim_config_dir = os.path.join(sim_share, "config")
    if not instructions_config:
        instructions_config = os.path.join(sim_share, "expert", "instructions.yaml")
    return [
        Node(
            package="sim2real_control",
            executable="safety_watchdog_node",
            name="safety_watchdog_node",
            output="screen",
            parameters=[
                {
                    "config_file": config_file,
                    "panda_config": panda_config,
                    "sim_config_dir": sim_config_dir,
                    "instructions_config": instructions_config,
                }
            ],
        )
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("config_file", default_value=""),
            DeclareLaunchArgument("panda_config", default_value=""),
            DeclareLaunchArgument("sim_config_dir", default_value=""),
            DeclareLaunchArgument("instructions_config", default_value=""),
            OpaqueFunction(function=_launch),
        ]
    )
