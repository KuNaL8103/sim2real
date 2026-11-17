"""Launch the VLA inference node.

The default backend is ``fake`` (a CPU-only hold-current-state policy), so no
model starts unless explicitly asked. Selecting the real checkpoint is a launch
argument; this file never starts or connects a controller.

Examples:
    ros2 launch sim2real_vla vla_inference.launch.py
    ros2 launch sim2real_vla vla_inference.launch.py policy_backend:=server \
        checkpoint_path:=outputs/remote/overfit_remote/checkpoint-00001000
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory("sim2real_vla")
    default_config = os.path.join(package_share, "config", "default.yaml")

    config_file = LaunchConfiguration("config_file")
    policy_backend = LaunchConfiguration("policy_backend")
    checkpoint_path = LaunchConfiguration("checkpoint_path")
    server_python = LaunchConfiguration("server_python")
    latency_compensation = LaunchConfiguration("latency_compensation")

    return LaunchDescription(
        [
            DeclareLaunchArgument("config_file", default_value=default_config),
            DeclareLaunchArgument("policy_backend", default_value="fake"),
            DeclareLaunchArgument("checkpoint_path", default_value=""),
            DeclareLaunchArgument("server_python", default_value=""),
            DeclareLaunchArgument("latency_compensation", default_value="true"),
            Node(
                package="sim2real_vla",
                executable="vla_inference_node",
                name="vla_inference_node",
                output="screen",
                parameters=[
                    {
                        "config_file": config_file,
                        "policy_backend": policy_backend,
                        "checkpoint_path": checkpoint_path,
                        "server_python": server_python,
                        "latency_compensation": ParameterValue(
                            latency_compensation, value_type=bool
                        ),
                    }
                ],
            ),
        ]
    )
