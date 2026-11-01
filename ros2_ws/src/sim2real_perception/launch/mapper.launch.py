"""Launch the workspace mapper, optionally with the simulator and RViz.

Arguments:
  backend:=fake|mujoco        forwarded to sim.launch.py
  include_sim:=true|false     start sim2real_sim as well (default true)
  use_rviz:=true|false        start RViz with the mapping displays
  config_file:=<path>         default: the installed configs/mapping/default.yaml
  publish_rate_hz:=<float>    0.0 means "use the config value"

The sim launch is always started with ``use_rviz:=false`` so only this launch
file owns RViz.  Both cameras feed one TSDF volume; no cross-camera sync is
needed.
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

BACKEND = LaunchConfiguration("backend")
INCLUDE_SIM = LaunchConfiguration("include_sim")
USE_RVIZ = LaunchConfiguration("use_rviz")
CONFIG_FILE = LaunchConfiguration("config_file")
PUBLISH_RATE = LaunchConfiguration("publish_rate_hz")


def generate_launch_description() -> LaunchDescription:
    perception_share = get_package_share_directory("sim2real_perception")
    sim_share = get_package_share_directory("sim2real_sim")
    default_config = os.path.join(perception_share, "config", "default.yaml")
    # The mapper RViz config intentionally omits the Camera/Image displays:
    # under llvmpipe they add GL load next to MuJoCo and can crash WSLg.
    rviz_config = os.path.join(perception_share, "rviz", "mapper.rviz")

    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(sim_share, "launch", "sim.launch.py")),
        launch_arguments={
            "backend": BACKEND,
            # Suppress the sim launch's own RViz without clobbering the parent
            # use_rviz argument; this launch owns the single RViz instance.
            "headless": "true",
        }.items(),
        condition=IfCondition(INCLUDE_SIM),
    )
    mapper = Node(
        package="sim2real_perception",
        executable="workspace_mapper_node",
        name="workspace_mapper_node",
        output="screen",
        parameters=[
            {
                "config_file": CONFIG_FILE,
                "publish_rate_hz": ParameterValue(PUBLISH_RATE, value_type=float),
            }
        ],
    )
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        condition=IfCondition(USE_RVIZ),
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("backend", default_value="fake"),
            DeclareLaunchArgument("include_sim", default_value="true"),
            DeclareLaunchArgument("use_rviz", default_value="false"),
            DeclareLaunchArgument("config_file", default_value=default_config),
            DeclareLaunchArgument("publish_rate_hz", default_value="0.0"),
            sim_launch,
            mapper,
            # On WSLg, initializing RViz's GLX context in parallel with MuJoCo's
            # EGL context can drop the X11 connection. Start RViz after the
            # simulator has finished initializing. The fake backend is
            # unaffected; the delay only adds 8 s to the first visualization.
            TimerAction(period=8.0, actions=[rviz]),
        ]
    )
