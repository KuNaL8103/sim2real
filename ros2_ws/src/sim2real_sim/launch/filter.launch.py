"""Launch the trajectory filter.

Arguments:
  config_path:=<path>     filter YAML (installed share by default)
  curated_root:=<path>    curated dataset root (data/curated by default)
  rejected_root:=<path>   rejection log root (data/rejected by default)
  dataset_name:=<name>    curated dataset name

The node subscribes to /sim2real/staging_episode, publishes
/sim2real/filter_metrics for every episode and /sim2real/filtered_episode for
accepted episodes, and writes a curated version when
/sim2real/finalize_dataset is called or at shutdown.
"""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("config_path", default_value=""),
            DeclareLaunchArgument("curated_root", default_value="data/curated"),
            DeclareLaunchArgument("rejected_root", default_value="data/rejected"),
            DeclareLaunchArgument("dataset_name", default_value="staged_curated"),
            Node(
                package="sim2real_sim",
                executable="trajectory_filter_node",
                name="trajectory_filter_node",
                output="screen",
                parameters=[
                    {
                        "config_path": LaunchConfiguration("config_path"),
                        "curated_root": LaunchConfiguration("curated_root"),
                        "rejected_root": LaunchConfiguration("rejected_root"),
                        "dataset_name": LaunchConfiguration("dataset_name"),
                    }
                ],
            ),
        ]
    )
