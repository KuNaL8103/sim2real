"""Launch the scripted-expert rollout collector.

Arguments:
  backend:=fake|mujoco        in-process backend (never a /joint_states owner)
  output_dir:=<path>          staging parent directory (data/staging by default)
  config_dir:=<path>          simulator configs (installed share by default)
  expert_config_dir:=<path>   expert configs (installed share by default)
  max_rollouts_per_call:=64
  perturbed_fraction:=-1.0    negative means "use perturbations.yaml"

This node publishes only /sim2real/staging_episode and serves
/sim2real/generate_rollouts. It does not publish /joint_states.
"""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("backend", default_value="fake"),
            DeclareLaunchArgument("output_dir", default_value="data/staging"),
            DeclareLaunchArgument("config_dir", default_value=""),
            DeclareLaunchArgument("randomization_dir", default_value=""),
            DeclareLaunchArgument("expert_config_dir", default_value=""),
            DeclareLaunchArgument("max_rollouts_per_call", default_value="64"),
            DeclareLaunchArgument("perturbed_fraction", default_value="-1.0"),
            Node(
                package="sim2real_sim",
                executable="expert_collector_node",
                name="expert_collector_node",
                output="screen",
                parameters=[
                    {
                        "backend": LaunchConfiguration("backend"),
                        "output_dir": LaunchConfiguration("output_dir"),
                        "config_dir": LaunchConfiguration("config_dir"),
                        "randomization_dir": LaunchConfiguration("randomization_dir"),
                        "expert_config_dir": LaunchConfiguration("expert_config_dir"),
                        "max_rollouts_per_call": LaunchConfiguration("max_rollouts_per_call"),
                        "perturbed_fraction": LaunchConfiguration("perturbed_fraction"),
                    }
                ],
            ),
        ]
    )
