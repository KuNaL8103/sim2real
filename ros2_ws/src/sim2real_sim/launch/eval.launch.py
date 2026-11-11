"""Launch the closed-loop evaluation node.

Arguments:
  backend:=fake|mujoco             in-process backend (never a /joint_states owner)
  policy_type:=scripted|noisy_expert|random|zero|checkpoint
  checkpoint_path:=<path>          required for policy_type:=checkpoint
  num_episodes:=20
  eval_seed:=20000                 evaluation-only seed range 20000-20999
  randomization_profile:=default
  output_dir:=data/eval
  n_action_steps:=10
  auto_generate:=false             fit failures + request targeted rollouts

This node publishes only /sim2real/eval_summary and serves /sim2real/run_eval.
It does not publish /joint_states.
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
            DeclareLaunchArgument("policy_type", default_value="noisy_expert"),
            DeclareLaunchArgument("checkpoint_path", default_value=""),
            DeclareLaunchArgument("num_episodes", default_value="20"),
            DeclareLaunchArgument("eval_seed", default_value="20000"),
            DeclareLaunchArgument("randomization_profile", default_value="default"),
            DeclareLaunchArgument("output_dir", default_value="data/eval"),
            DeclareLaunchArgument("n_action_steps", default_value="10"),
            DeclareLaunchArgument("auto_generate", default_value="false"),
            DeclareLaunchArgument("config_dir", default_value=""),
            DeclareLaunchArgument("evaluation_config", default_value=""),
            DeclareLaunchArgument("filter_config", default_value=""),
            Node(
                package="sim2real_sim",
                executable="closed_loop_eval_node",
                name="closed_loop_eval_node",
                output="screen",
                parameters=[
                    {
                        "backend": LaunchConfiguration("backend"),
                        "policy_type": LaunchConfiguration("policy_type"),
                        "checkpoint_path": LaunchConfiguration("checkpoint_path"),
                        "num_episodes": LaunchConfiguration("num_episodes"),
                        "eval_seed": LaunchConfiguration("eval_seed"),
                        "randomization_profile": LaunchConfiguration("randomization_profile"),
                        "output_dir": LaunchConfiguration("output_dir"),
                        "n_action_steps": LaunchConfiguration("n_action_steps"),
                        "auto_generate": LaunchConfiguration("auto_generate"),
                        "config_dir": LaunchConfiguration("config_dir"),
                        "evaluation_config": LaunchConfiguration("evaluation_config"),
                        "filter_config": LaunchConfiguration("filter_config"),
                    }
                ],
            ),
        ]
    )
