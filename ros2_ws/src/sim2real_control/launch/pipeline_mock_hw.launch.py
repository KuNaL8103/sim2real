"""Integrated mock-hardware pipeline: ros2_control + safety + fake inference.

What is **real** in this launch:

* the ROS 2 graph: publishers, subscribers, services, QoS, and message types;
* ``safety_watchdog_node`` and its validation/EMA/retime/hold/e-stop logic;
* ``vla_inference_node``'s observation assembly and chunk publication.

What is **mocked or fake**:

* ``mock_components/GenericSystem`` mirrors position commands to states with no
  dynamics, latency, torque, velocity enforcement, or collisions;
* ``fake_observation_publisher`` publishes synthetic 8x8 images, a synthetic
  table point cloud, and a fixed instruction;
* the policy is ``policy_backend:=fake`` (hold-current-state), not a trained
  model.

``sim_node`` is **not** started: the ros2_control ``joint_state_broadcaster`` is
the single owner of ``/joint_states``. This is not physical safety and proves
nothing about a real Franka.
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from sim2real_control.single_owner import check_single_owner


def _launch(context) -> list:
    start_sim = context.perform_substitution(LaunchConfiguration("start_sim")).lower()
    check_single_owner(sim_node=start_sim in {"1", "true", "yes"}, ros2_control=True)

    control_share = get_package_share_directory("sim2real_control")
    sim_share = get_package_share_directory("sim2real_sim")
    vla_share = get_package_share_directory("sim2real_vla")
    urdf_path = os.path.join(control_share, "urdf", "panda_mock.urdf")
    controllers = os.path.join(control_share, "config", "ros2_control.yaml")
    safety_config = context.perform_substitution(
        LaunchConfiguration("safety_config")
    ) or os.path.join(control_share, "config", "sim_mock.yaml")
    vla_config = context.perform_substitution(LaunchConfiguration("vla_config")) or os.path.join(
        vla_share, "config", "default.yaml"
    )
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
                "panda_config": os.path.join(sim_share, "config", "panda.yaml"),
                "sim_config_dir": os.path.join(sim_share, "config"),
                "instructions_config": os.path.join(sim_share, "expert", "instructions.yaml"),
            }
        ],
    )
    inference = Node(
        package="sim2real_vla",
        executable="vla_inference_node",
        name="vla_inference_node",
        output="screen",
        parameters=[
            {
                "config_file": vla_config,
                "policy_backend": "fake",
                "checkpoint_path": "",
                "server_python": "",
                "latency_compensation": ParameterValue(True, value_type=bool),
            }
        ],
    )
    observations = Node(
        package="sim2real_control",
        executable="fake_observation_publisher",
        name="fake_observation_publisher",
        output="screen",
        parameters=[
            {"instruction": context.perform_substitution(LaunchConfiguration("instruction"))}
        ],
        condition=IfCondition(LaunchConfiguration("start_fake_observations")),
    )
    return [
        controller_manager,
        robot_state_publisher,
        TimerAction(period=2.0, actions=[broadcaster]),
        TimerAction(period=3.0, actions=[trajectory]),
        safety,
        inference,
        observations,
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
                description="Empty uses the installed sim_mock.yaml (masked; simulation only)",
            ),
            DeclareLaunchArgument("vla_config", default_value=""),
            DeclareLaunchArgument("start_fake_observations", default_value="true"),
            DeclareLaunchArgument(
                "instruction", default_value="pick up the red cube and place it in the left zone"
            ),
            OpaqueFunction(function=_launch),
        ]
    )
