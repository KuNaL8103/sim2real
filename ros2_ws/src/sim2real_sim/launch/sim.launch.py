"""Launch the simulator node, optionally with RViz.

Arguments:
  backend:=fake|mujoco|isaac   (isaac fails with a clear UNAVAILABLE message)
  use_rviz:=true|false
  headless:=true|false         (suppresses RViz; images are still published)
  config_dir:=<path>           (defaults to the installed share/sim2real_sim/config)

RViz is OPTIONAL. TF comes from backend forward kinematics, so a URDF and
robot_state_publisher are not required for TF; only the RViz RobotModel display
needs a URDF, which this launch file does not provide.
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory("sim2real_sim")
    default_config = os.path.join(package_share, "config")
    default_rviz = os.path.join(package_share, "rviz", "sim2real.rviz")

    backend = LaunchConfiguration("backend")
    use_rviz = LaunchConfiguration("use_rviz")
    headless = LaunchConfiguration("headless")
    config_dir = LaunchConfiguration("config_dir")

    return LaunchDescription(
        [
            DeclareLaunchArgument("backend", default_value="fake"),
            DeclareLaunchArgument("use_rviz", default_value="false"),
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument("config_dir", default_value=default_config),
            Node(
                package="sim2real_sim",
                executable="sim_node",
                name="sim_node",
                output="screen",
                parameters=[
                    {
                        "backend": backend,
                        "config_dir": config_dir,
                        "publish_images": True,
                        "control_rate_hz": 20.0,
                        "camera_rate_hz": 10.0,
                        "tf_source": "backend_fk",
                    }
                ],
            ),
            # On WSLg, creating RViz's GLX context in parallel with MuJoCo's
            # EGL context can drop the X11 connection. Start RViz after the
            # simulator has initialized. The fake backend is unaffected.
            TimerAction(
                period=8.0,
                actions=[
                    Node(
                        package="rviz2",
                        executable="rviz2",
                        name="rviz2",
                        output="screen",
                        arguments=["-d", default_rviz],
                        condition=IfCondition(
                            PythonExpression(
                                [
                                    "'",
                                    use_rviz,
                                    "' == 'true' and '",
                                    headless,
                                    "' != 'true'",
                                ]
                            )
                        ),
                    )
                ],
            ),
        ]
    )
