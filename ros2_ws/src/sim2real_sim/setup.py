import os
from glob import glob

from setuptools import find_packages, setup

package_name = "sim2real_sim"
here = os.path.abspath(os.path.dirname(__file__))
repo_sim_configs = os.path.join(here, "..", "..", "..", "configs", "sim")
# colcon requires relative data_files sources; resolve the canonical repo configs.
sim_configs = [
    os.path.relpath(path, here) for path in sorted(glob(os.path.join(repo_sim_configs, "*.yaml")))
]
repo_randomization_configs = os.path.join(here, "..", "..", "..", "configs", "randomization")
randomization_configs = [
    os.path.relpath(path, here)
    for path in sorted(glob(os.path.join(repo_randomization_configs, "*.yaml")))
]
repo_expert_configs = os.path.join(here, "..", "..", "..", "configs", "expert")
expert_configs = [
    os.path.relpath(path, here)
    for path in sorted(glob(os.path.join(repo_expert_configs, "*.yaml")))
]

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
        # Ship the canonical configs so `ros2 launch` works from the install tree.
        (
            os.path.join("share", package_name, "config"),
            sim_configs,
        ),
        (
            os.path.join("share", package_name, "randomization"),
            randomization_configs,
        ),
        (
            os.path.join("share", package_name, "expert"),
            expert_configs,
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Kunal",
    maintainer_email="kunaldabbe2003@gmail.com",
    description=(
        "ROS 2 node adapters for simulator backends (fake and MuJoCo first; Isaac Sim optional)."
    ),
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "sim_node = sim2real_sim.sim_node:main",
            "expert_collector_node = sim2real_sim.expert_collector_node:main",
        ]
    },
)
