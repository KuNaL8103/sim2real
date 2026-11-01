import os
from glob import glob

from setuptools import find_packages, setup

package_name = "sim2real_perception"
here = os.path.abspath(os.path.dirname(__file__))
repo_mapping_configs = os.path.join(here, "..", "..", "..", "configs", "mapping")
mapping_configs = [
    os.path.relpath(path, here)
    for path in sorted(glob(os.path.join(repo_mapping_configs, "*.yaml")))
]

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), mapping_configs),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Kunal",
    maintainer_email="kunaldabbe2003@gmail.com",
    description="ROS 2 node adapters for RGB-D mapping, point clouds, and TSDF outputs.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "workspace_mapper_node = sim2real_perception.workspace_mapper_node:main",
        ]
    },
)
