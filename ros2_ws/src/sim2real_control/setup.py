import os
from glob import glob

from setuptools import find_packages, setup

package_name = "sim2real_control"
here = os.path.abspath(os.path.dirname(__file__))
repo_safety_configs = os.path.join(here, "..", "..", "..", "configs", "safety")
safety_configs = [
    os.path.relpath(path, here)
    for path in sorted(glob(os.path.join(repo_safety_configs, "*.yaml")))
]

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), safety_configs + glob("config/*.yaml")),
        (os.path.join("share", package_name, "urdf"), glob("urdf/*.urdf")),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="Kunal",
    maintainer_email="kunaldabbe2003@gmail.com",
    description=(
        "ROS 2 node adapters for trajectory dispatch, safety monitoring, and emergency stop."
    ),
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "safety_watchdog_node = sim2real_control.safety_watchdog_node:main",
            ("fake_observation_publisher = sim2real_control.fake_observation_publisher:main"),
        ]
    },
)
