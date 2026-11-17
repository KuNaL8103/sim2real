import os
from glob import glob

from setuptools import find_packages, setup

package_name = "sim2real_vla"
here = os.path.abspath(os.path.dirname(__file__))
repo_inference_configs = os.path.join(here, "..", "..", "..", "configs", "inference")
# colcon requires relative data_files sources; resolve the canonical repo configs.
inference_configs = [
    os.path.relpath(path, here)
    for path in sorted(glob(os.path.join(repo_inference_configs, "*.yaml")))
]

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        # Ship the canonical config so `ros2 launch` works from the install tree.
        (os.path.join("share", package_name, "config"), inference_configs),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="Kunal",
    maintainer_email="kunaldabbe2003@gmail.com",
    description=(
        "ROS 2 node adapters for VLA inference, task instructions, and action chunk publication."
    ),
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "vla_inference_node = sim2real_vla.vla_inference_node:main",
        ]
    },
)
