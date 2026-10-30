"""Pure-Python core library for the sim2real pipeline.

Everything in this package must stay importable without ROS, a simulator, a
GPU, or network access. ROS packages in ``ros2_ws`` are thin adapters around
this library.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
