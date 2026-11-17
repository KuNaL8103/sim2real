"""Pure single-owner guard for ``/joint_states``.

The sim node and ros2_control both publish ``/joint_states``.  Running both
silently corrupts every consumer's state estimate, so the launch files refuse
the combination instead of relying on operator discipline.
"""

from __future__ import annotations


class JointStateOwnerError(RuntimeError):
    """Raised when two ``/joint_states`` owners would be started."""


def check_single_owner(*, sim_node: bool, ros2_control: bool) -> None:
    if sim_node and ros2_control:
        raise JointStateOwnerError(
            "refusing to start sim_node and the ros2_control /joint_states "
            "broadcaster together: /joint_states must have exactly one owner"
        )


__all__ = ["JointStateOwnerError", "check_single_owner"]
