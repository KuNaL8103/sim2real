"""QoS decisions shared by the ROS adapter and the documentation test.

The strings here are compared against the QoS cells in
``docs/ARCHITECTURE.md``; change both together.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QosSpec:
    reliability: str
    durability: str
    depth: int
    history: str = "keep_last"


JOINT_STATES_QOS = QosSpec(reliability="reliable", durability="volatile", depth=10)
TF_QOS = QosSpec(reliability="reliable", durability="volatile", depth=10)
TF_STATIC_QOS = QosSpec(reliability="reliable", durability="transient_local", depth=1)
CAMERA_QOS = QosSpec(reliability="reliable", durability="volatile", depth=5)
# Filter topics: a burst of completed episodes must not be dropped.
FILTER_QOS = QosSpec(reliability="reliable", durability="volatile", depth=50)
# Mapping publishers: the latest map replaces the previous one; do not queue.
MAP_QOS = QosSpec(reliability="reliable", durability="volatile", depth=1)
# The mapper subscribes to the sim's reliable keep-last 5 camera topics.
MAP_CAMERA_SUB_QOS = QosSpec(reliability="reliable", durability="volatile", depth=5)
MAP_STATIC_TF_QOS = QosSpec(reliability="reliable", durability="transient_local", depth=1)
