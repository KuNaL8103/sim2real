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
# Evaluation summaries are published on request; keep the last few so a late
# subscriber can still see the most recent run.
EVAL_SUMMARY_QOS = QosSpec(reliability="reliable", durability="volatile", depth=10)
# Task instructions are commands: the latest replaces the previous one. Volatile
# durability keeps compatibility with the usual operator publisher (a
# transient-local subscriber would refuse a volatile publisher).
TASK_INSTRUCTION_QOS = QosSpec(reliability="reliable", durability="volatile", depth=1)
# Raw VLA chunks are consumed by the Task 11 safety node. Keep a short history
# so a brief scheduling stall does not lose every chunk; the safety node still
# owns staleness checks.
VLA_RAW_ACTION_QOS = QosSpec(reliability="reliable", durability="volatile", depth=10)
# Status is diagnostic: latest history is enough, and a late subscriber may
# still see the last few seconds of state transitions.
VLA_STATUS_QOS = QosSpec(reliability="reliable", durability="volatile", depth=10)
