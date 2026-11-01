"""State machine: legal path, illegal transition, guard failure, timeout."""

from __future__ import annotations

import pytest

from sim2real.expert.config import load_expert_config
from sim2real.expert.state_machine import (
    ExpertState,
    PickPlaceStateMachine,
    StateMachineError,
)

CONFIG_DIR = "configs/expert"


def machine() -> PickPlaceStateMachine:
    return PickPlaceStateMachine.from_config(load_expert_config(CONFIG_DIR))


def test_legal_path_reaches_done() -> None:
    instance = machine()
    context = instance.context(pose_error_m=0.0)
    assert instance.fire("start", context) is ExpertState.APPROACH_PREGRASP
    for _ in range(2):
        assert instance.fire("reached", context) in {
            ExpertState.DESCEND,
            ExpertState.GRASP,
        }
    assert instance.fire("grasped", instance.context(grasp_confirmed=True)) is ExpertState.LIFT
    for expected in (
        ExpertState.TRANSIT,
        ExpertState.PLACE_DESCEND,
        ExpertState.RELEASE,
    ):
        assert instance.fire("reached", context) is expected
    assert instance.fire("released", instance.context(gripper_open=True)) is ExpertState.RETREAT
    assert instance.fire("reached", instance.context(success=True)) is ExpertState.DONE
    assert instance.terminal
    assert [entry.to_state for entry in instance.log][-1] == "DONE"
    # Every entry has a from/to/reason and monotonic time.
    assert all(entry.reason for entry in instance.log)
    times = [entry.time_s for entry in instance.log]
    assert times == sorted(times)


def test_illegal_transition_raises() -> None:
    instance = machine()
    with pytest.raises(StateMachineError):
        instance.fire("grasped")
    instance.fire("start", instance.context())
    with pytest.raises(StateMachineError):
        instance.fire("released")
    instance.fail("planner exploded")
    with pytest.raises(StateMachineError):
        instance.fire("reached")


def test_guard_failure_becomes_failed_with_reason() -> None:
    instance = machine()
    context = instance.context(pose_error_m=0.0)
    instance.fire("start", context)
    instance.fire("reached", context)
    instance.fire("reached", context)
    assert instance.state is ExpertState.GRASP
    instance.fire("grasped", instance.context(grasp_confirmed=False))
    assert instance.state is ExpertState.FAILED
    assert "grasp_confirmed" in instance.failure_reason
    assert instance.terminal


def test_per_state_timeout_becomes_timeout() -> None:
    instance = machine()
    instance.fire("start", instance.context())
    timeout = instance.timeouts_s["APPROACH_PREGRASP"]
    instance.tick(timeout - 0.01)
    assert instance.state is ExpertState.APPROACH_PREGRASP
    instance.tick(0.02)
    assert instance.state is ExpertState.TIMEOUT
    assert "APPROACH_PREGRASP" in instance.failure_reason
    assert instance.terminal


def test_transition_log_is_json_safe() -> None:
    instance = machine()
    instance.fire("start", instance.context())
    payload = instance.to_dict()
    assert payload["state"] == "APPROACH_PREGRASP"
    assert payload["transitions"][0]["from"] == "IDLE"
    assert isinstance(payload["transitions"][0]["reason"], str)
