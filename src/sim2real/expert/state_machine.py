"""Explicit pick-and-place state machine.

The machine owns only state, time-in-state, the transition table, guards, and a
log. It does no physics, no IK, and reads no globals, which makes every legal,
illegal, guard-failure, and timeout path testable on CPU.

States::

    IDLE -> APPROACH_PREGRASP -> DESCEND -> GRASP -> LIFT -> TRANSIT
         -> PLACE_DESCEND -> RELEASE -> RETREAT -> DONE
    any non-terminal -> FAILED (guard or planner failure)
    any non-terminal -> TIMEOUT (per-state timeout)

Every transition is recorded as ``{from, to, event, reason, time_s}``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from sim2real.expert.config import ExpertConfig


class StateMachineError(RuntimeError):
    """Raised for an illegal transition or a transition out of a terminal state."""


class ExpertState(StrEnum):
    IDLE = "IDLE"
    APPROACH_PREGRASP = "APPROACH_PREGRASP"
    DESCEND = "DESCEND"
    GRASP = "GRASP"
    LIFT = "LIFT"
    TRANSIT = "TRANSIT"
    PLACE_DESCEND = "PLACE_DESCEND"
    RELEASE = "RELEASE"
    RETREAT = "RETREAT"
    DONE = "DONE"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"


TERMINAL = frozenset({ExpertState.DONE, ExpertState.FAILED, ExpertState.TIMEOUT})

# Explicit legal transition table: (from_state, event) -> (next_state, guard_name).
TRANSITIONS: dict[tuple[ExpertState, str], tuple[ExpertState, str | None]] = {
    (ExpertState.IDLE, "start"): (ExpertState.APPROACH_PREGRASP, None),
    (ExpertState.APPROACH_PREGRASP, "reached"): (ExpertState.DESCEND, "pose_reached"),
    (ExpertState.DESCEND, "reached"): (ExpertState.GRASP, "pose_reached"),
    (ExpertState.GRASP, "grasped"): (ExpertState.LIFT, "grasp_confirmed"),
    (ExpertState.LIFT, "reached"): (ExpertState.TRANSIT, "pose_reached"),
    (ExpertState.TRANSIT, "reached"): (ExpertState.PLACE_DESCEND, "pose_reached"),
    (ExpertState.PLACE_DESCEND, "reached"): (ExpertState.RELEASE, "pose_reached"),
    (ExpertState.RELEASE, "released"): (ExpertState.RETREAT, "gripper_open"),
    (ExpertState.RETREAT, "reached"): (ExpertState.DONE, "success"),
    (ExpertState.APPROACH_PREGRASP, "failed"): (ExpertState.FAILED, None),
    (ExpertState.DESCEND, "failed"): (ExpertState.FAILED, None),
    (ExpertState.GRASP, "failed"): (ExpertState.FAILED, None),
    (ExpertState.LIFT, "failed"): (ExpertState.FAILED, None),
    (ExpertState.TRANSIT, "failed"): (ExpertState.FAILED, None),
    (ExpertState.PLACE_DESCEND, "failed"): (ExpertState.FAILED, None),
    (ExpertState.RELEASE, "failed"): (ExpertState.FAILED, None),
    (ExpertState.RETREAT, "failed"): (ExpertState.FAILED, None),
}

_MOTION_STATES = (
    ExpertState.APPROACH_PREGRASP,
    ExpertState.DESCEND,
    ExpertState.LIFT,
    ExpertState.TRANSIT,
    ExpertState.PLACE_DESCEND,
    ExpertState.RETREAT,
)


def guard_pose_reached(context: dict[str, Any]) -> bool:
    tolerance = float(context.get("position_tolerance_m", 0.01))
    return float(context.get("pose_error_m", float("inf"))) <= tolerance


def guard_grasp_confirmed(context: dict[str, Any]) -> bool:
    return bool(context.get("grasp_confirmed", False))


def guard_gripper_open(context: dict[str, Any]) -> bool:
    return bool(context.get("gripper_open", False))


def guard_success(context: dict[str, Any]) -> bool:
    return bool(context.get("success", False))


GUARDS: dict[str, Callable[[dict[str, Any]], bool]] = {
    "pose_reached": guard_pose_reached,
    "grasp_confirmed": guard_grasp_confirmed,
    "gripper_open": guard_gripper_open,
    "success": guard_success,
}


@dataclass
class TransitionLog:
    from_state: str
    to_state: str
    event: str
    reason: str
    time_s: float

    def to_dict(self) -> dict[str, object]:
        return {
            "from": self.from_state,
            "to": self.to_state,
            "event": self.event,
            "reason": self.reason,
            "time_s": float(self.time_s),
        }


@dataclass
class PickPlaceStateMachine:
    """Deterministic state machine with per-state timeouts and guards."""

    timeouts_s: dict[str, float]
    position_tolerance_m: float = 0.01
    log: list[TransitionLog] = field(default_factory=list)
    state: ExpertState = ExpertState.IDLE
    time_in_state_s: float = 0.0
    total_time_s: float = 0.0
    failure_reason: str = ""

    @classmethod
    def from_config(cls, config: ExpertConfig) -> PickPlaceStateMachine:
        return cls(
            timeouts_s=dict(config.state_timeouts_s),
            position_tolerance_m=config.ik.position_tolerance_m * 5.0,
        )

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL

    @property
    def needs_motion(self) -> bool:
        return self.state in _MOTION_STATES

    def context(self, **values: Any) -> dict[str, Any]:
        merged: dict[str, Any] = {"position_tolerance_m": self.position_tolerance_m}
        merged.update(values)
        return merged

    def fire(self, event: str, context: dict[str, Any] | None = None) -> ExpertState:
        """Apply ``event``; an illegal event raises, a failed guard -> FAILED."""
        if self.terminal:
            raise StateMachineError(f"cannot fire {event!r} from terminal state {self.state.value}")
        try:
            next_state, guard_name = TRANSITIONS[(self.state, event)]
        except KeyError as exc:
            legal = sorted(name for (state, name) in TRANSITIONS if state is self.state)
            raise StateMachineError(
                f"illegal transition {self.state.value} --{event}-->; legal events: {legal}"
            ) from exc

        values = context or self.context()
        if guard_name is not None and not GUARDS[guard_name](values):
            reason = self._guard_failure_reason(guard_name, values)
            self._record(next_state=ExpertState.FAILED, event=event, reason=reason)
            return self.state

        self._record(next_state=next_state, event=event, reason=f"guard={guard_name or 'none'}")
        return self.state

    def fail(self, reason: str) -> ExpertState:
        """Move a non-terminal state directly to FAILED with a reason."""
        if self.terminal:
            raise StateMachineError(f"cannot fail from terminal state {self.state.value}")
        self._record(next_state=ExpertState.FAILED, event="failed", reason=reason)
        return self.state

    def tick(self, dt_s: float) -> ExpertState:
        """Advance time; per-state timeout moves a non-terminal state to TIMEOUT."""
        if dt_s < 0.0:
            raise ValueError("dt_s must be non-negative")
        self.time_in_state_s += float(dt_s)
        self.total_time_s += float(dt_s)
        if self.terminal:
            return self.state
        timeout = self.timeouts_s.get(self.state.value)
        if timeout is not None and self.time_in_state_s > timeout:
            self._record(
                next_state=ExpertState.TIMEOUT,
                event="timeout",
                reason=(
                    f"state {self.state.value} exceeded {timeout:.3f} s "
                    f"({self.time_in_state_s:.3f} s elapsed)"
                ),
            )
        return self.state

    def reset(self) -> None:
        self.log.clear()
        self.state = ExpertState.IDLE
        self.time_in_state_s = 0.0
        self.total_time_s = 0.0
        self.failure_reason = ""

    def _guard_failure_reason(self, guard_name: str, values: dict[str, Any]) -> str:
        if guard_name == "pose_reached":
            return (
                f"guard pose_reached failed: pose_error_m={values.get('pose_error_m')} "
                f"> {values.get('position_tolerance_m')}"
            )
        if guard_name == "grasp_confirmed":
            return "guard grasp_confirmed failed: target cube is not held"
        if guard_name == "gripper_open":
            return "guard gripper_open failed: gripper did not release"
        if guard_name == "success":
            return f"guard success failed: {values.get('success_reason', 'place check failed')}"
        return f"guard {guard_name} failed"

    def _record(self, *, next_state: ExpertState, event: str, reason: str) -> None:
        self.log.append(
            TransitionLog(
                from_state=self.state.value,
                to_state=next_state.value,
                event=event,
                reason=reason,
                time_s=self.total_time_s,
            )
        )
        self.state = next_state
        self.time_in_state_s = 0.0
        if next_state in {ExpertState.FAILED, ExpertState.TIMEOUT}:
            self.failure_reason = reason

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "time_in_state_s": float(self.time_in_state_s),
            "total_time_s": float(self.total_time_s),
            "failure_reason": self.failure_reason,
            "transitions": [entry.to_dict() for entry in self.log],
        }
