"""Safety state machine and per-reason counters (ROS-free).

States: STARTING -> OK <-> UNSAFE; ESTOP is latched.  A rejected chunk does not
change the state and never latches; only an input watchdog failure enters
UNSAFE and only an explicit reset leaves ESTOP.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .config import SafetyConfig
from .types import (
    DECISION_ACCEPTED,
    DECISION_NONE,
    DECISION_REJECTED,
    ESTOP_LATCHED,
    WATCHDOG_JOINT_STATE_STALE,
    WATCHDOG_MAP_STALE,
    WATCHDOG_MAP_UNAVAILABLE,
)

STARTING = "STARTING"
OK = "OK"
UNSAFE = "UNSAFE"
ESTOP = "ESTOP"


@dataclass
class SafetyGate:
    config: SafetyConfig
    state: str = STARTING
    failure_reason: str | None = None
    estop_latched: bool = False
    last_decision: str = DECISION_NONE
    last_reason: str | None = None
    reason_counts: Counter[str] = field(default_factory=Counter)
    accepted_chunks: int = 0
    rejected_chunks: int = 0

    def _set(self, state: str, reason: str | None) -> bool:
        changed = state != self.state or reason != self.failure_reason
        self.state = state
        self.failure_reason = reason
        return changed

    def update_inputs(
        self,
        *,
        joint_state_age_s: float | None,
        map_age_s: float | None,
        map_valid: bool,
    ) -> tuple[bool, str, str | None]:
        """Recompute OK/UNSAFE from input ages.

        Returns ``(changed, state, reason)``.  STARTING is retained while no
        joint state has ever arrived; a joint state with an unavailable map is
        UNSAFE because unknown is not safe.
        """
        if self.estop_latched:
            changed = self._set(ESTOP, ESTOP_LATCHED)
            return changed, self.state, self.failure_reason

        state = OK
        reason: str | None = None
        if joint_state_age_s is None:
            state = STARTING
            reason = None
        elif joint_state_age_s > min(
            self.config.max_joint_state_age_s, self.config.watchdog_timeout_s
        ):
            state = UNSAFE
            reason = WATCHDOG_JOINT_STATE_STALE
        elif self.config.map_required:
            if not map_valid:
                state = UNSAFE
                reason = WATCHDOG_MAP_UNAVAILABLE
            elif map_age_s is None or map_age_s > min(
                self.config.max_map_age_s, self.config.watchdog_timeout_s
            ):
                state = UNSAFE
                reason = WATCHDOG_MAP_STALE
        changed = self._set(state, reason)
        return changed, self.state, self.failure_reason

    def on_chunk(self, accepted: bool, reason: str | None) -> None:
        if accepted:
            self.last_decision = DECISION_ACCEPTED
            self.last_reason = None
            self.accepted_chunks += 1
        else:
            self.last_decision = DECISION_REJECTED
            self.last_reason = reason
            self.rejected_chunks += 1
            if reason:
                self.reason_counts[reason] += 1

    def trigger_estop(self) -> bool:
        self.estop_latched = True
        return self._set(ESTOP, ESTOP_LATCHED)

    def reset(self, *, inputs_fresh: bool) -> tuple[bool, str]:
        if not self.estop_latched:
            return False, "e-stop is not latched"
        if not inputs_fresh:
            return False, "inputs are not fresh and valid; reset refused"
        self.estop_latched = False
        self._set(STARTING, None)
        return True, "e-stop latch cleared; no motion is commanded until a valid chunk"

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "failure_reason": self.failure_reason,
            "estop_latched": self.estop_latched,
            "last_decision": self.last_decision,
            "last_reason": self.last_reason,
            "accepted_chunks": self.accepted_chunks,
            "rejected_chunks": self.rejected_chunks,
            "reason_counts": dict(sorted(self.reason_counts.items())),
        }


__all__ = ["ESTOP", "OK", "STARTING", "UNSAFE", "SafetyGate"]
