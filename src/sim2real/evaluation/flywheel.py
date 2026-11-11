"""Flywheel loop guards with a persistent, append-only, traceable ledger.

Task 9 **never** starts training or retraining.  The controller only emits a
decision: request targeted generation, stop, or (reserved) request retraining.
Every cycle is appended to ``data/flywheel/ledger.jsonl`` with links back to the
source datasets, checkpoint hash, evaluation run, and failure distribution, so a
future task can reconstruct exactly what happened.

Guards (all configurable in ``configs/evaluation/default.yaml``):

* ``max_cycles`` — hard cycle cap.
* ``max_total_rollouts`` — hard cumulative rollout cap.
* ``max_rollouts_per_request`` — per-request cap (also enforced by the collector).
* ``min_failures_to_trigger`` — do not target noise.
* ``min_success_improvement`` / ``max_consecutive_no_improvement`` — stop when
  the loop stops improving.
* ``cooldown_s`` — minimum wall time between cycles.
* ``STOP`` file — a kill switch that wins over everything.
* the same ``eval_run_id`` is refused twice (idempotency).

A corrupt or partial ledger line is reported and skipped; it never crashes the
controller.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sim2real.evaluation.config import DECISIONS, EvaluationConfig

SCHEMA_VERSION = 1
REQUIRED_FIELDS = (
    "schema_version",
    "cycle_id",
    "parent_cycle_id",
    "config_hash",
    "source_dataset_versions",
    "episode_ids",
    "episode_hash",
    "checkpoint_hash",
    "eval_run_id",
    "failure_distribution_id",
    "rollouts_requested",
    "rollouts_generated",
    "decision",
    "reason",
    "created_at",
    "success_rate",
)


class FlywheelError(RuntimeError):
    """Raised when the ledger cannot be written safely."""


@dataclass
class CycleRecord:
    cycle_id: str
    parent_cycle_id: str | None
    config_hash: str
    source_dataset_versions: tuple[str, ...]
    episode_ids: tuple[str, ...]
    episode_hash: str
    checkpoint_hash: str
    eval_run_id: str
    failure_distribution_id: str
    rollouts_requested: int
    rollouts_generated: int
    decision: str
    reason: str
    success_rate: float
    created_at: str = ""
    expert_success_rate: float = 0.0
    n_failures: int = 0
    policy_only_failures: int = 0
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.decision not in DECISIONS:
            raise FlywheelError(f"invalid decision {self.decision!r}")
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat()
        if self.rollouts_requested < 0 or self.rollouts_generated < 0:
            raise FlywheelError("rollout counts must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "cycle_id": self.cycle_id,
            "parent_cycle_id": self.parent_cycle_id,
            "config_hash": self.config_hash,
            "source_dataset_versions": list(self.source_dataset_versions),
            "episode_ids": list(self.episode_ids),
            "episode_hash": self.episode_hash,
            "checkpoint_hash": self.checkpoint_hash,
            "eval_run_id": self.eval_run_id,
            "failure_distribution_id": self.failure_distribution_id,
            "rollouts_requested": int(self.rollouts_requested),
            "rollouts_generated": int(self.rollouts_generated),
            "decision": self.decision,
            "reason": self.reason,
            "success_rate": float(self.success_rate),
            "expert_success_rate": float(self.expert_success_rate),
            "n_failures": int(self.n_failures),
            "policy_only_failures": int(self.policy_only_failures),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CycleRecord:
        missing = sorted(set(REQUIRED_FIELDS) - set(data))
        if missing:
            raise FlywheelError(f"ledger record missing fields {missing}")
        return cls(
            schema_version=int(data["schema_version"]),
            cycle_id=str(data["cycle_id"]),
            parent_cycle_id=(
                None if data.get("parent_cycle_id") is None else str(data["parent_cycle_id"])
            ),
            config_hash=str(data["config_hash"]),
            source_dataset_versions=tuple(str(v) for v in data["source_dataset_versions"]),
            episode_ids=tuple(str(v) for v in data["episode_ids"]),
            episode_hash=str(data["episode_hash"]),
            checkpoint_hash=str(data["checkpoint_hash"]),
            eval_run_id=str(data["eval_run_id"]),
            failure_distribution_id=str(data["failure_distribution_id"]),
            rollouts_requested=int(data["rollouts_requested"]),
            rollouts_generated=int(data["rollouts_generated"]),
            decision=str(data["decision"]),
            reason=str(data["reason"]),
            success_rate=float(data["success_rate"]),
            expert_success_rate=float(data.get("expert_success_rate", 0.0)),
            n_failures=int(data.get("n_failures", 0)),
            policy_only_failures=int(data.get("policy_only_failures", 0)),
            created_at=str(data["created_at"]),
        )


def _streak_without_improvement(records: list[CycleRecord], minimum: float) -> int:
    best = -1.0
    streak = 0
    for record in records:
        if record.decision != "REQUEST_TARGETED_GENERATION":
            continue
        improved = record.success_rate > best + minimum if best >= 0 else True
        best = max(best, record.success_rate)
        streak = 0 if improved else streak + 1
    return streak


class FlywheelController:
    def __init__(
        self,
        config: EvaluationConfig,
        *,
        ledger_path: str | Path | None = None,
        stop_file: str | Path | None = None,
        now: Callable[[], float] | None = None,
    ) -> None:
        self.config = config
        self.guards = config.flywheel
        self.ledger_path = Path(ledger_path or self.guards.ledger_path).expanduser()
        self.stop_file = Path(stop_file or self.guards.stop_file).expanduser()
        self._now = now or time.time

    # -- ledger ------------------------------------------------------------
    def load_ledger(self) -> tuple[list[CycleRecord], list[dict[str, Any]]]:
        if not self.ledger_path.is_file():
            return [], []
        records: list[CycleRecord] = []
        corrupt: list[dict[str, Any]] = []
        for number, line in enumerate(
            self.ledger_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if not isinstance(data, dict):
                    raise FlywheelError("line is not a JSON object")
                records.append(CycleRecord.from_dict(data))
            except (ValueError, KeyError, TypeError, FlywheelError) as exc:
                corrupt.append({"line": number, "reason": str(exc)})
        return records, corrupt

    def _append(self, record: CycleRecord) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.to_dict(), sort_keys=True, allow_nan=False)
        try:
            with self.ledger_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise FlywheelError(f"could not append to ledger {self.ledger_path}: {exc}") from exc

    # -- guards ------------------------------------------------------------
    def _stop(self, reason: str) -> tuple[str, str, int]:
        return ("STOP", reason, 0)

    def decide(
        self,
        *,
        eval_run_id: str,
        success_rate: float,
        expert_success_rate: float = 0.0,
        n_failures: int = 0,
        policy_only_failures: int = 0,
        checkpoint_hash: str = "none",
        failure_distribution_id: str = "none",
        source_dataset_versions: tuple[str, ...] | list[str] = (),
        episode_ids: tuple[str, ...] | list[str] = (),
        episode_hash: str = "",
        requested_rollouts: int | None = None,
        generated_rollouts: int = 0,
        parent_cycle_id: str | None = None,
    ) -> CycleRecord:
        records, corrupt = self.load_ledger()
        del corrupt  # reported by status(); a bad line must not block a decision
        guard = self.guards
        decision = "STOP"
        reason = ""
        requested = 0

        if self.stop_file.exists():
            reason = f"STOP file present: {self.stop_file}"
        elif any(record.eval_run_id == eval_run_id for record in records):
            reason = f"eval_run_id {eval_run_id!r} has already been processed (idempotency)"
        elif len(records) >= guard.max_cycles:
            reason = f"max_cycles={guard.max_cycles} reached"
        else:
            now = self._now()
            if guard.cooldown_s > 0 and records:
                last = records[-1].created_at
                try:
                    last_ts = datetime.fromisoformat(last).timestamp()
                except ValueError:
                    last_ts = now - guard.cooldown_s
                if now - last_ts < guard.cooldown_s:
                    reason = f"cooldown_s={guard.cooldown_s} has not elapsed"
            if not reason:
                streak = _streak_without_improvement(records, guard.min_success_improvement)
                if streak >= guard.max_consecutive_no_improvement:
                    reason = (
                        "no success improvement for "
                        f"{streak} consecutive cycles (max {guard.max_consecutive_no_improvement})"
                    )
            if not reason:
                if n_failures <= 0:
                    reason = "NO_TARGETING: zero failures"
                elif n_failures < guard.min_failures_to_trigger:
                    reason = (
                        f"failures={n_failures} below min_failures_to_trigger="
                        f"{guard.min_failures_to_trigger}; retraining would be a separate "
                        "Task 12 decision (Task 9 never trains)"
                    )
                else:
                    used = sum(record.rollouts_requested for record in records)
                    remaining = guard.max_total_rollouts - used
                    if remaining <= 0:
                        reason = f"max_total_rollouts={guard.max_total_rollouts} reached"
                    else:
                        wanted = (
                            guard.max_rollouts_per_request
                            if requested_rollouts is None
                            else int(requested_rollouts)
                        )
                        requested = max(0, min(wanted, remaining, guard.max_rollouts_per_request))
                        if requested == 0:
                            reason = "per-request cap or remaining budget is zero"
                        else:
                            decision = "REQUEST_TARGETED_GENERATION"
                            reason = (
                                f"{n_failures} failures (policy-only={policy_only_failures}); "
                                f"requesting {requested} targeted rollouts"
                            )

        cycle_id = f"cycle_{len(records) + 1:04d}"
        if parent_cycle_id is None and records:
            parent_cycle_id = records[-1].cycle_id
        record = CycleRecord(
            cycle_id=cycle_id,
            parent_cycle_id=parent_cycle_id,
            config_hash=self.config.config_hash(),
            source_dataset_versions=tuple(source_dataset_versions),
            episode_ids=tuple(episode_ids),
            episode_hash=episode_hash,
            checkpoint_hash=checkpoint_hash,
            eval_run_id=eval_run_id,
            failure_distribution_id=failure_distribution_id,
            rollouts_requested=requested,
            rollouts_generated=int(generated_rollouts),
            decision=decision,
            reason=reason,
            success_rate=float(success_rate),
            expert_success_rate=float(expert_success_rate),
            n_failures=int(n_failures),
            policy_only_failures=int(policy_only_failures),
        )
        self._append(record)
        return record

    # -- reporting ---------------------------------------------------------
    def status(self) -> dict[str, Any]:
        records, corrupt = self.load_ledger()
        counts: dict[str, int] = {}
        for record in records:
            counts[record.decision] = counts.get(record.decision, 0) + 1
        return {
            "ledger_path": str(self.ledger_path),
            "stop_file": str(self.stop_file),
            "stop_file_present": self.stop_file.exists(),
            "num_cycles": len(records),
            "total_rollouts_requested": sum(r.rollouts_requested for r in records),
            "total_rollouts_generated": sum(r.rollouts_generated for r in records),
            "decision_counts": dict(sorted(counts.items())),
            "last_cycle": records[-1].to_dict() if records else None,
            "corrupt_lines": corrupt,
            "guards": self.guards.to_dict(),
            "will_train": False,
        }


__all__ = [
    "CycleRecord",
    "FlywheelController",
    "FlywheelError",
    "SCHEMA_VERSION",
]
