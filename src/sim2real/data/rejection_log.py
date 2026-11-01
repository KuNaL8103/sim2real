"""Rejected-episode log.

Layout::

    data/rejected/<run_id>/
        rejections.jsonl          one structured row per rejected/invalid episode
        episodes/<episode_id>.json diagnostic reference to the staged episode

The log is separate from staging and curated output. An accepted episode is never
written here; the caller passes only non-accepted results.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sim2real.filtering.decision import FilterResult


class RejectionLogError(RuntimeError):
    """Raised when the rejection log cannot be written safely."""


@dataclass(frozen=True)
class RejectionEntry:
    episode_id: str
    decision: int
    reasons: list[dict[str, Any]]
    metrics: dict[str, Any] | None


class RejectionLog:
    def __init__(self, root: str | Path, run_id: str) -> None:
        if not run_id or "/" in run_id or "\\" in run_id:
            raise RejectionLogError(f"invalid run_id {run_id!r}")
        self.run_id = run_id
        self.root = Path(root).expanduser().resolve() / run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.episodes_dir = self.root / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "rejections.jsonl"

    def append(
        self,
        record: Any,
        result: FilterResult,
        *,
        staging_reference: str | None = None,
    ) -> RejectionEntry:
        episode_id = str(getattr(record, "episode_id", "") or result.episode_id)
        reasons = [reason.to_dict() for reason in result.reasons]
        metrics = result.metrics.to_dict() if result.metrics is not None else None
        row = {
            "episode_id": episode_id,
            "run_id": str(getattr(record, "run_id", "") or ""),
            "seed": int(getattr(record, "seed", -1) or -1),
            "decision": int(result.decision),
            "acceptance_score": result.acceptance_score,
            "reason_codes": [reason.code for reason in result.reasons],
            "reasons": reasons,
            "metrics": metrics,
            "validation_issues": [issue.to_dict() for issue in result.issues],
            "staging_reference": staging_reference,
            "record_ref": f"episodes/{episode_id or 'unknown'}.json",
            "logged_at": datetime.now(UTC).isoformat(),
        }
        text = json.dumps(row, sort_keys=True, allow_nan=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        diagnostic = {
            "episode_id": episode_id,
            "run_id": str(getattr(record, "run_id", "") or ""),
            "seed": int(getattr(record, "seed", -1) or -1),
            "backend": str(getattr(record, "backend", "") or ""),
            "outcome": int(getattr(record, "outcome", 0) or 0),
            "failure_reason": str(getattr(record, "failure_reason", "") or ""),
            "is_perturbed": bool(getattr(record, "is_perturbed", False)),
            "perturbation_name": getattr(record, "perturbation_name", None),
            "expected_quality": str(getattr(record, "expected_quality", "") or ""),
            "num_steps": len(getattr(record, "steps", []) or []),
            "staging_reference": staging_reference,
            "decision": int(result.decision),
            "reason_codes": [reason.code for reason in result.reasons],
        }
        self._write_json(self.episodes_dir / f"{episode_id or 'unknown'}.json", diagnostic)
        return RejectionEntry(episode_id, result.decision, reasons, metrics)

    def rows(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        text = json.dumps(payload, sort_keys=True, allow_nan=False)
        path.write_text(text + "\n", encoding="utf-8")


__all__ = ["RejectionEntry", "RejectionLog", "RejectionLogError"]
