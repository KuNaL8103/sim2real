"""Deterministic acceptance engine for filtered rollouts.

Decision rule:

1. Validation issues -> ``INVALID``; no metrics are computed.
2. A degenerate trajectory (zero path, zero speed, too short for the derivative)
   -> ``REJECT`` with ``DEGENERATE_TRAJECTORY``.
3. Hard gates (success, collisions, joint margin, velocity margin) and soft
   thresholds (jerk, SPARC, path length) must all pass.
4. The score is the weighted sum of clipped sub-scores in ``[0, 1]``; the
   episode is accepted only if it also reaches ``accept_score_threshold``.

The engine is a pure function of the record and the config. It never reads
``is_perturbed``, ``perturbation_name``, or ``expected_quality``, never uses
randomness, and emits reasons in a fixed order.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from sim2real import contracts
from sim2real.filtering.config import FilterConfig
from sim2real.filtering.metrics import (
    DegenerateTrajectoryError,
    TrajectoryMetrics,
    compute_metrics,
)
from sim2real.filtering.validation import ValidationIssue, validate_episode

NOT_SUCCESS = "NOT_SUCCESS"
COLLISION = "COLLISION"
JERK_TOO_HIGH = "JERK_TOO_HIGH"
SPARC_TOO_LOW = "SPARC_TOO_LOW"
PATH_TOO_LONG = "PATH_TOO_LONG"
JOINT_LIMIT_MARGIN = "JOINT_LIMIT_MARGIN"
VELOCITY_LIMIT_MARGIN = "VELOCITY_LIMIT_MARGIN"
DEGENERATE_TRAJECTORY = "DEGENERATE_TRAJECTORY"

REASON_ORDER = (
    # Validation codes first (imported constants are compared by value).
    "MALFORMED_EPISODE",
    "MISSING_FIELD",
    "SHAPE_MISMATCH",
    "ACTION_DIM",
    "NAN_VALUES",
    "TOO_SHORT",
    "NON_MONOTONIC_TIME",
    "DT_JITTER",
    "DUPLICATE_EPISODE",
    DEGENERATE_TRAJECTORY,
    NOT_SUCCESS,
    COLLISION,
    JOINT_LIMIT_MARGIN,
    VELOCITY_LIMIT_MARGIN,
    JERK_TOO_HIGH,
    SPARC_TOO_LOW,
    PATH_TOO_LONG,
)
REASON_RANK = {code: index for index, code in enumerate(REASON_ORDER)}


@dataclass(frozen=True)
class RejectionReason:
    code: str
    metric: str | None = None
    value: float | None = None
    threshold: float | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "detail": self.detail,
        }

    def text(self) -> str:
        if self.metric and self.value is not None and self.threshold is not None:
            return f"{self.code}: {self.metric}={self.value:.6g} (threshold {self.threshold:.6g})"
        return f"{self.code}: {self.detail}" if self.detail else self.code


@dataclass(frozen=True)
class FilterResult:
    episode_id: str
    decision: int
    acceptance_score: float
    reasons: list[RejectionReason]
    metrics: TrajectoryMetrics | None = None
    issues: list[ValidationIssue] = field(default_factory=list)
    config_hash: str = ""

    @property
    def accepted(self) -> bool:
        return self.decision == contracts.DECISION_ACCEPT

    @property
    def valid(self) -> bool:
        return self.decision != contracts.DECISION_INVALID

    def metadata(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "decision": int(self.decision),
            "acceptance_score": self.acceptance_score,
            "config_sha256": self.config_hash,
            "reason_codes": [reason.code for reason in self.reasons],
            "metrics": self.metrics.to_dict() if self.metrics is not None else None,
        }
        if self.issues:
            payload["validation_issues"] = [issue.to_dict() for issue in self.issues]
        return payload

    def metadata_json(self) -> str:
        return json.dumps(self.metadata(), sort_keys=True, allow_nan=False)


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(min(high, max(low, value)))


def lower_is_better_score(value: float, limit: float) -> float:
    """1 at zero, 0.5 exactly at the limit, 0 at twice the limit.

    A soft violation already rejects the episode; this score only ranks
    episodes that pass. Anchoring the threshold at 0.5 avoids requiring an
    arbitrary "ideal" second anchor.
    """
    if limit <= 0:
        return 0.0
    return _clip(1.0 - 0.5 * (value / limit))


def higher_is_better_score(value: float, limit: float) -> float:
    """For negative SPARC: 1 at zero, 0.5 at ``limit``, clipped outside."""
    if limit >= 0:
        return 0.0
    return _clip(0.5 + 0.5 * ((value - limit) / (0.0 - limit)))


def acceptance_score(metrics: TrajectoryMetrics, config: FilterConfig) -> float:
    weights = config.weights
    total = weights.dimensionless_squared_jerk + weights.sparc + weights.path_length_m
    if total <= 0:
        return 0.0
    score = (
        weights.dimensionless_squared_jerk
        * lower_is_better_score(
            metrics.dimensionless_squared_jerk,
            config.soft_thresholds.max_dimensionless_squared_jerk,
        )
        + weights.sparc * higher_is_better_score(metrics.sparc, config.soft_thresholds.min_sparc)
        + weights.path_length_m
        * lower_is_better_score(metrics.path_length_m, config.soft_thresholds.max_path_length_m)
    ) / total
    return _clip(score)


def _reason(
    code: str,
    metric: str | None = None,
    value: float | None = None,
    threshold: float | None = None,
    detail: str = "",
) -> RejectionReason:
    return RejectionReason(code, metric, value, threshold, detail)


def _hard_and_soft_reasons(
    metrics: TrajectoryMetrics, config: FilterConfig
) -> list[RejectionReason]:
    reasons: list[RejectionReason] = []
    hard = config.hard_gates
    soft = config.soft_thresholds
    if hard.require_success and not metrics.task_success:
        reasons.append(_reason(NOT_SUCCESS, "task_success", 0.0, 1.0, "episode did not succeed"))
    if metrics.collision_count > hard.max_collisions:
        reasons.append(
            _reason(
                COLLISION,
                "collision_count",
                float(metrics.collision_count),
                float(hard.max_collisions),
            )
        )
    if metrics.min_joint_limit_margin < hard.min_joint_margin:
        reasons.append(
            _reason(
                JOINT_LIMIT_MARGIN,
                "min_joint_limit_margin",
                metrics.min_joint_limit_margin,
                hard.min_joint_margin,
            )
        )
    if metrics.min_velocity_limit_margin < hard.min_velocity_margin:
        reasons.append(
            _reason(
                VELOCITY_LIMIT_MARGIN,
                "min_velocity_limit_margin",
                metrics.min_velocity_limit_margin,
                hard.min_velocity_margin,
            )
        )
    if metrics.dimensionless_squared_jerk > soft.max_dimensionless_squared_jerk:
        reasons.append(
            _reason(
                JERK_TOO_HIGH,
                "dimensionless_squared_jerk",
                metrics.dimensionless_squared_jerk,
                soft.max_dimensionless_squared_jerk,
            )
        )
    if metrics.sparc < soft.min_sparc:
        reasons.append(_reason(SPARC_TOO_LOW, "sparc", metrics.sparc, soft.min_sparc))
    if metrics.path_length_m > soft.max_path_length_m:
        reasons.append(
            _reason(
                PATH_TOO_LONG,
                "path_length_m",
                metrics.path_length_m,
                soft.max_path_length_m,
            )
        )
    return reasons


def _ordered(reasons: list[RejectionReason]) -> list[RejectionReason]:
    return sorted(
        reasons,
        key=lambda item: (REASON_RANK.get(item.code, len(REASON_RANK)), item.code),
    )


def decide(
    record: Any,
    config: FilterConfig,
    *,
    seen_ids: set[str] | None = None,
) -> FilterResult:
    """Validate, measure, and decide one episode. Never raises for bad input."""
    episode_id = str(getattr(record, "episode_id", "") or "")
    issues = validate_episode(record, config.validation, seen_ids=seen_ids)
    if issues:
        reasons = [_reason(issue.code, issue.field, None, None, issue.message) for issue in issues]
        return FilterResult(
            episode_id=episode_id,
            decision=contracts.DECISION_INVALID,
            acceptance_score=0.0,
            reasons=_ordered(reasons),
            metrics=None,
            issues=issues,
            config_hash=config.config_hash(),
        )

    try:
        metrics = compute_metrics(record, config)
    except DegenerateTrajectoryError as exc:
        return FilterResult(
            episode_id=episode_id,
            decision=contracts.DECISION_REJECT,
            acceptance_score=0.0,
            reasons=[_reason(DEGENERATE_TRAJECTORY, None, None, None, str(exc))],
            metrics=None,
            config_hash=config.config_hash(),
        )
    except Exception as exc:  # noqa: BLE001 - malformed metrics must not crash a batch
        return FilterResult(
            episode_id=episode_id,
            decision=contracts.DECISION_INVALID,
            acceptance_score=0.0,
            reasons=[_reason("MALFORMED_EPISODE", None, None, None, str(exc))],
            metrics=None,
            config_hash=config.config_hash(),
        )

    score = acceptance_score(metrics, config)
    reasons = _hard_and_soft_reasons(metrics, config)
    if not math.isfinite(score):
        reasons.append(_reason("MALFORMED_EPISODE", "acceptance_score", None, None, "non-finite"))
    accepted = not reasons and score >= config.accept_score_threshold
    return FilterResult(
        episode_id=episode_id,
        decision=contracts.DECISION_ACCEPT if accepted else contracts.DECISION_REJECT,
        acceptance_score=score,
        reasons=_ordered(reasons),
        metrics=metrics,
        config_hash=config.config_hash(),
    )


__all__ = [
    "COLLISION",
    "DEGENERATE_TRAJECTORY",
    "FilterResult",
    "JERK_TOO_HIGH",
    "JOINT_LIMIT_MARGIN",
    "NOT_SUCCESS",
    "PATH_TOO_LONG",
    "REASON_ORDER",
    "RejectionReason",
    "SPARC_TOO_LOW",
    "VELOCITY_LIMIT_MARGIN",
    "acceptance_score",
    "decide",
    "higher_is_better_score",
    "lower_is_better_score",
]
