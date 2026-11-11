"""Per-episode and per-run records for closed-loop evaluation.

A record is JSON-serialisable with an explicit ``schema_version``.  Validation
rejects non-finite numbers, wrong shapes, unknown termination reasons, and an
inconsistent ``success`` flag.  ``json.dumps(..., allow_nan=False)`` is the final
guard: a NaN can never be written into a record or a manifest.

``task_completion`` is defined as ``1 - clip(final_distance / start_distance,
0, 1)`` where both distances are the target cube's xy distance to the zone
centre.  It is therefore 1.0 when the cube reaches the zone centre and 0.0 when
it is no closer than it started.  A degenerate start distance (< 1e-9 m) is
handled explicitly rather than dividing by zero.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
TERMINATION_REASONS = (
    "SUCCESS",
    "TIMEOUT",
    "COLLISION",
    "LIMIT_VIOLATION",
    "POLICY_ERROR",
    "BACKEND_ERROR",
    "PHYSICS_INSTABILITY",
)
_DERIVE_SALT = 0x4556414C  # "EVAL"


class EvalRecordError(ValueError):
    """Raised when an evaluation record is malformed."""


def derive_eval_seeds(eval_seed: int, count: int) -> list[int]:
    """Derive a stable list of episode seeds from ``eval_seed``."""
    if isinstance(eval_seed, bool) or not isinstance(eval_seed, int) or eval_seed < 0:
        raise EvalRecordError("eval_seed must be a non-negative integer")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise EvalRecordError("count must be a non-negative integer")
    seeds: list[int] = []
    for index in range(count):
        state = np.random.SeedSequence([eval_seed, index, _DERIVE_SALT]).generate_state(
            1, dtype=np.uint32
        )
        seeds.append(int(state[0]) & 0x7FFFFFFF)
    return seeds


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if total <= 0:
        return (0.0, 0.0)
    if successes < 0 or successes > total:
        raise EvalRecordError("successes must be in [0, total]")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def task_completion(initial_xy: np.ndarray, final_xy: np.ndarray, zone_xy: np.ndarray) -> float:
    start = float(np.linalg.norm(np.asarray(initial_xy) - np.asarray(zone_xy)))
    final = float(np.linalg.norm(np.asarray(final_xy) - np.asarray(zone_xy)))
    if start < 1.0e-9:
        return 1.0 if final < 1.0e-9 else 0.0
    return float(np.clip(1.0 - final / start, 0.0, 1.0))


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise EvalRecordError(f"{name}: expected shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise EvalRecordError(f"{name}: values must be finite")
    return array


@dataclass
class EvalStep:
    stamp_ns: int
    step_index: int
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    ee_pose: np.ndarray
    action: np.ndarray
    gripper_position: float
    unintended_collision: bool

    def __post_init__(self) -> None:
        self.joint_positions = _array(self.joint_positions, (7,), "joint_positions")
        self.joint_velocities = _array(self.joint_velocities, (7,), "joint_velocities")
        self.ee_pose = _array(self.ee_pose, (4, 4), "ee_pose")
        self.action = _array(self.action, (8,), "action")
        if not math.isfinite(float(self.gripper_position)):
            raise EvalRecordError("gripper_position must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stamp_ns": int(self.stamp_ns),
            "step_index": int(self.step_index),
            "joint_positions": [float(v) for v in self.joint_positions],
            "joint_velocities": [float(v) for v in self.joint_velocities],
            "ee_pose": [[float(v) for v in row] for row in self.ee_pose],
            "action": [float(v) for v in self.action],
            "gripper_position": float(self.gripper_position),
            "unintended_collision": bool(self.unintended_collision),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalStep:
        return cls(
            stamp_ns=int(data["stamp_ns"]),
            step_index=int(data["step_index"]),
            joint_positions=data["joint_positions"],
            joint_velocities=data["joint_velocities"],
            ee_pose=data["ee_pose"],
            action=data["action"],
            gripper_position=float(data["gripper_position"]),
            unintended_collision=bool(data["unintended_collision"]),
        )


@dataclass
class EvalEpisodeRecord:
    episode_id: str
    eval_run_id: str
    seed: int
    episode_index: int
    randomization_profile: str
    randomization: dict[str, Any]
    initial_object_pose: tuple[float, float, float]
    friction_sliding: float
    object_mass: float
    light_intensity: float
    color_temperature_k: float
    camera_jitter_norm: float
    task_instruction: str
    success: bool
    task_completion: float
    collision_count: int
    termination_reason: str
    policy_id: str
    checkpoint_hash: str
    config_hash: str
    git_commit: str
    git_dirty: bool
    n_action_steps: int
    renderer: str
    metrics: dict[str, Any] | None = None
    metrics_error: str = ""
    failure: dict[str, Any] = field(default_factory=dict)
    steps: list[EvalStep] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    @property
    def outcome(self) -> int:
        """Task-5-compatible success code for ``compute_metrics``."""
        return 1 if self.success else 2

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise EvalRecordError(f"unsupported schema_version {self.schema_version}")
        if not self.episode_id or not self.eval_run_id:
            raise EvalRecordError("episode_id and eval_run_id must be non-empty")
        if self.seed < 0 or self.episode_index < 0:
            raise EvalRecordError("seed and episode_index must be >= 0")
        if self.termination_reason not in TERMINATION_REASONS:
            raise EvalRecordError(
                f"invalid termination_reason {self.termination_reason!r}; "
                f"expected one of {list(TERMINATION_REASONS)}"
            )
        if bool(self.success) != (self.termination_reason == "SUCCESS"):
            raise EvalRecordError("success must be true exactly when termination_reason is SUCCESS")
        completion = float(self.task_completion)
        if not math.isfinite(completion) or not (0.0 <= completion <= 1.0):
            raise EvalRecordError("task_completion must be finite and in [0, 1]")
        if self.collision_count < 0:
            raise EvalRecordError("collision_count must be >= 0")
        if self.n_action_steps < 1:
            raise EvalRecordError("n_action_steps must be >= 1")
        for name in (
            "friction_sliding",
            "object_mass",
            "light_intensity",
            "color_temperature_k",
            "camera_jitter_norm",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise EvalRecordError(f"{name} must be finite")
        if len(self.initial_object_pose) != 3 or not all(
            math.isfinite(float(v)) for v in self.initial_object_pose
        ):
            raise EvalRecordError("initial_object_pose must be three finite values")
        for step in self.steps:
            step.__post_init__()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "episode_id": self.episode_id,
            "eval_run_id": self.eval_run_id,
            "seed": int(self.seed),
            "episode_index": int(self.episode_index),
            "randomization_profile": self.randomization_profile,
            "randomization": self.randomization,
            "initial_object_pose": [float(v) for v in self.initial_object_pose],
            "friction_sliding": float(self.friction_sliding),
            "object_mass": float(self.object_mass),
            "light_intensity": float(self.light_intensity),
            "color_temperature_k": float(self.color_temperature_k),
            "camera_jitter_norm": float(self.camera_jitter_norm),
            "task_instruction": self.task_instruction,
            "success": bool(self.success),
            "task_completion": float(self.task_completion),
            "collision_count": int(self.collision_count),
            "termination_reason": self.termination_reason,
            "metrics": self.metrics,
            "metrics_error": self.metrics_error,
            "failure": self.failure,
            "policy_id": self.policy_id,
            "checkpoint_hash": self.checkpoint_hash,
            "config_hash": self.config_hash,
            "git_commit": self.git_commit,
            "git_dirty": bool(self.git_dirty),
            "n_action_steps": int(self.n_action_steps),
            "renderer": self.renderer,
            "steps": [step.to_dict() for step in self.steps],
        }

    def to_json(self) -> str:
        text = json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)
        validate_record_dict(json.loads(text))
        return text

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalEpisodeRecord:
        required = {
            "schema_version",
            "episode_id",
            "eval_run_id",
            "seed",
            "episode_index",
            "randomization_profile",
            "randomization",
            "initial_object_pose",
            "friction_sliding",
            "object_mass",
            "light_intensity",
            "color_temperature_k",
            "camera_jitter_norm",
            "task_instruction",
            "success",
            "task_completion",
            "collision_count",
            "termination_reason",
            "policy_id",
            "checkpoint_hash",
            "config_hash",
            "git_commit",
            "git_dirty",
            "n_action_steps",
            "renderer",
            "steps",
        }
        missing = sorted(required - set(data))
        if missing:
            raise EvalRecordError(f"record missing required fields: {missing}")
        return cls(
            schema_version=int(data["schema_version"]),
            episode_id=str(data["episode_id"]),
            eval_run_id=str(data["eval_run_id"]),
            seed=int(data["seed"]),
            episode_index=int(data["episode_index"]),
            randomization_profile=str(data["randomization_profile"]),
            randomization=dict(data["randomization"]),
            initial_object_pose=tuple(float(v) for v in data["initial_object_pose"]),
            friction_sliding=float(data["friction_sliding"]),
            object_mass=float(data["object_mass"]),
            light_intensity=float(data["light_intensity"]),
            color_temperature_k=float(data["color_temperature_k"]),
            camera_jitter_norm=float(data["camera_jitter_norm"]),
            task_instruction=str(data["task_instruction"]),
            success=bool(data["success"]),
            task_completion=float(data["task_completion"]),
            collision_count=int(data["collision_count"]),
            termination_reason=str(data["termination_reason"]),
            metrics=data.get("metrics"),
            metrics_error=str(data.get("metrics_error", "")),
            failure=dict(data.get("failure", {})),
            policy_id=str(data["policy_id"]),
            checkpoint_hash=str(data["checkpoint_hash"]),
            config_hash=str(data["config_hash"]),
            git_commit=str(data["git_commit"]),
            git_dirty=bool(data["git_dirty"]),
            n_action_steps=int(data["n_action_steps"]),
            renderer=str(data["renderer"]),
            steps=[EvalStep.from_dict(item) for item in data["steps"]],
        )

    @classmethod
    def from_json(cls, text: str) -> EvalEpisodeRecord:
        return cls.from_dict(json.loads(text))


def validate_record_dict(data: dict[str, Any]) -> None:
    """Validate a plain record dict without rebuilding NumPy arrays."""
    EvalEpisodeRecord.from_dict(data)


def metrics_for_record(record: EvalEpisodeRecord, filter_config: Any) -> dict[str, Any]:
    """Reuse Task 5 metrics; return a DEGENERATE marker instead of raising.

    Imports are lazy so the evaluation package (and the ROS targeted-generation
    path) can be imported where scipy is unavailable.
    """
    from sim2real.filtering.metrics import DegenerateTrajectoryError, compute_metrics

    if len(record.steps) < 2:
        return {"degenerate": True, "reason": "episode is too short for metrics"}
    try:
        metrics = compute_metrics(record, filter_config)
    except DegenerateTrajectoryError as exc:
        return {"degenerate": True, "reason": str(exc)}
    payload = metrics.to_dict()
    payload["degenerate"] = False
    return payload


@dataclass
class EvalRun:
    run_id: str
    config_hash: str
    eval_seed: int
    seeds: list[int]
    policy_id: str
    policy_info: dict[str, Any]
    randomization_profile: str
    n_action_steps: int
    renderer: str
    created_at: str = ""
    aggregates: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise EvalRecordError("unsupported run schema_version")
        if not self.run_id:
            raise EvalRecordError("run_id must be non-empty")
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "eval_seed": int(self.eval_seed),
            "seeds": [int(seed) for seed in self.seeds],
            "policy_id": self.policy_id,
            "policy_info": self.policy_info,
            "randomization_profile": self.randomization_profile,
            "n_action_steps": int(self.n_action_steps),
            "renderer": self.renderer,
            "created_at": self.created_at,
            "aggregates": self.aggregates,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)


def aggregate_records(records: list[EvalEpisodeRecord]) -> dict[str, Any]:
    """Success rate with a Wilson interval, termination counts, mean completion."""
    total = len(records)
    successes = sum(1 for record in records if record.success)
    counts: dict[str, int] = {}
    for record in records:
        counts[record.termination_reason] = counts.get(record.termination_reason, 0) + 1
    low, high = wilson_interval(successes, total)
    mean_completion = (
        float(np.mean([record.task_completion for record in records])) if records else 0.0
    )
    return {
        "num_episodes": total,
        "num_success": successes,
        "success_rate": (successes / total) if total else 0.0,
        "wilson_95": [low, high],
        "termination_counts": dict(sorted(counts.items())),
        "mean_task_completion": mean_completion,
    }


__all__ = [
    "SCHEMA_VERSION",
    "TERMINATION_REASONS",
    "EvalEpisodeRecord",
    "EvalRecordError",
    "EvalRun",
    "EvalStep",
    "aggregate_records",
    "derive_eval_seeds",
    "metrics_for_record",
    "task_completion",
    "validate_record_dict",
    "wilson_interval",
]
