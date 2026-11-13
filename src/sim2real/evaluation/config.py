"""Validated configuration for closed-loop evaluation, failure analysis and the
flywheel guards.

Every threshold lives here so a run can be reproduced from one file plus a seed.
Loading is strict about unknown keys and non-finite values; a typo must not
silently fall back to a default.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = 1
FAILURE_DEFINITIONS = ("policy_only", "any_failure")
EXECUTION_MODES = ("hold", "linear")
DECISIONS = ("REQUEST_TARGETED_GENERATION", "REQUEST_RETRAIN", "STOP")


class EvaluationConfigError(ValueError):
    """Raised when the evaluation config is missing or malformed."""


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvaluationConfigError(f"{path}: expected a mapping")
    return value


def _unknown(mapping: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise EvaluationConfigError(f"{path}: unknown keys {unknown}")


def _number(value: Any, path: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationConfigError(f"{path}: expected a number")
    number = float(value)
    if not math.isfinite(number):
        raise EvaluationConfigError(f"{path}: expected a finite number")
    if minimum is not None and number < minimum:
        raise EvaluationConfigError(f"{path}: expected >= {minimum}")
    return number


def _integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvaluationConfigError(f"{path}: expected an integer")
    if minimum is not None and value < minimum:
        raise EvaluationConfigError(f"{path}: expected >= {minimum}")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise EvaluationConfigError(f"{path}: expected true or false")
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationConfigError(f"{path}: expected a non-empty string")
    return value.strip()


def _pair_int(value: Any, path: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise EvaluationConfigError(f"{path}: expected [low, high]")
    low = _integer(value[0], f"{path}[0]", minimum=0)
    high = _integer(value[1], f"{path}[1]", minimum=0)
    if low > high:
        raise EvaluationConfigError(f"{path}: expected low <= high")
    return (low, high)


@dataclass(frozen=True)
class EvaluationSettings:
    n_action_steps: int = 10
    control_rate_hz: float = 20.0
    policy_step_period_s: float = 0.1
    max_control_steps: int = 600
    collision_terminate: bool = False
    terminate_on_limit: bool = True
    render_images: bool = False
    renderer: str = "llvmpipe"
    execution: str = "linear"
    output_dir: str = "data/eval"
    randomization_profile: str = "default"
    task_instruction: str = "pick up the red cube and place it in the left zone"
    eval_seed: int = 20000
    num_episodes: int = 60
    evaluation_seed_range: tuple[int, int] = (20000, 20999)
    targeted_seed_range: tuple[int, int] = (30000, 30999)

    def __post_init__(self) -> None:
        if self.n_action_steps < 1:
            raise EvaluationConfigError("evaluation.n_action_steps: must be >= 1")
        if self.control_rate_hz <= 0:
            raise EvaluationConfigError("evaluation.control_rate_hz: must be > 0")
        if self.policy_step_period_s <= 0:
            raise EvaluationConfigError("evaluation.policy_step_period_s: must be > 0")
        if self.max_control_steps < 2:
            raise EvaluationConfigError("evaluation.max_control_steps: must be >= 2")
        if self.execution not in EXECUTION_MODES:
            raise EvaluationConfigError(
                f"evaluation.execution: expected one of {list(EXECUTION_MODES)}"
            )
        if self.eval_seed < 0:
            raise EvaluationConfigError("evaluation.eval_seed: must be >= 0")
        if self.num_episodes < 0:
            raise EvaluationConfigError("evaluation.num_episodes: must be >= 0")
        low, high = self.evaluation_seed_range
        if not (low <= self.eval_seed <= high):
            raise EvaluationConfigError(
                "evaluation.eval_seed: must lie inside evaluation.seeds.evaluation"
            )
        if low < 20000 or high > 20999:
            raise EvaluationConfigError(
                "evaluation.seeds.evaluation: must stay inside the reserved 20000-20999 range"
            )
        t_low, t_high = self.targeted_seed_range
        if t_low < 30000 or t_high > 30999:
            raise EvaluationConfigError(
                "evaluation.seeds.targeted_generation: must stay inside 30000-30999"
            )

    @property
    def hold_steps(self) -> int:
        """How many control steps one policy action is held for."""
        value = self.control_rate_hz * self.policy_step_period_s
        rounded = int(round(value))
        if rounded < 1 or abs(value - rounded) > 1e-9:
            raise EvaluationConfigError(
                "evaluation: control_rate_hz * policy_step_period_s must be a positive integer"
            )
        return rounded

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_action_steps": self.n_action_steps,
            "control_rate_hz": self.control_rate_hz,
            "policy_step_period_s": self.policy_step_period_s,
            "max_control_steps": self.max_control_steps,
            "collision_terminate": self.collision_terminate,
            "terminate_on_limit": self.terminate_on_limit,
            "render_images": self.render_images,
            "renderer": self.renderer,
            "execution": self.execution,
            "output_dir": self.output_dir,
            "randomization_profile": self.randomization_profile,
            "task_instruction": self.task_instruction,
            "eval_seed": self.eval_seed,
            "num_episodes": self.num_episodes,
            "evaluation_seed_range": list(self.evaluation_seed_range),
            "targeted_seed_range": list(self.targeted_seed_range),
        }


@dataclass(frozen=True)
class FailureAnalysisConfig:
    failure_definition: str = "policy_only"
    min_failures_gmm: int = 30
    min_failures_kde: int = 10
    max_components: int = 5
    reg_covar: float = 1.0e-4
    bandwidth_floor: float = 0.05
    neighbor_sigma_fraction: float = 0.05
    targeted_fraction: float = 0.7
    max_rejection_tries: int = 1000
    random_state: int = 0
    output_dir: str = "data/failure_distributions"

    def __post_init__(self) -> None:
        if self.failure_definition not in FAILURE_DEFINITIONS:
            raise EvaluationConfigError(
                f"failure_analysis.failure_definition: expected one of {list(FAILURE_DEFINITIONS)}"
            )
        if self.min_failures_gmm < 1 or self.min_failures_kde < 1:
            raise EvaluationConfigError("failure_analysis: minimum counts must be >= 1")
        if self.min_failures_kde > self.min_failures_gmm:
            raise EvaluationConfigError(
                "failure_analysis: min_failures_kde must be <= min_failures_gmm"
            )
        if self.max_components < 1:
            raise EvaluationConfigError("failure_analysis.max_components: must be >= 1")
        if self.reg_covar <= 0 or self.bandwidth_floor <= 0:
            raise EvaluationConfigError(
                "failure_analysis: reg_covar and bandwidth_floor must be > 0"
            )
        if not (0 < self.neighbor_sigma_fraction <= 1):
            raise EvaluationConfigError(
                "failure_analysis.neighbor_sigma_fraction: must be in (0, 1]"
            )
        if not (0 <= self.targeted_fraction <= 1):
            raise EvaluationConfigError("failure_analysis.targeted_fraction: must be in [0, 1]")
        if self.max_rejection_tries < 1:
            raise EvaluationConfigError("failure_analysis.max_rejection_tries: must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_definition": self.failure_definition,
            "min_failures_gmm": self.min_failures_gmm,
            "min_failures_kde": self.min_failures_kde,
            "max_components": self.max_components,
            "reg_covar": self.reg_covar,
            "bandwidth_floor": self.bandwidth_floor,
            "neighbor_sigma_fraction": self.neighbor_sigma_fraction,
            "targeted_fraction": self.targeted_fraction,
            "max_rejection_tries": self.max_rejection_tries,
            "random_state": self.random_state,
            "output_dir": self.output_dir,
        }


@dataclass(frozen=True)
class FlywheelConfig:
    ledger_path: str = "data/flywheel/ledger.jsonl"
    stop_file: str = "data/flywheel/STOP"
    max_cycles: int = 10
    max_total_rollouts: int = 200
    min_failures_to_trigger: int = 3
    min_success_improvement: float = 0.05
    max_consecutive_no_improvement: int = 3
    cooldown_s: float = 0.0
    max_rollouts_per_request: int = 32

    def __post_init__(self) -> None:
        if self.max_cycles < 1:
            raise EvaluationConfigError("flywheel.max_cycles: must be >= 1")
        if self.max_total_rollouts < 1:
            raise EvaluationConfigError("flywheel.max_total_rollouts: must be >= 1")
        if self.min_failures_to_trigger < 1:
            raise EvaluationConfigError("flywheel.min_failures_to_trigger: must be >= 1")
        if not (0 <= self.min_success_improvement <= 1):
            raise EvaluationConfigError("flywheel.min_success_improvement: must be in [0, 1]")
        if self.max_consecutive_no_improvement < 1:
            raise EvaluationConfigError("flywheel.max_consecutive_no_improvement: must be >= 1")
        if self.cooldown_s < 0:
            raise EvaluationConfigError("flywheel.cooldown_s: must be >= 0")
        if self.max_rollouts_per_request < 1:
            raise EvaluationConfigError("flywheel.max_rollouts_per_request: must be >= 1")
        if self.max_rollouts_per_request > self.max_total_rollouts:
            raise EvaluationConfigError(
                "flywheel.max_rollouts_per_request: must be <= max_total_rollouts"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ledger_path": self.ledger_path,
            "stop_file": self.stop_file,
            "max_cycles": self.max_cycles,
            "max_total_rollouts": self.max_total_rollouts,
            "min_failures_to_trigger": self.min_failures_to_trigger,
            "min_success_improvement": self.min_success_improvement,
            "max_consecutive_no_improvement": self.max_consecutive_no_improvement,
            "cooldown_s": self.cooldown_s,
            "max_rollouts_per_request": self.max_rollouts_per_request,
        }


@dataclass(frozen=True)
class EvaluationConfig:
    settings: EvaluationSettings = field(default_factory=EvaluationSettings)
    failure_analysis: FailureAnalysisConfig = field(default_factory=FailureAnalysisConfig)
    flywheel: FlywheelConfig = field(default_factory=FlywheelConfig)
    source_path: str = ""
    raw: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def schema_version(self) -> int:
        return SCHEMA_VERSION

    def config_hash(self) -> str:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "evaluation": self.settings.to_dict(),
            "failure_analysis": self.failure_analysis.to_dict(),
            "flywheel": self.flywheel.to_dict(),
        }
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_settings(data: dict[str, Any], path: str) -> EvaluationSettings:
    allowed = {
        "n_action_steps",
        "control_rate_hz",
        "policy_step_period_s",
        "max_control_steps",
        "collision_terminate",
        "terminate_on_limit",
        "render_images",
        "renderer",
        "execution",
        "output_dir",
        "randomization_profile",
        "task_instruction",
        "eval_seed",
        "num_episodes",
        "seeds",
    }
    _unknown(data, allowed, path)
    seeds = _mapping(data.get("seeds", {}), f"{path}.seeds")
    _unknown(seeds, {"evaluation", "targeted_generation"}, f"{path}.seeds")
    default = EvaluationSettings()
    kwargs = {
        "n_action_steps": _integer(
            data.get("n_action_steps", default.n_action_steps), f"{path}.n_action_steps", minimum=1
        ),
        "control_rate_hz": _number(
            data.get("control_rate_hz", default.control_rate_hz),
            f"{path}.control_rate_hz",
            minimum=1.0,
        ),
        "policy_step_period_s": _number(
            data.get("policy_step_period_s", default.policy_step_period_s),
            f"{path}.policy_step_period_s",
            minimum=1.0e-6,
        ),
        "max_control_steps": _integer(
            data.get("max_control_steps", default.max_control_steps),
            f"{path}.max_control_steps",
            minimum=2,
        ),
        "collision_terminate": _boolean(
            data.get("collision_terminate", default.collision_terminate),
            f"{path}.collision_terminate",
        ),
        "terminate_on_limit": _boolean(
            data.get("terminate_on_limit", default.terminate_on_limit),
            f"{path}.terminate_on_limit",
        ),
        "render_images": _boolean(
            data.get("render_images", default.render_images), f"{path}.render_images"
        ),
        "renderer": _text(data.get("renderer", default.renderer), f"{path}.renderer"),
        "execution": _text(data.get("execution", default.execution), f"{path}.execution"),
        "output_dir": _text(data.get("output_dir", default.output_dir), f"{path}.output_dir"),
        "randomization_profile": _text(
            data.get("randomization_profile", default.randomization_profile),
            f"{path}.randomization_profile",
        ),
        "task_instruction": _text(
            data.get("task_instruction", default.task_instruction), f"{path}.task_instruction"
        ),
        "eval_seed": _integer(
            data.get("eval_seed", default.eval_seed), f"{path}.eval_seed", minimum=0
        ),
        "num_episodes": _integer(
            data.get("num_episodes", default.num_episodes), f"{path}.num_episodes", minimum=0
        ),
        "evaluation_seed_range": _pair_int(
            seeds.get("evaluation", list(default.evaluation_seed_range)),
            f"{path}.seeds.evaluation",
        ),
        "targeted_seed_range": _pair_int(
            seeds.get("targeted_generation", list(default.targeted_seed_range)),
            f"{path}.seeds.targeted_generation",
        ),
    }
    return EvaluationSettings(**kwargs)


def _parse_failure_analysis(data: dict[str, Any], path: str) -> FailureAnalysisConfig:
    _unknown(data, set(FailureAnalysisConfig.__dataclass_fields__), path)
    default = FailureAnalysisConfig()
    definition = data.get("failure_definition", default.failure_definition)
    if definition not in FAILURE_DEFINITIONS:
        raise EvaluationConfigError(
            f"{path}.failure_definition: expected one of {list(FAILURE_DEFINITIONS)}"
        )
    return FailureAnalysisConfig(
        failure_definition=str(definition),
        min_failures_gmm=_integer(
            data.get("min_failures_gmm", default.min_failures_gmm),
            f"{path}.min_failures_gmm",
            minimum=1,
        ),
        min_failures_kde=_integer(
            data.get("min_failures_kde", default.min_failures_kde),
            f"{path}.min_failures_kde",
            minimum=1,
        ),
        max_components=_integer(
            data.get("max_components", default.max_components), f"{path}.max_components", minimum=1
        ),
        reg_covar=_number(
            data.get("reg_covar", default.reg_covar), f"{path}.reg_covar", minimum=1.0e-12
        ),
        bandwidth_floor=_number(
            data.get("bandwidth_floor", default.bandwidth_floor),
            f"{path}.bandwidth_floor",
            minimum=1.0e-12,
        ),
        neighbor_sigma_fraction=_number(
            data.get("neighbor_sigma_fraction", default.neighbor_sigma_fraction),
            f"{path}.neighbor_sigma_fraction",
            minimum=1.0e-12,
        ),
        targeted_fraction=_number(
            data.get("targeted_fraction", default.targeted_fraction),
            f"{path}.targeted_fraction",
        ),
        max_rejection_tries=_integer(
            data.get("max_rejection_tries", default.max_rejection_tries),
            f"{path}.max_rejection_tries",
            minimum=1,
        ),
        random_state=_integer(
            data.get("random_state", default.random_state), f"{path}.random_state"
        ),
        output_dir=_text(data.get("output_dir", default.output_dir), f"{path}.output_dir"),
    )


def _parse_flywheel(data: dict[str, Any], path: str) -> FlywheelConfig:
    _unknown(data, set(FlywheelConfig.__dataclass_fields__), path)
    default = FlywheelConfig()
    return FlywheelConfig(
        ledger_path=_text(data.get("ledger_path", default.ledger_path), f"{path}.ledger_path"),
        stop_file=_text(data.get("stop_file", default.stop_file), f"{path}.stop_file"),
        max_cycles=_integer(
            data.get("max_cycles", default.max_cycles), f"{path}.max_cycles", minimum=1
        ),
        max_total_rollouts=_integer(
            data.get("max_total_rollouts", default.max_total_rollouts),
            f"{path}.max_total_rollouts",
            minimum=1,
        ),
        min_failures_to_trigger=_integer(
            data.get("min_failures_to_trigger", default.min_failures_to_trigger),
            f"{path}.min_failures_to_trigger",
            minimum=1,
        ),
        min_success_improvement=_number(
            data.get("min_success_improvement", default.min_success_improvement),
            f"{path}.min_success_improvement",
        ),
        max_consecutive_no_improvement=_integer(
            data.get("max_consecutive_no_improvement", default.max_consecutive_no_improvement),
            f"{path}.max_consecutive_no_improvement",
            minimum=1,
        ),
        cooldown_s=_number(
            data.get("cooldown_s", default.cooldown_s), f"{path}.cooldown_s", minimum=0.0
        ),
        max_rollouts_per_request=_integer(
            data.get("max_rollouts_per_request", default.max_rollouts_per_request),
            f"{path}.max_rollouts_per_request",
            minimum=1,
        ),
    )


def load_evaluation_config(
    path: str | Path = "configs/evaluation/default.yaml",
) -> EvaluationConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise EvaluationConfigError(f"missing evaluation config: {config_path}")
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise EvaluationConfigError(f"invalid YAML in {config_path}: {exc}") from exc
    root = _mapping(data, str(config_path))
    _unknown(
        root, {"schema_version", "evaluation", "failure_analysis", "flywheel"}, str(config_path)
    )
    if root.get("schema_version") != SCHEMA_VERSION:
        raise EvaluationConfigError(f"{config_path}: schema_version must be {SCHEMA_VERSION}")
    settings = _parse_settings(_mapping(root.get("evaluation", {}), "evaluation"), "evaluation")
    failure = _parse_failure_analysis(
        _mapping(root.get("failure_analysis", {}), "failure_analysis"), "failure_analysis"
    )
    flywheel = _parse_flywheel(_mapping(root.get("flywheel", {}), "flywheel"), "flywheel")
    return EvaluationConfig(
        settings=settings,
        failure_analysis=failure,
        flywheel=flywheel,
        source_path=str(config_path),
        raw=root,
    )


__all__ = [
    "DECISIONS",
    "FAILURE_DEFINITIONS",
    "EXECUTION_MODES",
    "SCHEMA_VERSION",
    "EvaluationConfig",
    "EvaluationConfigError",
    "EvaluationSettings",
    "FailureAnalysisConfig",
    "FlywheelConfig",
    "load_evaluation_config",
]
