"""Strict pipeline-orchestrator configuration.

One YAML file describes the pipeline stages, their executors, and every hard
bound.  Unknown keys, wrong types, non-finite numbers, and out-of-range values
are rejected with a path-qualified message, so a typo cannot silently change
what runs.

The orchestrator never reads a value it did not validate here.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sim2real.orchestrator.stages import EXECUTORS, STAGE_ORDER, Stage

SCHEMA_VERSION = 1
BUILD_MODES = ("core_validation", "lerobot")
EVAL_POLICIES = ("zero", "random", "scripted", "noisy_expert", "checkpoint")
GEN_BACKENDS = ("fake", "mujoco")


class PipelineConfigError(ValueError):
    """Raised when a pipeline config is missing, malformed, or out of bounds."""


def _mapping(value: Any, path: str, *, allowed: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PipelineConfigError(f"{path}: expected a mapping, got {type(value).__name__}")
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise PipelineConfigError(
            f"{path}: unknown key(s) {unknown}; allowed keys are {sorted(allowed)}"
        )
    return value


def _require(mapping: dict[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise PipelineConfigError(f"{path}.{key}: missing required key")
    return mapping[key]


def _int(value: Any, path: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PipelineConfigError(f"{path}: expected an integer, got {type(value).__name__}")
    if value < minimum or value > maximum:
        raise PipelineConfigError(f"{path}: expected {minimum} <= value <= {maximum}, got {value}")
    return value


def _float(value: Any, path: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PipelineConfigError(f"{path}: expected a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise PipelineConfigError(f"{path}: expected a finite number")
    if number < minimum or number > maximum:
        raise PipelineConfigError(f"{path}: expected {minimum} <= value <= {maximum}, got {number}")
    return number


def _str(value: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise PipelineConfigError(f"{path}: expected a string, got {type(value).__name__}")
    if not allow_empty and not value.strip():
        raise PipelineConfigError(f"{path}: expected a non-empty string")
    return value


def _bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise PipelineConfigError(f"{path}: expected a boolean, got {type(value).__name__}")
    return value


def _enum(value: Any, path: str, choices: tuple[str, ...]) -> str:
    text = _str(value, path)
    if text not in choices:
        raise PipelineConfigError(f"{path}: expected one of {list(choices)}, got {text!r}")
    return text


@dataclass(frozen=True)
class GenerationConfig:
    backend: str
    num_episodes: int
    seed: int
    profile: str
    perturbed_fraction: float
    instruction: str


@dataclass(frozen=True)
class FilteringConfig:
    config: str
    dataset_name: str


@dataclass(frozen=True)
class BuildConfig:
    mode: str
    config: str
    embodiments: str
    decimation: int
    val_fraction: float
    output: str
    estimated_write_bytes: int


@dataclass(frozen=True)
class TrainingConfig:
    config: str
    checkpoint: str
    gpu_estimated_s: float
    estimated_download_bytes: int


@dataclass(frozen=True)
class EvaluationStageConfig:
    config: str
    policy: str
    paired_expert: bool
    num_episodes: int
    seed: int
    ledger_path: str
    stop_file: str


@dataclass(frozen=True)
class FailureAnalysisConfig:
    config: str


@dataclass(frozen=True)
class TargetedGenerationConfig:
    num_rollouts: int
    seed: int


@dataclass(frozen=True)
class PipelineConfig:
    source_path: str
    fingerprint: str
    run_root: str
    max_cycles: int
    max_episodes_per_cycle: int
    max_wall_time_s: float
    max_write_bytes: int
    gpu_time_limit_s: float
    generation: GenerationConfig
    filtering: FilteringConfig
    build: BuildConfig
    training: TrainingConfig
    evaluation: EvaluationStageConfig
    failure_analysis: FailureAnalysisConfig
    targeted_generate: TargetedGenerationConfig
    stages: dict[str, str]

    def executor(self, stage: Stage) -> str:
        return self.stages[stage.value]

    def stage_config_hash(self, stage: Stage) -> str:
        """Hash only the config that can change this stage's result.

        A global hash would invalidate every stage whenever an unrelated
        executor changes, which would defeat resume.  The run manifest still
        records the global fingerprint.
        """
        sections: dict[str, Any] = {
            Stage.GENERATE: {"generation": _asdict(self.generation)},
            Stage.FILTER: {"filtering": _asdict(self.filtering)},
            Stage.BUILD: {"build": _asdict(self.build)},
            Stage.TRAIN: {"training": _asdict(self.training)},
            Stage.EVALUATE: {"evaluation": _asdict(self.evaluation)},
            Stage.ANALYZE_FAILURES: {
                "failure_analysis": _asdict(self.failure_analysis),
                "evaluation": _asdict(self.evaluation),
            },
            Stage.TARGETED_GENERATE: {"targeted_generate": _asdict(self.targeted_generate)},
        }[stage]
        sections["executor"] = self.executor(stage)
        sections["schema_version"] = SCHEMA_VERSION
        return _sha256_json(sections)


def _asdict(value: Any) -> dict[str, Any]:
    return {key: getattr(value, key) for key in value.__dataclass_fields__}


def _sha256_json(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_TOP_KEYS = (
    "schema_version",
    "pipeline",
)
_PIPELINE_KEYS = (
    "run_root",
    "max_cycles",
    "max_episodes_per_cycle",
    "max_wall_time_s",
    "max_write_bytes",
    "gpu_time_limit_s",
    "generation",
    "filtering",
    "build",
    "training",
    "evaluation",
    "failure_analysis",
    "targeted_generate",
    "stages",
)


def parse_config(raw: dict[str, Any], *, source_path: str = "<memory>") -> PipelineConfig:
    """Validate a raw YAML mapping and return a typed config."""
    top = _mapping(raw, "pipeline config", allowed=_TOP_KEYS)
    schema_version = _int(
        _require(top, "schema_version", "pipeline"), "schema_version", minimum=1, maximum=1
    )
    del schema_version
    body = _mapping(_require(top, "pipeline", "pipeline"), "pipeline", allowed=_PIPELINE_KEYS)

    run_root = _str(_require(body, "run_root", "pipeline"), "pipeline.run_root")
    max_cycles = _int(
        _require(body, "max_cycles", "pipeline"), "pipeline.max_cycles", minimum=1, maximum=100
    )
    max_episodes = _int(
        _require(body, "max_episodes_per_cycle", "pipeline"),
        "pipeline.max_episodes_per_cycle",
        minimum=1,
        maximum=1000,
    )
    max_wall_time_s = _float(
        _require(body, "max_wall_time_s", "pipeline"),
        "pipeline.max_wall_time_s",
        minimum=0.001,
        maximum=86400.0,
    )
    max_write_bytes = _int(
        _require(body, "max_write_bytes", "pipeline"),
        "pipeline.max_write_bytes",
        minimum=1,
        maximum=10**12,
    )
    gpu_time_limit_s = _float(
        _require(body, "gpu_time_limit_s", "pipeline"),
        "pipeline.gpu_time_limit_s",
        minimum=0.0,
        maximum=86400.0,
    )

    generation_raw = _mapping(
        _require(body, "generation", "pipeline"),
        "pipeline.generation",
        allowed=(
            "backend",
            "num_episodes",
            "seed",
            "profile",
            "perturbed_fraction",
            "instruction",
        ),
    )
    generation = GenerationConfig(
        backend=_enum(
            _require(generation_raw, "backend", "pipeline.generation"),
            "pipeline.generation.backend",
            GEN_BACKENDS,
        ),
        num_episodes=_int(
            _require(generation_raw, "num_episodes", "pipeline.generation"),
            "pipeline.generation.num_episodes",
            minimum=1,
            maximum=max_episodes,
        ),
        seed=_int(
            _require(generation_raw, "seed", "pipeline.generation"),
            "pipeline.generation.seed",
            minimum=0,
            maximum=10**9,
        ),
        profile=_str(
            _require(generation_raw, "profile", "pipeline.generation"),
            "pipeline.generation.profile",
        ),
        perturbed_fraction=_float(
            _require(generation_raw, "perturbed_fraction", "pipeline.generation"),
            "pipeline.generation.perturbed_fraction",
            minimum=0.0,
            maximum=1.0,
        ),
        instruction=_str(
            _require(generation_raw, "instruction", "pipeline.generation"),
            "pipeline.generation.instruction",
        ),
    )

    filtering_raw = _mapping(
        _require(body, "filtering", "pipeline"),
        "pipeline.filtering",
        allowed=("config", "dataset_name"),
    )
    filtering = FilteringConfig(
        config=_str(
            _require(filtering_raw, "config", "pipeline.filtering"), "pipeline.filtering.config"
        ),
        dataset_name=_str(
            _require(filtering_raw, "dataset_name", "pipeline.filtering"),
            "pipeline.filtering.dataset_name",
        ),
    )

    build_raw = _mapping(
        _require(body, "build", "pipeline"),
        "pipeline.build",
        allowed=(
            "mode",
            "config",
            "embodiments",
            "decimation",
            "val_fraction",
            "output",
            "estimated_write_bytes",
        ),
    )
    build = BuildConfig(
        mode=_enum(
            _require(build_raw, "mode", "pipeline.build"), "pipeline.build.mode", BUILD_MODES
        ),
        config=_str(_require(build_raw, "config", "pipeline.build"), "pipeline.build.config"),
        embodiments=_str(
            _require(build_raw, "embodiments", "pipeline.build"), "pipeline.build.embodiments"
        ),
        decimation=_int(
            _require(build_raw, "decimation", "pipeline.build"),
            "pipeline.build.decimation",
            minimum=1,
            maximum=20,
        ),
        val_fraction=_float(
            _require(build_raw, "val_fraction", "pipeline.build"),
            "pipeline.build.val_fraction",
            minimum=0.0,
            maximum=0.9,
        ),
        output=_str(_require(build_raw, "output", "pipeline.build"), "pipeline.build.output"),
        estimated_write_bytes=_int(
            _require(build_raw, "estimated_write_bytes", "pipeline.build"),
            "pipeline.build.estimated_write_bytes",
            minimum=0,
            maximum=10**12,
        ),
    )

    training_raw = _mapping(
        _require(body, "training", "pipeline"),
        "pipeline.training",
        allowed=("config", "checkpoint", "gpu_estimated_s", "estimated_download_bytes"),
    )
    training = TrainingConfig(
        config=_str(
            _require(training_raw, "config", "pipeline.training"), "pipeline.training.config"
        ),
        checkpoint=_str(
            _require(training_raw, "checkpoint", "pipeline.training"),
            "pipeline.training.checkpoint",
            allow_empty=True,
        ),
        gpu_estimated_s=_float(
            _require(training_raw, "gpu_estimated_s", "pipeline.training"),
            "pipeline.training.gpu_estimated_s",
            minimum=0.0,
            maximum=86400.0,
        ),
        estimated_download_bytes=_int(
            _require(training_raw, "estimated_download_bytes", "pipeline.training"),
            "pipeline.training.estimated_download_bytes",
            minimum=0,
            maximum=10**12,
        ),
    )

    evaluation_raw = _mapping(
        _require(body, "evaluation", "pipeline"),
        "pipeline.evaluation",
        allowed=(
            "config",
            "policy",
            "paired_expert",
            "num_episodes",
            "seed",
            "ledger_path",
            "stop_file",
        ),
    )
    evaluation = EvaluationStageConfig(
        config=_str(
            _require(evaluation_raw, "config", "pipeline.evaluation"), "pipeline.evaluation.config"
        ),
        policy=_enum(
            _require(evaluation_raw, "policy", "pipeline.evaluation"),
            "pipeline.evaluation.policy",
            EVAL_POLICIES,
        ),
        paired_expert=_bool(
            _require(evaluation_raw, "paired_expert", "pipeline.evaluation"),
            "pipeline.evaluation.paired_expert",
        ),
        num_episodes=_int(
            _require(evaluation_raw, "num_episodes", "pipeline.evaluation"),
            "pipeline.evaluation.num_episodes",
            minimum=1,
            maximum=max_episodes,
        ),
        seed=_int(
            _require(evaluation_raw, "seed", "pipeline.evaluation"),
            "pipeline.evaluation.seed",
            minimum=0,
            maximum=10**9,
        ),
        ledger_path=_str(
            _require(evaluation_raw, "ledger_path", "pipeline.evaluation"),
            "pipeline.evaluation.ledger_path",
            allow_empty=True,
        ),
        stop_file=_str(
            _require(evaluation_raw, "stop_file", "pipeline.evaluation"),
            "pipeline.evaluation.stop_file",
            allow_empty=True,
        ),
    )

    analysis_raw = _mapping(
        _require(body, "failure_analysis", "pipeline"),
        "pipeline.failure_analysis",
        allowed=("config",),
    )
    failure_analysis = FailureAnalysisConfig(
        config=_str(
            _require(analysis_raw, "config", "pipeline.failure_analysis"),
            "pipeline.failure_analysis.config",
        )
    )

    targeted_raw = _mapping(
        _require(body, "targeted_generate", "pipeline"),
        "pipeline.targeted_generate",
        allowed=("num_rollouts", "seed"),
    )
    targeted = TargetedGenerationConfig(
        num_rollouts=_int(
            _require(targeted_raw, "num_rollouts", "pipeline.targeted_generate"),
            "pipeline.targeted_generate.num_rollouts",
            minimum=1,
            maximum=max_episodes,
        ),
        seed=_int(
            _require(targeted_raw, "seed", "pipeline.targeted_generate"),
            "pipeline.targeted_generate.seed",
            minimum=30000,
            maximum=30999,
        ),
    )

    stages_raw = _mapping(
        _require(body, "stages", "pipeline"),
        "pipeline.stages",
        allowed=tuple(stage.value for stage in STAGE_ORDER),
    )
    stages: dict[str, str] = {}
    for stage in STAGE_ORDER:
        stages[stage.value] = _enum(
            _require(stages_raw, stage.value, "pipeline.stages"),
            f"pipeline.stages.{stage.value}",
            EXECUTORS,
        )

    if build.mode == "lerobot" and stages[Stage.BUILD.value] != "local":
        raise PipelineConfigError(
            "pipeline.build.mode='lerobot' requires pipeline.stages.BUILD='local' "
            "(the fake executor cannot write a real LeRobot dataset)"
        )
    if training.checkpoint and stages[Stage.TRAIN.value] == "skip":
        raise PipelineConfigError(
            "pipeline.training.checkpoint is set but TRAIN is 'skip'; "
            "refusing to silently discard a supplied checkpoint"
        )

    fingerprint = _sha256_json(raw)
    return PipelineConfig(
        source_path=source_path,
        fingerprint=fingerprint,
        run_root=run_root,
        max_cycles=max_cycles,
        max_episodes_per_cycle=max_episodes,
        max_wall_time_s=max_wall_time_s,
        max_write_bytes=max_write_bytes,
        gpu_time_limit_s=gpu_time_limit_s,
        generation=generation,
        filtering=filtering,
        build=build,
        training=training,
        evaluation=evaluation,
        failure_analysis=failure_analysis,
        targeted_generate=targeted,
        stages=stages,
    )


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    source = Path(path).expanduser()
    if not source.is_file():
        raise PipelineConfigError(f"pipeline config not found: {source}")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - malformed YAML
        raise PipelineConfigError(f"{source}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise PipelineConfigError(f"{source}: expected a YAML mapping at the top level")
    return parse_config(raw, source_path=str(source.resolve()))


__all__ = [
    "BuildConfig",
    "EvaluationStageConfig",
    "FailureAnalysisConfig",
    "FilteringConfig",
    "GenerationConfig",
    "PipelineConfig",
    "PipelineConfigError",
    "TargetedGenerationConfig",
    "TrainingConfig",
    "load_pipeline_config",
    "parse_config",
]
