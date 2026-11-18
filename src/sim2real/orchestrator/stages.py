"""Pipeline stage declarations shared by the config and the orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Stage(StrEnum):
    GENERATE = "GENERATE"
    FILTER = "FILTER"
    BUILD = "BUILD"
    TRAIN = "TRAIN"
    EVALUATE = "EVALUATE"
    ANALYZE_FAILURES = "ANALYZE_FAILURES"
    TARGETED_GENERATE = "TARGETED_GENERATE"


STAGE_ORDER: tuple[Stage, ...] = (
    Stage.GENERATE,
    Stage.FILTER,
    Stage.BUILD,
    Stage.TRAIN,
    Stage.EVALUATE,
    Stage.ANALYZE_FAILURES,
    Stage.TARGETED_GENERATE,
)

EXECUTORS = ("local", "external", "fake", "skip")

#: Executors that may not proceed silently.
APPROVAL_EXECUTORS = ("external",)


class StageStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    SKIPPED = "SKIPPED"  # completed earlier, inputs unchanged
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"  # an approval/checkpoint gate stopped the stage
    QUARANTINED = "QUARANTINED"  # output failed integrity checks


@dataclass(frozen=True)
class StageSpec:
    environment: str
    description: str
    inputs: str
    outputs: str
    uses_gpu: bool


STAGE_SPECS: dict[Stage, StageSpec] = {
    Stage.GENERATE: StageSpec(
        environment=".venv",
        description="Seeded randomization + scripted-expert collection into staging",
        inputs="pipeline.generation, configs/randomization, configs/expert, configs/sim",
        outputs="artifacts/GENERATE/staging/<run>/ (episode NPZ + sidecar + manifest)",
        uses_gpu=False,
    ),
    Stage.FILTER: StageSpec(
        environment=".venv",
        description="Trajectory filtering into curated_v1 + rejection log",
        inputs="GENERATE staging, configs/filtering",
        outputs="curated/<dataset>/v<M.m.p>/, rejected/<run>/rejections.jsonl",
        uses_gpu=False,
    ),
    Stage.BUILD: StageSpec(
        environment=".venv",
        description=(
            "Unified dataset validation, stats/split, LeRobot build or mixing "
            "(lerobot mode runs in .venv-ml)"
        ),
        inputs="FILTER curated version, configs/dataset",
        outputs="artifacts/BUILD/summary.json (+ a LeRobot dataset in lerobot mode)",
        uses_gpu=False,
    ),
    Stage.TRAIN: StageSpec(
        environment="external GPU host (.venv-ml equivalent)",
        description="LoRA fine-tuning (external executor) or a labelled test double",
        inputs="BUILD dataset hash, configs/training",
        outputs="checkpoint directory + checkpoint_metadata.json (verified)",
        uses_gpu=True,
    ),
    Stage.EVALUATE: StageSpec(
        environment=".venv",
        description="Closed-loop evaluation with a fake/expert/checkpoint policy",
        inputs="checkpoint, configs/evaluation, configs/sim",
        outputs="artifacts/EVALUATE/eval/{records.jsonl,manifest.json,pairs.json}",
        uses_gpu=False,
    ),
    Stage.ANALYZE_FAILURES: StageSpec(
        environment=".venv",
        description="Failure-distribution fit (GMM/KDE/neighbour/none)",
        inputs="EVALUATE records + pairs, configs/evaluation",
        outputs="artifacts/ANALYZE_FAILURES/<distribution>.json",
        uses_gpu=False,
    ),
    Stage.TARGETED_GENERATE: StageSpec(
        environment=".venv",
        description="Failure-targeted rollout generation with seeds 30000-30999",
        inputs="ANALYZE_FAILURES artifact, configs/randomization, configs/expert",
        outputs="artifacts/TARGETED_GENERATE/staging/<run>/",
        uses_gpu=False,
    ),
}


__all__ = [
    "APPROVAL_EXECUTORS",
    "EXECUTORS",
    "STAGE_ORDER",
    "STAGE_SPECS",
    "Stage",
    "StageSpec",
    "StageStatus",
]
