"""Explicit, bounded pipeline orchestration.

Stages: GENERATE, FILTER, BUILD, TRAIN, EVALUATE, ANALYZE_FAILURES,
TARGETED_GENERATE.  The package is ROS-free and has no torch dependency.
"""

from sim2real.orchestrator.config import (
    PipelineConfig,
    PipelineConfigError,
    load_pipeline_config,
)
from sim2real.orchestrator.pipeline_orchestrator import (
    CYCLE_APPROVAL_FLAG,
    SUPPLY_CHECKPOINT_FLAG,
    ApprovalRequiredError,
    CheckpointVerificationError,
    CorruptStageOutputError,
    CycleLimitError,
    OrchestratorError,
    PipelineOrchestrator,
    StageError,
    StageRecord,
)
from sim2real.orchestrator.seed_ledger import (
    SeedLedgerError,
    check_evaluation_seed,
    check_generation_seeds,
    check_targeted_seeds,
    describe_ledger,
)
from sim2real.orchestrator.stages import (
    EXECUTORS,
    STAGE_ORDER,
    STAGE_SPECS,
    Stage,
    StageSpec,
    StageStatus,
)

__all__ = [
    "CYCLE_APPROVAL_FLAG",
    "EXECUTORS",
    "STAGE_ORDER",
    "STAGE_SPECS",
    "SUPPLY_CHECKPOINT_FLAG",
    "ApprovalRequiredError",
    "CheckpointVerificationError",
    "CorruptStageOutputError",
    "CycleLimitError",
    "OrchestratorError",
    "PipelineConfig",
    "PipelineConfigError",
    "PipelineOrchestrator",
    "SeedLedgerError",
    "Stage",
    "StageError",
    "StageRecord",
    "StageSpec",
    "StageStatus",
    "check_evaluation_seed",
    "check_generation_seeds",
    "check_targeted_seeds",
    "describe_ledger",
    "load_pipeline_config",
]
