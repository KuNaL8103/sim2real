"""Language-conditioned scripted pick-and-place expert and rollout generation.

Pure Python only: no ``rclpy``, no MuJoCo, no rendering. The ROS adapter
(``expert_collector_node``) turns these records into messages.
"""

from sim2real.expert.config import ExpertConfig, ExpertConfigError, load_expert_config
from sim2real.expert.conversion import episode_to_rollout_fields
from sim2real.expert.ik import DampedLeastSquaresIK, IKResult
from sim2real.expert.instruction import (
    TaskSpec,
    UnparseableInstructionError,
    parse_instruction,
)
from sim2real.expert.perturbations import (
    Perturbation,
    PerturbationSet,
    load_perturbations,
    select_perturbation,
)
from sim2real.expert.planner import (
    CartesianInterpolationPlanner,
    JointPlan,
    MoveItPlanner,
    PlannerUnavailable,
)
from sim2real.expert.rollout import (
    EpisodeRecord,
    StepRecord,
    derive_episode_seed,
    run_episode,
    run_episodes,
)
from sim2real.expert.staging import StagingBuffer, StagingError, estimate_storage

__all__ = [
    "CartesianInterpolationPlanner",
    "DampedLeastSquaresIK",
    "EpisodeRecord",
    "ExpertConfig",
    "ExpertConfigError",
    "IKResult",
    "JointPlan",
    "MoveItPlanner",
    "Perturbation",
    "PerturbationSet",
    "PlannerUnavailable",
    "StagingBuffer",
    "StagingError",
    "StepRecord",
    "TaskSpec",
    "UnparseableInstructionError",
    "derive_episode_seed",
    "episode_to_rollout_fields",
    "estimate_storage",
    "load_expert_config",
    "load_perturbations",
    "parse_instruction",
    "run_episode",
    "run_episodes",
    "select_perturbation",
]
