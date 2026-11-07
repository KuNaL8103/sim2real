"""Offline VLA fine-tuning (Task 8).

The package imports without torch, PEFT, LeRobot, ROS, or a GPU.  The real
SmolVLA adapter imports those lazily and raises ``AdapterUnavailable`` with an
install hint when the optional ``.venv-ml`` environment is absent.
"""

from .actions import ActionChunk, ActionChunkError, encode_chunk, horizon_seconds
from .adapters import (
    AdapterError,
    AdapterUnavailable,
    FakePolicyAdapter,
    OpenVLAAdapter,
    PolicyAdapter,
    SmolVLAAdapter,
    make_adapter,
)
from .checkpoint import (
    CheckpointError,
    build_metadata,
    config_hash,
    dataset_fingerprint,
    load_checkpoint,
    save_checkpoint,
)
from .config import (
    ConfigError,
    DataConfig,
    LoraConfig,
    ModelConfig,
    TrackingConfig,
    TrainingConfig,
    TrainingParams,
    load_training_config,
    parse_training_config,
)
from .lora import (
    DEFAULT_SMOLVLA_TARGET_MODULES,
    LoraTargetError,
    resolve_target_modules,
    trainable_parameter_report,
)
from .normalization import (
    FeatureStats,
    NormalizationBundle,
    NormalizationError,
    Normalizer,
    bundle_from_stats,
)
from .tracking import TrackingError, make_tracker
from .trainer import RunResult, Trainer, TrainerError, load_split_manifest, select_episode_ids
from .validation import L1Report, ValidationError, evaluate_l1

__all__ = [
    "ActionChunk",
    "ActionChunkError",
    "AdapterError",
    "AdapterUnavailable",
    "CheckpointError",
    "ConfigError",
    "DEFAULT_SMOLVLA_TARGET_MODULES",
    "DataConfig",
    "FakePolicyAdapter",
    "FeatureStats",
    "L1Report",
    "LoraConfig",
    "LoraTargetError",
    "ModelConfig",
    "NormalizationBundle",
    "NormalizationError",
    "Normalizer",
    "OpenVLAAdapter",
    "PolicyAdapter",
    "RunResult",
    "SmolVLAAdapter",
    "TrackingConfig",
    "TrackingError",
    "Trainer",
    "TrainerError",
    "TrainingConfig",
    "TrainingParams",
    "ValidationError",
    "build_metadata",
    "bundle_from_stats",
    "config_hash",
    "dataset_fingerprint",
    "encode_chunk",
    "evaluate_l1",
    "horizon_seconds",
    "load_checkpoint",
    "load_split_manifest",
    "load_training_config",
    "make_adapter",
    "make_tracker",
    "parse_training_config",
    "resolve_target_modules",
    "save_checkpoint",
    "select_episode_ids",
    "trainable_parameter_report",
]
