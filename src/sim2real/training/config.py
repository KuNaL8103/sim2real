"""Validated configuration for offline VLA fine-tuning.

This module is NumPy-free and torch-free.  It only parses and validates YAML so
the ROS-free core suite and CI can import it without the optional ML
environment.  Every error is path-qualified (for example
``training.batch_size: expected >= 1``) so a typo cannot silently fall back to a
default.

The verified SmolVLA facts encoded here come from LeRobot 0.6.1 and the
``lerobot/smolvla_base`` checkpoint (revision
``5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6``); see ``docs/TRAINING.md``.
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
FAMILIES = ("smolvla", "openvla7b")
PRECISIONS = ("fp32", "bf16", "fp16")
ACTION_REPRESENTATIONS = ("joint_position", "ee_delta")
LORA_BIASES = ("none", "all", "lora_only")
TRACKING_BACKENDS = ("none", "wandb", "mlflow")

# Verified keys of lerobot/smolvla_base: camera1..camera3.  The default training
# config feeds the two cameras this repository actually records (overhead and
# wrist) into camera1 and camera2; the third slot is absent.  This is recorded
# as a limitation in docs/TRAINING.md, not hidden.
SMOLVLA_CAMERA_KEYS = (
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.images.camera3",
)


class ConfigError(ValueError):
    """Raised when a training config is missing or invalid."""


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{path}: expected a mapping")
    return value


def _unknown(mapping: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigError(f"{path}: unknown keys {unknown}")


def _string(value: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{path}: expected a string")
    if not allow_empty and not value.strip():
        raise ConfigError(f"{path}: expected a non-empty string")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{path}: expected true or false")
    return value


def _integer(
    value: Any,
    path: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path}: expected an integer")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{path}: expected >= {minimum}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{path}: expected <= {maximum}")
    return value


def _number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    inclusive_max: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path}: expected a number")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"{path}: expected a finite number")
    if minimum is not None and number < minimum:
        raise ConfigError(f"{path}: expected >= {minimum}")
    if maximum is not None:
        if inclusive_max and number > maximum:
            raise ConfigError(f"{path}: expected <= {maximum}")
        if not inclusive_max and number >= maximum:
            raise ConfigError(f"{path}: expected < {maximum}")
    return number


def _string_list(value: Any, path: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ConfigError(f"{path}: expected a list of strings")
    result = tuple(_string(item, f"{path}[{index}]") for index, item in enumerate(value))
    if not allow_empty and not result:
        raise ConfigError(f"{path}: expected at least one entry")
    if len(set(result)) != len(result):
        raise ConfigError(f"{path}: duplicate entries")
    return result


@dataclass(frozen=True)
class ModelConfig:
    family: str
    checkpoint: str
    revision: str | None
    expected_cameras: tuple[str, ...]
    validated: bool = True
    max_state_dim: int = 32
    max_action_dim: int = 32
    empty_cameras: int = 0

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ConfigError(f"model.family: expected one of {list(FAMILIES)}")
        if self.validated and not self.revision:
            raise ConfigError("model.revision: a pinned revision is required for a validated model")
        for key in self.expected_cameras:
            if not key.startswith("observation.images."):
                raise ConfigError(
                    f"model.expected_cameras: expected observation.images.* keys, got {key!r}"
                )
        if self.max_state_dim <= 0:
            raise ConfigError("model.max_state_dim: expected >= 1")
        if self.max_action_dim <= 0:
            raise ConfigError("model.max_action_dim: expected >= 1")
        if self.empty_cameras < 0:
            raise ConfigError("model.empty_cameras: expected >= 0")


@dataclass(frozen=True)
class LoraConfig:
    enabled: bool
    r: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]
    bias: str = "none"
    method_type: str = "LORA"
    init_type: str | None = None

    def __post_init__(self) -> None:
        if self.r <= 0:
            raise ConfigError("lora.r: expected > 0")
        if self.alpha <= 0:
            raise ConfigError("lora.alpha: expected > 0")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError("lora.dropout: expected in [0, 1)")
        if self.enabled and not self.target_modules:
            raise ConfigError("lora.target_modules: expected at least one verified module pattern")
        if self.bias not in LORA_BIASES:
            raise ConfigError(f"lora.bias: expected one of {list(LORA_BIASES)}")
        if self.method_type.upper() != "LORA":
            raise ConfigError("lora.method_type: only 'LORA' is validated in Task 8")


@dataclass(frozen=True)
class DataConfig:
    dataset_root: str
    val_fraction: float
    split_seed: int
    exclude_perturbed: bool
    rename_map: dict[str, str]
    action_representation: str
    decimation: int
    fps: float
    mixture_path: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.val_fraction < 1.0:
            raise ConfigError("data.val_fraction: expected in [0, 1)")
        if self.split_seed < 0:
            raise ConfigError("data.split_seed: expected >= 0")
        if self.action_representation not in ACTION_REPRESENTATIONS:
            raise ConfigError(
                f"data.action_representation: expected one of {list(ACTION_REPRESENTATIONS)}"
            )
        if self.decimation < 1:
            raise ConfigError("data.decimation: expected >= 1")
        if self.fps <= 0:
            raise ConfigError("data.fps: expected > 0")
        if not self.rename_map:
            raise ConfigError("data.rename_map: expected at least one camera rename")
        for source, target in self.rename_map.items():
            if not source.startswith("observation.images."):
                raise ConfigError(
                    f"data.rename_map.{source}: expected an observation.images.* source key"
                )
            if not target.startswith("observation.images."):
                raise ConfigError(
                    f"data.rename_map.{source}: expected an observation.images.* target key"
                )
        if len(set(self.rename_map.values())) != len(self.rename_map):
            raise ConfigError("data.rename_map: target camera keys must be distinct")


@dataclass(frozen=True)
class TrainingParams:
    smoke_test: bool
    max_steps: int
    batch_size: int
    seed: int
    lr: float
    grad_accum: int
    precision: str
    gradient_checkpointing: bool
    save_every: int
    num_workers: int
    deterministic: bool
    chunk_size: int
    n_action_steps: int

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ConfigError("training.max_steps: expected >= 1")
        if self.batch_size < 1:
            raise ConfigError("training.batch_size: expected >= 1")
        if self.seed < 0:
            raise ConfigError("training.seed: expected >= 0")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ConfigError("training.lr: expected > 0")
        if self.grad_accum < 1:
            raise ConfigError("training.grad_accum: expected >= 1")
        if self.precision not in PRECISIONS:
            raise ConfigError(f"training.precision: expected one of {list(PRECISIONS)}")
        if self.save_every < 0:
            raise ConfigError("training.save_every: expected >= 0")
        if self.num_workers < 0:
            raise ConfigError("training.num_workers: expected >= 0")
        if self.chunk_size < 1:
            raise ConfigError("training.chunk_size: expected >= 1")
        if not 1 <= self.n_action_steps <= self.chunk_size:
            raise ConfigError("training.n_action_steps: expected in [1, chunk_size]")


@dataclass(frozen=True)
class TrackingConfig:
    backend: str = "none"
    offline: bool = True

    def __post_init__(self) -> None:
        if self.backend not in TRACKING_BACKENDS:
            raise ConfigError(f"tracking.backend: expected one of {list(TRACKING_BACKENDS)}")
        if self.backend != "none" and not self.offline:
            raise ConfigError(
                "tracking.offline: network tracking is not approved; set offline: true"
            )


@dataclass(frozen=True)
class TrainingConfig:
    model: ModelConfig
    lora: LoraConfig
    data: DataConfig
    training: TrainingParams
    tracking: TrackingConfig
    raw: dict[str, Any] = field(default_factory=dict, repr=False)
    source: str | None = None

    @property
    def horizon_seconds(self) -> float:
        """Time span of one action chunk: ``chunk_size / fps``."""
        return self.training.chunk_size / self.data.fps

    @property
    def fps(self) -> float:
        return self.data.fps

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.raw, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "model": {
                "family": self.model.family,
                "checkpoint": self.model.checkpoint,
                "revision": self.model.revision,
                "expected_cameras": list(self.model.expected_cameras),
                "validated": self.model.validated,
                "max_state_dim": self.model.max_state_dim,
                "max_action_dim": self.model.max_action_dim,
                "empty_cameras": self.model.empty_cameras,
            },
            "lora": {
                "enabled": self.lora.enabled,
                "r": self.lora.r,
                "alpha": self.lora.alpha,
                "dropout": self.lora.dropout,
                "target_modules": list(self.lora.target_modules),
                "bias": self.lora.bias,
                "method_type": self.lora.method_type,
                "init_type": self.lora.init_type,
            },
            "data": {
                "dataset_root": self.data.dataset_root,
                "val_fraction": self.data.val_fraction,
                "split_seed": self.data.split_seed,
                "exclude_perturbed": self.data.exclude_perturbed,
                "mixture_path": self.data.mixture_path,
                "rename_map": dict(self.data.rename_map),
                "action_representation": self.data.action_representation,
                "decimation": self.data.decimation,
                "fps": self.data.fps,
            },
            "training": {
                "smoke_test": self.training.smoke_test,
                "max_steps": self.training.max_steps,
                "batch_size": self.training.batch_size,
                "seed": self.training.seed,
                "lr": self.training.lr,
                "grad_accum": self.training.grad_accum,
                "precision": self.training.precision,
                "gradient_checkpointing": self.training.gradient_checkpointing,
                "save_every": self.training.save_every,
                "num_workers": self.training.num_workers,
                "deterministic": self.training.deterministic,
                "chunk_size": self.training.chunk_size,
                "n_action_steps": self.training.n_action_steps,
            },
            "tracking": {
                "backend": self.tracking.backend,
                "offline": self.tracking.offline,
            },
        }

    def validate_camera_rename_map(self) -> None:
        """Every camera the model needs must be covered by a rename target."""
        mapped = set(self.data.rename_map.values())
        missing = sorted(set(self.model.expected_cameras) - mapped)
        if missing:
            raise ConfigError(
                "data.rename_map: does not map every camera the model needs; "
                f"missing targets {missing}"
            )


def parse_training_config(raw: Any, *, source: str | None = None) -> TrainingConfig:
    if not isinstance(raw, dict):
        raise ConfigError("root: expected a mapping")
    _unknown(raw, {"schema_version", "model", "lora", "data", "training", "tracking"}, "root")
    schema_version = _integer(raw.get("schema_version"), "schema_version", minimum=1)
    if schema_version != SCHEMA_VERSION:
        raise ConfigError(f"schema_version: expected {SCHEMA_VERSION}, got {schema_version}")

    model_raw = _mapping(raw.get("model"), "model")
    _unknown(
        model_raw,
        {
            "family",
            "checkpoint",
            "revision",
            "expected_cameras",
            "validated",
            "max_state_dim",
            "max_action_dim",
            "empty_cameras",
        },
        "model",
    )
    family = _string(model_raw.get("family"), "model.family")
    if family not in FAMILIES:
        raise ConfigError(f"model.family: expected one of {list(FAMILIES)}")
    checkpoint = _string(model_raw.get("checkpoint"), "model.checkpoint")
    revision_raw = model_raw.get("revision")
    revision = None if revision_raw is None else _string(revision_raw, "model.revision")
    validated = _boolean(model_raw.get("validated", family == "smolvla"), "model.validated")
    if validated and not revision:
        raise ConfigError("model.revision: a pinned revision is required for a validated model")
    default_cameras = SMOLVLA_CAMERA_KEYS[:2] if family == "smolvla" else ()
    cameras_raw = model_raw.get("expected_cameras", list(default_cameras))
    expected_cameras = _string_list(cameras_raw, "model.expected_cameras")
    model = ModelConfig(
        family=family,
        checkpoint=checkpoint,
        revision=revision,
        expected_cameras=expected_cameras,
        validated=validated,
        max_state_dim=_integer(
            model_raw.get("max_state_dim", 32), "model.max_state_dim", minimum=1
        ),
        max_action_dim=_integer(
            model_raw.get("max_action_dim", 32), "model.max_action_dim", minimum=1
        ),
        empty_cameras=_integer(model_raw.get("empty_cameras", 0), "model.empty_cameras", minimum=0),
    )

    lora_raw = _mapping(raw.get("lora"), "lora")
    _unknown(
        lora_raw,
        {"enabled", "r", "alpha", "dropout", "target_modules", "bias", "method_type", "init_type"},
        "lora",
    )
    init_raw = lora_raw.get("init_type")
    lora = LoraConfig(
        enabled=_boolean(lora_raw.get("enabled"), "lora.enabled"),
        r=_integer(lora_raw.get("r"), "lora.r", minimum=1),
        alpha=_integer(lora_raw.get("alpha"), "lora.alpha", minimum=1),
        dropout=_number(
            lora_raw.get("dropout"), "lora.dropout", minimum=0.0, maximum=1.0, inclusive_max=False
        ),
        target_modules=_string_list(
            lora_raw.get("target_modules", []),
            "lora.target_modules",
            allow_empty=not bool(lora_raw.get("enabled")),
        ),
        bias=_string(lora_raw.get("bias", "none"), "lora.bias"),
        method_type=_string(lora_raw.get("method_type", "LORA"), "lora.method_type"),
        init_type=None if init_raw is None else _string(init_raw, "lora.init_type"),
    )

    data_raw = _mapping(raw.get("data"), "data")
    _unknown(
        data_raw,
        {
            "dataset_root",
            "val_fraction",
            "split_seed",
            "exclude_perturbed",
            "mixture_path",
            "rename_map",
            "action_representation",
            "decimation",
            "fps",
        },
        "data",
    )
    rename_raw = _mapping(data_raw.get("rename_map"), "data.rename_map")
    rename_map: dict[str, str] = {}
    for source_key, target_key in rename_raw.items():
        if not isinstance(source_key, str):
            raise ConfigError("data.rename_map: expected string keys")
        rename_map[source_key] = _string(target_key, f"data.rename_map.{source_key}")
    mixture_raw = data_raw.get("mixture_path")
    data = DataConfig(
        dataset_root=_string(data_raw.get("dataset_root"), "data.dataset_root"),
        val_fraction=_number(
            data_raw.get("val_fraction"),
            "data.val_fraction",
            minimum=0.0,
            maximum=1.0,
            inclusive_max=False,
        ),
        split_seed=_integer(data_raw.get("split_seed"), "data.split_seed", minimum=0),
        exclude_perturbed=_boolean(
            data_raw.get("exclude_perturbed", False), "data.exclude_perturbed"
        ),
        rename_map=rename_map,
        action_representation=_string(
            data_raw.get("action_representation"), "data.action_representation"
        ),
        decimation=_integer(data_raw.get("decimation"), "data.decimation", minimum=1),
        fps=_number(data_raw.get("fps"), "data.fps", minimum=0.0),
        mixture_path=None if mixture_raw is None else _string(mixture_raw, "data.mixture_path"),
    )

    training_raw = _mapping(raw.get("training"), "training")
    _unknown(
        training_raw,
        {
            "smoke_test",
            "max_steps",
            "batch_size",
            "seed",
            "lr",
            "grad_accum",
            "precision",
            "gradient_checkpointing",
            "save_every",
            "num_workers",
            "deterministic",
            "chunk_size",
            "n_action_steps",
        },
        "training",
    )
    if "seed" not in training_raw:
        raise ConfigError("training.seed: required")
    training = TrainingParams(
        smoke_test=_boolean(training_raw.get("smoke_test", False), "training.smoke_test"),
        max_steps=_integer(training_raw.get("max_steps"), "training.max_steps", minimum=1),
        batch_size=_integer(training_raw.get("batch_size"), "training.batch_size", minimum=1),
        seed=_integer(training_raw.get("seed"), "training.seed", minimum=0),
        lr=_number(training_raw.get("lr"), "training.lr", minimum=0.0),
        grad_accum=_integer(training_raw.get("grad_accum", 1), "training.grad_accum", minimum=1),
        precision=_string(training_raw.get("precision", "fp32"), "training.precision"),
        gradient_checkpointing=_boolean(
            training_raw.get("gradient_checkpointing", False), "training.gradient_checkpointing"
        ),
        save_every=_integer(training_raw.get("save_every", 0), "training.save_every", minimum=0),
        num_workers=_integer(training_raw.get("num_workers", 0), "training.num_workers", minimum=0),
        deterministic=_boolean(training_raw.get("deterministic", True), "training.deterministic"),
        chunk_size=_integer(training_raw.get("chunk_size"), "training.chunk_size", minimum=1),
        n_action_steps=_integer(
            training_raw.get("n_action_steps"), "training.n_action_steps", minimum=1
        ),
    )

    tracking_raw = _mapping(raw.get("tracking", {}), "tracking")
    _unknown(tracking_raw, {"backend", "offline"}, "tracking")
    tracking = TrackingConfig(
        backend=_string(tracking_raw.get("backend", "none"), "tracking.backend"),
        offline=_boolean(tracking_raw.get("offline", True), "tracking.offline"),
    )

    config = TrainingConfig(
        model=model,
        lora=lora,
        data=data,
        training=training,
        tracking=tracking,
        raw=dict(raw),
        source=source,
    )
    config.validate_camera_rename_map()
    return config


def load_training_config(path: str | Path) -> TrainingConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"{config_path}: training config not found")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - message formatting only
        raise ConfigError(f"{config_path}: invalid YAML: {exc}") from exc
    return parse_training_config(raw, source=str(config_path))


__all__ = [
    "ACTION_REPRESENTATIONS",
    "ConfigError",
    "DataConfig",
    "FAMILIES",
    "LoraConfig",
    "ModelConfig",
    "PRECISIONS",
    "SCHEMA_VERSION",
    "SMOLVLA_CAMERA_KEYS",
    "TRACKING_BACKENDS",
    "TrackingConfig",
    "TrainingConfig",
    "TrainingParams",
    "load_training_config",
    "parse_training_config",
]
