"""Validated configuration for the ROS 2 VLA inference node.

The loader is strict: unknown keys, wrong types, and out-of-range values are
rejected rather than silently replaced by a default. The same dataclass is used
by the pure-Python worker tests and by the ROS node, so a parameter cannot mean
one thing in a unit test and another in a launch file.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = 1
POLICY_BACKENDS = ("server", "fake")


class InferenceConfigError(ValueError):
    """Raised when the inference config is missing, unknown, or malformed."""


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InferenceConfigError(f"{path}: expected a mapping")
    return value


def _unknown(mapping: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise InferenceConfigError(f"{path}: unknown keys {unknown}")


def _number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InferenceConfigError(f"{path}: expected a number")
    number = float(value)
    if not math.isfinite(number):
        raise InferenceConfigError(f"{path}: expected a finite number")
    if minimum is not None:
        if exclusive_minimum and number <= minimum:
            raise InferenceConfigError(f"{path}: expected > {minimum}")
        if not exclusive_minimum and number < minimum:
            raise InferenceConfigError(f"{path}: expected >= {minimum}")
    if maximum is not None and number > maximum:
        raise InferenceConfigError(f"{path}: expected <= {maximum}")
    return number


def _integer(
    value: Any,
    path: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InferenceConfigError(f"{path}: expected an integer")
    if minimum is not None and value < minimum:
        raise InferenceConfigError(f"{path}: expected >= {minimum}")
    if maximum is not None and value > maximum:
        raise InferenceConfigError(f"{path}: expected <= {maximum}")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise InferenceConfigError(f"{path}: expected true or false")
    return value


def _text(value: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise InferenceConfigError(f"{path}: expected a string")
    text = value.strip()
    if not text and not allow_empty:
        raise InferenceConfigError(f"{path}: expected a non-empty string")
    return text


@dataclass(frozen=True)
class InferenceConfig:
    """All inference settings, with units in the field names."""

    policy_backend: str = "fake"
    checkpoint_path: str = ""
    server_python: str = ""
    server_script: str = ""
    # Cold start (torch import + base model load) is not a policy step. The
    # measured local time-to-ready was 107.38 s (Task 10b); 180 s leaves margin.
    # This is a timeout, not a promise.
    startup_timeout_s: float = 180.0
    call_timeout_s: float = 30.0
    policy_period_s: float = 0.1
    n_action_steps: int = 10
    max_state_age_s: float = 0.5
    max_camera_skew_s: float = 0.05
    max_observation_age_s: float = 1.0
    max_inference_rate_hz: float = 10.0
    sample_seed: int = -1
    latency_compensation: bool = True
    max_restarts: int = 3
    restart_backoff_s: float = 0.5
    max_instruction_chars: int = 512
    max_chunk_length: int = 100
    status_rate_hz: float = 1.0

    def __post_init__(self) -> None:
        if self.policy_backend not in POLICY_BACKENDS:
            raise InferenceConfigError(
                f"policy_backend: expected one of {list(POLICY_BACKENDS)}, "
                f"got {self.policy_backend!r}"
            )
        _text(self.checkpoint_path, "checkpoint_path", allow_empty=True)
        _text(self.server_python, "server_python", allow_empty=True)
        _text(self.server_script, "server_script", allow_empty=True)
        _number(self.startup_timeout_s, "startup_timeout_s", minimum=0.0, exclusive_minimum=True)
        _number(self.call_timeout_s, "call_timeout_s", minimum=0.0, exclusive_minimum=True)
        _number(self.policy_period_s, "policy_period_s", minimum=0.0, exclusive_minimum=True)
        if self.policy_period_s > 10.0:
            raise InferenceConfigError("policy_period_s: expected <= 10.0")
        _integer(self.n_action_steps, "n_action_steps", minimum=1, maximum=1000)
        _number(self.max_state_age_s, "max_state_age_s", minimum=0.0, exclusive_minimum=True)
        _number(self.max_camera_skew_s, "max_camera_skew_s", minimum=0.0)
        _number(
            self.max_observation_age_s,
            "max_observation_age_s",
            minimum=0.0,
            exclusive_minimum=True,
        )
        _number(
            self.max_inference_rate_hz,
            "max_inference_rate_hz",
            minimum=0.0,
            exclusive_minimum=True,
        )
        _integer(self.sample_seed, "sample_seed", minimum=-1)
        _integer(self.max_restarts, "max_restarts", minimum=0, maximum=100)
        _number(self.restart_backoff_s, "restart_backoff_s", minimum=0.0)
        if self.restart_backoff_s > 60.0:
            raise InferenceConfigError("restart_backoff_s: expected <= 60.0")
        _integer(self.max_instruction_chars, "max_instruction_chars", minimum=1, maximum=4096)
        _integer(self.max_chunk_length, "max_chunk_length", minimum=1, maximum=1000)
        _number(
            self.status_rate_hz,
            "status_rate_hz",
            minimum=0.0,
            maximum=100.0,
            exclusive_minimum=True,
        )
        if self.n_action_steps > self.max_chunk_length:
            raise InferenceConfigError(
                "n_action_steps: must be <= max_chunk_length "
                f"({self.n_action_steps} > {self.max_chunk_length})"
            )
        if self.policy_backend == "server" and not self.checkpoint_path:
            raise InferenceConfigError("checkpoint_path: required when policy_backend is 'server'")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_values(cls, values: dict[str, Any]) -> InferenceConfig:
        """Build from a flat mapping (ROS parameters or YAML's ``inference`` block)."""
        allowed = set(cls.__dataclass_fields__)
        _unknown(values, allowed, "inference")
        data = {name: values[name] for name in allowed if name in values}
        return cls(
            policy_backend=_text(data.get("policy_backend", cls.policy_backend), "policy_backend"),
            checkpoint_path=_text(
                data.get("checkpoint_path", cls.checkpoint_path),
                "checkpoint_path",
                allow_empty=True,
            ),
            server_python=_text(
                data.get("server_python", cls.server_python),
                "server_python",
                allow_empty=True,
            ),
            server_script=_text(
                data.get("server_script", cls.server_script),
                "server_script",
                allow_empty=True,
            ),
            startup_timeout_s=_number(
                data.get("startup_timeout_s", cls.startup_timeout_s),
                "startup_timeout_s",
                minimum=0.0,
                exclusive_minimum=True,
            ),
            call_timeout_s=_number(
                data.get("call_timeout_s", cls.call_timeout_s),
                "call_timeout_s",
                minimum=0.0,
                exclusive_minimum=True,
            ),
            policy_period_s=_number(
                data.get("policy_period_s", cls.policy_period_s),
                "policy_period_s",
                minimum=0.0,
                exclusive_minimum=True,
            ),
            n_action_steps=_integer(
                data.get("n_action_steps", cls.n_action_steps),
                "n_action_steps",
                minimum=1,
                maximum=1000,
            ),
            max_state_age_s=_number(
                data.get("max_state_age_s", cls.max_state_age_s),
                "max_state_age_s",
                minimum=0.0,
                exclusive_minimum=True,
            ),
            max_camera_skew_s=_number(
                data.get("max_camera_skew_s", cls.max_camera_skew_s),
                "max_camera_skew_s",
                minimum=0.0,
            ),
            max_observation_age_s=_number(
                data.get("max_observation_age_s", cls.max_observation_age_s),
                "max_observation_age_s",
                minimum=0.0,
                exclusive_minimum=True,
            ),
            max_inference_rate_hz=_number(
                data.get("max_inference_rate_hz", cls.max_inference_rate_hz),
                "max_inference_rate_hz",
                minimum=0.0,
                exclusive_minimum=True,
            ),
            sample_seed=_integer(
                data.get("sample_seed", cls.sample_seed),
                "sample_seed",
                minimum=-1,
            ),
            latency_compensation=_boolean(
                data.get("latency_compensation", cls.latency_compensation),
                "latency_compensation",
            ),
            max_restarts=_integer(
                data.get("max_restarts", cls.max_restarts),
                "max_restarts",
                minimum=0,
                maximum=100,
            ),
            restart_backoff_s=_number(
                data.get("restart_backoff_s", cls.restart_backoff_s),
                "restart_backoff_s",
                minimum=0.0,
            ),
            max_instruction_chars=_integer(
                data.get("max_instruction_chars", cls.max_instruction_chars),
                "max_instruction_chars",
                minimum=1,
                maximum=4096,
            ),
            max_chunk_length=_integer(
                data.get("max_chunk_length", cls.max_chunk_length),
                "max_chunk_length",
                minimum=1,
                maximum=1000,
            ),
            status_rate_hz=_number(
                data.get("status_rate_hz", cls.status_rate_hz),
                "status_rate_hz",
                minimum=0.0,
                maximum=100.0,
                exclusive_minimum=True,
            ),
        )


def load_inference_config(path: str | Path) -> InferenceConfig:
    config_path = Path(path).expanduser()
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InferenceConfigError(f"{config_path}: cannot read config: {exc}") from exc
    except yaml.YAMLError as exc:
        raise InferenceConfigError(f"{config_path}: invalid YAML: {exc}") from exc
    root = _mapping(data, str(config_path))
    _unknown(root, {"schema_version", "inference"}, str(config_path))
    schema = root.get("schema_version", SCHEMA_VERSION)
    if schema != SCHEMA_VERSION:
        raise InferenceConfigError(
            f"{config_path}: unsupported schema_version {schema!r} (expected {SCHEMA_VERSION})"
        )
    block = _mapping(root.get("inference", {}), f"{config_path}: inference")
    return InferenceConfig.from_values(block)


def default_inference_config() -> InferenceConfig:
    return InferenceConfig()


__all__ = [
    "InferenceConfig",
    "InferenceConfigError",
    "POLICY_BACKENDS",
    "SCHEMA_VERSION",
    "default_inference_config",
    "load_inference_config",
]
