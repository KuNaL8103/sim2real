"""Pure reset-service logic, importable without ``rclpy``.

Task 2 applies a nominal reset only. Randomization fields are echoed in the
response and explicitly marked as ignored; Task 3 will implement them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from sim2real.metadata import validate_metadata_json
from sim2real.sim.backend import ResetConfig, SimBackend


@dataclass(frozen=True)
class ResetOutcome:
    success: bool
    message: str
    applied_seed: int
    applied_config_json: str


def _json(payload: dict[str, object]) -> str:
    text = json.dumps(payload, sort_keys=True, allow_nan=False)
    # Never return a payload that the metadata validator would reject.
    validate_metadata_json(text)
    return text


def reset_environment(
    backend: SimBackend,
    seed: int,
    *,
    randomization_enabled: bool = False,
    randomization_profile: str = "nominal",
    task_instruction: str = "",
    object_pose_override: np.ndarray | None = None,
) -> ResetOutcome:
    """Reset ``backend`` nominally and return a validated applied-config JSON string."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        return ResetOutcome(
            success=False,
            message="seed must be an integer",
            applied_seed=0,
            applied_config_json=_json({"error": "seed must be an integer"}),
        )
    if seed < 0:
        return ResetOutcome(
            success=False,
            message=f"seed must be non-negative, got {seed}",
            applied_seed=0,
            applied_config_json=_json({"error": "negative seed", "requested_seed": seed}),
        )

    config = ResetConfig(
        seed=seed,
        randomization_enabled=bool(randomization_enabled),
        randomization_profile=randomization_profile or "nominal",
        task_instruction=task_instruction,
        object_pose_override=object_pose_override,
    )
    try:
        observation = backend.reset(seed=seed, config=config)
    except Exception as exc:  # noqa: BLE001 - the service must report, not crash
        return ResetOutcome(
            success=False,
            message=f"reset failed: {type(exc).__name__}: {exc}",
            applied_seed=0,
            applied_config_json=_json({"error": str(exc), "requested_seed": seed}),
        )

    metadata = config.applied_metadata()
    metadata["message"] = "nominal reset applied; randomization fields ignored (Task 3)"
    metadata["stamp_ns"] = int(observation.stamp_ns)
    return ResetOutcome(
        success=True,
        message=(
            "nominal reset applied; randomization is not implemented (Task 3), "
            "so randomization fields were ignored"
        ),
        applied_seed=seed,
        applied_config_json=_json(metadata),
    )
