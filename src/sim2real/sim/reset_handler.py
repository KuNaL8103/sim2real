"""Pure reset-service logic, importable without ``rclpy``.

Task 3 flow (always called inside the node's render-quiesce window):

1. validate the seed, profile, and any object-pose override **before** touching
   the backend, so a bad request cannot leave a half-applied scene;
2. load the profile and build a deterministic sample for
   ``(seed, episode_index)``;
3. ``backend.reset`` restores nominal state and model parameters;
4. ``apply_randomization`` applies the sample and returns an honest
   applied/skipped report;
5. an explicit object-pose override is applied as a separate, validated
   object-pose-only step;
6. on any application error, reset to nominal again and report the rollback.

The reset service uses ``episode_index=0``. The sampler itself is a pure
function of ``(seed, episode_index)``; later tasks (the expert/collector) can
pass an explicit index. A per-node counter is deliberately **not** used, so the
same seed and profile always produce identical output regardless of earlier
resets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from sim2real.metadata import validate_metadata_json
from sim2real.randomization.applier import (
    APPLIED,
    PARTIAL,
    SKIPPED,
    AppliedRandomization,
    apply_randomization,
)
from sim2real.randomization.config import load_profile
from sim2real.randomization.sampler import ObjectPoseSample, sample
from sim2real.sim.backend import ConfigError, ResetConfig, SimBackend

SERVICE_EPISODE_INDEX = 0


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


def _failure(
    seed: int,
    message: str,
    *,
    error: str,
    profile: str,
    rolled_back: bool = False,
) -> ResetOutcome:
    return ResetOutcome(
        success=False,
        message=message,
        applied_seed=0,
        applied_config_json=_json(
            {
                "error": error,
                "requested_seed": int(seed),
                "requested_profile": profile,
                "rolled_back_to_nominal": bool(rolled_back),
            }
        ),
    )


def _randomization_dir(backend: SimBackend, override: str | Path | None) -> Path:
    if override is not None:
        return Path(override).expanduser()
    if backend.config_dir is not None:
        return Path(backend.config_dir).parent / "randomization"
    return Path("configs/randomization")


def _validate_override(
    override: np.ndarray, workspace_min: np.ndarray, workspace_max: np.ndarray
) -> np.ndarray:
    pose = np.asarray(override, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ConfigError("object_pose_override: expected a 4x4 homogeneous transform")
    if not np.all(np.isfinite(pose)):
        raise ConfigError("object_pose_override: values must be finite")
    position = pose[:3, 3]
    outside = (position < workspace_min) | (position > workspace_max)
    if np.any(outside):
        raise ConfigError(
            "object_pose_override: position "
            f"{position.tolist()} lies outside the scene workspace "
            f"min={workspace_min.tolist()} max={workspace_max.tolist()}"
        )
    return pose


def reset_environment(
    backend: SimBackend,
    seed: int,
    *,
    randomization_enabled: bool = False,
    randomization_profile: str = "nominal",
    task_instruction: str = "",
    object_pose_override: np.ndarray | None = None,
    randomization_dir: str | Path | None = None,
    episode_index: int = SERVICE_EPISODE_INDEX,
) -> ResetOutcome:
    """Reset ``backend`` and apply a validated randomization sample.

    Returns a :class:`ResetOutcome` whose ``applied_config_json`` always passes
    :func:`sim2real.metadata.validate_metadata_json`.
    """
    profile = (randomization_profile or "nominal").strip()
    if not isinstance(seed, int) or isinstance(seed, bool):
        return _failure(
            seed if isinstance(seed, int) else 0,
            "seed must be an integer",
            error="seed must be an integer",
            profile=profile,
        )
    if seed < 0:
        return _failure(
            seed,
            f"seed must be non-negative, got {seed}",
            error="negative seed",
            profile=profile,
        )

    scene = backend.configs.scene
    directory = _randomization_dir(backend, randomization_dir)
    primary_object = next(iter(scene.cubes))
    override: np.ndarray | None = None
    try:
        if randomization_enabled:
            config = load_profile(profile, scene=scene, profiles_dir=directory)
        else:
            config = load_profile("none", scene=scene, profiles_dir=directory)
        # The request seed is authoritative; the profile seed is only the
        # default for direct sampler use. The profile hash is unchanged.
        config = replace(config, seed=seed)
        if object_pose_override is not None:
            override = _validate_override(
                object_pose_override, scene.workspace_min, scene.workspace_max
            )
        sampled = sample(config, episode_index)
    except (ConfigError, ValueError) as exc:
        return _failure(
            seed,
            f"invalid randomization request: {exc}",
            error=str(exc),
            profile=profile,
        )

    request = ResetConfig(
        seed=seed,
        randomization_enabled=bool(randomization_enabled),
        randomization_profile=profile,
        task_instruction=task_instruction,
        object_pose_override=override,
    )

    try:
        backend.reset(seed=seed, config=request)
        report = apply_randomization(backend, sampled)
        if override is not None:
            override_sample = replace(
                sampled,
                nominal=False,
                disabled_groups=tuple(
                    group for group in sampled.disabled_groups if group != "object_pose"
                ),
                object_pose=ObjectPoseSample(poses={primary_object: override}),
            )
            override_report = apply_randomization(
                backend, override_sample, only_groups=("object_pose",)
            )
            merged = dict(report.groups)
            merged["object_pose"] = override_report.groups["object_pose"]
            report = AppliedRandomization(groups=merged, nominal=report.nominal)
            poses = dict(sampled.object_pose.poses)
            poses[primary_object] = override
            sampled = replace(
                sampled,
                nominal=False,
                disabled_groups=tuple(
                    group for group in sampled.disabled_groups if group != "object_pose"
                ),
                object_pose=ObjectPoseSample(poses=poses),
            )
        observation = backend.get_observation()
    except Exception as exc:  # noqa: BLE001 - the service must report, not crash
        rolled_back = False
        try:
            backend.reset(seed=seed, config=request)
            rolled_back = True
        except Exception:  # noqa: BLE001 - report the original failure
            rolled_back = False
        suffix = "rolled back to nominal" if rolled_back else "rollback also failed"
        return _failure(
            seed,
            f"randomization failed: {type(exc).__name__}: {exc}; {suffix}",
            error=str(exc),
            profile=profile,
            rolled_back=rolled_back,
        )

    counts = {
        status: sum(1 for result in report.groups.values() if result.status == status)
        for status in (APPLIED, PARTIAL, SKIPPED)
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "seed": int(seed),
        "episode_index": int(episode_index),
        "profile": profile,
        "config_hash": config.config_hash,
        "randomization_enabled": bool(randomization_enabled),
        "randomization_applied": any(
            result.status in {APPLIED, PARTIAL} for result in report.groups.values()
        ),
        "task_instruction": task_instruction,
        "sample": sampled.to_dict(),
        "report": report.to_dict(),
        "object_pose_override": {
            "requested": object_pose_override is not None,
            "applied": override is not None,
            "object": primary_object if override is not None else None,
        },
        "stamp_ns": int(observation.stamp_ns),
    }
    try:
        applied_json = _json(payload)
    except Exception as exc:  # noqa: BLE001 - never return invalid metadata
        return _failure(
            seed,
            f"applied config could not be serialized: {exc}",
            error=str(exc),
            profile=profile,
        )

    if sampled.nominal:
        message = f"nominal reset applied; profile={profile}; randomization disabled"
    else:
        message = (
            f"randomization reset applied; profile={profile}; "
            f"applied={counts[APPLIED]} partial={counts[PARTIAL]} skipped={counts[SKIPPED]}"
        )
    if override is not None:
        message += f"; object pose override applied to {primary_object}"
    return ResetOutcome(
        success=True,
        message=message,
        applied_seed=seed,
        applied_config_json=applied_json,
    )
