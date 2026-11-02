"""Validation and index construction for unified episodes and mixtures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .embodiments import EmbodimentSpec
from .schema import UnifiedEpisode


class DataValidationError(ValueError):
    """Raised with one or more path-qualified validation failures."""


@dataclass(frozen=True)
class DatasetIndex:
    entries: tuple[dict[str, Any], ...]
    tasks: tuple[dict[str, Any], ...]
    total_frames: int
    num_episodes: int


def _fail(issues: list[str], message: str) -> None:
    issues.append(message)


def validate_episode(
    episode: UnifiedEpisode,
    spec: EmbodimentSpec,
    representation: str,
) -> None:
    """Raise :class:`DataValidationError` if one episode is malformed."""
    issues: list[str] = []
    label = episode.original_episode_id or "<episode>"
    if episode.embodiment_id != spec.embodiment_id:
        _fail(issues, f"{label}: embodiment_id {episode.embodiment_id!r} != {spec.embodiment_id!r}")
    if not episode.task or not str(episode.task).strip():
        _fail(issues, f"{label}: task string is empty")
    if not np.isfinite(episode.fps) or episode.fps <= 0:
        _fail(issues, f"{label}: fps must be a positive finite number")

    state = np.asarray(episode.observation_state)
    action = np.asarray(episode.action)
    timestamps = np.asarray(episode.timestamps)
    if state.ndim != 2 or state.shape[1] != spec.state.dim:
        _fail(issues, f"{label}: observation.state shape {state.shape} != (T, {spec.state.dim})")
    if action.ndim != 2 or action.shape[1] != spec.action(representation).dim:
        _fail(
            issues,
            f"{label}: action shape {action.shape} != (T, {spec.action(representation).dim})",
        )
    if state.ndim == 2 and action.ndim == 2 and state.shape[0] != action.shape[0]:
        _fail(
            issues, f"{label}: state/action lengths differ ({state.shape[0]} vs {action.shape[0]})"
        )
    if state.size and not np.all(np.isfinite(state)):
        _fail(issues, f"{label}: observation.state contains NaN or infinity")
    if action.size and not np.all(np.isfinite(action)):
        _fail(issues, f"{label}: action contains NaN or infinity")
    if timestamps.ndim != 1 or timestamps.size == 0:
        _fail(issues, f"{label}: timestamps must be a non-empty 1-D array")
    else:
        if not np.all(np.isfinite(timestamps)):
            _fail(issues, f"{label}: timestamps contain NaN or infinity")
        if timestamps.size > 1 and not np.all(np.diff(timestamps) > 0):
            _fail(issues, f"{label}: timestamps must be strictly increasing")
        if state.ndim == 2 and timestamps.size != state.shape[0]:
            _fail(issues, f"{label}: timestamps length {timestamps.size} != state length")

    length = int(state.shape[0]) if state.ndim == 2 else 0
    for camera, frames in episode.images.items():
        if len(frames) != length:
            _fail(issues, f"{label}: camera {camera!r} has {len(frames)} frames, expected {length}")
        for index, image in enumerate(frames):
            if image.camera != camera:
                _fail(
                    issues,
                    f"{label}: camera {camera!r} frame {index} labels itself {image.camera!r}",
                )
            if index and frames[index - 1].step_index >= image.step_index:
                _fail(issues, f"{label}: camera {camera!r} step indices are not increasing")
    unknown_cameras = sorted(set(episode.images) - set(spec.cameras))
    if unknown_cameras:
        _fail(issues, f"{label}: unknown cameras {unknown_cameras}")

    if issues:
        raise DataValidationError("\n".join(issues))


def build_dataset_index(
    episodes: list[UnifiedEpisode],
    spec: EmbodimentSpec,
    representation: str,
) -> DatasetIndex:
    """Validate every episode and assign contiguous global indices/tasks."""
    if not episodes:
        raise DataValidationError("dataset contains no episodes")
    seen: set[tuple[str, str]] = set()
    entries: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    task_index: dict[str, int] = {}
    offset = 0
    for episode_index, episode in enumerate(episodes):
        validate_episode(episode, spec, representation)
        key = (episode.source, episode.original_episode_id)
        if key in seen:
            raise DataValidationError(f"duplicate episode key {key}")
        seen.add(key)
        if episode.task not in task_index:
            task_index[episode.task] = len(tasks)
            tasks.append({"task_index": task_index[episode.task], "instruction": episode.task})
        length = episode.num_frames
        entries.append(
            {
                "episode_index": episode_index,
                "episode_id": episode.original_episode_id,
                "source": episode.source,
                "task": episode.task,
                "task_index": task_index[episode.task],
                "length": length,
                "start_index": offset,
                "end_index": offset + length,
                "fps": float(episode.fps),
                "num_images": episode.num_images,
            }
        )
        offset += length
    return DatasetIndex(
        entries=tuple(entries),
        tasks=tuple(tasks),
        total_frames=offset,
        num_episodes=len(entries),
    )


def validate_image_policy(
    episodes: list[UnifiedEpisode],
    *,
    renderer: str,
    decimation: int,
    fps: float,
    allow_imageless: bool = False,
    max_joint_error_rad: float = 1e-6,
    max_object_error_m: float = 1e-6,
) -> dict[str, Any]:
    """Enforce the one-renderer, fidelity, and imageless rules."""
    if decimation < 1:
        raise DataValidationError("decimation must be >= 1")
    if not renderer.strip():
        raise DataValidationError("renderer must be a non-empty string")
    renderers: set[str] = set()
    imageless = 0
    for episode in episodes:
        label = episode.original_episode_id
        if not episode.images:
            imageless += 1
            if not allow_imageless:
                raise DataValidationError(
                    f"{label}: episode has no images; pass allow_imageless=true to export "
                    "an images:none dataset"
                )
            continue
        episode_renderer = str(episode.metadata.get("renderer", renderer))
        renderers.add(episode_renderer)
        joint_error = episode.metadata.get("replay_max_joint_error_rad")
        if joint_error is not None and float(joint_error) > max_joint_error_rad:
            raise DataValidationError(
                f"{label}: replay joint error {joint_error} rad exceeds {max_joint_error_rad} rad"
            )
        object_error = episode.metadata.get("replay_max_object_error_m")
        if object_error is not None and float(object_error) > max_object_error_m:
            raise DataValidationError(
                f"{label}: replay object error {object_error} m exceeds {max_object_error_m} m"
            )
    if len(renderers) > 1:
        raise DataValidationError(
            f"dataset mixes renderers {sorted(renderers)}; one renderer per dataset is required"
        )
    mode = "none" if not renderers else "image"
    return {
        "renderer": next(iter(renderers), renderer),
        "declared_renderer": renderer,
        "decimation": int(decimation),
        "fps": float(fps),
        "cameras": sorted({camera for episode in episodes for camera in episode.images}),
        "mode": mode,
        "imageless_episodes": imageless,
        "num_episodes": len(episodes),
    }


def validate_mixture_compatibility(
    action_by_source: dict[str, str],
    *,
    allow_cross_embodiment: bool = False,
) -> None:
    """Reject concatenating incompatible action semantics without a manifest."""
    if not action_by_source:
        raise DataValidationError("mixture has no sources")
    representations = set(action_by_source.values())
    if len(representations) > 1 and not allow_cross_embodiment:
        raise DataValidationError(
            "sources use different action semantics "
            f"{sorted(representations)}; mix them only through a mixture manifest"
        )


__all__ = [
    "DataValidationError",
    "DatasetIndex",
    "build_dataset_index",
    "validate_episode",
    "validate_image_policy",
    "validate_mixture_compatibility",
]
