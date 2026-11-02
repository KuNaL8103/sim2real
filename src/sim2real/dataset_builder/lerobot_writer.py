"""Thin adapter over the official LeRobot dataset-creation API.

The writer is the only module that imports LeRobot or torch.  It does not
hand-write Parquet/MP4 metadata: ``LeRobotDataset.create``, ``add_frame``,
``save_episode``, and ``finalize`` remain the source of truth.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .embodiments import EmbodimentSpec
from .schema import UnifiedEpisode
from .validation import validate_episode


class MLLibraryUnavailable(RuntimeError):
    """Raised when the ML environment is not installed."""


class LerobotWriterError(ValueError):
    """Raised for an invalid dataset request."""


@dataclass(frozen=True)
class BuildResult:
    path: Path
    repo_id: str
    num_episodes: int
    num_frames: int
    fps: int
    stats_keys: tuple[str, ...]
    image_mode: str
    renderer: str


def _replay_code_sha256() -> str | None:
    path = Path(__file__).resolve().parents[3] / "scripts" / "render_replay.py"
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gl_settings() -> dict[str, str]:
    keys = ("MUJOCO_GL", "GALLIUM_DRIVER", "MESA_D3D12_DEFAULT_ADAPTER_NAME")
    return {key: os.environ[key] for key in keys if key in os.environ}


def _require_lerobot() -> Any:
    try:
        from lerobot.datasets import LeRobotDataset
    except Exception as exc:  # pragma: no cover - exercised in the ml environment
        raise MLLibraryUnavailable(
            "LeRobot is not installed; run the writer in .venv-ml "
            "(see scripts/setup_envs.sh --with-ml)"
        ) from exc
    return LeRobotDataset


def _features(
    episode: UnifiedEpisode,
    spec: EmbodimentSpec,
    representation: str,
    *,
    use_videos: bool,
) -> dict[str, dict[str, Any]]:
    action = spec.action(representation)
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": tuple(spec.state.shape),
            "names": list(spec.state.names or ()),
        },
        "action": {
            "dtype": "float32",
            "shape": (action.dim,),
            "names": [f"action_{index}" for index in range(action.dim)],
        },
    }
    dtype = "video" if use_videos else "image"
    for camera, frames in episode.images.items():
        if not frames:
            raise LerobotWriterError(f"camera {camera!r} has no frames")
        shape = frames[0].load().shape
        features[f"observation.images.{camera}"] = {
            "dtype": dtype,
            "shape": shape,
            "names": ["height", "width", "channels"],
        }
    return features


def build_lerobot_dataset(
    episodes: list[UnifiedEpisode],
    spec: EmbodimentSpec,
    out_root: str | Path,
    dataset_name: str,
    *,
    representation: str = "joint_position",
    use_videos: bool = False,
    renderer: str = "unknown",
    decimation: int = 1,
    split_manifest: dict[str, Any] | None = None,
    max_episodes: int | None = None,
) -> BuildResult:
    """Write a local LeRobot v3 dataset and return its verified metadata."""
    LeRobotDataset = _require_lerobot()
    if not episodes:
        raise LerobotWriterError("no episodes to write")
    if max_episodes is not None:
        if max_episodes <= 0:
            raise LerobotWriterError("max_episodes must be positive")
        episodes = episodes[:max_episodes]
    for episode in episodes:
        validate_episode(episode, spec, representation)
    fps_values = {round(float(episode.fps), 9) for episode in episodes}
    if len(fps_values) != 1:
        raise LerobotWriterError(f"episodes have mixed fps values: {sorted(fps_values)}")
    fps_float = next(iter(fps_values))
    if abs(fps_float - round(fps_float)) > 1e-9:
        raise LerobotWriterError(f"LeRobot requires an integer fps, got {fps_float}")
    fps = int(round(fps_float))
    features = _features(episodes[0], spec, representation, use_videos=use_videos)
    for episode in episodes[1:]:
        other = _features(episode, spec, representation, use_videos=use_videos)
        if set(other) != set(features):
            raise LerobotWriterError("episodes have different image camera sets")
        for key in features:
            if tuple(other[key]["shape"]) != tuple(features[key]["shape"]):
                raise LerobotWriterError(f"episodes have different shape for {key}")

    root = Path(out_root).expanduser().resolve() / dataset_name
    if root.exists():
        raise LerobotWriterError(f"refusing to overwrite existing dataset {root}")
    dataset = LeRobotDataset.create(
        repo_id=dataset_name,
        fps=fps,
        features=features,
        root=root,
        robot_type=spec.robot_type,
        use_videos=use_videos,
    )
    try:
        for episode in episodes:
            for index in range(episode.num_frames):
                frame = {
                    "observation.state": np.asarray(
                        episode.observation_state[index], dtype=np.float32
                    ),
                    "action": np.asarray(episode.action[index], dtype=np.float32),
                    "task": episode.task,
                }
                for camera, frames in episode.images.items():
                    image = frames[index].load()
                    frame[f"observation.images.{camera}"] = image
                dataset.add_frame(frame)
            dataset.save_episode()
    finally:
        dataset.finalize()

    stats = getattr(dataset.meta, "stats", None) or {}
    first_image = None
    if episodes[0].images:
        first_image = next(iter(episodes[0].images.values()))[0].load()
    replay_joint_errors = [
        float(episode.metadata["replay_max_joint_error_rad"])
        for episode in episodes
        if episode.metadata.get("replay_max_joint_error_rad") is not None
    ]
    replay_object_errors = [
        float(episode.metadata["replay_max_object_error_m"])
        for episode in episodes
        if episode.metadata.get("replay_max_object_error_m") is not None
    ]
    build_meta = {
        "schema_version": 1,
        "repo_id": dataset_name,
        "fps": fps,
        "renderer": renderer,
        "gl_settings": _gl_settings(),
        "decimation": int(decimation),
        "resolution": None if first_image is None else list(first_image.shape[:2]),
        "use_videos": bool(use_videos),
        "image_mode": "video" if use_videos else ("image" if episodes[0].images else "none"),
        "cameras": sorted(episodes[0].images),
        "num_episodes": len(episodes),
        "num_frames": sum(episode.num_frames for episode in episodes),
        "stats_keys": sorted(stats),
        "sources": sorted({episode.source for episode in episodes}),
        "embodiments": sorted({episode.embodiment_id for episode in episodes}),
        "replay_code_sha256": _replay_code_sha256(),
        "replay_max_joint_error_rad": max(replay_joint_errors) if replay_joint_errors else None,
        "replay_max_object_error_m": max(replay_object_errors) if replay_object_errors else None,
    }
    (root / "meta" / "sim2real_build.json").write_text(
        json.dumps(build_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if split_manifest is not None:
        (root / "meta" / "sim2real_split.json").write_text(
            json.dumps(split_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return BuildResult(
        path=root,
        repo_id=dataset_name,
        num_episodes=len(episodes),
        num_frames=sum(episode.num_frames for episode in episodes),
        fps=fps,
        stats_keys=tuple(sorted(stats)),
        image_mode=build_meta["image_mode"],
        renderer=renderer,
    )


def load_lerobot_dataset(path: str | Path, dataset_name: str) -> Any:
    LeRobotDataset = _require_lerobot()
    return LeRobotDataset(repo_id=dataset_name, root=Path(path))


def compare_episode(
    dataset: Any,
    episode: UnifiedEpisode,
    frame_index: int = 0,
) -> dict[str, Any]:
    """Compare one reloaded frame against the unified source frame."""
    if not 0 <= frame_index < episode.num_frames:
        raise LerobotWriterError("frame_index out of range")
    sample = dataset[frame_index]
    state = np.asarray(sample["observation.state"].detach().cpu().numpy(), dtype=np.float32)
    action = np.asarray(sample["action"].detach().cpu().numpy(), dtype=np.float32)
    result: dict[str, Any] = {
        "state_max_abs_diff": float(np.max(np.abs(state - episode.observation_state[frame_index]))),
        "action_max_abs_diff": float(np.max(np.abs(action - episode.action[frame_index]))),
        "task": str(sample.get("task", "")),
    }
    for camera, _frames in episode.images.items():
        key = f"observation.images.{camera}"
        if key in sample:
            image = np.asarray(sample[key].detach().cpu().numpy())
            result[f"{key}_shape"] = tuple(int(value) for value in image.shape)
    return result


def stats_summary(dataset: Any) -> dict[str, list[str]]:
    stats = getattr(dataset.meta, "stats", None) or {}
    return {key: sorted(value) for key, value in stats.items()}


__all__ = [
    "BuildResult",
    "LerobotWriterError",
    "MLLibraryUnavailable",
    "build_lerobot_dataset",
    "compare_episode",
    "load_lerobot_dataset",
    "stats_summary",
]
