"""RLDS/Open X-Embodiment adapter with a CI-safe fixture source.

``FixtureEpisodeSource`` is deliberately tiny and dependency-free.  The TFDS
source is guarded and is never constructed in CI.  A conversion always requires
an explicit ``max_episodes``; there is no unbounded path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..embodiments import EmbodimentSpec
from ..schema import ImageFrame, UnifiedEpisode, UnifiedFeatureSpec


class RLDSError(ValueError):
    """Raised when an RLDS-like source is malformed."""


class RLDSEpisodeSource(Protocol):
    dataset_name: str
    license_tag: str

    def num_episodes(self) -> int: ...

    def get_episode(self, index: int) -> list[dict[str, Any]]: ...


class FixtureEpisodeSource:
    """Read a tiny JSON fixture with the same episode/step shape as RLDS."""

    def __init__(self, path: str | Path, *, dataset_name: str = "bridge_fixture") -> None:
        self.path = Path(path)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("episodes"), list):
            raise RLDSError(f"{self.path}: expected a mapping with an episodes list")
        self.dataset_name = str(payload.get("dataset_name", dataset_name))
        self.license_tag = str(payload.get("license", "CC-BY-4.0"))
        self._episodes: list[dict[str, Any]] = payload["episodes"]

    def num_episodes(self) -> int:
        return len(self._episodes)

    def get_episode(self, index: int) -> list[dict[str, Any]]:
        if not 0 <= index < len(self._episodes):
            raise RLDSError(f"episode index {index} out of range")
        episode = self._episodes[index]
        steps = episode.get("steps")
        if not isinstance(steps, list):
            raise RLDSError(f"fixture episode {index}: steps must be a list")
        return steps


class TFDSEpisodeSource:
    """Guarded TFDS source; never imported or used by CI."""

    def __init__(
        self,
        dataset_name: str,
        *,
        data_dir: str | Path | None = None,
        split: str = "train",
        license_tag: str = "CC-BY-4.0",
        download: bool = False,
    ) -> None:
        try:
            import tensorflow_datasets as tfds
        except Exception as exc:  # pragma: no cover - CI never has TF
            raise RLDSError(
                "TensorFlow Datasets is not installed; use --source fixture in CI"
            ) from exc
        self._tfds = tfds
        self.dataset_name = dataset_name
        self.license_tag = license_tag
        self._dataset = tfds.load(
            dataset_name,
            data_dir=None if data_dir is None else str(data_dir),
            split=split,
            download=download,
        )
        try:
            self._count = int(self._dataset.cardinality().numpy())
        except Exception:
            self._count = 0

    def num_episodes(self) -> int:
        if self._count:
            return self._count
        raise RLDSError(f"{self.dataset_name}: TFDS cardinality is unknown; pass a finite split")

    def get_episode(self, index: int) -> list[dict[str, Any]]:
        if not 0 <= index < self.num_episodes():
            raise RLDSError(f"episode index {index} out of range")
        sample = next(iter(self._dataset.skip(index).take(1)))
        steps = sample.get("steps")
        if steps is None:
            raise RLDSError(f"{self.dataset_name}: sample has no steps")
        return [dict(step) for step in self._tfds.as_numpy(steps)]


@dataclass(frozen=True)
class BridgeAdapterConfig:
    max_episodes: int
    seed: int = 0
    resolution: tuple[int, int] = (256, 256)
    camera_map: dict[str, str] = field(
        default_factory=lambda: {"third_person": "image_0", "wrist": "image_3"}
    )
    on_missing_instruction: str = "fill_default"
    default_instruction: str = "perform the task"
    missing_image: str = "skip"

    def __post_init__(self) -> None:
        if self.max_episodes <= 0:
            raise RLDSError("max_episodes must be positive")
        if self.on_missing_instruction not in {"skip", "fill_default"}:
            raise RLDSError("on_missing_instruction must be 'skip' or 'fill_default'")
        if self.missing_image not in {"skip", "zeros"}:
            raise RLDSError("missing_image must be 'skip' or 'zeros'")
        if len(self.resolution) != 2 or any(int(value) <= 0 for value in self.resolution):
            raise RLDSError("resolution must be two positive integers")
        if not self.camera_map:
            raise RLDSError("camera_map must not be empty")


@dataclass
class BridgeAdapterReport:
    available: int = 0
    selected: int = 0
    converted: int = 0
    skipped_missing_language: int = 0
    skipped_nan: int = 0
    skipped_missing_image: int = 0
    skipped_boundary: int = 0
    filled_default_instruction: int = 0
    zero_filled_images: int = 0
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "selected": self.selected,
            "converted": self.converted,
            "skipped_missing_language": self.skipped_missing_language,
            "skipped_nan": self.skipped_nan,
            "skipped_missing_image": self.skipped_missing_image,
            "skipped_boundary": self.skipped_boundary,
            "filled_default_instruction": self.filled_default_instruction,
            "zero_filled_images": self.zero_filled_images,
            "reasons": list(self.reasons),
        }


def select_episode_indices(available: int, max_episodes: int, seed: int) -> list[int]:
    if available <= 0:
        raise RLDSError("source has no episodes")
    if max_episodes <= 0:
        raise RLDSError("max_episodes must be positive")
    count = min(available, max_episodes)
    rng = np.random.default_rng(seed)
    selected = rng.permutation(available)[:count]
    return sorted(int(index) for index in selected)


def resize_nearest(image: np.ndarray, resolution: tuple[int, int]) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise RLDSError("image must be an HxWx3 uint8 array")
    height, width = resolution
    if array.shape[:2] == (height, width):
        return array.copy()
    rows = np.linspace(0, array.shape[0] - 1, height).astype(np.int64)
    cols = np.linspace(0, array.shape[1] - 1, width).astype(np.int64)
    return array[rows][:, cols].copy()


def adapt_rlds_source(
    source: RLDSEpisodeSource,
    spec: EmbodimentSpec,
    config: BridgeAdapterConfig,
    *,
    action_representation: str = "bridge_ee_delta",
) -> tuple[list[UnifiedEpisode], BridgeAdapterReport]:
    """Convert a finite, seeded subset of an RLDS-like source."""
    action_spec = spec.action(action_representation)
    available = int(source.num_episodes())
    selected = select_episode_indices(available, config.max_episodes, config.seed)
    report = BridgeAdapterReport(available=available, selected=len(selected))
    episodes: list[UnifiedEpisode] = []
    for index in selected:
        steps = list(source.get_episode(index))
        if not steps:
            report.skipped_boundary += 1
            report.reasons.append(f"episode {index}: empty")
            continue
        if not bool(steps[0].get("is_first", False)):
            report.skipped_boundary += 1
            report.reasons.append(f"episode {index}: first step is not is_first")
            continue
        last_flags = [
            bool(step.get("is_last", False)) or bool(step.get("is_terminal", False))
            for step in steps
        ]
        if not last_flags[-1] or any(last_flags[:-1]):
            report.skipped_boundary += 1
            report.reasons.append(f"episode {index}: is_last/is_terminal boundary is malformed")
            continue
        language = ""
        for step in steps:
            candidate = step.get("language_instruction")
            if candidate:
                language = str(candidate).strip()
                break
        if not language:
            if config.on_missing_instruction == "skip":
                report.skipped_missing_language += 1
                report.reasons.append(f"episode {index}: missing language instruction")
                continue
            language = config.default_instruction
            report.filled_default_instruction += 1

        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        images: dict[str, list[ImageFrame]] = {camera: [] for camera in config.camera_map}
        bad = False
        for step_index, step in enumerate(steps):
            observation = step.get("observation") or {}
            state = np.asarray(observation.get("state", step.get("state")), dtype=np.float32)
            action = np.asarray(step.get("action"), dtype=np.float32)
            if state.shape != (spec.state.dim,) or action.shape != (action_spec.dim,):
                report.skipped_nan += 1
                report.reasons.append(f"episode {index}: malformed state/action shape")
                bad = True
                break
            if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
                report.skipped_nan += 1
                report.reasons.append(f"episode {index}: NaN/inf in state or action")
                bad = True
                break
            states.append(state)
            actions.append(action)
            for camera, key in config.camera_map.items():
                raw = observation.get(key)
                if raw is None:
                    if config.missing_image == "skip":
                        report.skipped_missing_image += 1
                        report.reasons.append(f"episode {index}: missing {key}")
                        bad = True
                        break
                    raw = np.zeros((*config.resolution, 3), dtype=np.uint8)
                    report.zero_filled_images += 1
                raw_array = np.asarray(raw)
                if raw_array.dtype != np.uint8:
                    raw_array = raw_array.astype(np.uint8)
                images[camera].append(
                    ImageFrame(
                        step_index=step_index,
                        camera=camera,
                        array=resize_nearest(raw_array, config.resolution),
                    )
                )
            if bad:
                break
        if bad:
            continue
        states_array = np.stack(states).astype(np.float32)
        actions_array = np.stack(actions).astype(np.float32)
        report.converted += 1
        episodes.append(
            UnifiedEpisode(
                observation_state=states_array,
                action=actions_array,
                images={camera: tuple(frames) for camera, frames in images.items()},
                timestamps=np.arange(len(steps), dtype=np.float64) / spec.fps,
                task=language,
                embodiment_id=spec.embodiment_id,
                source=source.dataset_name,
                fps=spec.fps,
                original_episode_id=f"{source.dataset_name}_{index}",
                license_tag=source.license_tag,
                features={
                    "observation.state": spec.state,
                    "action": UnifiedFeatureSpec(
                        name="action",
                        dtype="float32",
                        shape=(action_spec.dim,),
                        names=tuple(f"action_{dim}" for dim in range(action_spec.dim)),
                        units=action_spec.units,
                        frame=action_spec.frame,
                    ),
                },
                metadata={
                    "renderer": "bridge_source",
                    "image_source": "rlds",
                    "bridge_episode_index": index,
                },
            )
        )
    return episodes, report


__all__ = [
    "BridgeAdapterConfig",
    "BridgeAdapterReport",
    "FixtureEpisodeSource",
    "RLDSEpisodeSource",
    "RLDSError",
    "TFDSEpisodeSource",
    "adapt_rlds_source",
    "resize_nearest",
    "select_episode_indices",
]
