"""Synthetic unified episodes shared by dataset-builder tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sim2real.dataset_builder import ImageFrame, UnifiedEpisode, load_embodiments

ROOT = Path(__file__).resolve().parents[2]
EMBODIMENTS = ROOT / "configs" / "dataset" / "embodiments.yaml"


def panda_spec():
    return load_embodiments(EMBODIMENTS)["panda_sim"]


def bridge_spec():
    return load_embodiments(EMBODIMENTS)["widowx_bridge"]


def make_episode(
    *,
    episode_id: str = "ep0",
    source: str = "mujoco",
    frames: int = 4,
    cameras: tuple[str, ...] = ("overhead", "wrist"),
    image_size: tuple[int, int] = (4, 4),
    fps: float = 10.0,
    renderer: str = "llvmpipe",
    with_images: bool = True,
    state: np.ndarray | None = None,
    action: np.ndarray | None = None,
    task: str = "pick up the red cube",
    metadata: dict | None = None,
) -> UnifiedEpisode:
    spec = panda_spec()
    if state is None:
        state = np.arange(frames * 8, dtype=np.float32).reshape(frames, 8) * 0.01
    if action is None:
        action = state + 0.05
    images: dict[str, tuple[ImageFrame, ...]] = {}
    if with_images:
        for camera_index, camera in enumerate(cameras):
            images[camera] = tuple(
                ImageFrame(
                    step_index=index,
                    camera=camera,
                    array=np.full(
                        (*image_size, 3),
                        (index * 10 + camera_index) % 256,
                        dtype=np.uint8,
                    ),
                )
                for index in range(frames)
            )
    episode_metadata = {"renderer": renderer}
    if metadata:
        episode_metadata.update(metadata)
    return UnifiedEpisode(
        observation_state=np.asarray(state, dtype=np.float32),
        action=np.asarray(action, dtype=np.float32),
        images=images,
        timestamps=np.arange(frames, dtype=np.float64) / fps,
        task=task,
        embodiment_id=spec.embodiment_id,
        source=source,
        fps=fps,
        original_episode_id=episode_id,
        license_tag=spec.license_tag,
        features={"observation.state": spec.state},
        metadata=episode_metadata,
    )
