"""Read ``curated_v1`` output and attach replay-rendered images.

The adapter deliberately reads the existing curated layout rather than inventing
a second dataset path.  Replay images are expected to be the PNG references
written by ``scripts/render_replay.py``; each referenced file is SHA-256 checked
before the episode is accepted.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..embodiments import EmbodimentSpec
from ..schema import ImageFrame, UnifiedEpisode, UnifiedFeatureSpec


class CuratedAdapterError(ValueError):
    """Raised when curated data or its replay references cannot be read."""


@dataclass(frozen=True)
class CuratedAdapter:
    curated_root: str | Path
    embodiment: EmbodimentSpec
    decimation: int = 2
    action_representation: str = "joint_position"
    staging_root: str | Path | None = None
    allow_imageless: bool = False
    declared_renderer: str = "unknown"
    cameras: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.decimation < 1:
            raise CuratedAdapterError("decimation must be >= 1")
        self.embodiment.action(self.action_representation)
        if self.cameras is not None:
            self.embodiment.require_cameras(self.cameras)

    def load(self, *, max_episodes: int | None = None) -> list[UnifiedEpisode]:
        root = resolve_curated_version(self.curated_root)
        info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
        episode_rows = _read_jsonl(root / "meta" / "episodes.jsonl")
        if not episode_rows:
            raise CuratedAdapterError(f"{root}: no episodes")
        if max_episodes is not None:
            if max_episodes <= 0:
                raise CuratedAdapterError("max_episodes must be positive")
            episode_rows = episode_rows[:max_episodes]
        tasks = {
            int(row["task_index"]): str(row["instruction"])
            for row in _read_jsonl(root / "meta" / "tasks.jsonl")
        }
        fps = float(info.get("fps", self.embodiment.control_rate_hz))
        if abs(fps - self.embodiment.control_rate_hz) > 1e-6:
            raise CuratedAdapterError(
                f"curated fps {fps} does not match embodiment control rate "
                f"{self.embodiment.control_rate_hz}"
            )
        output: list[UnifiedEpisode] = []
        for row in episode_rows:
            output.append(self._load_one(root, info, row, tasks))
        return output

    def _load_one(
        self,
        root: Path,
        info: dict[str, Any],
        row: dict[str, Any],
        tasks: dict[int, str],
    ) -> UnifiedEpisode:
        episode_id = str(row["episode_id"])
        episode_index = int(row["episode_index"])
        task_index = int(row["task_index"])
        task = tasks.get(task_index)
        if not task:
            raise CuratedAdapterError(
                f"{episode_id}: task_index {task_index} is not in tasks.jsonl"
            )
        control_rate = float(info.get("fps", self.embodiment.control_rate_hz))
        table = _read_episode_rows(root, episode_index)
        length = int(row.get("length", len(table)))
        if len(table) != length:
            raise CuratedAdapterError(
                f"{episode_id}: episodes.jsonl says {length} frames, parquet has {len(table)}"
            )
        selected = np.arange(0, length, self.decimation, dtype=np.int64)
        if selected.size == 0:
            raise CuratedAdapterError(
                f"{episode_id}: no frames selected at decimation {self.decimation}"
            )
        timestamps = np.asarray([float(table[i]["timestamp"]) for i in selected], dtype=np.float64)
        states = np.asarray([table[i]["observation.state"] for i in selected], dtype=np.float32)
        actions = np.asarray([table[i]["action"] for i in selected], dtype=np.float32)
        staging = self._staging_episode(row)
        images, image_meta = self._load_images(episode_id, staging, selected)
        if self.action_representation == "ee_delta":
            poses, commands = self._load_ee_poses(staging, selected)
            actions = derive_ee_delta(
                poses,
                commands,
                open_m=self.embodiment.gripper_open_m,
                closed_m=self.embodiment.gripper_closed_m,
            )
        elif self.action_representation != "joint_position":
            raise CuratedAdapterError(
                f"curated adapter does not support {self.action_representation!r}"
            )
        action_spec = self.embodiment.action(self.action_representation)
        features = {
            "observation.state": self.embodiment.state,
            "action": UnifiedFeatureSpec(
                name="action",
                dtype="float32",
                shape=(action_spec.dim,),
                names=_action_names(action_spec, self.embodiment),
                units=action_spec.units,
                frame=action_spec.frame,
            ),
        }
        metadata = {
            "renderer": image_meta.get("renderer", self.declared_renderer),
            "source_run_id": str(row.get("source_run_id", "")),
            "seed": int(row.get("seed", 0)),
            "image_mode": "image" if images else "none",
            "decimation": self.decimation,
            "curated_root": str(root),
        }
        metadata.update(image_meta)
        return UnifiedEpisode(
            observation_state=states,
            action=actions,
            images=images,
            timestamps=timestamps,
            task=task,
            embodiment_id=self.embodiment.embodiment_id,
            source=str(row.get("backend", "curated")),
            fps=control_rate / self.decimation,
            original_episode_id=episode_id,
            license_tag=self.embodiment.license_tag,
            features=features,
            metadata=metadata,
        )

    def _staging_episode(self, row: dict[str, Any]) -> Path | None:
        if self.staging_root is None:
            return None
        root = Path(self.staging_root)
        run_id = str(row.get("source_run_id", ""))
        candidates = [root / run_id] if run_id else []
        if (root / "manifest.jsonl").is_file():
            candidates.append(root)
        sidecar_name = f"episode_{row['episode_id']}.json"
        for candidate in candidates:
            if (candidate / "manifest.jsonl").is_file() or (candidate / sidecar_name).is_file():
                return candidate / sidecar_name
        return None

    def _load_images(
        self,
        episode_id: str,
        sidecar: Path | None,
        selected: np.ndarray,
    ) -> tuple[dict[str, tuple[ImageFrame, ...]], dict[str, Any]]:
        if sidecar is None or not sidecar.is_file():
            if self.allow_imageless:
                return {}, {"image_mode": "none", "renderer": self.declared_renderer}
            raise CuratedAdapterError(
                f"{episode_id}: no staging sidecar; cannot locate replay images "
                "(pass staging_root or allow_imageless=true)"
            )
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        entries = list(payload.get("images") or [])
        if not entries:
            if self.allow_imageless:
                return {}, {
                    "image_mode": "none",
                    "renderer": payload.get("renderer", self.declared_renderer),
                }
            raise CuratedAdapterError(f"{episode_id}: staging sidecar has no replay images")
        run_dir = sidecar.parent
        by_step_camera: dict[tuple[int, str], dict[str, Any]] = {}
        for entry in entries:
            key = (int(entry["step_index"]), str(entry["camera"]))
            if key in by_step_camera:
                raise CuratedAdapterError(f"{episode_id}: duplicate image for {key}")
            by_step_camera[key] = entry
        grouped: dict[str, list[ImageFrame]] = {}
        renderers: set[str] = set()
        selected_cameras = self.cameras or self.embodiment.cameras
        for step in selected:
            for camera in selected_cameras:
                entry = by_step_camera.get((int(step), camera))
                if entry is None:
                    raise CuratedAdapterError(
                        f"{episode_id}: missing replay image step={step} camera={camera!r}"
                    )
                path = run_dir / str(entry["path"])
                expected = str(entry["sha256"])
                actual = _sha256(path)
                if actual != expected:
                    raise CuratedAdapterError(
                        f"{episode_id}: image sha256 mismatch for {path}: "
                        f"expected {expected}, got {actual}"
                    )
                renderer = str(
                    entry.get("renderer", payload.get("renderer", self.declared_renderer))
                )
                renderers.add(renderer)
                grouped.setdefault(camera, []).append(
                    ImageFrame(
                        step_index=int(step),
                        camera=camera,
                        path=path,
                        sha256=expected,
                    )
                )
        if len(renderers) > 1:
            raise CuratedAdapterError(
                f"{episode_id}: replay images mix renderers {sorted(renderers)}"
            )
        replay = payload.get("replay") or {}
        return {key: tuple(value) for key, value in grouped.items()}, {
            "renderer": next(iter(renderers), self.declared_renderer),
            "replay_max_joint_error_rad": replay.get("max_joint_error_rad"),
            "replay_max_object_error_m": replay.get("max_object_error_m"),
        }

    def _load_ee_poses(
        self,
        sidecar: Path | None,
        selected: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if sidecar is None or not sidecar.is_file():
            raise CuratedAdapterError("ee_delta requires a staging episode with ee_pose")
        npz = (
            sidecar.parent
            / f"episode_{sidecar.name.removeprefix('episode_').removesuffix('.json')}.npz"
        )
        if not npz.is_file():
            raise CuratedAdapterError(f"missing staging arrays {npz}")
        with np.load(npz, allow_pickle=False) as arrays:
            if "ee_pose" not in arrays or "action" not in arrays:
                raise CuratedAdapterError(f"{npz}: needs ee_pose and action arrays for ee_delta")
            poses = np.asarray(arrays["ee_pose"], dtype=np.float64)[selected]
            commands = np.asarray(arrays["action"], dtype=np.float64)[selected, 7]
        return poses, commands


def resolve_curated_version(path: str | Path) -> Path:
    root = Path(path)
    if (root / "meta" / "info.json").is_file():
        return root
    versions = sorted(
        (
            (tuple(int(part) for part in child.name[1:].split(".")), child)
            for child in root.iterdir()
            if child.is_dir()
            and child.name.startswith("v")
            and all(part.isdigit() for part in child.name[1:].split("."))
        ),
        key=lambda item: item[0],
    )
    if not versions:
        raise CuratedAdapterError(f"{root}: no meta/info.json and no v* version directory")
    return versions[-1][1]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _read_episode_rows(root: Path, episode_index: int) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    direct = sorted(root.glob(f"data/chunk-*/episode_{episode_index:06d}.parquet"))
    if direct:
        return pq.read_table(direct[0]).to_pylist()
    for path in sorted(root.glob("data/chunk-*/*.parquet")):
        table = pq.read_table(path, filters=[("episode_index", "=", episode_index)])
        if table.num_rows:
            return table.to_pylist()
    raise CuratedAdapterError(f"{root}: no parquet rows for episode_index {episode_index}")


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise CuratedAdapterError(f"missing replay image {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _action_names(action: Any, embodiment: EmbodimentSpec) -> tuple[str, ...]:
    if action.name == "joint_position":
        return embodiment.state.names  # type: ignore[return-value]
    return tuple(f"action_{index}" for index in range(action.dim))


def rotation_vector_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return the axis-angle rotation vector for a proper rotation matrix."""
    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise CuratedAdapterError(f"rotation matrix must be 3x3, got {rotation.shape}")
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    theta = math.acos(cosine)
    if theta < 1e-10:
        return np.zeros(3, dtype=np.float64)
    if math.pi - theta > 1e-6:
        axis = np.array(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ],
            dtype=np.float64,
        ) / (2.0 * math.sin(theta))
        return axis * theta
    # Near 180 degrees the skew part vanishes; use the symmetric eigenvector.
    symmetric = (rotation + np.eye(3)) / 2.0
    values, vectors = np.linalg.eigh(symmetric)
    axis = vectors[:, int(np.argmax(values))]
    axis = axis / np.linalg.norm(axis)
    nonzero = np.flatnonzero(np.abs(axis) > 1e-12)
    if nonzero.size and axis[nonzero[0]] < 0:
        axis = -axis
    return axis * theta


def derive_ee_delta(
    ee_poses: np.ndarray,
    gripper_commands: np.ndarray,
    *,
    open_m: float,
    closed_m: float,
) -> np.ndarray:
    """Derive ``[dx, dy, dz, drx, dry, drz, gripper 0..1]``.

    Translation and the rotation vector use the world axes of the recorded
    ``ee_pose``.  The rotation vector is ``R_{t+1} R_t^{-1}``.  The gripper is
    the absolute command at the target frame, normalized by the embodiment's
    open/closed span; the last frame is a zero delta with the last gripper value.
    """
    poses = np.asarray(ee_poses, dtype=np.float64)
    commands = np.asarray(gripper_commands, dtype=np.float64).reshape(-1)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise CuratedAdapterError(f"ee_poses must be (T,4,4), got {poses.shape}")
    if commands.shape[0] != poses.shape[0]:
        raise CuratedAdapterError("gripper_commands length must match ee_poses")
    if open_m <= closed_m:
        raise CuratedAdapterError("open_m must be greater than closed_m")
    normalized = np.clip((commands - closed_m) / (open_m - closed_m), 0.0, 1.0)
    output = np.zeros((poses.shape[0], 7), dtype=np.float32)
    for index in range(max(0, poses.shape[0] - 1)):
        output[index, :3] = poses[index + 1, :3, 3] - poses[index, :3, 3]
        output[index, 3:6] = rotation_vector_from_matrix(
            poses[index + 1, :3, :3] @ poses[index, :3, :3].T
        )
        output[index, 6] = normalized[index + 1]
    if poses.shape[0]:
        output[-1, 6] = normalized[-1]
    return output


__all__ = [
    "CuratedAdapter",
    "CuratedAdapterError",
    "derive_ee_delta",
    "resolve_curated_version",
    "rotation_vector_from_matrix",
]
