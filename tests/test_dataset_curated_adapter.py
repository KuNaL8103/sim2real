"""Curated-v1 adapter: action round trip, decimation alignment, SHA checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sim2real.dataset_builder import derive_ee_delta, rotation_vector_from_matrix
from sim2real.dataset_builder.adapters.curated_v1 import (
    CuratedAdapter,
    CuratedAdapterError,
)
from sim2real.expert.images import write_png
from tests.fixtures.dataset_builder import panda_spec


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_curated(tmp_path: Path, *, frames: int = 6, images: bool = True):
    version = tmp_path / "curated" / "demo" / "v1.0.0"
    (version / "meta").mkdir(parents=True)
    (version / "data" / "chunk-000").mkdir(parents=True)
    (version / "meta" / "info.json").write_text(
        json.dumps({"fps": 20.0, "features": {}}), encoding="utf-8"
    )
    (version / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "instruction": "pick up the red cube"}) + "\n",
        encoding="utf-8",
    )
    (version / "meta" / "episodes.jsonl").write_text(
        json.dumps(
            {
                "episode_id": "ep0",
                "episode_index": 0,
                "task_index": 0,
                "length": frames,
                "source_run_id": "run0",
                "seed": 100,
                "backend": "fake",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = []
    for index in range(frames):
        rows.append(
            {
                "timestamp": index / 20.0,
                "frame_index": index,
                "episode_index": 0,
                "index": index,
                "task_index": 0,
                "observation.state": [float(index)] * 8,
                "action": [float(index) + 0.5] * 8,
                "next.done": index == frames - 1,
            }
        )
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, version / "data" / "chunk-000" / "episode_000000.parquet")

    run = tmp_path / "staging" / "run0"
    run.mkdir(parents=True)
    image_entries = []
    selected = list(range(0, frames, 2))
    for step in selected:
        for camera_index, camera in enumerate(("overhead", "wrist")):
            path = run / "images" / "ep0" / f"{step:04d}_{camera}.png"
            write_png(path, np.full((4, 4, 3), step * 10 + camera_index, dtype=np.uint8))
            image_entries.append(
                {
                    "step_index": step,
                    "camera": camera,
                    "path": str(path.relative_to(run)),
                    "sha256": _sha(path),
                    "skipped": False,
                }
            )
    sidecar = {
        "episode_id": "ep0",
        "run_id": "run0",
        "episode_index": 0,
        "seed": 100,
        "backend": "fake",
        "task_instruction": "pick up the red cube",
        "task_object_id": "cube_red",
        "task_zone_id": "zone_left",
        "control_rate_hz": 20.0,
        "outcome": 1,
        "failure_reason": "",
        "is_perturbed": False,
        "perturbation_name": None,
        "expected_quality": "clean",
        "randomization_seed": 1,
        "randomization_report": {"profile": "default"},
        "state_per_step": ["DONE"] * frames,
        "final_object_poses": {},
        "transitions": [],
        "started_at": "",
        "success_metrics": {},
        "metadata": {},
        "renderer": "llvmpipe",
        "replay": {"max_joint_error_rad": 0.0, "max_object_error_m": 0.0},
        "images": image_entries if images else [],
    }
    (run / "episode_ep0.json").write_text(json.dumps(sidecar), encoding="utf-8")
    poses = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], frames, axis=0)
    poses[:, 0, 3] = np.arange(frames) * 0.01
    actions = np.zeros((frames, 8), dtype=np.float64)
    actions[:, 7] = np.linspace(0.0, 0.04, frames)
    np.savez(run / "episode_ep0.npz", ee_pose=poses, action=actions)
    return version, run


def test_joint_position_round_trip_and_decimation_alignment(tmp_path):
    version, run = _make_curated(tmp_path)
    adapter = CuratedAdapter(
        curated_root=version,
        embodiment=panda_spec(),
        decimation=2,
        action_representation="joint_position",
        staging_root=tmp_path / "staging",
        declared_renderer="llvmpipe",
    )
    episodes = adapter.load()
    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.fps == 10.0
    np.testing.assert_allclose(episode.observation_state[:, 0], [0, 2, 4])
    np.testing.assert_allclose(episode.action[:, 0], [0.5, 2.5, 4.5])
    assert [frame.step_index for frame in episode.images["overhead"]] == [0, 2, 4]
    assert [frame.step_index for frame in episode.images["wrist"]] == [0, 2, 4]
    np.testing.assert_allclose(episode.timestamps, [0.0, 0.1, 0.2])
    assert episode.metadata["replay_max_joint_error_rad"] == 0.0
    assert episode.metadata["renderer"] == "llvmpipe"

    bad = episode.images["overhead"][0]
    assert bad.path is not None
    bad.path.write_bytes(b"changed")
    with pytest.raises(CuratedAdapterError, match="sha256 mismatch"):
        adapter.load()


def test_ee_delta_known_motion_and_rotation_vector():
    poses = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], 3, axis=0)
    poses[:, 0, 3] = [0.0, 0.1, 0.2]
    commands = np.array([0.0, 0.02, 0.04])
    delta = derive_ee_delta(poses, commands, open_m=0.04, closed_m=0.0)
    np.testing.assert_allclose(delta[:, 0], [0.1, 0.1, 0.0], atol=1e-6)
    np.testing.assert_allclose(delta[:, 1:6], np.zeros((3, 5)), atol=1e-6)
    np.testing.assert_allclose(delta[:, 6], [0.5, 1.0, 1.0], atol=1e-6)

    theta = 0.1
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(rotation_vector_from_matrix(rotation), [0.0, 0.0, theta])


def test_allow_imageless(tmp_path):
    version, _run = _make_curated(tmp_path, images=False)
    adapter = CuratedAdapter(
        curated_root=version,
        embodiment=panda_spec(),
        decimation=2,
        staging_root=tmp_path / "staging",
        allow_imageless=True,
    )
    episode = adapter.load()[0]
    assert episode.images == {}
    assert episode.metadata["image_mode"] == "none"
