#!/usr/bin/env python3
"""Render MuJoCo keyframes for failed clean episodes (diagnosis only).

This replays the recorded actions of a staged episode from the same reset seed
and saves overhead + wrist PNGs at the end of APPROACH_PREGRASP, GRASP, LIFT,
and RELEASE. It does not modify expert code or tune anything.

    uv run python scripts/debug_keyframes.py \
        --staging data/staging/census --seeds 3004,3008,3036 \
        --out log/task5_keyframes

Run this ALONE: MuJoCo rendering is software GL here, and EGL contexts are
thread-bound. No colcon build or other heavy job should run at the same time.

The PNG writer is a ~20-line stdlib implementation so this script needs no image
library.
"""

from __future__ import annotations

import argparse
import json
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from sim2real.expert.rollout import derive_episode_seed
from sim2real.expert.staging import StagingBuffer
from sim2real.sim import create_backend
from sim2real.sim.reset_handler import reset_environment

STATES = ("APPROACH_PREGRASP", "GRASP", "LIFT", "RELEASE")


def _png(path: Path, rgb: np.ndarray) -> None:
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError("expected HxWx3 RGB")
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    payload = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(payload)


def _staging_run(path: Path) -> StagingBuffer:
    if (path / "manifest.jsonl").is_file():
        return StagingBuffer(path.parent, path.name)
    runs = sorted(
        candidate for candidate in path.iterdir() if (candidate / "manifest.jsonl").is_file()
    )
    if not runs:
        raise SystemExit(f"no manifest.jsonl under {path}")
    return StagingBuffer(runs[-1].parent, runs[-1].name)


def _transition_indices(states: list[str]) -> dict[str, int]:
    found: dict[str, int] = {}
    for state in STATES:
        matches = [index for index, value in enumerate(states) if value == state]
        if matches:
            found[state] = matches[-1]
    return found


def _replay_and_render(record: Any, backend: Any, output: Path) -> dict[str, Any]:
    profile = str(record.randomization_report.get("profile", "default"))
    seed = derive_episode_seed(record.seed, record.episode_index)
    randomization_dir = Path(backend.config_dir).parent / "randomization"
    reset = reset_environment(
        backend,
        seed,
        randomization_enabled=profile != "none",
        randomization_profile=profile,
        task_instruction=record.task_instruction,
        randomization_dir=randomization_dir,
        episode_index=record.episode_index,
    )
    if not reset.success:
        raise RuntimeError(f"reset failed: {reset.message}")

    states = [step.state for step in record.steps]
    targets = _transition_indices(states)
    images: list[dict[str, Any]] = []
    episode_dir = output / f"seed_{record.seed}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    max_joint_error = 0.0
    for index in range(1, len(record.steps)):
        action = record.steps[index].action
        observation = backend.step(action[:7], float(action[7])).observation
        recorded = np.asarray(record.steps[index].joint_positions, dtype=np.float64)
        error = float(np.max(np.abs(observation.robot_state.joint_positions - recorded)))
        max_joint_error = max(max_joint_error, error)
        state = record.steps[index].state
        if targets.get(state) != index:
            continue
        for camera in ("overhead", "wrist"):
            frame = backend.render(camera)
            name = f"{index:03d}_{state}_{camera}.png"
            _png(episode_dir / name, frame.rgb)
            images.append(
                {
                    "step_index": index,
                    "state": state,
                    "camera": camera,
                    "path": str((episode_dir / name).relative_to(output)),
                }
            )
    return {
        "seed": int(record.seed),
        "episode_id": record.episode_id,
        "outcome": int(record.outcome),
        "failure_reason": record.failure_reason,
        "transition_steps": targets,
        "max_joint_replay_error_rad": max_joint_error,
        "images": images,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", required=True)
    parser.add_argument("--seeds", default="3004,3008,3036")
    parser.add_argument("--out", default="log/task5_keyframes")
    parser.add_argument("--config-dir", default="configs/sim")
    args = parser.parse_args(argv)

    wanted = [int(value) for value in args.seeds.split(",") if value.strip()]
    buffer = _staging_run(Path(args.staging).expanduser().resolve())
    records = {record.seed: record for record in buffer.iter_records() if record.seed in wanted}
    if not records:
        raise SystemExit(f"none of the requested seeds are in {buffer.root}")
    failed = [
        record for record in records.values() if record.outcome != 1 and not record.is_perturbed
    ]
    if not failed:
        print("no failed clean episodes among the requested seeds")
        return 0

    output = Path(args.out).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    backend = create_backend("mujoco", args.config_dir)
    report: list[dict[str, Any]] = []
    try:
        # At most three: rendering is slow (software GL).
        for record in sorted(failed, key=lambda item: item.seed)[:3]:
            print(f"replaying seed {record.seed}: {record.failure_reason}")
            report.append(_replay_and_render(record, backend, output))
    finally:
        backend.close()

    summary = output / "summary.json"
    summary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {sum(len(item['images']) for item in report)} images under {output}")
    print(f"summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
