#!/usr/bin/env python3
"""Replay staged episodes and write RGB PNGs at a configured decimation.

The staging record contains everything needed to rebuild the exact scene:
``seed``, ``episode_index``, the applied randomization report (including the
profile), the recorded actions, the control rate, and the final object poses.
Replay uses the same ``reset_environment`` path and the same backend, steps the
stored actions, and renders with the existing PNG format
(``sim2real.expert.images.write_png``).

    uv run python scripts/render_replay.py --staging data/staging/run \
        --episode-id run_ep0000 --decimation 2

Images are written under ``<run>/images/<episode_id>/`` and referenced in the
episode sidecar and manifest with their SHA-256. Existing image files are never
overwritten. The script refuses an estimate above 1 GiB or 10 minutes unless
``--i-approved`` is given. Depth is optional and OFF by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sim2real.expert.images import write_png
from sim2real.expert.rollout import derive_episode_seed
from sim2real.expert.staging import StagingBuffer
from sim2real.sim import create_backend
from sim2real.sim.reset_handler import reset_environment

MAX_UNAPPROVED_GIB = 1.0
MAX_UNAPPROVED_MINUTES = 10.0
DEFAULT_RENDER_SECONDS = 0.25
DEFAULT_CAMERAS = ("overhead", "wrist")


def renderer_name() -> str:
    """Record the renderer label used for a replay.

    This is an explicit label, not a probe: the default backend uses llvmpipe;
    the D3D12 paths are opt-in environment settings.  ``SIM2REAL_RENDERER`` can
    override the label for a different machine.
    """
    override = os.environ.get("SIM2REAL_RENDERER")
    if override:
        return override
    if os.environ.get("GALLIUM_DRIVER") == "d3d12":
        if os.environ.get("MESA_D3D12_DEFAULT_ADAPTER_NAME") == "NVIDIA":
            return "d3d12+nvidia"
        return "d3d12"
    return "llvmpipe"


class ReplayError(RuntimeError):
    """Raised when a record cannot be replayed."""


class ReplayRefused(RuntimeError):
    """Raised when the estimate needs explicit approval."""


@dataclass(frozen=True)
class ReplayEstimate:
    episodes: int
    frames: int
    bytes: int
    gib: float
    minutes: float


def estimate_replay(
    records: Sequence[Any],
    *,
    decimation: int = 1,
    resolution: tuple[int, int] = (256, 256),
    render_seconds: float = DEFAULT_RENDER_SECONDS,
    cameras: Sequence[str] = DEFAULT_CAMERAS,
) -> ReplayEstimate:
    if decimation < 1:
        raise ReplayError("decimation must be >= 1")
    height, width = resolution
    frames = sum(math.ceil(len(record.steps) / decimation) * len(cameras) for record in records)
    total = frames * width * height * 3  # RGB uint8 only; depth is OFF.
    return ReplayEstimate(
        episodes=len(records),
        frames=frames,
        bytes=total,
        gib=total / (1024.0**3),
        minutes=frames * render_seconds / 60.0,
    )


def ensure_approved(estimate: ReplayEstimate, approved: bool) -> None:
    if approved:
        return
    if estimate.gib > MAX_UNAPPROVED_GIB or estimate.minutes > MAX_UNAPPROVED_MINUTES:
        raise ReplayRefused(
            f"estimated {estimate.gib:.3f} GiB / {estimate.minutes:.1f} min exceeds "
            f"the unapproved limits ({MAX_UNAPPROVED_GIB} GiB / "
            f"{MAX_UNAPPROVED_MINUTES} min); pass --i-approved after approval"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate(record: Any) -> None:
    if not getattr(record, "steps", None):
        raise ReplayError(f"{getattr(record, 'episode_id', '?')}: no recorded steps")
    if getattr(record, "episode_index", None) is None:
        raise ReplayError(f"{record.episode_id}: missing episode_index")
    if getattr(record, "seed", None) is None:
        raise ReplayError(f"{record.episode_id}: missing seed")
    first = np.asarray(record.steps[0].action)
    if first.shape != (8,) or not np.all(np.isfinite(first)):
        raise ReplayError(f"{record.episode_id}: action must be a finite 8-vector")


def replay_record(
    record: Any,
    backend: Any,
    out_root: str | Path,
    *,
    decimation: int = 1,
    cameras: Sequence[str] = DEFAULT_CAMERAS,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Replay one record and write PNGs; never overwrite existing images."""
    _validate(record)
    if decimation < 1:
        raise ReplayError("decimation must be >= 1")
    profile = str(record.randomization_report.get("profile", "default"))
    randomization_dir = Path(backend.config_dir).parent / "randomization"
    reset = reset_environment(
        backend,
        derive_episode_seed(record.seed, record.episode_index),
        randomization_enabled=profile != "none",
        randomization_profile=profile,
        task_instruction=record.task_instruction,
        randomization_dir=randomization_dir,
        episode_index=record.episode_index,
    )
    if not reset.success:
        raise ReplayError(f"{record.episode_id}: reset failed: {reset.message}")
    set_context = getattr(backend, "set_collision_context", None)
    if callable(set_context):
        set_context(record.task_object_id)

    output_root = Path(out_root).expanduser().resolve()
    episode_dir = output_root / "images" / record.episode_id
    images: list[dict[str, Any]] = []
    max_joint_error = 0.0
    started = time.perf_counter()
    for index in range(len(record.steps)):
        if index > 0:
            action = np.asarray(record.steps[index].action, dtype=np.float64)
            observation = backend.step(action[:7], float(action[7])).observation
            recorded = np.asarray(record.steps[index].joint_positions, dtype=np.float64)
            max_joint_error = max(
                max_joint_error,
                float(np.max(np.abs(observation.robot_state.joint_positions - recorded))),
            )
        if index % decimation != 0:
            continue
        for camera in cameras:
            frame = backend.render(camera)
            path = episode_dir / f"{index:04d}_{camera}.png"
            skipped = path.exists() and not overwrite
            if not skipped:
                write_png(path, frame.rgb)
            images.append(
                {
                    "step_index": int(index),
                    "camera": str(camera),
                    "path": str(path.relative_to(output_root)),
                    "sha256": _sha256(path),
                    "skipped": bool(skipped),
                }
            )

    max_object_error: float | None = None
    if record.final_object_poses:
        current = backend.get_observation().object_poses
        errors = [
            float(
                np.linalg.norm(
                    np.asarray(current[name], dtype=np.float64)[:3, 3]
                    - np.asarray(recorded, dtype=np.float64)[:3, 3]
                )
            )
            for name, recorded in record.final_object_poses.items()
            if name in current
        ]
        max_object_error = max(errors) if errors else None
    return {
        "episode_id": record.episode_id,
        "frames": len(images),
        "renderer": renderer_name(),
        "max_joint_error_rad": max_joint_error,
        "max_object_error_m": max_object_error,
        "seconds": time.perf_counter() - started,
        "images": images,
    }


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def update_manifest(
    buffer: StagingBuffer,
    episode_id: str,
    images: list[dict[str, Any]],
    *,
    replay: dict[str, Any] | None = None,
) -> None:
    """Record image references, renderer, and replay fidelity in the metadata."""
    sidecar_path = buffer.sidecar_path(episode_id)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["images"] = images
    if replay is not None:
        sidecar["renderer"] = replay.get("renderer", renderer_name())
        sidecar["replay"] = {
            key: replay.get(key)
            for key in (
                "renderer",
                "max_joint_error_rad",
                "max_object_error_m",
                "frames",
                "seconds",
            )
        }
    _atomic_write(sidecar_path, json.dumps(sidecar, sort_keys=True, allow_nan=False) + "\n")
    rows: list[dict[str, Any]] = []
    for line in buffer.manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("episode_id")) == episode_id:
            row["images"] = images
            if replay is not None:
                row["renderer"] = replay.get("renderer", renderer_name())
                row["replay"] = sidecar["replay"]
        rows.append(row)
    _atomic_write(
        buffer.manifest_path,
        "\n".join(json.dumps(row, sort_keys=True, allow_nan=False) for row in rows) + "\n",
    )


def _staging_buffer(path: Path) -> StagingBuffer:
    if (path / "manifest.jsonl").is_file():
        return StagingBuffer(path.parent, path.name)
    runs = sorted(
        candidate for candidate in path.iterdir() if (candidate / "manifest.jsonl").is_file()
    )
    if not runs:
        raise ReplayError(f"no manifest.jsonl under {path}")
    return StagingBuffer(runs[-1].parent, runs[-1].name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", required=True)
    parser.add_argument("--episode-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--decimation", type=int, default=1)
    parser.add_argument("--out", default=None)
    parser.add_argument("--backend", default="mujoco", choices=("mujoco", "fake"))
    parser.add_argument("--config-dir", default="configs/sim")
    parser.add_argument("--render-seconds", type=float, default=DEFAULT_RENDER_SECONDS)
    parser.add_argument("--resolution", default="256x256")
    parser.add_argument("--i-approved", action="store_true")
    args = parser.parse_args(argv)

    width, height = (int(value) for value in args.resolution.lower().split("x"))
    buffer = _staging_buffer(Path(args.staging).expanduser().resolve())
    records = [
        record
        for record in buffer.iter_records()
        if not args.episode_id or record.episode_id in set(args.episode_id)
    ]
    if args.limit is not None:
        records = records[: max(0, args.limit)]
    if not records:
        raise ReplayError("no matching staged episodes")

    estimate = estimate_replay(
        records,
        decimation=args.decimation,
        resolution=(height, width),
        render_seconds=args.render_seconds,
    )
    print(
        f"estimate: {estimate.episodes} episodes, {estimate.frames} frames, "
        f"{estimate.gib:.3f} GiB, {estimate.minutes:.1f} min"
    )
    ensure_approved(estimate, args.i_approved)

    output = Path(args.out).expanduser().resolve() if args.out else buffer.root
    backend = create_backend(args.backend, args.config_dir)
    try:
        for record in records:
            report = replay_record(
                record,
                backend,
                output,
                decimation=args.decimation,
                cameras=DEFAULT_CAMERAS,
            )
            if output == buffer.root:
                update_manifest(buffer, record.episode_id, report["images"], replay=report)
            else:
                print("note: --out differs from the staging root; manifest not updated")
            print(
                json.dumps(
                    {key: value for key, value in report.items() if key != "images"},
                    sort_keys=True,
                )
            )
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
