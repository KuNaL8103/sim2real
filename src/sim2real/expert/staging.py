"""Atomic on-disk staging for expert rollout episodes.

Layout::

    data/staging/<run_id>/
        episode_<episode_id>.npz      arrays (joints, ee, actions, stamps)
        episode_<episode_id>.json     sidecar (metadata, transitions, report)
        manifest.jsonl                one JSON row per episode

Writes are atomic: arrays and sidecar are written to temporary files in the same
directory and then ``os.replace``d into place. A crash before the rename leaves
no final file visible. Corrupt or incomplete entries are logged and skipped by
:meth:`StagingBuffer.iter_records`; one bad file never stops a batch.

Images are **off by default**. When enabled they are stored as uint8 RGB and
float16 depth at the configured decimation and resolution. The caller must print
and clear :func:`estimate_storage` *before* a large run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import uuid
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from sim2real.expert.rollout import EpisodeRecord, StepRecord
from sim2real.metadata import validate_metadata_json

LOGGER = logging.getLogger(__name__)
MANIFEST_NAME = "manifest.jsonl"


class StagingError(RuntimeError):
    """Raised for an invalid staging root or malformed entry."""


class DuplicateEpisodeError(StagingError):
    """Raised when an episode id already exists and overwrite was not requested."""


@dataclass(frozen=True)
class StagedEpisode:
    episode_id: str
    path: Path
    sidecar_path: Path
    sha256: str
    outcome: int
    seed: int
    written_at: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(payload: dict[str, Any], path: Path) -> None:
    text = json.dumps(payload, sort_keys=True, allow_nan=False)
    validate_metadata_json(text)
    path.write_text(text + "\n", encoding="utf-8")


def _record_sidecar(record: EpisodeRecord) -> dict[str, Any]:
    return {
        "episode_id": record.episode_id,
        "run_id": record.run_id,
        "episode_index": int(record.episode_index),
        "seed": int(record.seed),
        "randomization_seed": int(record.randomization_seed),
        "backend": record.backend,
        "task_instruction": record.task_instruction,
        "task_object_id": record.task_object_id,
        "task_zone_id": record.task_zone_id,
        "control_rate_hz": float(record.control_rate_hz),
        "outcome": int(record.outcome),
        "failure_reason": record.failure_reason,
        "is_perturbed": bool(record.is_perturbed),
        "perturbation_name": record.perturbation_name,
        "expected_quality": record.expected_quality,
        "started_at": record.started_at,
        "transitions": record.transitions,
        "randomization_report": record.randomization_report,
        "success_metrics": record.success_metrics,
        "state_per_step": [step.state for step in record.steps],
        "metadata": record.metadata_dict(),
    }


def _arrays(record: EpisodeRecord) -> dict[str, np.ndarray]:
    return {
        "step_index": np.asarray([step.step_index for step in record.steps], dtype=np.int64),
        "stamp_ns": np.asarray([step.stamp_ns for step in record.steps], dtype=np.int64),
        "joint_positions": np.asarray(
            [step.joint_positions for step in record.steps], dtype=np.float64
        ).reshape(-1, 7),
        "joint_velocities": np.asarray(
            [step.joint_velocities for step in record.steps], dtype=np.float64
        ).reshape(-1, 7),
        "ee_pose": np.asarray([step.ee_pose for step in record.steps], dtype=np.float64).reshape(
            -1, 4, 4
        ),
        "action": np.asarray([step.action for step in record.steps], dtype=np.float64).reshape(
            -1, 8
        ),
        "gripper_position": np.asarray(
            [step.gripper_position for step in record.steps], dtype=np.float64
        ),
        "unintended_collision": np.asarray(
            [step.unintended_collision for step in record.steps], dtype=np.uint8
        ),
    }


def _record_from_arrays(sidecar: dict[str, Any], arrays: Any) -> EpisodeRecord:
    count = int(arrays["stamp_ns"].shape[0])
    states = sidecar.get("state_per_step") or ["UNKNOWN"] * count
    steps: list[StepRecord] = []
    for index in range(count):
        steps.append(
            StepRecord(
                stamp_ns=int(arrays["stamp_ns"][index]),
                step_index=int(arrays["step_index"][index]),
                joint_positions=np.asarray(arrays["joint_positions"][index], dtype=np.float64),
                joint_velocities=np.asarray(arrays["joint_velocities"][index], dtype=np.float64),
                ee_pose=np.asarray(arrays["ee_pose"][index], dtype=np.float64),
                action=np.asarray(arrays["action"][index], dtype=np.float64),
                action_space=0,
                gripper_position=float(arrays["gripper_position"][index]),
                unintended_collision=bool(arrays["unintended_collision"][index]),
                state=str(states[index]),
            )
        )
    return EpisodeRecord(
        episode_id=str(sidecar["episode_id"]),
        run_id=str(sidecar["run_id"]),
        episode_index=int(sidecar["episode_index"]),
        seed=int(sidecar["seed"]),
        randomization_seed=int(sidecar["randomization_seed"]),
        backend=str(sidecar["backend"]),
        task_instruction=str(sidecar["task_instruction"]),
        task_object_id=str(sidecar["task_object_id"]),
        task_zone_id=str(sidecar["task_zone_id"]),
        control_rate_hz=float(sidecar["control_rate_hz"]),
        outcome=int(sidecar["outcome"]),
        failure_reason=str(sidecar["failure_reason"]),
        is_perturbed=bool(sidecar["is_perturbed"]),
        perturbation_name=sidecar.get("perturbation_name"),
        expected_quality=str(sidecar.get("expected_quality", "clean")),
        steps=steps,
        transitions=list(sidecar.get("transitions") or []),
        randomization_report=dict(sidecar.get("randomization_report") or {}),
        success_metrics=dict(sidecar.get("success_metrics") or {}),
        started_at=str(sidecar.get("started_at", "")),
        metadata=dict(sidecar.get("metadata") or {}),
    )


def estimate_storage(
    num_steps: int,
    num_episodes: int,
    resolution: tuple[int, int],
    decimation: int,
    cameras_per_step: int = 2,
    render_seconds_per_frame: float = 0.25,
) -> dict[str, float]:
    """Estimate image bytes and wall time BEFORE a run writes anything.

    RGB is 3 bytes/pixel, depth is float16 (2 bytes/pixel). Decimation keeps
    every ``decimation``-th step. Estimates are deliberately simple upper bounds.
    """
    if num_steps < 0 or num_episodes < 0 or decimation < 1:
        raise ValueError("invalid estimate arguments")
    height, width = resolution
    frames = math.ceil(num_steps / decimation) * cameras_per_step * num_episodes
    bytes_per_frame = height * width * (3 + 2)
    total_bytes = frames * bytes_per_frame
    seconds = frames * render_seconds_per_frame
    return {
        "frames": float(frames),
        "bytes": float(total_bytes),
        "gib": total_bytes / (1024.0**3),
        "seconds": seconds,
        "minutes": seconds / 60.0,
    }


class StagingBuffer:
    """Write, iterate, and verify atomically staged episodes."""

    def __init__(self, output_dir: str | Path, run_id: str) -> None:
        if not run_id or "/" in run_id or "\\" in run_id:
            raise StagingError(f"invalid run_id {run_id!r}")
        self.run_id = run_id
        self.root = Path(output_dir).expanduser().resolve() / run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / MANIFEST_NAME
        self.last_errors: list[str] = []

    # -- paths -------------------------------------------------------------
    def episode_path(self, episode_id: str) -> Path:
        return self.root / f"episode_{episode_id}.npz"

    def sidecar_path(self, episode_id: str) -> Path:
        return self.root / f"episode_{episode_id}.json"

    # -- writing -----------------------------------------------------------
    def write(self, record: EpisodeRecord, *, overwrite: bool = False) -> StagedEpisode:
        final_npz = self.episode_path(record.episode_id)
        final_json = self.sidecar_path(record.episode_id)
        if not overwrite and (final_npz.exists() or final_json.exists()):
            raise DuplicateEpisodeError(
                f"episode {record.episode_id!r} already exists in {self.root}"
            )
        if record.episode_id in self._manifest_ids() and not overwrite:
            raise DuplicateEpisodeError(
                f"episode {record.episode_id!r} already appears in {self.manifest_path}"
            )

        token = uuid.uuid4().hex
        tmp_npz = self.root / f".{record.episode_id}.{token}.npz.tmp"
        tmp_json = self.root / f".{record.episode_id}.{token}.json.tmp"
        try:
            with tmp_npz.open("wb") as handle:
                np.savez(handle, **_arrays(record))
                handle.flush()
                os.fsync(handle.fileno())
            _json_dump(_record_sidecar(record), tmp_json)
            digest = _sha256(tmp_npz)
            os.replace(tmp_npz, final_npz)
            os.replace(tmp_json, final_json)
        except Exception:
            for path in (tmp_npz, tmp_json):
                with suppress(FileNotFoundError):
                    path.unlink()
            raise

        written_at = datetime.now(UTC).isoformat()
        row = {
            "episode_id": record.episode_id,
            "seed": int(record.seed),
            "outcome": int(record.outcome),
            "path": str(final_npz.relative_to(self.root)),
            "sha256": digest,
            "written_at": written_at,
        }
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            self._remove_manifest_rows(record.episode_id, keep_last=1)
        return StagedEpisode(
            episode_id=record.episode_id,
            path=final_npz,
            sidecar_path=final_json,
            sha256=digest,
            outcome=int(record.outcome),
            seed=int(record.seed),
            written_at=written_at,
        )

    def _manifest_ids(self) -> set[str]:
        if not self.manifest_path.is_file():
            return set()
        ids: set[str] = set()
        for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
            try:
                ids.add(str(json.loads(line)["episode_id"]))
            except (KeyError, ValueError, TypeError):
                continue
        return ids

    def _remove_manifest_rows(self, episode_id: str, *, keep_last: int) -> None:
        """Keep only the newest ``keep_last`` rows for ``episode_id``."""
        if not self.manifest_path.is_file():
            return
        matching: list[str] = []
        kept: list[str] = []
        for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
            try:
                row_id = str(json.loads(line)["episode_id"])
            except (KeyError, ValueError, TypeError):
                kept.append(line)
                continue
            if row_id == episode_id:
                matching.append(line)
            else:
                kept.append(line)
        kept.extend(matching[-keep_last:] if keep_last > 0 else [])
        temporary = self.root / f".manifest.{uuid.uuid4().hex}.tmp"
        temporary.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        os.replace(temporary, self.manifest_path)

    def manifest_rows(self) -> list[dict[str, Any]]:
        if not self.manifest_path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        for number, line in enumerate(
            self.manifest_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                self.last_errors.append(f"manifest line {number}: {exc}")
        return rows

    # -- reading -----------------------------------------------------------
    def iter_records(self) -> Iterator[EpisodeRecord]:
        """Yield valid records; log and skip corrupt/incomplete entries."""
        self.last_errors = []
        for row in self.manifest_rows():
            try:
                record = self._read_row(row)
            except Exception as exc:  # noqa: BLE001 - a bad file must not stop iteration
                message = f"skipping {row.get('episode_id', row.get('path'))!r}: {exc}"
                LOGGER.error("staging: %s", message)
                self.last_errors.append(message)
                continue
            yield record

    def _read_row(self, row: dict[str, Any]) -> EpisodeRecord:
        episode_id = str(row["episode_id"])
        relative = Path(str(row["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise StagingError(f"refusing unsafe manifest path {relative}")
        npz_path = self.root / relative
        sidecar_path = self.sidecar_path(episode_id)
        if not npz_path.is_file():
            raise StagingError(f"missing array file {npz_path}")
        if not sidecar_path.is_file():
            raise StagingError(f"missing sidecar {sidecar_path}")
        expected = str(row.get("sha256", ""))
        if expected and _sha256(npz_path) != expected:
            raise StagingError(f"sha256 mismatch for {npz_path}")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        with np.load(npz_path, allow_pickle=False) as arrays:
            required = ("step_index", "stamp_ns", "joint_positions", "action")
            for key in required:
                if key not in arrays:
                    raise StagingError(f"{npz_path} is missing array {key!r}")
            count = arrays["stamp_ns"].shape[0]
            if count == 0:
                raise StagingError(f"{npz_path} has no steps")
            if arrays["joint_positions"].shape != (count, 7):
                raise StagingError(f"{npz_path} has inconsistent joint_positions shape")
            if arrays["action"].shape != (count, 8):
                raise StagingError(f"{npz_path} has inconsistent action shape")
            loaded = {key: arrays[key] for key in arrays.files}
        return _record_from_arrays(sidecar, loaded)

    def verify(self) -> dict[str, Any]:
        """Return counts and problems without raising on corrupt entries."""
        rows = self.manifest_rows()
        ids = [str(row.get("episode_id", "")) for row in rows]
        duplicates = sorted({name for name in ids if ids.count(name) > 1})
        corrupt = list(self.last_errors)
        valid = 0
        missing = 0
        for row in rows:
            try:
                self._read_row(row)
                valid += 1
            except FileNotFoundError:
                missing += 1
                corrupt.append(f"missing files for {row.get('episode_id')}")
            except Exception as exc:  # noqa: BLE001 - report, never crash
                corrupt.append(f"{row.get('episode_id')}: {exc}")
        return {
            "run_id": self.run_id,
            "root": str(self.root),
            "manifest_rows": len(rows),
            "valid": valid,
            "corrupt": len(corrupt),
            "missing": missing,
            "duplicates": duplicates,
            "problems": corrupt,
        }

    def disk_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.root.glob("*") if path.is_file())
