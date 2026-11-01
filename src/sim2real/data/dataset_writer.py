"""Versioned ``curated_v1`` dataset writer.

This layout is LeRobot-*style*: the column names and the episode/task metadata
follow LeRobot's conventions, but it has **not** been validated against the
LeRobot loader. That validation belongs to Task 7.

Layout::

    data/curated/<dataset_name>/v<MAJOR.MINOR.PATCH>/
        meta/info.json
        meta/episodes.jsonl
        meta/tasks.jsonl
        dataset_card.md
        data/chunk-000/episode_<index>.parquet

Rules:

* a version is written into a ``.partial-*`` directory and atomically renamed;
* an existing version is never overwritten;
* each write creates a **new minor version** containing the previous accepted
  episodes plus the new ones (a complete snapshot, not an incremental patch);
* episode indices stay contiguous and duplicate episode ids are refused;
* staging images are never copied. When staging has no images the dataset
  records ``images: none`` and the dataset card says so prominently.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sim2real.filtering.config import FilterConfig
from sim2real.filtering.decision import FilterResult

SCHEMA_VERSION = "curated_v1"
IMAGE_NONE_TEXT = (
    "This dataset contains NO images. Staging ran with image capture disabled, "
    "so the parquet files hold states and actions only. It cannot train a "
    "vision-language-action model until RGB-D frames are rendered and attached "
    "in a later task."
)


class DatasetWriterError(RuntimeError):
    """Raised for invalid dataset names, duplicate episodes, or unsafe writes."""


class VersionExistsError(DatasetWriterError):
    """Raised when the requested version directory already exists."""


@dataclass(frozen=True)
class DatasetWriteResult:
    version: str
    path: Path
    num_episodes: int
    num_new_episodes: int
    info: dict[str, Any]


def _code_version() -> str:
    """Best-effort git commit without spawning a process."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        git = parent / ".git"
        if git.is_dir():
            head = git / "HEAD"
            if head.is_file():
                text = head.read_text(encoding="utf-8").strip()
                if text.startswith("ref: "):
                    ref = git / text[5:]
                    if ref.is_file():
                        return ref.read_text(encoding="utf-8").strip()
                return text
        elif git.is_file():
            return git.read_text(encoding="utf-8").strip()
    from sim2real import __version__

    return f"sim2real-{__version__}"


def _state_vector(step: Any) -> list[float]:
    return [float(value) for value in step.joint_positions] + [float(step.gripper_position)]


def _action_vector(step: Any) -> list[float]:
    return [float(value) for value in step.action]


def _episode_rows(
    record: Any,
    result: FilterResult,
    episode_index: int,
    index_offset: int,
    task_index: int,
) -> list[dict[str, Any]]:
    count = len(record.steps)
    start_ns = int(record.steps[0].stamp_ns)
    rows: list[dict[str, Any]] = []
    for frame_index, step in enumerate(record.steps):
        rows.append(
            {
                "timestamp": (int(step.stamp_ns) - start_ns) * 1e-9,
                "frame_index": frame_index,
                "episode_index": episode_index,
                "index": index_offset + frame_index,
                "task_index": task_index,
                "observation.state": _state_vector(step),
                "action": _action_vector(step),
                "next.done": frame_index == count - 1,
            }
        )
    return rows


def _episode_meta(
    record: Any,
    result: FilterResult,
    episode_index: int,
    task_index: int,
) -> dict[str, Any]:
    metrics = result.metrics
    return {
        "episode_index": episode_index,
        "episode_id": str(record.episode_id),
        "seed": int(record.seed),
        "task": str(record.task_instruction),
        "task_index": task_index,
        "length": len(record.steps),
        "duration_s": (float(result.metrics.duration_s) if result.metrics is not None else None),
        "outcome": int(record.outcome),
        "is_perturbed": bool(record.is_perturbed),
        "perturbation_name": record.perturbation_name,
        "expected_quality": record.expected_quality,
        "decision": int(result.decision),
        "acceptance_score": result.acceptance_score,
        "reason_codes": [reason.code for reason in result.reasons],
        "metrics": metrics.to_dict() if metrics is not None else None,
        "source_run_id": str(record.run_id),
        "backend": str(record.backend),
        "filter_config_sha256": result.config_hash,
    }


def _features() -> dict[str, Any]:
    return {
        "timestamp": {
            "dtype": "float64",
            "shape": [1],
            "units": "s",
            "frame": "episode_start",
        },
        "frame_index": {"dtype": "int64", "shape": [1], "units": "frame", "frame": "none"},
        "episode_index": {"dtype": "int64", "shape": [1], "units": "index", "frame": "none"},
        "index": {"dtype": "int64", "shape": [1], "units": "index", "frame": "none"},
        "task_index": {"dtype": "int64", "shape": [1], "units": "index", "frame": "none"},
        "observation.state": {
            "dtype": "float64",
            "shape": [8],
            "units": ["rad"] * 7 + ["m"],
            "frame": "panda_joint1..7 + panda_finger_joint1",
        },
        "action": {
            "dtype": "float64",
            "shape": [8],
            "units": ["rad"] * 7 + ["m"],
            "frame": "panda_joint1..7 + panda_finger_joint1",
        },
        "next.done": {"dtype": "bool", "shape": [1], "units": "flag", "frame": "none"},
    }


def _normalization_placeholders() -> dict[str, Any]:
    return {
        "mode": "none",
        "note": "No dataset statistics were computed. Task 7/8 must fit normalisation.",
        "features": {
            name: {"mean": None, "std": None, "min": None, "max": None}
            for name in ("observation.state", "action")
        },
    }


def _chunk_tables(version_dir: Path) -> list[Any]:
    import pyarrow.parquet as pq

    chunk_dir = version_dir / "data" / "chunk-000"
    if not chunk_dir.is_dir():
        return []
    return [pq.read_table(path) for path in sorted(chunk_dir.glob("*.parquet"))]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "\n".join(json.dumps(row, sort_keys=True, allow_nan=False) for row in rows)
    path.write_text(text + ("\n" if rows else ""), encoding="utf-8")


class CuratedDatasetWriter:
    def __init__(self, root: str | Path, dataset_name: str) -> None:
        if not dataset_name or any(character in dataset_name for character in "/\\"):
            raise DatasetWriterError(f"invalid dataset_name {dataset_name!r}")
        self.root = Path(root).expanduser().resolve()
        self.dataset_name = dataset_name
        self.dataset_root = self.root / dataset_name

    # -- versions ----------------------------------------------------------
    def versions(self) -> list[tuple[tuple[int, int, int], str]]:
        found: list[tuple[tuple[int, int, int], str]] = []
        if not self.dataset_root.is_dir():
            return found
        for path in self.dataset_root.iterdir():
            if not path.is_dir() or not path.name.startswith("v"):
                continue
            parts = path.name[1:].split(".")
            if len(parts) != 3 or not all(part.isdigit() for part in parts):
                continue
            found.append(((int(parts[0]), int(parts[1]), int(parts[2])), path.name))
        return sorted(found)

    def latest_version_dir(self) -> Path | None:
        versions = self.versions()
        return self.dataset_root / versions[-1][1] if versions else None

    def next_version(self) -> str:
        versions = self.versions()
        if not versions:
            return "v1.0.0"
        major, minor, _patch = versions[-1][0]
        return f"v{major}.{minor + 1}.0"

    # -- writing -----------------------------------------------------------
    def write(
        self,
        accepted: list[tuple[Any, FilterResult]],
        *,
        config: FilterConfig,
        requested_version: str | None = None,
        images: dict[str, Any] | None = None,
    ) -> DatasetWriteResult:
        if not accepted:
            raise DatasetWriterError("no accepted episodes to write")
        latest = self.latest_version_dir()
        version = requested_version or self.next_version()
        if not _valid_version(version):
            raise DatasetWriterError(f"invalid version {version!r}")
        final = self.dataset_root / version
        if final.exists():
            raise VersionExistsError(f"refusing to overwrite existing version {final}")

        prior_meta = _read_jsonl(latest / "meta" / "episodes.jsonl") if latest else []
        prior_tasks = _read_jsonl(latest / "meta" / "tasks.jsonl") if latest else []
        prior_ids = {str(row["episode_id"]) for row in prior_meta}
        new_ids = {str(record.episode_id) for record, _ in accepted}
        duplicates = sorted(prior_ids & new_ids)
        if duplicates:
            raise DatasetWriterError(f"episode ids already curated: {duplicates}")
        if len(new_ids) != len(accepted):
            raise DatasetWriterError("duplicate episode ids in the accepted batch")

        tasks: list[dict[str, Any]] = [
            {"task_index": int(row["task_index"]), "instruction": str(row["instruction"])}
            for row in prior_tasks
        ]
        task_index = {row["instruction"]: row["task_index"] for row in tasks}
        for record, _result in accepted:
            instruction = str(record.task_instruction)
            if instruction not in task_index:
                task_index[instruction] = len(tasks)
                tasks.append({"task_index": task_index[instruction], "instruction": instruction})

        prior_tables = _chunk_tables(latest) if latest else []
        prior_episode_count = len(prior_meta)
        prior_row_count = sum(table.num_rows for table in prior_tables)
        if prior_meta:
            expected_rows = sum(int(row["length"]) for row in prior_meta)
            if prior_row_count != expected_rows:
                raise DatasetWriterError(
                    f"existing version has {prior_row_count} parquet rows but "
                    f"episodes.jsonl declares {expected_rows}"
                )

        new_rows: list[dict[str, Any]] = []
        new_meta: list[dict[str, Any]] = []
        for offset, (record, result) in enumerate(accepted):
            episode_index = prior_episode_count + offset
            rows = _episode_rows(
                record,
                result,
                episode_index,
                prior_row_count + len(new_rows),
                task_index[str(record.task_instruction)],
            )
            new_rows.extend(rows)
            new_meta.append(
                _episode_meta(
                    record,
                    result,
                    episode_index,
                    task_index[str(record.task_instruction)],
                )
            )

        self.dataset_root.mkdir(parents=True, exist_ok=True)
        partial = self.dataset_root / f".partial-{uuid.uuid4().hex}"
        try:
            self._write_tree(
                partial,
                prior_tables=prior_tables,
                new_rows=new_rows,
                episodes=prior_meta + new_meta,
                tasks=tasks,
                accepted=new_meta,
                config=config,
                version=version,
                images=images,
            )
            os.rename(partial, final)
        except Exception:
            shutil.rmtree(partial, ignore_errors=True)
            raise

        info = json.loads((final / "meta" / "info.json").read_text(encoding="utf-8"))
        return DatasetWriteResult(
            version=version,
            path=final,
            num_episodes=len(prior_meta) + len(new_meta),
            num_new_episodes=len(new_meta),
            info=info,
        )

    def _write_tree(
        self,
        target: Path,
        *,
        prior_tables: list[Any],
        new_rows: list[dict[str, Any]],
        episodes: list[dict[str, Any]],
        tasks: list[dict[str, Any]],
        accepted: list[dict[str, Any]],
        config: FilterConfig,
        version: str,
        images: dict[str, Any] | None,
    ) -> None:
        (target / "meta").mkdir(parents=True)
        chunk = target / "data" / "chunk-000"
        chunk.mkdir(parents=True)

        table = _parquet_table(prior_tables, new_rows)
        import pyarrow.parquet as pq

        offset = 0
        for row in episodes:
            length = int(row["length"])
            piece = table.slice(offset, length)
            episode_index = int(row["episode_index"])
            pq.write_table(
                piece,
                chunk / f"episode_{episode_index:06d}.parquet",
                compression="snappy",
            )
            offset += length
        if offset != table.num_rows:
            raise DatasetWriterError(
                f"episode lengths sum to {offset} but parquet has {table.num_rows} rows"
            )
        _write_jsonl(target / "meta" / "episodes.jsonl", episodes)
        _write_jsonl(target / "meta" / "tasks.jsonl", tasks)

        image_section = images or {
            "mode": "none",
            "reason": "staging has no images; no frames were copied",
        }
        config_copy = config.to_dict()
        info = {
            "schema_version": SCHEMA_VERSION,
            "dataset_name": self.dataset_name,
            "version": version,
            "fps": _fps(episodes),
            "robot_type": "franka_panda",
            "features": _features(),
            "normalization": _normalization_placeholders(),
            "filter_config": config_copy,
            "filter_config_sha256": config.config_hash(),
            "source_run_ids": sorted({str(row["source_run_id"]) for row in episodes}),
            "created_at": datetime.now(UTC).isoformat(),
            "code_version": _code_version(),
            "images": image_section,
            "num_episodes": len(episodes),
            "num_frames": int(table.num_rows),
            "notes": [
                "LeRobot-STYLE, not validated against the LeRobot loader until Task 7",
                "Columns follow LeRobot naming; no video or image tensors are present",
            ],
        }
        text = json.dumps(info, sort_keys=True, allow_nan=False)
        (target / "meta" / "info.json").write_text(text + "\n", encoding="utf-8")
        (target / "dataset_card.md").write_text(
            _dataset_card(info, accepted, episodes), encoding="utf-8"
        )


def _valid_version(version: str) -> bool:
    if not version.startswith("v"):
        return False
    parts = version[1:].split(".")
    return len(parts) == 3 and all(part.isdigit() for part in parts)


def _fps(episodes: list[dict[str, Any]]) -> int:
    rates = [
        (int(row["length"]) - 1) / float(row["duration_s"])
        for row in episodes
        if row.get("duration_s") and int(row["length"]) > 1
    ]
    if not rates:
        return 20
    return int(round(sum(rates) / len(rates)))


def _parquet_table(prior_tables: list[Any], new_rows: list[dict[str, Any]]) -> Any:
    import pyarrow as pa

    schema = pa.schema(
        [
            ("timestamp", pa.float64()),
            ("frame_index", pa.int64()),
            ("episode_index", pa.int64()),
            ("index", pa.int64()),
            ("task_index", pa.int64()),
            ("observation.state", pa.list_(pa.float64(), 8)),
            ("action", pa.list_(pa.float64(), 8)),
            ("next.done", pa.bool_()),
        ]
    )
    arrays = {
        "timestamp": [float(row["timestamp"]) for row in new_rows],
        "frame_index": [int(row["frame_index"]) for row in new_rows],
        "episode_index": [int(row["episode_index"]) for row in new_rows],
        "index": [int(row["index"]) for row in new_rows],
        "task_index": [int(row["task_index"]) for row in new_rows],
        "observation.state": [list(row["observation.state"]) for row in new_rows],
        "action": [list(row["action"]) for row in new_rows],
        "next.done": [bool(row["next.done"]) for row in new_rows],
    }
    new_table = pa.Table.from_pydict(arrays, schema=schema)
    if not prior_tables:
        return new_table
    return pa.concat_tables([*prior_tables, new_table])


def _image_note(info: dict[str, Any]) -> str:
    if info["images"].get("mode") == "none":
        return IMAGE_NONE_TEXT
    return "Staging image references are recorded in meta/info.json; arrays are not copied."


def _dataset_card(
    info: dict[str, Any], accepted: list[dict[str, Any]], episodes: list[dict[str, Any]]
) -> str:
    return f"""# {info["dataset_name"]} ({info["version"]})

`curated_v1`, generated by the sim2real trajectory filter on {info["created_at"]}.

## What this is

- robot: `{info["robot_type"]}`
- episodes: **{info["num_episodes"]}**, frames: **{info["num_frames"]}**, fps: **{info["fps"]}**
- state/action: 7 arm joints (rad) plus one gripper finger joint (m)
- coordinate conventions: MuJoCo world frame, z up; end-effector poses use the
  `panda_hand` frame with the TCP offset from `configs/sim/panda.yaml`
- source runs: {", ".join(info["source_run_ids"]) or "none"}

## Images: {info["images"].get("mode", "unknown")}

**{_image_note(info)}**

## Filtering

The acceptance decision is deterministic and never reads perturbation labels.
Metrics and thresholds are documented in `docs/FILTER_METRICS.md`.
Filter config SHA-256: `{info["filter_config_sha256"]}`.

## Limitations

- **LeRobot-STYLE only.** The column names follow LeRobot conventions, but this
  layout has not been validated against the LeRobot loader (Task 7).
- Episodes are accepted from the scripted expert, which fails on some object
  poses; accepted data may skew toward easy poses.
- No normalisation statistics are computed (`normalization.mode = none`).
- No images, video, or camera calibration are included.
"""


__all__ = [
    "IMAGE_NONE_TEXT",
    "SCHEMA_VERSION",
    "CuratedDatasetWriter",
    "DatasetWriteResult",
    "DatasetWriterError",
    "VersionExistsError",
]
