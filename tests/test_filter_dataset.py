"""Dataset writer, rejection log, summarizer invariant, and fake end-to-end tests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from sim2real import contracts
from sim2real.data import (
    CuratedDatasetWriter,
    DatasetWriterError,
    RejectionLog,
    VersionExistsError,
)
from sim2real.expert import (
    load_expert_config,
    load_perturbations,
    parse_instruction,
    run_episodes,
)
from sim2real.expert.staging import StagingBuffer
from sim2real.filtering.config import load_filter_config
from sim2real.filtering.decision import decide
from sim2real.filtering.pipeline import FilterPipeline
from sim2real.filtering.synthetic import MID_RANGE_ARM, make_episode
from sim2real.sim import create_backend

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_filter_config(ROOT / "configs" / "filtering" / "default.yaml")
INSTRUCTION = "pick up the red cube and place it in the left zone"


def _accepted(episode_id: str, seed: int = 0):
    episode = make_episode(episode_id=episode_id, seed=seed)
    result = decide(episode, CONFIG)
    assert result.accepted, result.reasons
    return episode, result


def test_curated_layout_info_and_parquet_schema(tmp_path):
    writer = CuratedDatasetWriter(tmp_path / "curated", "unit")
    first, first_result = _accepted("ep-0001", 1)
    second, second_result = _accepted("ep-0002", 2)
    written = writer.write([(first, first_result), (second, second_result)], config=CONFIG)

    version_dir = written.path
    assert version_dir.name == "v1.0.0"
    for relative in (
        "meta/info.json",
        "meta/episodes.jsonl",
        "meta/tasks.jsonl",
        "dataset_card.md",
    ):
        assert (version_dir / relative).is_file(), relative

    info = json.loads((version_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["schema_version"] == "curated_v1"
    assert info["dataset_name"] == "unit"
    assert info["version"] == "v1.0.0"
    assert info["fps"] == 20
    assert info["robot_type"] == "franka_panda"
    assert info["filter_config_sha256"] == CONFIG.config_hash()
    assert info["filter_config"]["soft_thresholds"]
    assert info["images"]["mode"] == "none"
    assert info["features"]["observation.state"]["shape"] == [8]
    assert info["features"]["action"]["units"][-1] == "m"
    assert info["normalization"]["mode"] == "none"

    episodes = [
        json.loads(line)
        for line in (version_dir / "meta" / "episodes.jsonl").read_text().splitlines()
    ]
    assert [row["episode_id"] for row in episodes] == ["ep-0001", "ep-0002"]
    assert [row["episode_index"] for row in episodes] == [0, 1]
    assert all(row["decision"] == contracts.DECISION_ACCEPT for row in episodes)
    tasks = [
        json.loads(line) for line in (version_dir / "meta" / "tasks.jsonl").read_text().splitlines()
    ]
    assert tasks == [{"task_index": 0, "instruction": INSTRUCTION}]

    import pyarrow.parquet as pq

    parquet_files = sorted((version_dir / "data" / "chunk-000").glob("*.parquet"))
    assert [path.name for path in parquet_files] == [
        "episode_000000.parquet",
        "episode_000001.parquet",
    ]
    table = pq.read_table(parquet_files)
    assert table.schema.names == [
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
        "observation.state",
        "action",
        "next.done",
    ]
    assert table.num_rows == len(first.steps) + len(second.steps)
    assert table.schema.field("observation.state").type.list_size == 8
    assert table.schema.field("action").type.list_size == 8
    rows = table.to_pylist()
    assert [row["episode_index"] for row in rows] == sorted(row["episode_index"] for row in rows)
    assert [row["index"] for row in rows] == list(range(len(rows)))
    assert rows[len(first.steps) - 1]["next.done"] is True
    assert rows[-1]["next.done"] is True
    assert rows[0]["observation.state"][:7] == pytest.approx(list(MID_RANGE_ARM))

    card = (version_dir / "dataset_card.md").read_text(encoding="utf-8")
    assert "NO images" in card
    assert "LeRobot-STYLE" in card


def test_writer_appends_minor_version_and_refuses_duplicates(tmp_path):
    writer = CuratedDatasetWriter(tmp_path / "curated", "unit")
    first, first_result = _accepted("ep-0001")
    initial = writer.write([(first, first_result)], config=CONFIG)
    assert initial.version == "v1.0.0"

    second, second_result = _accepted("ep-0002")
    appended = writer.write([(second, second_result)], config=CONFIG)
    assert appended.version == "v1.1.0"
    assert appended.num_episodes == 2
    assert appended.num_new_episodes == 1
    episodes = [
        json.loads(line)
        for line in (appended.path / "meta" / "episodes.jsonl").read_text().splitlines()
    ]
    assert [row["episode_index"] for row in episodes] == [0, 1]

    import pyarrow.parquet as pq

    table = pq.read_table(sorted((appended.path / "data" / "chunk-000").glob("*.parquet")))
    assert table.num_rows == len(first.steps) + len(second.steps)
    assert table.column("episode_index").to_pylist()[: len(first.steps)] == [0] * len(first.steps)
    assert table.column("episode_index").to_pylist()[-1] == 1

    with pytest.raises(VersionExistsError):
        writer.write([(second, second_result)], config=CONFIG, requested_version="v1.0.0")
    with pytest.raises(DatasetWriterError):
        writer.write([(first, first_result)], config=CONFIG)


def test_mid_write_failure_leaves_no_partial_version(tmp_path, monkeypatch):
    writer = CuratedDatasetWriter(tmp_path / "curated", "unit")
    episode, result = _accepted("ep-0001")

    def explode(*args, **kwargs):
        raise RuntimeError("simulated disk failure")

    monkeypatch.setattr(writer, "_write_tree", explode)
    with pytest.raises(RuntimeError, match="simulated disk failure"):
        writer.write([(episode, result)], config=CONFIG)
    dataset_root = tmp_path / "curated" / "unit"
    assert not any(path.name.startswith("v") for path in dataset_root.iterdir())
    assert not any(path.name.startswith(".partial") for path in dataset_root.iterdir())


def test_rejection_log_is_structured_and_disjoint_from_curated(tmp_path):
    accepted, accepted_result = _accepted("ep-accepted")
    rejected = make_episode(episode_id="ep-rejected", outcome=contracts.OUTCOME_FAILURE, seed=9)
    rejected_result = decide(rejected, CONFIG)
    assert rejected_result.decision == contracts.DECISION_REJECT

    nan_joints = np.tile(np.asarray(MID_RANGE_ARM), (len(rejected.steps), 1))
    nan_joints[0, 0] = np.nan
    invalid = make_episode(episode_id="ep-invalid", joint_positions=nan_joints)
    invalid_result = decide(invalid, CONFIG)
    assert invalid_result.decision == contracts.DECISION_INVALID

    writer = CuratedDatasetWriter(tmp_path / "curated", "unit")
    written = writer.write([(accepted, accepted_result)], config=CONFIG)
    log = RejectionLog(tmp_path / "rejected", "run-1")
    log.append(rejected, rejected_result, staging_reference="run/ep-rejected.npz")
    log.append(invalid, invalid_result, staging_reference="run/ep-invalid.npz")

    lines = (tmp_path / "rejected" / "run-1" / "rejections.jsonl").read_text().splitlines()
    assert len(lines) == 2
    rows = [json.loads(line) for line in lines]
    assert {row["episode_id"] for row in rows} == {"ep-rejected", "ep-invalid"}
    rejected_row = next(row for row in rows if row["episode_id"] == "ep-rejected")
    assert rejected_row["reasons"][0]["code"] == "NOT_SUCCESS"
    assert rejected_row["metrics"]["dimensionless_squared_jerk"] > 0
    invalid_row = next(row for row in rows if row["episode_id"] == "ep-invalid")
    assert "NAN_VALUES" in invalid_row["reason_codes"]
    assert invalid_row["metrics"] is None
    assert (tmp_path / "rejected" / "run-1" / "episodes" / "ep-rejected.json").is_file()

    curated_ids = {
        json.loads(line)["episode_id"]
        for line in (written.path / "meta" / "episodes.jsonl").read_text().splitlines()
    }
    rejected_ids = {row["episode_id"] for row in rows}
    assert curated_ids.isdisjoint(rejected_ids)


def _load_summarizer():
    path = ROOT / "scripts" / "summarize_staging.py"
    spec = importlib.util.spec_from_file_location("summarize_staging_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_summarizer_categories_sum_to_episode_count(tmp_path):
    expert = load_expert_config(ROOT / "configs" / "expert")
    perturbations = load_perturbations(expert)
    names = ["clean", *perturbations.enabled_names]
    buffer = StagingBuffer(tmp_path / "staging", "run")
    for index, name in enumerate(names):
        buffer.write(
            make_episode(
                episode_id=f"ep-{name}",
                seed=index,
                is_perturbed=name != "clean",
                perturbation_name=None if name == "clean" else name,
            )
        )
    summary = _load_summarizer().build_summary(buffer, buffer.root)
    assert summary["episodes"] == len(names)
    assert sum(summary["perturbations"].values()) == summary["episodes"]
    assert set(perturbations.enabled_names).issubset(summary["perturbations"])
    assert "clean" in summary["perturbations"]
    assert sum(summary["outcomes"].values()) == summary["episodes"]


def test_fake_backend_end_to_end_acceptance(tmp_path):
    expert = load_expert_config(ROOT / "configs" / "expert")
    perturbations = load_perturbations(expert)
    backend = create_backend("fake", str(ROOT / "configs" / "sim"))
    try:
        task = parse_instruction(INSTRUCTION, backend.configs.scene, expert.instructions_file)
        seeds = list(range(5000, 5030))
        records = run_episodes(
            backend,
            task,
            seeds,
            instruction=INSTRUCTION,
            run_id="e2e",
            perturbations=perturbations,
            perturbed_fraction=0.4,
            config=expert,
        )
    finally:
        backend.close()

    buffer = StagingBuffer(tmp_path / "staging", "e2e")
    for record in records:
        buffer.write(record)

    pipeline = FilterPipeline(CONFIG)
    counts: dict[str, list[int]] = {}
    invalid = 0
    for record in buffer.iter_records():
        result = pipeline.process(record)
        invalid += int(result.decision == contracts.DECISION_INVALID)
        name = record.perturbation_name or "clean"
        counts.setdefault(name, [0, 0])
        counts[name][0] += int(result.accepted)
        counts[name][1] += 1

    print("fake end-to-end acceptance:", counts)
    assert invalid == 0
    assert counts["clean"][0] / counts["clean"][1] > 0.5
    assert counts.get("bad_grasp", [0, 0])[0] == 0
    assert counts.get("bad_grasp", [0, 0])[1] > 0


@pytest.mark.sim
def test_mujoco_end_to_end_filter_completes(tmp_path):
    expert = load_expert_config(ROOT / "configs" / "expert")
    backend = create_backend("mujoco", str(ROOT / "configs" / "sim"))
    try:
        task = parse_instruction(INSTRUCTION, backend.configs.scene, expert.instructions_file)
        records = run_episodes(
            backend,
            task,
            list(range(5100, 5106)),
            instruction=INSTRUCTION,
            run_id="e2e_mujoco",
            config=expert,
        )
    finally:
        backend.close()

    mujoco_config = load_filter_config(ROOT / "configs" / "filtering" / "mujoco.yaml")
    pipeline = FilterPipeline(mujoco_config)
    results = [pipeline.process(record) for record in records]
    print("mujoco end-to-end:", [result.decision for result in results])
    assert len(results) == 6
    assert all(result.decision != contracts.DECISION_INVALID for result in results)
    assert all(result.metrics is not None for result in results)
