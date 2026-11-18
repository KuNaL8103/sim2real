"""CPU tests for the explicit pipeline orchestrator.

These tests use the fake backend, fake policies, and the labelled fake TRAIN
executor.  Nothing here loads torch, a GPU, or a simulator GUI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from sim2real.evaluation import FlywheelController, load_evaluation_config
from sim2real.orchestrator import (
    ApprovalRequiredError,
    CheckpointVerificationError,
    CorruptStageOutputError,
    CycleLimitError,
    PipelineConfigError,
    PipelineOrchestrator,
    SeedLedgerError,
    Stage,
    StageError,
    StageStatus,
    load_pipeline_config,
)
from sim2real.orchestrator.pipeline_orchestrator import RunFlags

REPO_ROOT = Path(__file__).resolve().parents[1]
FAKE_CONFIG = REPO_ROOT / "configs" / "pipeline" / "fake_e2e.yaml"


def _raw_config() -> dict:
    return yaml.safe_load(FAKE_CONFIG.read_text(encoding="utf-8"))


def _write_config(tmp_path: Path, mutate=None) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    raw = _raw_config()
    raw["pipeline"]["run_root"] = str(tmp_path / "runs")
    if mutate is not None:
        mutate(raw)
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _all_fake(mutate=None):
    def apply(raw):
        raw["pipeline"]["stages"] = {stage.value: "fake" for stage in Stage}
        raw["pipeline"]["max_wall_time_s"] = 900.0
        if mutate is not None:
            mutate(raw)

    return apply


def _orchestrator(tmp_path: Path, mutate=None, *, run_id: str = "run_test") -> PipelineOrchestrator:
    config = load_pipeline_config(_write_config(tmp_path, mutate))
    return PipelineOrchestrator(config, repo_root=REPO_ROOT, run_id=run_id)


@pytest.fixture(scope="module")
def completed_pipeline(tmp_path_factory):
    """One real fake-backend cycle: real scripts, fake TRAIN only."""
    root = tmp_path_factory.mktemp("pipeline_e2e")
    config = load_pipeline_config(_write_config(root, None))
    orchestrator = PipelineOrchestrator(config, repo_root=REPO_ROOT, run_id="run_e2e")
    manifests = orchestrator.run(cycles=1, dry_run=False)
    return orchestrator, manifests[0]


# 1. full fake cycle with complete lineage
def test_full_fake_cycle_writes_complete_lineage(completed_pipeline) -> None:
    orchestrator, manifest = completed_pipeline
    assert manifest["status"] == "COMPLETED"
    assert set(manifest["stages"]) == {stage.value for stage in Stage}
    for name, stage in manifest["stages"].items():
        assert stage["status"] in {StageStatus.COMPLETED.value, StageStatus.SKIPPED.value}, name
        assert stage["stage_id"]
    assert manifest["seeds"]["generation"] == list(range(20950, 20960))
    assert len(manifest["source_episode_ids"]) == 10
    assert manifest["dataset"]["dataset_hash"]
    assert manifest["checkpoint"]["metadata_hash"]
    assert manifest["checkpoint"]["verification"] == "fake_executor"
    assert manifest["evaluation_run_id"] == "eval_cycle_0001"
    assert manifest["failure_distribution_id"]
    assert manifest["flywheel"]["decision"] == "REQUEST_TARGETED_GENERATION"
    assert manifest["config_hash"]
    assert (orchestrator.manifests_dir / "cycle_0001.json").is_file()
    # The real filter and evaluation ran; the curated dataset is a versioned dir.
    curated = Path(manifest["dataset"]["curated_version_dir"])
    assert curated.is_dir()
    assert (curated / "meta" / "info.json").is_file()
    assert (Path(manifest["stages"]["EVALUATE"]["outputs"][0]["path"]) / "records.jsonl").is_file()
    # The flywheel ledger lives inside the run directory.
    ledger = orchestrator.run_root / "flywheel" / "ledger.jsonl"
    assert ledger.is_file()
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 1


# 2. seed-ledger enforcement
def test_seed_ledger_refuses_owned_range(tmp_path: Path) -> None:
    def owned(raw):
        raw["pipeline"]["generation"]["seed"] = 20900  # safety range
        raw["pipeline"]["stages"] = {stage.value: "fake" for stage in Stage}

    orchestrator = _orchestrator(tmp_path, owned)
    with pytest.raises(SeedLedgerError, match="20900"):
        orchestrator.run(cycles=1, dry_run=False)

    def mujoco_e2e(raw):
        raw["pipeline"]["generation"]["seed"] = 20950
        raw["pipeline"]["generation"]["backend"] = "mujoco"
        raw["pipeline"]["stages"] = {stage.value: "fake" for stage in Stage}

    orchestrator = _orchestrator(tmp_path / "mujoco", mujoco_e2e)
    with pytest.raises(SeedLedgerError, match="fake-backend"):
        orchestrator.run(cycles=1, dry_run=False)


# 3. cycle limits and the new-checkpoint gate
def test_cycle_limits_and_new_checkpoint_gate(tmp_path: Path) -> None:
    def one_cycle(raw):
        raw["pipeline"]["max_cycles"] = 1
        raw["pipeline"]["stages"] = {stage.value: "fake" for stage in Stage}

    orchestrator = _orchestrator(tmp_path / "cap", one_cycle)
    with pytest.raises(CycleLimitError, match="max_cycles"):
        orchestrator.run(cycles=2, dry_run=False)

    def tiny_wall(raw):
        raw["pipeline"]["max_wall_time_s"] = 0.001
        raw["pipeline"]["stages"] = {stage.value: "fake" for stage in Stage}

    orchestrator = _orchestrator(tmp_path / "wall", tiny_wall)
    with pytest.raises(CycleLimitError, match="max_wall_time_s"):
        orchestrator.run(cycles=1, dry_run=False)

    orchestrator = _orchestrator(tmp_path / "gate", _all_fake())
    manifest = orchestrator.run(cycles=1, dry_run=False)[0]
    checkpoint = manifest["checkpoint"]["path"]
    with pytest.raises(CycleLimitError, match="--approve-cycle"):
        orchestrator._check_cycle_gate(2, RunFlags(approve_cycle=False))
    # A fake TRAIN cycle is allowed once the cycle itself is explicitly approved.
    orchestrator._check_cycle_gate(2, RunFlags(approve_cycle=True))
    # The same checkpoint is not "a new verified checkpoint".
    with pytest.raises(CycleLimitError, match="not new"):
        orchestrator._check_cycle_gate(
            2, RunFlags(approve_cycle=True, supplied_checkpoint=checkpoint)
        )


# 4. resume: skip unchanged stages, re-run after an input change
def test_resume_skips_completed_stages_and_reruns_after_change(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path, _all_fake())

    def crash_on_train(stage, context, executor):
        if stage is Stage.TRAIN:
            raise RuntimeError("simulated crash after FILTER")
        return orchestrator._run_fake(stage, context)

    orchestrator._stage_runner = crash_on_train
    with pytest.raises(StageError, match="simulated crash"):
        orchestrator.run(cycles=1, dry_run=False)

    first = json.loads((orchestrator.stages_dir / "GENERATE.json").read_text())
    assert first["status"] == StageStatus.COMPLETED.value
    assert json.loads((orchestrator.stages_dir / "FILTER.json").read_text())["status"] == (
        StageStatus.COMPLETED.value
    )
    assert json.loads((orchestrator.stages_dir / "TRAIN.json").read_text())["status"] == (
        StageStatus.FAILED.value
    )

    # Restart with the same config and run id: completed stages must be reused.
    restarted = _orchestrator(tmp_path, _all_fake())
    manifest = restarted.run(cycles=1, dry_run=False)[0]
    assert manifest["stages"]["GENERATE"]["status"] == StageStatus.SKIPPED.value
    assert manifest["stages"]["FILTER"]["status"] == StageStatus.SKIPPED.value
    assert manifest["stages"]["TRAIN"]["status"] == StageStatus.COMPLETED.value
    skipped_generate_id = restarted.stages_dir / "GENERATE.json"

    # Change the generation seed: GENERATE and FILTER must re-run.
    def new_seed(raw):
        raw["pipeline"]["stages"] = {stage.value: "fake" for stage in Stage}
        raw["pipeline"]["generation"]["seed"] = 20960

    changed = PipelineOrchestrator(
        load_pipeline_config(_write_config(tmp_path, new_seed)),
        repo_root=REPO_ROOT,
        run_id="run_test",
    )
    manifest = changed.run(cycles=1, dry_run=False)[0]
    assert manifest["stages"]["GENERATE"]["status"] == StageStatus.COMPLETED.value
    assert manifest["stages"]["FILTER"]["status"] == StageStatus.COMPLETED.value
    assert skipped_generate_id.is_file()


# 5. corrupt output is quarantined and the stage fails loudly
def test_corrupt_stage_output_is_quarantined(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path, _all_fake())
    orchestrator.run(cycles=1, dry_run=False)
    curated = json.loads((orchestrator.stages_dir / "FILTER.json").read_text())["outputs"][0][
        "path"
    ]
    marker = Path(curated) / "fake_filter.json"
    marker.write_text(marker.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")

    restarted = _orchestrator(tmp_path, _all_fake())
    with pytest.raises(CorruptStageOutputError, match="corrupt/incomplete"):
        restarted.run(cycles=1, dry_run=False)
    assert restarted.quarantine_dir.is_dir()
    assert any("FILTER" in path.name for path in restarted.quarantine_dir.iterdir())
    status = json.loads((restarted.stages_dir / "FILTER.json").read_text())
    assert status["status"] == StageStatus.QUARANTINED.value
    # GENERATE completed earlier and was reused, not quarantined.
    assert json.loads((restarted.stages_dir / "GENERATE.json").read_text())["status"] in {
        StageStatus.COMPLETED.value,
        StageStatus.SKIPPED.value,
    }


# 6. approval gates: external TRAIN needs a checkpoint; a bad one is refused
def test_external_train_requires_a_verified_checkpoint(tmp_path: Path) -> None:
    def external(raw):
        raw["pipeline"]["stages"] = {stage.value: "fake" for stage in Stage}
        raw["pipeline"]["stages"]["TRAIN"] = "external"
        raw["pipeline"]["training"]["gpu_estimated_s"] = 0.0
        raw["pipeline"]["training"]["estimated_download_bytes"] = 0

    orchestrator = _orchestrator(tmp_path / "missing", external)
    with pytest.raises(ApprovalRequiredError) as excinfo:
        orchestrator.run(cycles=1, dry_run=False)
    message = str(excinfo.value)
    assert "--supply-checkpoint" in message
    assert "docs/REMOTE_TRAINING.md" in message
    assert (orchestrator.artifacts_dir / "TRAIN" / "remote_command.sh").is_file()

    bogus = tmp_path / "bogus_checkpoint"
    bogus.mkdir()
    (bogus / "checkpoint_metadata.json").write_text("{}\n", encoding="utf-8")

    def bogus_config(raw):
        external(raw)
        raw["pipeline"]["training"]["checkpoint"] = str(bogus)

    orchestrator = _orchestrator(tmp_path / "bogus", bogus_config)
    with pytest.raises(CheckpointVerificationError):
        orchestrator.run(cycles=1, dry_run=False)


# 7. config validation: bounds, types, unknown keys
def test_config_validation_rejects_bad_values(tmp_path: Path) -> None:
    def unknown(raw):
        raw["pipeline"]["generation"]["seeed"] = 1

    with pytest.raises(PipelineConfigError, match="unknown key"):
        load_pipeline_config(_write_config(tmp_path / "a", unknown))

    def wrong_type(raw):
        raw["pipeline"]["generation"]["num_episodes"] = "10"

    with pytest.raises(PipelineConfigError, match="expected an integer"):
        load_pipeline_config(_write_config(tmp_path / "b", wrong_type))

    def out_of_bounds(raw):
        raw["pipeline"]["max_cycles"] = 0

    with pytest.raises(PipelineConfigError, match="max_cycles"):
        load_pipeline_config(_write_config(tmp_path / "c", out_of_bounds))

    def bad_fraction(raw):
        raw["pipeline"]["generation"]["perturbed_fraction"] = 1.5

    with pytest.raises(PipelineConfigError, match="perturbed_fraction"):
        load_pipeline_config(_write_config(tmp_path / "d", bad_fraction))

    def bad_targeted_seed(raw):
        raw["pipeline"]["targeted_generate"]["seed"] = 1000

    with pytest.raises(PipelineConfigError, match="targeted_generate.seed"):
        load_pipeline_config(_write_config(tmp_path / "e", bad_targeted_seed))

    def impossible_build(raw):
        raw["pipeline"]["build"]["mode"] = "lerobot"
        raw["pipeline"]["stages"]["BUILD"] = "fake"

    with pytest.raises(PipelineConfigError, match="requires pipeline.stages.BUILD"):
        load_pipeline_config(_write_config(tmp_path / "f", impossible_build))


# 8. dry run writes nothing except an optional plan file
def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path, _all_fake())
    plan = orchestrator.run(dry_run=True)
    assert len(plan) == len(Stage)
    assert plan[0]["stage"] == "GENERATE"
    assert not orchestrator.run_root.exists()

    plan_path = tmp_path / "plan.json"
    orchestrator.run(dry_run=True, plan_file=plan_path)
    assert plan_path.is_file()
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert payload["dry_run"] is True
    assert {row["stage"] for row in payload["stages"]} == {stage.value for stage in Stage}
    assert not orchestrator.run_root.exists()


# 9. lineage query returns config, dataset, episodes, checkpoint and eval ids
def test_lineage_query_returns_the_chain(completed_pipeline) -> None:
    orchestrator, manifest = completed_pipeline
    checkpoint = manifest["checkpoint"]["path"]
    report = orchestrator.lineage(checkpoint)
    assert report["count"] == 1
    match = report["matches"][0]
    assert match["config_hash"] == manifest["config_hash"]
    assert match["dataset"]["dataset_hash"] == manifest["dataset"]["dataset_hash"]
    assert match["episode_ids"] == manifest["source_episode_ids"]
    assert match["checkpoint"]["metadata_hash"] == manifest["checkpoint"]["metadata_hash"]
    assert match["evaluation_run_id"] == manifest["evaluation_run_id"]
    assert match["failure_distribution_id"] == manifest["failure_distribution_id"]

    by_hash = orchestrator.lineage(manifest["checkpoint"]["metadata_hash"])
    assert by_hash["count"] == 1

    assert orchestrator.lineage("does-not-exist")["count"] == 0


# extra: the shipped fake profile is honest about its ledger and safety default
def test_shipped_profiles_and_ledger_are_consistent() -> None:
    fake = load_pipeline_config(FAKE_CONFIG)
    assert fake.generation.backend == "fake"
    assert fake.executor(Stage.TRAIN) == "fake"
    assert fake.evaluation.policy == "zero"
    # core_validation must stay in the core environment even when .venv-ml exists
    fake_orchestrator = PipelineOrchestrator(fake, repo_root=REPO_ROOT, run_id="env_check")
    assert fake_orchestrator._environment(Stage.BUILD) == ".venv"
    default = load_pipeline_config(REPO_ROOT / "configs" / "pipeline" / "default.yaml")
    assert default.executor(Stage.TRAIN) == "external"
    assert default.training.gpu_estimated_s > default.gpu_time_limit_s
    assert default.evaluation.policy == "checkpoint"
    default_orchestrator = PipelineOrchestrator(default, repo_root=REPO_ROOT, run_id="env_check")
    assert default_orchestrator._environment(Stage.BUILD) == ".venv-ml"

    # The flywheel controller remains the single ledger implementation.
    evaluation = load_evaluation_config(REPO_ROOT / "configs" / "evaluation" / "default.yaml")
    controller = FlywheelController(
        evaluation, ledger_path="/tmp/sim2real/flywheel_test.jsonl", stop_file="/tmp/sim2real/STOP"
    )
    assert controller.status()["will_train"] is False
