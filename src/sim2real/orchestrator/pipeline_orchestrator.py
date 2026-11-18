"""Explicit pipeline orchestrator: GENERATE -> FILTER -> BUILD -> TRAIN ->
EVALUATE -> ANALYZE_FAILURES -> TARGETED_GENERATE.

The orchestrator is pure Python and ROS-free.  Each stage either runs the
existing script (``local``), writes the remote-training bundle/command and waits
for a verified checkpoint (``external``), uses a labelled deterministic test
double (``fake``), or is skipped explicitly (``skip``).

Safety properties:

* every stage has a deterministic id derived from its config slice and inputs;
* completed stages are skipped only when that id and the recorded output hashes
  still match, so a restarted run resumes instead of recomputing;
* a corrupt or incomplete artifact is quarantined and the stage fails loudly;
* every cycle writes one atomic JSON manifest with full lineage;
* hard bounds on cycles, episodes, wall time, writes and GPU time are enforced;
* the default CLI mode is a write-free dry run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sim2real.evaluation import EvaluationConfig, FlywheelController, load_evaluation_config
from sim2real.evaluation.records import derive_eval_seeds
from sim2real.orchestrator.config import PipelineConfig
from sim2real.orchestrator.files import (
    atomic_write_json,
    git_state,
    hash_tree,
    quarantine,
    read_json,
    sha256_bytes,
    sha256_file,
    tree_size_bytes,
)
from sim2real.orchestrator.seed_ledger import (
    SeedLedgerError,
    check_evaluation_seed,
    check_generation_seeds,
    check_targeted_seeds,
)
from sim2real.orchestrator.stages import (
    STAGE_ORDER,
    STAGE_SPECS,
    Stage,
    StageStatus,
)

SCHEMA_VERSION = 1
CYCLE_APPROVAL_FLAG = "--approve-cycle"
SUPPLY_CHECKPOINT_FLAG = "--supply-checkpoint"


class OrchestratorError(RuntimeError):
    """Base class for orchestrator failures."""


class StageError(OrchestratorError):
    """A stage failed and the cycle stopped."""


class ApprovalRequiredError(StageError):
    """A gate stopped the stage; the message says exactly what is required."""


class CheckpointVerificationError(StageError):
    """A supplied checkpoint did not pass verification."""


class CorruptStageOutputError(StageError):
    """A recorded artifact is missing or changed; it was quarantined."""


class CycleLimitError(OrchestratorError):
    """A hard cycle/wall-time/checkpoint gate refused another cycle."""


@dataclass
class StageRecord:
    stage: Stage
    executor: str
    environment: str
    stage_id: str
    status: StageStatus
    inputs: dict[str, Any]
    outputs: list[dict[str, Any]] = field(default_factory=list)
    detail: str = ""
    error: str = ""
    started_at: str = ""
    finished_at: str = ""
    wall_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "executor": self.executor,
            "environment": self.environment,
            "stage_id": self.stage_id,
            "status": self.status.value,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "detail": self.detail,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "wall_s": round(self.wall_s, 3),
        }


@dataclass
class RunFlags:
    approve_cycle: bool = False
    allow_train: bool = False
    allow_gpu: bool = False
    allow_large_write: bool = False
    allow_download: bool = False
    supplied_checkpoint: str = ""


def _timestamp() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def _last_json_object(text: str) -> dict[str, Any]:
    """Return the last JSON object printed by a script."""
    for index in range(len(text) - 1, -1, -1):
        if text[index] != "{":
            continue
        try:
            value = json.loads(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise StageError("script produced no parseable JSON summary")


def _manifest_episode_ids(staging_dir: Path) -> list[str]:
    manifest = staging_dir / "manifest.jsonl"
    if not manifest.is_file():
        return []
    ids: list[str] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ids.append(str(json.loads(line)["episode_id"]))
    return ids


class PipelineOrchestrator:
    def __init__(
        self,
        config: PipelineConfig,
        *,
        repo_root: str | Path | None = None,
        root: str | Path | None = None,
        run_id: str | None = None,
        stage_runner: Callable[[Stage, dict[str, Any], str], dict[str, Any]] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.repo_root = Path(repo_root or Path(__file__).resolve().parents[3]).resolve()
        configured_root = Path(root or config.run_root).expanduser()
        self.root = (
            configured_root if configured_root.is_absolute() else self.repo_root / configured_root
        )
        self.run_id = run_id or f"run_{config.fingerprint[:12]}"
        self.run_root = self.root / self.run_id
        self.clock = clock
        self._stage_runner = stage_runner
        self._deadline: float | None = None
        self._flags = RunFlags()
        self._evaluation_config: EvaluationConfig | None = None

    # -- layout ------------------------------------------------------------
    @property
    def stages_dir(self) -> Path:
        return self.run_root / "stages"

    @property
    def manifests_dir(self) -> Path:
        return self.run_root / "manifests"

    @property
    def artifacts_dir(self) -> Path:
        return self.run_root / "artifacts"

    @property
    def quarantine_dir(self) -> Path:
        return self.run_root / "quarantine"

    def _status_path(self, stage: Stage) -> Path:
        return self.stages_dir / f"{stage.value}.json"

    def _environment(self, stage: Stage) -> str:
        if stage is Stage.BUILD:
            return ".venv-ml" if self.config.build.mode == "lerobot" else ".venv"
        if stage is Stage.TRAIN and self.config.executor(stage) != "external":
            return ".venv-ml"
        return STAGE_SPECS[stage].environment

    def _python(self, environment: str) -> str:
        if environment == ".venv-ml":
            candidate = self.repo_root / ".venv-ml" / "bin" / "python"
            if not candidate.is_file():
                raise ApprovalRequiredError(
                    "BUILD mode='lerobot' / local TRAIN requires .venv-ml, which is absent.\n"
                    "Install it with: scripts/setup_envs.sh --with-ml (requires approval)."
                )
            return str(candidate)
        return sys.executable

    # -- plan / dry run ----------------------------------------------------
    def plan(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for stage in STAGE_ORDER:
            spec = STAGE_SPECS[stage]
            executor = self.config.executor(stage)
            rows.append(
                {
                    "stage": stage.value,
                    "executor": executor,
                    "environment": self._environment(stage),
                    "description": spec.description,
                    "inputs": spec.inputs,
                    "outputs": spec.outputs,
                    "uses_gpu": spec.uses_gpu,
                    "approval": self._approval_note(stage, executor),
                }
            )
        return rows

    def _approval_note(self, stage: Stage, executor: str) -> str:
        notes: list[str] = []
        if stage is Stage.TRAIN and executor == "external":
            notes.append(f"supply a verified checkpoint with {SUPPLY_CHECKPOINT_FLAG} <path>")
        if stage is Stage.TRAIN and executor == "local":
            notes.append("--allow-train")
        if (
            stage is Stage.TRAIN
            and self.config.training.gpu_estimated_s > self.config.gpu_time_limit_s
        ):
            notes.append("--allow-gpu (estimated GPU time exceeds gpu_time_limit_s)")
        if (
            stage is Stage.BUILD
            and self.config.build.estimated_write_bytes > self.config.max_write_bytes
        ):
            notes.append("--allow-large-write (estimated write exceeds max_write_bytes)")
        if (
            stage is Stage.TRAIN
            and self.config.training.estimated_download_bytes > self.config.max_write_bytes
        ):
            notes.append("--allow-download (estimated download exceeds max_write_bytes)")
        if stage is Stage.EVALUATE and self.config.evaluation.policy == "checkpoint":
            notes.append("--allow-gpu (real checkpoint policy)")
        return ", ".join(notes) if notes else "none"

    def write_plan(self, path: str | Path) -> Path:
        target = Path(path).expanduser()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "config_path": self.config.source_path,
            "config_hash": self.config.fingerprint,
            "run_id": self.run_id,
            "run_root": str(self.run_root),
            "dry_run": True,
            "stages": self.plan(),
        }
        atomic_write_json(target, payload)
        return target

    # -- public entry point ------------------------------------------------
    def run(
        self,
        *,
        cycles: int = 1,
        dry_run: bool = True,
        plan_file: str | Path | None = None,
        approve_cycle: bool = False,
        supplied_checkpoint: str = "",
        allow_train: bool = False,
        allow_gpu: bool = False,
        allow_large_write: bool = False,
        allow_download: bool = False,
    ) -> list[dict[str, Any]]:
        if dry_run:
            if plan_file is not None:
                self.write_plan(plan_file)
            return self.plan()
        if cycles < 1:
            raise CycleLimitError("cycles must be >= 1")
        if cycles > self.config.max_cycles:
            raise CycleLimitError(
                f"requested {cycles} cycles but max_cycles={self.config.max_cycles}"
            )
        self._flags = RunFlags(
            approve_cycle=approve_cycle,
            allow_train=allow_train,
            allow_gpu=allow_gpu,
            allow_large_write=allow_large_write,
            allow_download=allow_download,
            supplied_checkpoint=supplied_checkpoint,
        )
        manifests: list[dict[str, Any]] = []
        for index in range(1, cycles + 1):
            self._check_cycle_gate(index, self._flags)
            manifests.append(self.run_cycle(index))
        return manifests

    # -- cycles ------------------------------------------------------------
    def _previous_manifest(self, index: int) -> dict[str, Any] | None:
        if index <= 1:
            return None
        path = self.manifests_dir / f"cycle_{index - 1:04d}.json"
        return read_json(path) if path.is_file() else None

    def _check_cycle_gate(self, index: int, flags: RunFlags) -> None:
        if index <= 1:
            return
        if not flags.approve_cycle:
            raise CycleLimitError(
                f"cycle {index} refused: another cycle requires {CYCLE_APPROVAL_FLAG}"
            )
        previous = self._previous_manifest(index)
        previous_hash = ""
        if previous is not None:
            previous_hash = str((previous.get("checkpoint") or {}).get("metadata_hash") or "")
        if flags.supplied_checkpoint:
            supplied_hash = self._checkpoint_metadata_hash(flags.supplied_checkpoint)
            if supplied_hash and supplied_hash == previous_hash:
                raise CycleLimitError(
                    f"cycle {index} refused: the supplied checkpoint is not new "
                    f"(metadata hash {supplied_hash[:12]}… equals the previous cycle)"
                )
            return
        if self.config.executor(Stage.TRAIN) != "fake":
            raise CycleLimitError(
                f"cycle {index} refused: another cycle requires a new verified checkpoint "
                f"({SUPPLY_CHECKPOINT_FLAG} <path>) and {CYCLE_APPROVAL_FLAG}"
            )

    def _checkpoint_metadata_hash(self, checkpoint: str | Path) -> str:
        metadata = Path(checkpoint).expanduser() / "checkpoint_metadata.json"
        if not metadata.is_file():
            return ""
        try:
            payload = read_json(metadata)
        except (OSError, ValueError):
            return ""
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return sha256_bytes(canonical.encode("utf-8"))

    def _check_deadline(self) -> None:
        if self._deadline is not None and self.clock() > self._deadline:
            raise CycleLimitError(
                f"max_wall_time_s={self.config.max_wall_time_s} exceeded; stopping the cycle"
            )

    def run_cycle(self, cycle_index: int) -> dict[str, Any]:
        cycle_id = f"cycle_{cycle_index:04d}"
        started = self.clock()
        self._deadline = started + self.config.max_wall_time_s
        context: dict[str, Any] = {
            "cycle_id": cycle_id,
            "cycle_index": cycle_index,
            "run_id": self.run_id,
        }
        records: list[StageRecord] = []
        failure: BaseException | None = None
        try:
            for stage in STAGE_ORDER:
                self._check_deadline()
                record = self._run_stage(stage, context)
                records.append(record)
                if record.status in {
                    StageStatus.FAILED,
                    StageStatus.BLOCKED,
                    StageStatus.QUARANTINED,
                }:
                    raise StageError(
                        f"{stage.value} {record.status.value}: {record.error or record.detail}"
                    )
            provisional = self._build_manifest(
                cycle_id,
                records,
                context,
                wall_s=self.clock() - started,
                error="",
            )
            context["flywheel"] = self.record_flywheel_decision(provisional)
        except BaseException as exc:  # noqa: BLE001 - record the failure, re-raise
            failure = exc
        manifest = self._build_manifest(
            cycle_id,
            records,
            context,
            wall_s=self.clock() - started,
            error="" if failure is None else f"{type(failure).__name__}: {failure}",
        )
        atomic_write_json(self.manifests_dir / f"{cycle_id}.json", manifest)
        if failure is not None:
            raise failure
        return manifest

    # -- stage execution ---------------------------------------------------
    def _run_stage(self, stage: Stage, context: dict[str, Any]) -> StageRecord:
        executor = self.config.executor(stage)
        environment = self._environment(stage)
        inputs = self._stage_inputs(stage, context)
        stage_id = self._stage_id(stage, inputs)
        record = StageRecord(
            stage=stage,
            executor=executor,
            environment=environment,
            stage_id=stage_id,
            status=StageStatus.PENDING,
            inputs=inputs,
        )

        resumed = self._resume(stage, stage_id)
        if resumed is not None:
            context.update(resumed)
            record.status = StageStatus.SKIPPED
            record.outputs = read_json(self._status_path(stage)).get("outputs", [])
            record.detail = "inputs unchanged; reused completed stage"
            return record

        record.started_at = _timestamp()
        started = self.clock()
        self._write_status(record)
        try:
            produced = self._dispatch(stage, context, executor)
        except StageError as exc:
            record.status = (
                StageStatus.BLOCKED
                if isinstance(exc, ApprovalRequiredError)
                else StageStatus.FAILED
            )
            record.error = str(exc)
            record.finished_at = _timestamp()
            record.wall_s = self.clock() - started
            self._write_status(record)
            raise
        except Exception as exc:  # noqa: BLE001 - convert to a stage failure
            record.status = StageStatus.FAILED
            record.error = f"{type(exc).__name__}: {exc}"
            record.finished_at = _timestamp()
            record.wall_s = self.clock() - started
            self._write_status(record)
            raise StageError(f"{stage.value} failed: {record.error}") from exc

        record.outputs = produced.get("outputs", [])
        record.detail = str(produced.get("detail", ""))
        context.update(produced.get("context", {}))
        record.status = StageStatus.COMPLETED
        record.finished_at = _timestamp()
        record.wall_s = self.clock() - started
        self._write_status(record)
        return record

    def _stage_id(self, stage: Stage, inputs: dict[str, Any]) -> str:
        payload = {
            "stage": stage.value,
            "config": self.config.stage_config_hash(stage),
            "inputs": inputs,
            "schema_version": SCHEMA_VERSION,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return sha256_bytes(canonical.encode("utf-8"))[:32]

    def _write_status(self, record: StageRecord) -> None:
        atomic_write_json(self._status_path(record.stage), record.to_dict())

    def _resume(self, stage: Stage, stage_id: str) -> dict[str, Any] | None:
        path = self._status_path(stage)
        if not path.is_file():
            return None
        try:
            status = read_json(path)
        except (OSError, ValueError) as exc:
            self._quarantine(stage, [path], f"unreadable status: {exc}")
            raise CorruptStageOutputError(
                f"{stage.value} status file is unreadable: {exc}"
            ) from exc
        if status.get("stage_id") != stage_id:
            return None
        if status.get("status") not in {StageStatus.COMPLETED.value, StageStatus.SKIPPED.value}:
            return None
        problems = self._verify_outputs(status.get("outputs", []))
        if problems:
            targets = [item["path"] for item in status.get("outputs", []) if item.get("path")]
            moved = self._quarantine(stage, [*targets, path], "; ".join(problems))
            record = StageRecord(
                stage=stage,
                executor=str(status.get("executor", "")),
                environment=str(status.get("environment", "")),
                stage_id=stage_id,
                status=StageStatus.QUARANTINED,
                inputs=status.get("inputs", {}),
                outputs=status.get("outputs", []),
                error="; ".join(problems),
                detail=f"quarantined to {moved}",
            )
            self._write_status(record)
            raise CorruptStageOutputError(
                f"{stage.value} recorded output is corrupt/incomplete ({'; '.join(problems)}); "
                f"moved to {moved}"
            )
        reconstructed = self._context_from_outputs(stage, status.get("outputs", []), status)
        return reconstructed

    def _verify_outputs(self, outputs: list[dict[str, Any]]) -> list[str]:
        problems: list[str] = []
        for output in outputs:
            path = Path(str(output.get("path", "")))
            if not path.exists():
                problems.append(f"missing {path}")
                continue
            expected = str(output.get("sha256", ""))
            actual = hash_tree(path) if output.get("is_dir") else sha256_file(path)
            if expected and actual != expected:
                problems.append(f"hash mismatch for {path}")
        return problems

    def _quarantine(self, stage: Stage, paths: list[str | Path], reason: str) -> Path:
        from datetime import UTC, datetime

        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        target = self.quarantine_dir / f"{stage.value}-{stamp}"
        target.mkdir(parents=True, exist_ok=True)
        import shutil

        moved = 0
        for path in paths:
            candidate = Path(path)
            if not candidate.exists():
                continue
            destination = target / candidate.name
            shutil.move(str(candidate), str(destination))
            moved += 1
        if moved == 0:
            (target / "MISSING.txt").write_text(
                "all recorded artifacts were already missing\n", encoding="utf-8"
            )
        (target / "QUARANTINE_REASON.txt").write_text(reason + "\n", encoding="utf-8")
        return target

    def _reset_partial_dir(self, stage: Stage, path: str | Path) -> None:
        """Quarantine a leftover partial output from a failed attempt."""
        if Path(path).exists():
            quarantine(path, self.quarantine_dir, f"{stage.value}-partial")

    # -- inputs per stage --------------------------------------------------
    def _stage_inputs(self, stage: Stage, context: dict[str, Any]) -> dict[str, Any]:
        cfg = self.config
        if stage is Stage.GENERATE:
            seeds = [cfg.generation.seed + index for index in range(cfg.generation.num_episodes)]
            check_generation_seeds(seeds, backend=cfg.generation.backend)
            context["generation_seeds"] = seeds
            return {
                "seeds": seeds,
                "backend": cfg.generation.backend,
                "profile": cfg.generation.profile,
                "perturbed_fraction": cfg.generation.perturbed_fraction,
                "instruction": cfg.generation.instruction,
            }
        if stage is Stage.FILTER:
            staging = Path(context.get("staging_dir", ""))
            return {
                "staging_hash": hash_tree(staging) if staging.exists() else "",
                "filter_config_sha256": self._config_file_sha(cfg.filtering.config),
                "dataset_name": cfg.filtering.dataset_name,
            }
        if stage is Stage.BUILD:
            curated = Path(context.get("curated_version_dir", ""))
            return {
                "curated_hash": hash_tree(curated) if curated.exists() else "",
                "mode": cfg.build.mode,
                "decimation": cfg.build.decimation,
                "val_fraction": cfg.build.val_fraction,
                "dataset_config_sha256": self._config_file_sha(cfg.build.config),
            }
        if stage is Stage.TRAIN:
            supplied = self._flags.supplied_checkpoint or cfg.training.checkpoint
            return {
                # Each cycle is a new training attempt by definition, even when
                # the dataset is unchanged; the checkpoint must be new before
                # another cycle is allowed.
                "cycle_id": context["cycle_id"],
                "dataset_hash": context.get("dataset_hash", ""),
                "training_config_sha256": self._config_file_sha(cfg.training.config),
                "supplied_checkpoint": str(supplied),
                "supplied_metadata_hash": (
                    self._checkpoint_metadata_hash(supplied) if supplied else ""
                ),
            }
        if stage is Stage.EVALUATE:
            check_evaluation_seed(cfg.evaluation.seed, backend=cfg.generation.backend)
            seeds = derive_eval_seeds(cfg.evaluation.seed, cfg.evaluation.num_episodes)
            context["evaluation_seeds"] = seeds
            return {
                "dataset_hash": context.get("dataset_hash", ""),
                "checkpoint_hash": context.get("checkpoint_metadata_hash", ""),
                "policy": cfg.evaluation.policy,
                "num_episodes": cfg.evaluation.num_episodes,
                "seed": cfg.evaluation.seed,
                "evaluation_config_sha256": self._config_file_sha(cfg.evaluation.config),
            }
        if stage is Stage.ANALYZE_FAILURES:
            eval_dir = Path(context.get("eval_dir", ""))
            return {
                "eval_hash": hash_tree(eval_dir) if eval_dir.exists() else "",
                "evaluation_config_sha256": self._config_file_sha(cfg.failure_analysis.config),
            }
        if stage is Stage.TARGETED_GENERATE:
            distribution = Path(context.get("failure_distribution_path", ""))
            seeds = [
                cfg.targeted_generate.seed + index
                for index in range(cfg.targeted_generate.num_rollouts)
            ]
            check_targeted_seeds(seeds)
            return {
                "distribution_hash": (sha256_file(distribution) if distribution.is_file() else ""),
                "num_rollouts": cfg.targeted_generate.num_rollouts,
                "seed": cfg.targeted_generate.seed,
            }
        raise OrchestratorError(f"unknown stage {stage!r}")  # pragma: no cover

    def _config_file_sha(self, path: str) -> str:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.repo_root / candidate
        return sha256_file(candidate) if candidate.is_file() else ""

    # -- dispatch ----------------------------------------------------------
    def _dispatch(self, stage: Stage, context: dict[str, Any], executor: str) -> dict[str, Any]:
        if self._stage_runner is not None and executor in {"local", "fake"}:
            return self._stage_runner(stage, context, executor)
        if executor == "skip":
            return {"outputs": [], "detail": "executor=skip", "context": {}}
        if executor == "fake":
            return self._run_fake(stage, context)
        if executor == "external":
            return self._run_external(stage, context)
        if executor == "local":
            self._enforce_local_gates(stage)
            return self._run_local(stage, context)
        raise OrchestratorError(f"unknown executor {executor!r}")  # pragma: no cover

    def _enforce_local_gates(self, stage: Stage) -> None:
        cfg = self.config
        if stage is Stage.TRAIN and not self._flags.allow_train:
            raise ApprovalRequiredError(
                "TRAIN executor=local runs a real training job and requires --allow-train.\n"
                "The default CLI mode is --dry-run. Ask the operator before training."
            )
        if (
            stage in {Stage.TRAIN, Stage.EVALUATE}
            and cfg.training.gpu_estimated_s > cfg.gpu_time_limit_s
            and not self._flags.allow_gpu
        ):
            raise ApprovalRequiredError(
                f"TRAIN uses the GPU for an estimated {cfg.training.gpu_estimated_s:.0f} s, "
                f"above gpu_time_limit_s={cfg.gpu_time_limit_s:.0f}; pass --allow-gpu"
            )
        if (
            stage is Stage.EVALUATE
            and cfg.evaluation.policy == "checkpoint"
            and not self._flags.allow_gpu
        ):
            raise ApprovalRequiredError(
                "EVALUATE policy=checkpoint loads the real model on the GPU; pass --allow-gpu "
                "(local timings are throttled, not a performance figure)"
            )
        if (
            stage is Stage.BUILD
            and cfg.build.estimated_write_bytes > cfg.max_write_bytes
            and not self._flags.allow_large_write
        ):
            raise ApprovalRequiredError(
                f"BUILD estimates {cfg.build.estimated_write_bytes} bytes of output, above "
                f"max_write_bytes={cfg.max_write_bytes}; pass --allow-large-write"
            )
        if (
            stage is Stage.TRAIN
            and cfg.training.estimated_download_bytes > cfg.max_write_bytes
            and not self._flags.allow_download
        ):
            raise ApprovalRequiredError(
                f"TRAIN estimates {cfg.training.estimated_download_bytes} bytes of download, above "
                f"max_write_bytes={cfg.max_write_bytes}; pass --allow-download"
            )

    # -- local execution ---------------------------------------------------
    def _run_script(self, args: list[str], *, environment: str = ".venv") -> str:
        python = self._python(environment)
        timeout = None
        if self._deadline is not None:
            timeout = max(1.0, self._deadline - self.clock())
        result = subprocess.run(
            [python, *args],
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip().splitlines()[-12:]
            raise StageError(
                f"script {' '.join(args)} exited {result.returncode}:\n" + "\n".join(tail)
            )
        return result.stdout

    def _run_local(self, stage: Stage, context: dict[str, Any]) -> dict[str, Any]:
        if stage is Stage.GENERATE:
            return self._local_generate(context)
        if stage is Stage.FILTER:
            return self._local_filter(context)
        if stage is Stage.BUILD:
            return self._local_build(context)
        if stage is Stage.TRAIN:
            return self._local_train(context)
        if stage is Stage.EVALUATE:
            return self._local_evaluate(context)
        if stage is Stage.ANALYZE_FAILURES:
            return self._local_analyze(context)
        if stage is Stage.TARGETED_GENERATE:
            return self._local_targeted(context)
        raise OrchestratorError(f"no local runner for {stage.value}")  # pragma: no cover

    def _local_generate(self, context: dict[str, Any]) -> dict[str, Any]:
        gen = self.config.generation
        staging_root = self.artifacts_dir / "GENERATE" / "staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        run_name = f"gen_{context['cycle_id']}"
        self._reset_partial_dir(Stage.GENERATE, staging_root / run_name)
        stdout = self._run_script(
            [
                "scripts/collect_rollouts.py",
                "--backend",
                gen.backend,
                "--num",
                str(gen.num_episodes),
                "--seed",
                str(gen.seed),
                "--out",
                str(staging_root),
                "--run-id",
                run_name,
                "--profile",
                gen.profile,
                "--perturbed-fraction",
                str(gen.perturbed_fraction),
                "--instruction",
                gen.instruction,
            ]
        )
        summary = _last_json_object(stdout)
        staging_dir = Path(summary["staging_dir"])
        episode_ids = _manifest_episode_ids(staging_dir)
        return {
            "outputs": [self._artifact("staging", staging_dir)],
            "detail": f"{summary.get('episodes')} episodes, counts={summary.get('counts')}",
            "context": {
                "staging_dir": str(staging_dir),
                "source_episode_ids": episode_ids,
                "generation_summary": summary,
            },
        }

    def _local_filter(self, context: dict[str, Any]) -> dict[str, Any]:
        cfg = self.config
        staged = Path(context["staging_dir"])
        curated_root = self.run_root / "curated"
        rejected_root = self.run_root / "rejected"
        curated_root.mkdir(parents=True, exist_ok=True)
        stdout = self._run_script(
            [
                "scripts/filter_staging.py",
                "--staging",
                str(staged),
                "--config",
                cfg.filtering.config,
                "--curated",
                str(curated_root),
                "--rejected",
                str(rejected_root),
                "--dataset-name",
                cfg.filtering.dataset_name,
            ]
        )
        summary = _last_json_object(stdout)
        version_dir = summary.get("dataset")
        if not version_dir:
            raise StageError(
                "filter accepted no episodes; a downstream BUILD would have no data "
                f"(reason counts: {summary.get('reason_counts')})"
            )
        return {
            "outputs": [self._artifact("curated", Path(version_dir))],
            "detail": (
                f"accepted {summary.get('accepted')}/{summary.get('episodes')} "
                f"into {summary.get('dataset_version')}"
            ),
            "context": {
                "curated_version_dir": str(version_dir),
                "curated_version": str(summary.get("dataset_version")),
                "filter_summary": summary,
            },
        }

    def _local_build(self, context: dict[str, Any]) -> dict[str, Any]:
        cfg = self.config
        curated = Path(context["curated_version_dir"])
        staged = Path(context["staging_dir"])
        artifacts = self.artifacts_dir / "BUILD"
        artifacts.mkdir(parents=True, exist_ok=True)
        args = [
            "scripts/build_lerobot_dataset.py",
            "--curated",
            str(curated),
            "--staging",
            str(staged),
            "--config",
            cfg.build.config,
            "--embodiments",
            cfg.build.embodiments,
            "--decimation",
            str(cfg.build.decimation),
            "--val-fraction",
            str(cfg.build.val_fraction),
            "--allow-imageless",
        ]
        if cfg.build.mode == "core_validation":
            args.append("--dry-run")
        else:
            out_root = Path(cfg.build.output)
            if not out_root.is_absolute():
                out_root = self.repo_root / out_root
            self._reset_partial_dir(Stage.BUILD, out_root / cfg.filtering.dataset_name)
            args.extend(["--out", str(out_root), "--name", cfg.filtering.dataset_name])
        stdout = self._run_script(args, environment=self._environment(Stage.BUILD))
        summary = _last_json_object(stdout)
        summary_path = artifacts / "summary.json"
        atomic_write_json(summary_path, summary)
        dataset_hash = hash_tree(curated)
        return {
            "outputs": [
                self._artifact("summary", summary_path),
                self._artifact("curated", curated),
            ],
            "detail": (
                f"mode={cfg.build.mode}, episodes={summary.get('episodes')}, "
                f"frames={summary.get('frames')}, image_mode={summary.get('image_mode')}"
            ),
            "context": {
                "dataset_hash": dataset_hash,
                "build_summary": summary,
                "build_mode": cfg.build.mode,
            },
        }

    def _local_train(self, context: dict[str, Any]) -> dict[str, Any]:
        cfg = self.config
        out_dir = self.artifacts_dir / "TRAIN"
        out_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_root = out_dir / "checkpoints"
        self._run_script(
            [
                "scripts/train_vla.py",
                "--config",
                cfg.training.config,
                "--output-dir",
                str(checkpoint_root),
            ],
            environment=".venv-ml",
        )
        checkpoints = sorted(checkpoint_root.glob("checkpoint-*"))
        if not checkpoints:
            raise StageError("training produced no checkpoint")
        checkpoint = checkpoints[-1]
        return {
            "outputs": [self._artifact("checkpoint", checkpoint)],
            "detail": f"local training checkpoint {checkpoint.name}",
            "context": {
                "checkpoint_path": str(checkpoint),
                "checkpoint_metadata_hash": self._checkpoint_metadata_hash(checkpoint),
                "checkpoint_verification": "local_run",
            },
        }

    def _local_evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
        cfg = self.config
        artifacts = self.artifacts_dir / "EVALUATE"
        artifacts.mkdir(parents=True, exist_ok=True)
        out_dir = artifacts / "eval"
        partial = artifacts / ".partial-eval"
        self._reset_partial_dir(Stage.EVALUATE, partial)
        run_id = f"eval_{context['cycle_id']}"
        args = [
            "scripts/run_eval.py",
            "--backend",
            cfg.generation.backend,
            "--policy",
            cfg.evaluation.policy,
            "--num",
            str(cfg.evaluation.num_episodes),
            "--seed",
            str(cfg.evaluation.seed),
            "--config",
            cfg.evaluation.config,
            "--out",
            str(partial),
            "--run-id",
            run_id,
        ]
        if cfg.evaluation.paired_expert and cfg.evaluation.policy != "scripted":
            args.append("--paired-expert")
        if cfg.evaluation.policy == "checkpoint":
            checkpoint = context.get("checkpoint_path") or cfg.training.checkpoint
            if not checkpoint:
                raise StageError("policy=checkpoint but no checkpoint is available")
            args.extend(["--checkpoint", str(checkpoint)])
        self._run_script(args)
        os.replace(partial, out_dir)
        manifest_path = out_dir / "manifest.json"
        manifest = read_json(manifest_path)
        return {
            "outputs": [self._artifact("evaluation", out_dir)],
            "detail": (
                f"run_id={run_id}, "
                f"success_rate={manifest.get('aggregates', {}).get('success_rate')}"
            ),
            "context": {
                "eval_dir": str(out_dir),
                "eval_run_id": run_id,
                "eval_manifest": manifest,
            },
        }

    def _local_analyze(self, context: dict[str, Any]) -> dict[str, Any]:
        cfg = self.config
        artifacts = self.artifacts_dir / "ANALYZE_FAILURES"
        self._reset_partial_dir(Stage.ANALYZE_FAILURES, artifacts)
        artifacts.mkdir(parents=True, exist_ok=True)
        stdout = self._run_script(
            [
                "scripts/fit_failure_distribution.py",
                "--eval-run",
                str(context["eval_dir"]),
                "--out",
                str(artifacts),
                "--config",
                cfg.failure_analysis.config,
            ]
        )
        summary = _last_json_object(stdout)
        candidates = sorted(
            path for path in artifacts.glob("*.json") if path.name != "summary.json"
        )
        if not candidates:
            raise StageError("failure analysis produced no distribution artifact")
        artifact = candidates[-1]
        artifact_payload = read_json(artifact)
        distribution_id = str(artifact_payload.get("id", ""))
        distribution_type = str(artifact_payload.get("type", ""))
        return {
            "outputs": [self._artifact("failure_distribution", artifact)],
            "detail": (
                f"distribution={distribution_type or summary.get('type')} id={distribution_id}"
            ),
            "context": {
                "failure_distribution_path": str(artifact),
                "failure_distribution_id": distribution_id,
                "failure_summary": summary,
            },
        }

    def _local_targeted(self, context: dict[str, Any]) -> dict[str, Any]:
        cfg = self.config
        staging_root = self.artifacts_dir / "TARGETED_GENERATE" / "staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        run_name = f"targeted_{context['cycle_id']}"
        self._reset_partial_dir(Stage.TARGETED_GENERATE, staging_root / run_name)
        stdout = self._run_script(
            [
                "scripts/collect_rollouts.py",
                "--backend",
                cfg.generation.backend,
                "--num",
                str(cfg.targeted_generate.num_rollouts),
                "--seed",
                str(cfg.targeted_generate.seed),
                "--out",
                str(staging_root),
                "--run-id",
                run_name,
                "--instruction",
                cfg.generation.instruction,
                "--failure-distribution",
                str(context["failure_distribution_path"]),
                "--evaluation-config",
                cfg.evaluation.config,
            ]
        )
        summary = _last_json_object(stdout)
        staged = Path(summary.get("staging_dir", staging_root / run_name))
        return {
            "outputs": [self._artifact("targeted_staging", staged)],
            "detail": f"targeted rollouts={summary.get('generated', summary.get('episodes'))}",
            "context": {
                "targeted_staging_dir": str(staged),
                "targeted_seeds": summary.get("seeds", []),
                "targeted_summary": summary,
            },
        }

    # -- external training -------------------------------------------------
    def _run_external(self, stage: Stage, context: dict[str, Any]) -> dict[str, Any]:
        if stage is not Stage.TRAIN:
            raise OrchestratorError(
                f"executor=external is only supported for TRAIN, not {stage.value}"
            )
        cfg = self.config
        out_dir = self.artifacts_dir / "TRAIN"
        out_dir.mkdir(parents=True, exist_ok=True)
        if cfg.training.gpu_estimated_s > cfg.gpu_time_limit_s and not self._flags.allow_gpu:
            raise ApprovalRequiredError(
                f"TRAIN executor=external uses the GPU for an estimated "
                f"{cfg.training.gpu_estimated_s:.0f} s, above "
                f"gpu_time_limit_s={cfg.gpu_time_limit_s:.0f}; pass --allow-gpu"
            )
        if (
            cfg.training.estimated_download_bytes > cfg.max_write_bytes
            and not self._flags.allow_download
        ):
            raise ApprovalRequiredError(
                f"TRAIN estimates {cfg.training.estimated_download_bytes} bytes of download, "
                f"above max_write_bytes={cfg.max_write_bytes}; pass --allow-download"
            )
        command_path = out_dir / "remote_command.sh"
        command_path.write_text(self._remote_command(context), encoding="utf-8")
        checkpoint = self._flags.supplied_checkpoint or cfg.training.checkpoint
        if not checkpoint:
            raise ApprovalRequiredError(
                "TRAIN executor=external needs a checkpoint from the remote GPU host.\n"
                f"Planned bundle/command written to: {command_path}\n"
                "Next steps:\n"
                "  1. uv run python scripts/export_remote_bundle.py "
                "(see docs/REMOTE_TRAINING.md)\n"
                "  2. Run the bundle on a GPU host that passes the clock/memory gate.\n"
                "  3. Re-run this orchestrator with "
                f"{SUPPLY_CHECKPOINT_FLAG} <checkpoint-dir> (and {CYCLE_APPROVAL_FLAG} "
                "for cycles after the first).\n"
                "The orchestrator will verify the checkpoint before EVALUATE."
            )
        self._verify_checkpoint(checkpoint)
        metadata_hash = self._checkpoint_metadata_hash(checkpoint)
        if not metadata_hash:
            raise CheckpointVerificationError(
                f"checkpoint_metadata.json missing or unreadable in {checkpoint}"
            )
        return {
            "outputs": [
                self._artifact("remote_command", command_path),
                self._artifact("checkpoint", Path(checkpoint)),
            ],
            "detail": f"verified remote checkpoint {checkpoint}",
            "context": {
                "checkpoint_path": str(checkpoint),
                "checkpoint_metadata_hash": metadata_hash,
                "checkpoint_verification": "remote_verifier",
            },
        }

    def _remote_command(self, context: dict[str, Any]) -> str:
        cfg = self.config
        return (
            "#!/usr/bin/env bash\n"
            "# Generated by the pipeline orchestrator from docs/REMOTE_TRAINING.md.\n"
            "# Not executed locally. Run on a GPU host whose driver supports CUDA 12.6+.\n"
            "set -euo pipefail\n"
            "git clone <your-repo-url> sim2real && cd sim2real\n"
            f"git checkout {git_state(self.repo_root)[0]}\n"
            "ML_TORCH_BACKEND=cu126 ./scripts/setup_envs.sh --with-ml\n"
            f".venv-ml/bin/python scripts/train_vla.py --config {cfg.training.config} "
            "--max-steps <N> --output-dir outputs/task8/remote\n"
            "# Bring back the checkpoint directory, checkpoint_metadata.json, trainer_state.json\n"
            "# and losses.jsonl, then supply the checkpoint to this orchestrator.\n"
        )

    def _verify_checkpoint(self, checkpoint: str | Path) -> None:
        script = self.repo_root / "scripts" / "verify_remote_checkpoint.py"
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--checkpoint",
                str(checkpoint),
                "--config",
                self.config.training.config,
                "--handshake",
                "skip",
            ],
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if result.returncode != 0:
            tail = (result.stdout + "\n" + result.stderr).strip().splitlines()[-12:]
            raise CheckpointVerificationError(
                f"checkpoint verification refused {checkpoint}:\n" + "\n".join(tail)
            )

    # -- deterministic fake execution (tests / plan rehearsal) -------------
    def _run_fake(self, stage: Stage, context: dict[str, Any]) -> dict[str, Any]:
        cfg = self.config
        stage_dir = self.artifacts_dir / stage.value
        stage_dir.mkdir(parents=True, exist_ok=True)
        marker = {"stage": stage.value, "cycle_id": context["cycle_id"], "fake": True}

        if stage is Stage.GENERATE:
            staging = stage_dir / "staging" / f"gen_{context['cycle_id']}"
            staging.mkdir(parents=True, exist_ok=True)
            lines = []
            for index in range(cfg.generation.num_episodes):
                seed = cfg.generation.seed + index
                lines.append(
                    json.dumps(
                        {
                            "episode_id": f"gen_{context['cycle_id']}_ep{index:04d}",
                            "seed": seed,
                            "outcome": 1,
                        },
                        sort_keys=True,
                    )
                )
            (staging / "manifest.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
            atomic_write_json(staging / "fake_generation.json", marker)
            episode_ids = [json.loads(line)["episode_id"] for line in lines]
            return {
                "outputs": [self._artifact("staging", staging)],
                "detail": f"fake generation: {len(episode_ids)} episodes",
                "context": {
                    "staging_dir": str(staging),
                    "source_episode_ids": episode_ids,
                },
            }

        if stage is Stage.FILTER:
            version = stage_dir / cfg.filtering.dataset_name / "v0.1.0"
            version.mkdir(parents=True, exist_ok=True)
            accepted = list(context.get("source_episode_ids", []))
            atomic_write_json(
                version / "meta" / "info.json",
                {"fake": True, "version": "v0.1.0", "episodes": accepted},
            )
            atomic_write_json(version / "fake_filter.json", marker)
            return {
                "outputs": [self._artifact("curated", version)],
                "detail": f"fake filter: {len(accepted)} accepted",
                "context": {
                    "curated_version_dir": str(version),
                    "curated_version": "v0.1.0",
                },
            }

        if stage is Stage.BUILD:
            bundle = stage_dir / "dataset"
            bundle.mkdir(parents=True, exist_ok=True)
            atomic_write_json(
                bundle / "info.json",
                {
                    "fake": True,
                    "mode": cfg.build.mode,
                    "episodes": len(context.get("source_episode_ids", [])),
                },
            )
            atomic_write_json(bundle / "fake_build.json", marker)
            summary = {
                "mode": cfg.build.mode,
                "episodes": len(context.get("source_episode_ids", [])),
                "image_mode": "none",
                "fake": True,
            }
            summary_path = stage_dir / "summary.json"
            atomic_write_json(summary_path, summary)
            return {
                "outputs": [
                    self._artifact("dataset", bundle),
                    self._artifact("summary", summary_path),
                ],
                "detail": "fake build",
                "context": {"dataset_hash": hash_tree(bundle), "build_summary": summary},
            }

        if stage is Stage.TRAIN:
            checkpoint = stage_dir / "checkpoint"
            (checkpoint / "adapter").mkdir(parents=True, exist_ok=True)
            metadata = {
                "fake": True,
                "cycle_id": context["cycle_id"],
                "dataset_hash": context.get("dataset_hash", ""),
                "step": 0,
                "adapter_hash": sha256_bytes(context["cycle_id"].encode("utf-8")),
            }
            atomic_write_json(checkpoint / "checkpoint_metadata.json", metadata)
            (checkpoint / "adapter" / "adapter_model.fake").write_text(
                "fake adapter; not a trained model\n", encoding="utf-8"
            )
            return {
                "outputs": [self._artifact("checkpoint", checkpoint)],
                "detail": "fake TRAIN executor; not a real checkpoint",
                "context": {
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_metadata_hash": self._checkpoint_metadata_hash(checkpoint),
                    "checkpoint_verification": "fake_executor",
                },
            }

        if stage is Stage.EVALUATE:
            eval_dir = stage_dir / "eval"
            eval_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "fake": True,
                "run_id": f"eval_{context['cycle_id']}",
                "aggregates": {"success_rate": 0.0, "num_episodes": cfg.evaluation.num_episodes},
            }
            atomic_write_json(eval_dir / "manifest.json", manifest)
            (eval_dir / "records.jsonl").write_text(
                "".join(
                    json.dumps({"fake": True, "index": index}) + "\n"
                    for index in range(cfg.evaluation.num_episodes)
                ),
                encoding="utf-8",
            )
            atomic_write_json(eval_dir / "fake_eval.json", marker)
            return {
                "outputs": [self._artifact("evaluation", eval_dir)],
                "detail": "fake evaluation",
                "context": {
                    "eval_dir": str(eval_dir),
                    "eval_run_id": manifest["run_id"],
                    "eval_manifest": manifest,
                },
            }

        if stage is Stage.ANALYZE_FAILURES:
            distribution_id = sha256_bytes(
                f"{context['cycle_id']}:{context.get('eval_run_id', '')}".encode()
            )[:16]
            artifact = stage_dir / f"{distribution_id}.json"
            payload = {
                "fake": True,
                "type": "kde",
                "id": distribution_id,
                "n_failures": cfg.evaluation.num_episodes,
                "source_eval_run_id": context.get("eval_run_id", ""),
            }
            atomic_write_json(artifact, payload)
            return {
                "outputs": [self._artifact("failure_distribution", artifact)],
                "detail": f"fake failure fit id={distribution_id}",
                "context": {
                    "failure_distribution_path": str(artifact),
                    "failure_distribution_id": distribution_id,
                },
            }

        if stage is Stage.TARGETED_GENERATE:
            staging = stage_dir / "staging" / f"targeted_{context['cycle_id']}"
            staging.mkdir(parents=True, exist_ok=True)
            seeds = [
                cfg.targeted_generate.seed + index
                for index in range(cfg.targeted_generate.num_rollouts)
            ]
            lines = [
                json.dumps(
                    {
                        "episode_id": f"targeted_{context['cycle_id']}_ep{index:04d}",
                        "seed": seed,
                        "outcome": 1,
                    },
                    sort_keys=True,
                )
                for index, seed in enumerate(seeds)
            ]
            (staging / "manifest.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
            atomic_write_json(staging / "fake_targeted.json", marker)
            return {
                "outputs": [self._artifact("targeted_staging", staging)],
                "detail": f"fake targeted generation: {len(seeds)} rollouts",
                "context": {
                    "targeted_staging_dir": str(staging),
                    "targeted_seeds": seeds,
                },
            }
        raise OrchestratorError(f"no fake runner for {stage.value}")  # pragma: no cover

    # -- artifacts / manifests ---------------------------------------------
    def _artifact(self, kind: str, path: Path) -> dict[str, Any]:
        resolved = Path(path)
        is_dir = resolved.is_dir()
        return {
            "kind": kind,
            "path": str(resolved),
            "is_dir": is_dir,
            "sha256": hash_tree(resolved) if is_dir else sha256_file(resolved),
            "size_bytes": tree_size_bytes(resolved),
        }

    def _context_from_outputs(
        self, stage: Stage, outputs: list[dict[str, Any]], status: dict[str, Any]
    ) -> dict[str, Any]:
        """Reconstruct the small context slice a completed stage provided.

        The manifest is the durable record; on resume we read the stage status
        and re-derive only the paths the next stage needs.
        """
        ctx: dict[str, Any] = {}
        by_kind = {str(item.get("kind")): str(item.get("path")) for item in outputs}
        if stage is Stage.GENERATE and "staging" in by_kind:
            ctx["staging_dir"] = by_kind["staging"]
            ctx["source_episode_ids"] = _manifest_episode_ids(Path(by_kind["staging"]))
            ctx["generation_seeds"] = status.get("inputs", {}).get("seeds", [])
        elif stage is Stage.FILTER and "curated" in by_kind:
            ctx["curated_version_dir"] = by_kind["curated"]
            ctx["curated_version"] = Path(by_kind["curated"]).name
        elif stage is Stage.BUILD:
            if "dataset" in by_kind:
                ctx["dataset_hash"] = hash_tree(Path(by_kind["dataset"]))
            elif "curated" in by_kind:
                ctx["dataset_hash"] = hash_tree(Path(by_kind["curated"]))
        elif stage is Stage.TRAIN and "checkpoint" in by_kind:
            ctx["checkpoint_path"] = by_kind["checkpoint"]
            ctx["checkpoint_metadata_hash"] = self._checkpoint_metadata_hash(by_kind["checkpoint"])
        elif stage is Stage.EVALUATE and "evaluation" in by_kind:
            ctx["eval_dir"] = by_kind["evaluation"]
            manifest = Path(by_kind["evaluation"]) / "manifest.json"
            if manifest.is_file():
                ctx["eval_manifest"] = read_json(manifest)
                ctx["eval_run_id"] = ctx["eval_manifest"].get("run_id", "")
        elif stage is Stage.ANALYZE_FAILURES and "failure_distribution" in by_kind:
            ctx["failure_distribution_path"] = by_kind["failure_distribution"]
            try:
                payload = read_json(by_kind["failure_distribution"])
            except (OSError, ValueError):
                payload = {}
            ctx["failure_distribution_id"] = payload.get("id", "")
        elif stage is Stage.TARGETED_GENERATE and "targeted_staging" in by_kind:
            ctx["targeted_staging_dir"] = by_kind["targeted_staging"]
        return ctx

    def _build_manifest(
        self,
        cycle_id: str,
        records: list[StageRecord],
        context: dict[str, Any],
        *,
        wall_s: float,
        error: str,
    ) -> dict[str, Any]:
        commit, dirty = git_state(self.repo_root)
        previous = self._previous_manifest(int(cycle_id.split("_")[1]))
        parent = None if previous is None else previous.get("cycle_id")
        by_stage = {record.stage.value: record for record in records}
        eval_manifest = context.get("eval_manifest") or {}
        aggregates = eval_manifest.get("aggregates") or {}
        checkpoint = {
            "path": context.get("checkpoint_path", ""),
            "metadata_hash": context.get("checkpoint_metadata_hash", ""),
            "verification": context.get("checkpoint_verification", "none"),
        }
        dataset = {
            "curated_version": context.get("curated_version", ""),
            "curated_version_dir": context.get("curated_version_dir", ""),
            "dataset_hash": context.get("dataset_hash", ""),
            "build_mode": context.get("build_mode", self.config.build.mode),
            "build_summary": context.get("build_summary", {}),
        }
        flywheel = context.get("flywheel") or {}
        return {
            "schema_version": SCHEMA_VERSION,
            "cycle_id": cycle_id,
            "parent_cycle_id": parent,
            "run_id": self.run_id,
            "config_path": self.config.source_path,
            "config_hash": self.config.fingerprint,
            "git_commit": commit,
            "git_dirty": dirty,
            "seeds": {
                "generation": context.get("generation_seeds", []),
                "evaluation": context.get("evaluation_seeds", []),
                "targeted": context.get("targeted_seeds", []),
            },
            "source_episode_ids": context.get("source_episode_ids", []),
            "dataset": dataset,
            "checkpoint": checkpoint,
            "evaluation_run_id": context.get("eval_run_id", ""),
            "evaluation": {
                "success_rate": aggregates.get("success_rate"),
                "manifest": eval_manifest,
            },
            "failure_distribution_id": context.get("failure_distribution_id", ""),
            "flywheel": flywheel,
            "stages": {name: record.to_dict() for name, record in by_stage.items()},
            "wall_time_s": round(wall_s, 3),
            "status": "FAILED" if error else "COMPLETED",
            "error": error,
        }

    # -- flywheel ----------------------------------------------------------
    def _get_evaluation_config(self) -> EvaluationConfig:
        if self._evaluation_config is None:
            self._evaluation_config = load_evaluation_config(self.config.evaluation.config)
        return self._evaluation_config

    def _ledger_path(self, value: str) -> str:
        """Resolve ledger/stop paths relative to the run directory.

        Keeping them inside the run keeps a test or a rehearsal from appending
        to the tracked ``data/flywheel`` ledger.
        """
        if not value:
            return ""
        path = Path(value).expanduser()
        return str(path if path.is_absolute() else self.run_root / path)

    def record_flywheel_decision(self, manifest: dict[str, Any]) -> dict[str, Any]:
        """Append one flywheel cycle for a completed EVALUATE/ANALYZE pair.

        Reuses :class:`FlywheelController` and its append-only ledger; this
        method never trains and never requests a rollout by itself.
        """
        evaluation = self._get_evaluation_config()
        ledger = self._ledger_path(self.config.evaluation.ledger_path) or None
        stop_file = self._ledger_path(self.config.evaluation.stop_file) or None
        controller = FlywheelController(
            evaluation,
            ledger_path=ledger,
            stop_file=stop_file,
        )
        aggregates = (manifest.get("evaluation") or {}).get("manifest", {}).get("aggregates", {})
        num_episodes = int(self.config.evaluation.num_episodes)
        success_rate = float(aggregates.get("success_rate") or 0.0)
        eval_run_id = manifest.get("evaluation_run_id") or f"eval_{manifest['cycle_id']}"
        existing, _ = controller.load_ledger()
        for record in existing:
            if record.eval_run_id == eval_run_id:
                # Resume of an already-decided cycle: reuse the append-only
                # record instead of appending a duplicate STOP.
                return record.to_dict()
        record = controller.decide(
            eval_run_id=eval_run_id,
            success_rate=success_rate,
            n_failures=int(round((1.0 - success_rate) * num_episodes)),
            policy_only_failures=int(round((1.0 - success_rate) * num_episodes)),
            checkpoint_hash=(manifest.get("checkpoint") or {}).get("metadata_hash") or "none",
            failure_distribution_id=manifest.get("failure_distribution_id") or "none",
            source_dataset_versions=tuple(
                value for value in [(manifest.get("dataset") or {}).get("curated_version")] if value
            ),
            episode_ids=tuple(manifest.get("source_episode_ids") or []),
            episode_hash=sha256_bytes(
                json.dumps(manifest.get("source_episode_ids") or [], sort_keys=True).encode("utf-8")
            ),
            requested_rollouts=self.config.targeted_generate.num_rollouts,
            generated_rollouts=len(manifest.get("seeds", {}).get("targeted") or []),
        )
        return record.to_dict()

    # -- lineage -----------------------------------------------------------
    def lineage(self, query: str) -> dict[str, Any]:
        """Find every manifest whose checkpoint path or metadata hash matches."""
        text = str(query)
        matches: list[dict[str, Any]] = []
        for path in sorted(self.manifests_dir.glob("cycle_*.json")):
            manifest = read_json(path)
            checkpoint = manifest.get("checkpoint") or {}
            checkpoint_path = str(checkpoint.get("path") or "")
            metadata_hash = str(checkpoint.get("metadata_hash") or "")
            hit = text in {checkpoint_path, metadata_hash}
            if not hit and checkpoint_path:
                hit = Path(checkpoint_path).name == text
            if not hit:
                continue
            matches.append(
                {
                    "manifest": str(path),
                    "cycle_id": manifest.get("cycle_id"),
                    "run_id": manifest.get("run_id"),
                    "config_hash": manifest.get("config_hash"),
                    "git_commit": manifest.get("git_commit"),
                    "episode_ids": manifest.get("source_episode_ids", []),
                    "dataset": manifest.get("dataset", {}),
                    "checkpoint": checkpoint,
                    "evaluation_run_id": manifest.get("evaluation_run_id", ""),
                    "failure_distribution_id": manifest.get("failure_distribution_id", ""),
                }
            )
        return {"query": text, "matches": matches, "count": len(matches)}


__all__ = [
    "CYCLE_APPROVAL_FLAG",
    "SUPPLY_CHECKPOINT_FLAG",
    "ApprovalRequiredError",
    "CheckpointVerificationError",
    "CorruptStageOutputError",
    "CycleLimitError",
    "OrchestratorError",
    "PipelineOrchestrator",
    "RunFlags",
    "StageError",
    "StageRecord",
    "StageStatus",
    "SeedLedgerError",
]
