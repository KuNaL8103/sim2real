"""Failure-targeted rollout generation (collector library).

Loads a hashed failure distribution, draws feature vectors with the seeded
mixture sampler, maps each vector back to a full valid randomization sample with
:func:`sim2real.randomization.features.from_feature_vector`, and runs the
scripted expert with that sample applied directly.

The Task 4 expert still fails roughly 40% of clean episodes, so targeted rollouts
are not automatically useful training data: roughly 60% are expected to be
successful and fewer pass the trajectory filter.  The caller must report the
measured yield, not promise it.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from sim2real import contracts
from sim2real.evaluation.config import EvaluationConfig
from sim2real.evaluation.failure_analysis import (
    FailureDistribution,
    FailureDistributionError,
    TargetedSampler,
    load_profile_config,
)
from sim2real.expert.config import ExpertConfig, load_expert_config
from sim2real.expert.ik import DampedLeastSquaresIK
from sim2real.expert.instruction import parse_instruction
from sim2real.expert.planner import CartesianInterpolationPlanner
from sim2real.expert.rollout import EpisodeRecord, _EpisodeRunner
from sim2real.expert.staging import StagingBuffer, StagingError
from sim2real.randomization.applier import apply_randomization
from sim2real.sim.backend import ResetConfig

TARGETED_SEED_LOW = 30000
TARGETED_SEED_HIGH = 30999


class TargetedGenerationError(ValueError):
    """Raised when a targeted-generation request is invalid."""


def targeted_seeds(seed: int, count: int) -> list[int]:
    """Derive ``count`` seeds inside the reserved 30000-30999 range."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise TargetedGenerationError("seed must be a non-negative integer")
    if not TARGETED_SEED_LOW <= seed <= TARGETED_SEED_HIGH:
        raise TargetedGenerationError(
            f"seed must be inside {TARGETED_SEED_LOW}-{TARGETED_SEED_HIGH}"
        )
    span = TARGETED_SEED_HIGH - TARGETED_SEED_LOW + 1
    seeds: list[int] = []
    for index in range(count):
        state = np.random.SeedSequence([seed, index, 0x54415247]).generate_state(1, dtype=np.uint32)
        seeds.append(TARGETED_SEED_LOW + int(state[0]) % span)
    return seeds


def _run_expert_with_sample(
    backend: Any,
    task: Any,
    seed: int,
    episode_index: int,
    sample: Any,
    *,
    instruction: str,
    config: ExpertConfig,
    run_id: str,
    distribution_id: str,
    max_steps: int | None = None,
) -> EpisodeRecord:
    record = EpisodeRecord(
        episode_id=f"{run_id}_ep{episode_index:04d}",
        run_id=run_id,
        episode_index=episode_index,
        seed=int(seed),
        randomization_seed=int(seed),
        backend=str(getattr(backend, "name", type(backend).__name__)),
        task_instruction=instruction,
        task_object_id=task.object_id,
        task_zone_id=task.zone_id,
        control_rate_hz=float(backend.control_rate_hz),
        outcome=contracts.OUTCOME_UNKNOWN,
        failure_reason="",
        is_perturbed=False,
        perturbation_name=None,
        expected_quality="clean",
        started_at=datetime.now(UTC).isoformat(),
    )
    try:
        backend.reset(seed=int(seed), config=ResetConfig(seed=int(seed)))
        report = apply_randomization(backend, sample)
        backend.get_observation()
        record.randomization_report = {
            "targeted": True,
            "failure_distribution_id": distribution_id,
            "sample": sample.to_dict(),
            "report": report.to_dict(),
        }
        zone = backend.configs.scene.zones[task.zone_id]
        zone_xy = np.asarray(zone.position, dtype=np.float64)[:2]
        cube_pose = np.asarray(sample.object_pose.poses[task.object_id], dtype=np.float64)
        in_region = bool(
            np.linalg.norm(cube_pose[:2, 3] - zone_xy)
            <= zone.radius * config.success.success_zone_scale
            + config.success.zone_radius_tolerance_m
        )
        record.metadata = {
            "targeted": True,
            "failure_distribution_id": distribution_id,
            "starts_in_success_region": in_region,
        }
        setter = getattr(backend, "set_collision_context", None)
        if callable(setter):
            setter(task.object_id)
        runner = _EpisodeRunner(
            backend=backend,
            task=task,
            record=record,
            config=config,
            ik_solver=DampedLeastSquaresIK(config.ik),
            planner=CartesianInterpolationPlanner(),
            perturbation=None,
            max_steps=int(config.max_steps if max_steps is None else max_steps),
        )
        return runner.run()
    except Exception as exc:  # noqa: BLE001 - a bad targeted episode must not stop the batch
        record.outcome = contracts.OUTCOME_ABORTED
        record.failure_reason = f"{type(exc).__name__}: {exc}"
        return record


def load_targeted_distribution(
    path: str | Path, *, expected_id: str | None = None
) -> FailureDistribution:
    """Load and validate a distribution, including its content hash."""
    try:
        distribution = FailureDistribution.load(path, expected_hash=None)
    except FailureDistributionError:
        raise
    if expected_id is not None and distribution.distribution_id != expected_id:
        raise TargetedGenerationError(
            f"failure distribution id {distribution.distribution_id} != {expected_id}"
        )
    if distribution.type == "none":
        raise TargetedGenerationError("failure distribution is NO_TARGETING; refusing to generate")
    return distribution


def run_targeted_rollouts(
    backend: Any,
    *,
    failure_distribution_path: str | Path,
    num_rollouts: int,
    seed: int,
    task_instruction: str,
    output_dir: str | Path,
    evaluation_config: EvaluationConfig,
    expert_config: ExpertConfig | None = None,
    randomization_dir: str | Path = "configs/randomization",
    max_rollouts_per_call: int | None = None,
    profile_config: Any | None = None,
    run_id: str | None = None,
    on_episode: Any | None = None,
) -> dict[str, Any]:
    """Run at most ``max_rollouts_per_call`` targeted rollouts and stage them."""
    config = expert_config or load_expert_config()
    cap = (
        evaluation_config.flywheel.max_rollouts_per_request
        if max_rollouts_per_call is None
        else int(max_rollouts_per_call)
    )
    if num_rollouts < 1 or num_rollouts > cap:
        raise TargetedGenerationError(
            f"num_rollouts={num_rollouts} is out of bounds; expected 1..{cap}"
        )
    distribution = load_targeted_distribution(failure_distribution_path)
    if distribution.failure_definition not in {"policy_only", "any_failure"}:
        raise TargetedGenerationError(
            f"unsupported failure_definition {distribution.failure_definition!r}"
        )
    seeds = targeted_seeds(int(seed), int(num_rollouts))
    if profile_config is None:
        profile_config = load_profile_config(
            evaluation_config.settings.randomization_profile,
            randomization_dir=randomization_dir,
        )
    sampler = TargetedSampler(
        distribution,
        evaluation_config,
        seed=int(seed),
        profile_config=profile_config,
    )
    features, sampler_report = sampler.sample(int(num_rollouts))
    pairs = sampler.samples_to_randomization(features, base_seed=int(seed))

    task = parse_instruction(task_instruction, backend.configs.scene, config.instructions_file)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_id = run_id or f"targeted_{stamp}_{seed}"
    buffer = StagingBuffer(output_dir, run_id)
    counts = {"success": 0, "failure": 0, "timeout": 0, "aborted": 0, "unknown": 0}
    outcome_names = {
        contracts.OUTCOME_SUCCESS: "success",
        contracts.OUTCOME_FAILURE: "failure",
        contracts.OUTCOME_TIMEOUT: "timeout",
        contracts.OUTCOME_ABORTED: "aborted",
        contracts.OUTCOME_UNKNOWN: "unknown",
    }
    staged: list[dict[str, Any]] = []
    sampled: list[dict[str, Any]] = []
    generated = 0
    for index, ((sample, sample_report), episode_seed) in enumerate(zip(pairs, seeds, strict=True)):
        record = _run_expert_with_sample(
            backend,
            task,
            episode_seed,
            index,
            sample,
            instruction=task_instruction,
            config=config,
            run_id=run_id,
            distribution_id=distribution.distribution_id,
        )
        try:
            buffer.write(record)
        except StagingError as exc:
            sampled.append(
                {
                    "episode_index": index,
                    "seed": episode_seed,
                    "staged": False,
                    "error": str(exc),
                }
            )
            continue
        if on_episode is not None:
            on_episode(record)
        generated += 1
        counts[outcome_names.get(record.outcome, "unknown")] += 1
        staged.append({"episode_id": record.episode_id, "seed": episode_seed})
        sampled.append(
            {
                "episode_index": index,
                "seed": episode_seed,
                "episode_id": record.episode_id,
                "outcome": outcome_names.get(record.outcome, "unknown"),
                "clipped_features": sample_report["clipped_features"],
                "separation_repaired": sample_report["separation_repaired"],
            }
        )
    num_failed = counts["failure"] + counts["timeout"] + counts["aborted"]
    yield_rate = counts["success"] / generated if generated else 0.0
    return {
        "success": True,
        "run_id": run_id,
        "backend": str(getattr(backend, "name", type(backend).__name__)),
        "failure_distribution_id": distribution.distribution_id,
        "failure_definition": distribution.failure_definition,
        "num_requested": int(num_rollouts),
        "num_generated": generated,
        "num_failed": num_failed,
        "counts": counts,
        "expert_yield": yield_rate,
        "seeds": seeds,
        "episode_ids": [item["episode_id"] for item in staged],
        "sampler_report": sampler_report,
        "episodes": sampled,
        "staging_dir": str(buffer.root),
    }


def summarize_result(result: dict[str, Any]) -> str:
    return (
        f"targeted {result['num_generated']}/{result['num_requested']} episodes; "
        f"success={result['counts']['success']} failure={result['counts']['failure']} "
        f"timeout={result['counts']['timeout']} aborted={result['counts']['aborted']}; "
        f"expert_yield={result['expert_yield']:.2f}; "
        f"distribution={result['failure_distribution_id']}; run_id={result['run_id']}"
    )


def result_metadata_json(result: dict[str, Any]) -> str:
    payload = {
        "run_id": result["run_id"],
        "backend": result["backend"],
        "failure_distribution_id": result["failure_distribution_id"],
        "failure_definition": result["failure_definition"],
        "num_requested": result["num_requested"],
        "num_generated": result["num_generated"],
        "num_failed": result["num_failed"],
        "counts": result["counts"],
        "expert_yield": result["expert_yield"],
        "seeds": result["seeds"],
        "episode_ids": result["episode_ids"],
        "sampler_report": result["sampler_report"],
        "staging_dir": result["staging_dir"],
    }
    return json.dumps(payload, sort_keys=True, allow_nan=False)


__all__ = [
    "TARGETED_SEED_HIGH",
    "TARGETED_SEED_LOW",
    "TargetedGenerationError",
    "load_targeted_distribution",
    "result_metadata_json",
    "run_targeted_rollouts",
    "summarize_result",
    "targeted_seeds",
]
