"""GenerateRollouts FAILURE_TARGETED handler and targeted-generation library."""

from __future__ import annotations

import json

import pytest

from sim2real.evaluation import (
    FailureDistribution,
    FailureDistributionError,
    TargetedGenerationError,
    load_evaluation_config,
    load_profile_config,
)
from sim2real.evaluation.targeted import (
    TARGETED_SEED_HIGH,
    TARGETED_SEED_LOW,
    load_targeted_distribution,
    run_targeted_rollouts,
    targeted_seeds,
)
from sim2real.randomization import FEATURE_NAMES, feature_bounds
from sim2real.sim import create_backend

CONFIG = load_evaluation_config()
PROFILE = load_profile_config("default")


def _distribution() -> FailureDistribution:
    return FailureDistribution(
        type="neighbor",
        parameters={"sigma": 0.05, "centers": [[0.5] * len(FEATURE_NAMES)] * 3},
        n_failures=3,
        feature_names=FEATURE_NAMES,
        bounds=feature_bounds(PROFILE),
        seed=0,
        source_eval_run_id="run",
        failure_definition="any_failure",
        config_hash="abc",
    )


def _run(tmp_path, path, *, count=3, seed=30000, cap=None):
    backend = create_backend("fake", "configs/sim")
    try:
        return run_targeted_rollouts(
            backend,
            failure_distribution_path=path,
            num_rollouts=count,
            seed=seed,
            task_instruction=CONFIG.settings.task_instruction,
            output_dir=tmp_path / "staging",
            evaluation_config=CONFIG,
            max_rollouts_per_call=cap,
        )
    finally:
        backend.close()


def test_missing_malformed_and_hash_mismatched_references_are_refused(tmp_path):
    with pytest.raises(FailureDistributionError):
        load_targeted_distribution(tmp_path / "missing.json")

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not json", encoding="utf-8")
    with pytest.raises(FailureDistributionError):
        load_targeted_distribution(malformed)

    distribution = _distribution()
    path = distribution.save(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["n_failures"] = 99
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(FailureDistributionError):
        load_targeted_distribution(tampered)
    with pytest.raises(FailureDistributionError):
        _run(tmp_path, tampered)


def test_no_targeting_distribution_is_refused(tmp_path):
    distribution = _distribution()
    distribution.type = "none"
    distribution.fallback_reason = "NO_TARGETING: zero valid failures"
    distribution.content_hash = distribution.compute_hash()
    path = distribution.save(tmp_path)
    with pytest.raises(TargetedGenerationError):
        load_targeted_distribution(path)


def test_num_rollouts_above_the_cap_is_rejected(tmp_path):
    path = _distribution().save(tmp_path)
    with pytest.raises(TargetedGenerationError):
        _run(tmp_path, path, count=3, cap=2)


def test_valid_reference_stages_episodes_with_the_distribution_id(tmp_path):
    distribution = _distribution()
    path = distribution.save(tmp_path)
    result = _run(tmp_path, path, count=3, seed=30000)
    assert result["success"] is True
    assert result["num_generated"] == 3
    assert result["failure_distribution_id"] == distribution.distribution_id
    assert result["failure_definition"] == "any_failure"
    assert all(TARGETED_SEED_LOW <= seed <= TARGETED_SEED_HIGH for seed in result["seeds"])
    staging = tmp_path / "staging" / result["run_id"]
    for episode_id in result["episode_ids"]:
        sidecar = json.loads((staging / f"episode_{episode_id}.json").read_text(encoding="utf-8"))
        assert sidecar["metadata"]["failure_distribution_id"] == distribution.distribution_id
        assert sidecar["metadata"]["starts_in_success_region"] in {True, False}


def test_targeted_seeds_stay_inside_the_reserved_range():
    seeds = targeted_seeds(30000, 200)
    assert all(TARGETED_SEED_LOW <= seed <= TARGETED_SEED_HIGH for seed in seeds)
    assert seeds == targeted_seeds(30000, 200)
    with pytest.raises(TargetedGenerationError):
        targeted_seeds(20000, 1)
