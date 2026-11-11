"""Failure distribution fitting, fallbacks, definitions and sampler bounds."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from sim2real.evaluation import (
    FailureDistribution,
    FailureDistributionError,
    TargetedSampler,
    fit_failure_distribution,
    load_evaluation_config,
    load_profile_config,
)
from sim2real.evaluation.failure_analysis import failure_feature_vectors
from sim2real.evaluation.records import EvalEpisodeRecord
from sim2real.randomization import FEATURE_DIM, from_feature_vector

CONFIG = load_evaluation_config()
PROFILE = load_profile_config("default")


def _record(seed: int, vector: np.ndarray | None, *, success: bool = False) -> EvalEpisodeRecord:
    if vector is None:
        randomization: dict = {}
    else:
        sample_obj, _ = from_feature_vector(vector, PROFILE, base_seed=seed, episode_index=seed)
        randomization = sample_obj.to_dict()
    return EvalEpisodeRecord(
        episode_id=f"run_ep{seed:04d}",
        eval_run_id="run",
        seed=seed,
        episode_index=seed,
        randomization_profile="default",
        randomization=randomization,
        initial_object_pose=(0.45, 0.0, 0.0),
        friction_sliding=0.9,
        object_mass=0.07,
        light_intensity=1.0,
        color_temperature_k=6500.0,
        camera_jitter_norm=0.01,
        task_instruction="pick",
        success=success,
        task_completion=0.0,
        collision_count=0,
        termination_reason="SUCCESS" if success else "TIMEOUT",
        policy_id="test",
        checkpoint_hash="none",
        config_hash="abc",
        git_commit="deadbeef",
        git_dirty=False,
        n_action_steps=10,
        renderer="fake-analytic",
    )


def _cluster(count: int, center: np.ndarray, spread: float = 0.01) -> list[np.ndarray]:
    rng = np.random.default_rng(123)
    return [
        np.clip(center + rng.normal(0.0, spread, size=FEATURE_DIM), 0.0, 1.0) for _ in range(count)
    ]


def test_gmm_path_targets_the_cluster_and_stays_in_bounds():
    center = np.full(FEATURE_DIM, 0.2)
    center[0] = 0.8
    vectors = _cluster(40, center)
    records = [_record(seed=20000 + i, vector=v) for i, v in enumerate(vectors)]
    paired = {record.seed: True for record in records}
    distribution = fit_failure_distribution(
        records, CONFIG, source_eval_run_id="run", paired=paired
    )
    assert distribution.type == "gmm"
    assert distribution.n_failures == 40
    sampler = TargetedSampler(distribution, CONFIG, seed=30000)
    matrix, report = sampler.sample(500)
    assert matrix.shape == (500, FEATURE_DIM)
    assert np.all((matrix >= 0.0) & (matrix <= 1.0))
    assert report["targeted_fraction_actual"] == pytest.approx(0.7, abs=0.01)
    targeted_distance = float(np.mean(np.linalg.norm(matrix - center, axis=1)))
    rng = np.random.default_rng(0)
    uniform = rng.uniform(0.0, 1.0, size=(500, FEATURE_DIM))
    uniform_distance = float(np.mean(np.linalg.norm(uniform - center, axis=1)))
    assert targeted_distance < uniform_distance


def test_kde_path_is_used_below_the_gmm_minimum():
    vectors = _cluster(15, np.full(FEATURE_DIM, 0.6))
    records = [_record(seed=i, vector=v) for i, v in enumerate(vectors)]
    distribution = fit_failure_distribution(
        records, CONFIG, source_eval_run_id="run", paired={r.seed: True for r in records}
    )
    assert distribution.type == "kde"
    sampler = TargetedSampler(distribution, CONFIG, seed=30001)
    matrix, _ = sampler.sample(200)
    assert np.all((matrix >= 0.0) & (matrix <= 1.0))


def test_neighbour_fallback_with_three_failures_is_bounded_and_deterministic():
    vectors = _cluster(3, np.array([0.5] * FEATURE_DIM))
    records = [_record(seed=i, vector=v) for i, v in enumerate(vectors)]
    paired = {r.seed: True for r in records}
    distribution = fit_failure_distribution(
        records, CONFIG, source_eval_run_id="run", paired=paired
    )
    assert distribution.type == "neighbor"
    first, _ = TargetedSampler(distribution, CONFIG, seed=30002).sample(50)
    second, _ = TargetedSampler(distribution, CONFIG, seed=30002).sample(50)
    assert np.array_equal(first, second)
    assert np.all((first >= 0.0) & (first <= 1.0))


def test_zero_failures_is_no_targeting():
    records = [_record(seed=i, vector=None, success=True) for i in range(5)]
    distribution = fit_failure_distribution(
        records, CONFIG, source_eval_run_id="run", paired={r.seed: True for r in records}
    )
    assert distribution.type == "none"
    assert "NO_TARGETING" in distribution.fallback_reason
    matrix, report = TargetedSampler(distribution, CONFIG, seed=1).sample(10)
    assert report["targeted_fraction_actual"] == 0.0
    assert np.all((matrix >= 0.0) & (matrix <= 1.0))


def test_invalid_and_duplicate_failure_data_degrades_without_crashing():
    base = np.full(FEATURE_DIM, 0.5)
    records = [_record(seed=i, vector=base) for i in range(35)]
    distribution = fit_failure_distribution(
        records, CONFIG, source_eval_run_id="run", paired={r.seed: True for r in records}
    )
    assert distribution.type == "neighbor"
    assert "duplicate" in distribution.fallback_reason

    malformed = [_record(seed=50000 + i, vector=None) for i in range(4)]
    matrix, report = failure_feature_vectors(
        malformed, CONFIG, paired={r.seed: True for r in malformed}
    )
    assert matrix.shape == (0, FEATURE_DIM)
    assert len(report["rejected"]) == 4
    with pytest.raises(FailureDistributionError):
        failure_feature_vectors(malformed, CONFIG, paired=None)


def test_failure_definition_policy_only_and_any_failure():
    vectors = _cluster(4, np.full(FEATURE_DIM, 0.5))
    records = [_record(seed=20 + i, vector=v) for i, v in enumerate(vectors)]
    paired = {20: True, 21: True, 22: False, 23: False}

    policy_only = fit_failure_distribution(records, CONFIG, source_eval_run_id="run", paired=paired)
    assert policy_only.failure_definition == "policy_only"
    assert policy_only.n_failures == 2

    any_config = replace(
        CONFIG,
        failure_analysis=replace(CONFIG.failure_analysis, failure_definition="any_failure"),
    )
    any_failure = fit_failure_distribution(
        records, any_config, source_eval_run_id="run", paired=paired
    )
    assert any_failure.n_failures == 4

    with pytest.raises(FailureDistributionError):
        fit_failure_distribution(records, CONFIG, source_eval_run_id="run", paired=None)


def test_sampler_bounds_and_targeted_fraction_with_boundary_failures():
    vectors = [np.zeros(FEATURE_DIM), np.ones(FEATURE_DIM)] + _cluster(
        35, np.full(FEATURE_DIM, 0.5)
    )
    records = [_record(seed=100 + i, vector=v) for i, v in enumerate(vectors)]
    paired = {r.seed: True for r in records}
    distribution = fit_failure_distribution(
        records, CONFIG, source_eval_run_id="run", paired=paired
    )
    matrix, report = TargetedSampler(distribution, CONFIG, seed=30003).sample(2000)
    assert matrix.shape == (2000, FEATURE_DIM)
    assert np.all((matrix >= 0.0) & (matrix <= 1.0))
    assert report["targeted_fraction_actual"] == pytest.approx(0.7, abs=0.001)
    assert np.array_equal(matrix, TargetedSampler(distribution, CONFIG, seed=30003).sample(2000)[0])


def test_distribution_save_load_round_trip_and_hash_mismatch(tmp_path):
    records = [
        _record(seed=i, vector=v) for i, v in enumerate(_cluster(12, np.full(FEATURE_DIM, 0.4)))
    ]
    distribution = fit_failure_distribution(
        records, CONFIG, source_eval_run_id="run", paired={r.seed: True for r in records}
    )
    path = distribution.save(tmp_path)
    loaded = FailureDistribution.load(path)
    assert loaded.to_dict() == distribution.to_dict()
    data = path.read_text(encoding="utf-8").replace('"n_failures": 12', '"n_failures": 13')
    tampered = tmp_path / "tampered.json"
    tampered.write_text(data, encoding="utf-8")
    with pytest.raises(FailureDistributionError):
        FailureDistribution.load(tampered)


def test_numpy_gmm_cross_check_against_sklearn():
    """Optional cross-check: higher-level EM and scikit-learn select the same model.

    Tolerance: both BIC argmins must select the same component count; BIC values
    at the selected count must agree within 2% relative; matched component means
    must agree within 0.05 in normalised feature units.  The test skips with a
    reason when scikit-learn is not installed, so CI never requires it.
    """
    pytest.importorskip(
        "sklearn", reason="scikit-learn is not installed in the core .venv (optional cross-check)"
    )
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    from sim2real.evaluation.failure_analysis import _fit_gmm_numpy, _fit_gmm_sklearn

    rng = np.random.default_rng(7)
    centers = np.array(
        [
            [0.2] * FEATURE_DIM,
            [0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2],
            [0.5, 0.8, 0.5, 0.8, 0.5, 0.8, 0.5, 0.8],
        ]
    )
    features = np.vstack(
        [center + rng.normal(0.0, 0.03, size=(40, FEATURE_DIM)) for center in centers]
    )
    features = np.clip(features, 0.0, 1.0)
    reg_covar = 1.0e-4
    seed = 0
    numpy_fits = [_fit_gmm_numpy(features, k, reg_covar, seed) for k in (1, 2, 3)]
    sklearn_fits = [_fit_gmm_sklearn(features, k, reg_covar, seed) for k in (1, 2, 3)]
    best_numpy = min(numpy_fits, key=lambda fit: fit["bic"])
    best_sklearn = min(sklearn_fits, key=lambda fit: fit["bic"])

    assert best_numpy["components"] == best_sklearn["components"]
    relative_bic = abs(best_numpy["bic"] - best_sklearn["bic"]) / max(1.0, abs(best_sklearn["bic"]))
    assert relative_bic <= 0.02

    numpy_means = np.asarray(best_numpy["means"], dtype=np.float64)
    sklearn_means = np.asarray(best_sklearn["means"], dtype=np.float64)
    rows, columns = linear_sum_assignment(cdist(numpy_means, sklearn_means))
    max_mean_difference = float(np.max(np.abs(numpy_means[rows] - sklearn_means[columns])))
    assert max_mean_difference <= 0.05
