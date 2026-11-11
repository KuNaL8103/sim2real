"""Failure-mode analysis over initial-state / randomization features.

The feature space is the normalised vector from
:mod:`sim2real.randomization.features`.  Failures are modelled with a Gaussian
mixture (selected by BIC), a Gaussian KDE, or a bounded neighbour perturbation,
with documented minimum sample counts and an explicit fallback chain.  Zero
failures produce ``NO_TARGETING``: the flywheel must not request rollouts without
a failure model.

Failure definition (``configs/evaluation/default.yaml``):

* ``policy_only`` (default) — seeds where the **expert succeeded but the policy
  failed**, taken from a paired evaluation.  Targeting ``any_failure`` would also
  sample regions where the scripted expert fails, producing demonstrations the
  trajectory filter rejects (wasted compute).  ``policy_only`` without paired
  data is refused.
* ``any_failure`` — every failed episode.

The fitted artifact is JSON with a content hash so the collector can refuse a
truncated or edited distribution.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sim2real.evaluation.config import EvaluationConfig, FailureAnalysisConfig
from sim2real.evaluation.records import EvalEpisodeRecord
from sim2real.randomization.config import RandomizationConfig, load_profile, scene_defaults
from sim2real.randomization.features import (
    FEATURE_DIM,
    FEATURE_NAMES,
    feature_bounds,
    from_feature_vector,
    sample_from_dict,
    to_feature_vector,
)
from sim2real.sim.config import load_sim_configs

DISTRIBUTION_TYPES = ("gmm", "kde", "neighbor", "none")
DEFAULT_SIM_CONFIG_DIR = "configs/sim"
DEFAULT_RANDOMIZATION_DIR = "configs/randomization"


class FailureDistributionError(ValueError):
    """Raised when a failure distribution cannot be fitted or validated."""


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _content_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def load_profile_config(
    profile: str,
    *,
    sim_config_dir: str | Path = DEFAULT_SIM_CONFIG_DIR,
    randomization_dir: str | Path = DEFAULT_RANDOMIZATION_DIR,
) -> RandomizationConfig:
    scene = load_sim_configs(sim_config_dir).scene
    return load_profile(profile, scene=scene_defaults(scene), profiles_dir=randomization_dir)


@dataclass
class FailureDistribution:
    type: str
    parameters: dict[str, Any]
    n_failures: int
    feature_names: tuple[str, ...]
    bounds: dict[str, tuple[float, float]]
    seed: int
    source_eval_run_id: str
    failure_definition: str
    config_hash: str
    fallback_reason: str = ""
    schema_version: int = 1
    content_hash: str = ""

    def __post_init__(self) -> None:
        if self.type not in DISTRIBUTION_TYPES:
            raise FailureDistributionError(f"invalid distribution type {self.type!r}")
        if self.feature_names != FEATURE_NAMES:
            raise FailureDistributionError("feature_names do not match the feature space")
        if self.type == "none" and not self.fallback_reason:
            self.fallback_reason = "NO_TARGETING: no failures"
        if not self.content_hash:
            self.content_hash = self.compute_hash()

    def compute_hash(self) -> str:
        payload = self.to_dict()
        payload.pop("content_hash", None)
        payload.pop("id", None)
        return _content_hash(payload)

    @property
    def distribution_id(self) -> str:
        return self.content_hash[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "id": self.distribution_id,
            "type": self.type,
            "parameters": self.parameters,
            "n_failures": int(self.n_failures),
            "feature_names": list(self.feature_names),
            "bounds": {
                name: [float(low), float(high)] for name, (low, high) in self.bounds.items()
            },
            "seed": int(self.seed),
            "source_eval_run_id": self.source_eval_run_id,
            "failure_definition": self.failure_definition,
            "config_hash": self.config_hash,
            "fallback_reason": self.fallback_reason,
            "content_hash": self.content_hash,
        }

    def save(self, output_dir: str | Path) -> Path:
        directory = Path(output_dir).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.distribution_id}.json"
        if path.exists():
            return path
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path, *, expected_hash: str | None = None) -> FailureDistribution:
        source = Path(path).expanduser()
        if not source.is_file():
            raise FailureDistributionError(f"failure distribution not found: {source}")
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise FailureDistributionError(f"malformed failure distribution: {exc}") from exc
        if not isinstance(data, dict):
            raise FailureDistributionError("failure distribution must be a JSON object")
        stored_hash = data.get("content_hash")
        if not isinstance(stored_hash, str) or not stored_hash:
            raise FailureDistributionError("failure distribution has no content_hash")
        payload = dict(data)
        payload.pop("content_hash", None)
        payload.pop("id", None)
        recomputed = _content_hash(payload)
        if recomputed != stored_hash:
            raise FailureDistributionError(
                f"failure distribution hash mismatch: {recomputed} != {stored_hash}"
            )
        if expected_hash is not None and stored_hash != expected_hash:
            raise FailureDistributionError("failure distribution does not match the expected hash")
        distribution = cls(
            type=str(data["type"]),
            parameters=dict(data["parameters"]),
            n_failures=int(data["n_failures"]),
            feature_names=tuple(str(name) for name in data["feature_names"]),
            bounds={
                str(name): (float(value[0]), float(value[1]))
                for name, value in data["bounds"].items()
            },
            seed=int(data["seed"]),
            source_eval_run_id=str(data["source_eval_run_id"]),
            failure_definition=str(data["failure_definition"]),
            config_hash=str(data["config_hash"]),
            fallback_reason=str(data.get("fallback_reason", "")),
            schema_version=int(data.get("schema_version", 1)),
            content_hash=stored_hash,
        )
        if distribution.compute_hash() != stored_hash:
            raise FailureDistributionError("failure distribution content is inconsistent")
        return distribution


# -- NumPy Gaussian mixture -------------------------------------------------


def _logpdf(X: np.ndarray, mean: np.ndarray, covariance: np.ndarray, reg: float) -> np.ndarray:
    dimension = X.shape[1]
    cov = covariance + np.eye(dimension) * reg
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0 or not math.isfinite(logdet):
        return np.full(X.shape[0], -np.inf)
    difference = X - mean
    try:
        solved = np.linalg.solve(cov, difference.T).T
    except np.linalg.LinAlgError:
        return np.full(X.shape[0], -np.inf)
    quadratic = np.einsum("ij,ij->i", difference, solved)
    return -0.5 * (dimension * math.log(2.0 * math.pi) + logdet + quadratic)


def _responsibilities(
    X: np.ndarray,
    weights: np.ndarray,
    means: list[np.ndarray],
    covariances: list[np.ndarray],
    reg: float,
) -> tuple[np.ndarray, float]:
    log_prob = np.empty((X.shape[0], len(means)))
    for index, (mean, covariance) in enumerate(zip(means, covariances, strict=True)):
        weight = max(float(weights[index]), 1.0e-12)
        log_prob[:, index] = math.log(weight) + _logpdf(X, mean, covariance, reg)
    maximum = np.max(log_prob, axis=1)
    shifted = np.exp(log_prob - maximum[:, None])
    total = np.sum(shifted, axis=1, keepdims=True)
    log_likelihood = float(np.sum(maximum + np.log(total[:, 0])))
    responsibilities = shifted / total
    return responsibilities, log_likelihood


def _fit_gmm_numpy(
    X: np.ndarray, components: int, reg_covar: float, seed: int, max_iter: int = 200
) -> dict[str, Any]:
    n, dimension = X.shape
    rng = np.random.default_rng(seed)
    if n < components:
        raise FailureDistributionError("not enough failures for the requested components")
    indices = rng.choice(n, size=components, replace=False)
    means = [X[index].copy() for index in indices]
    overall = np.cov(X, rowvar=False)
    overall = np.atleast_2d(overall) + np.eye(dimension) * reg_covar
    covariances = [overall.copy() for _ in range(components)]
    weights = np.full(components, 1.0 / components)
    previous = -np.inf
    log_likelihood = previous
    for _ in range(max_iter):
        responsibilities, log_likelihood = _responsibilities(
            X, weights, means, covariances, reg_covar
        )
        counts = responsibilities.sum(axis=0)
        weights = np.maximum(counts / n, 1.0e-9)
        weights /= weights.sum()
        for index in range(components):
            if counts[index] < 1.0e-9:
                continue
            means[index] = (responsibilities[:, index][:, None] * X).sum(axis=0) / counts[index]
            difference = X - means[index]
            weighted = difference * responsibilities[:, index][:, None]
            covariances[index] = (weighted.T @ difference) / counts[index] + np.eye(dimension) * (
                reg_covar
            )
        if abs(log_likelihood - previous) <= 1.0e-7 * max(1.0, abs(log_likelihood)):
            break
        previous = log_likelihood
    parameter_count = (
        components - 1 + components * dimension + components * dimension * (dimension + 1) / 2.0
    )
    bic = -2.0 * log_likelihood + parameter_count * math.log(max(n, 2))
    return {
        "implementation": "numpy_em",
        "components": components,
        "weights": [float(value) for value in weights],
        "means": [[float(value) for value in mean] for mean in means],
        "covariances": [
            [[float(value) for value in row] for row in covariance] for covariance in covariances
        ],
        "log_likelihood": float(log_likelihood),
        "bic": float(bic),
        "reg_covar": float(reg_covar),
    }


def _fit_gmm_sklearn(
    X: np.ndarray, components: int, reg_covar: float, seed: int
) -> dict[str, Any] | None:
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        return None
    model = GaussianMixture(
        n_components=components,
        covariance_type="full",
        reg_covar=reg_covar,
        random_state=seed,
        max_iter=200,
    )
    model.fit(X)
    return {
        "implementation": "sklearn",
        "components": components,
        "weights": [float(value) for value in model.weights_],
        "means": [[float(value) for value in mean] for mean in model.means_],
        "covariances": [
            [[float(value) for value in row] for row in covariance]
            for covariance in model.covariances_
        ],
        "log_likelihood": float(model.lower_bound_ * X.shape[0]),
        "bic": float(model.bic(X)),
        "reg_covar": float(reg_covar),
    }


def _select_gmm(X: np.ndarray, config: FailureAnalysisConfig) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    maximum = min(config.max_components, X.shape[0])
    for components in range(1, maximum + 1):
        candidate = _fit_gmm_sklearn(X, components, config.reg_covar, config.random_state)
        if candidate is None:
            candidate = _fit_gmm_numpy(X, components, config.reg_covar, config.random_state)
        if best is None or candidate["bic"] < best["bic"]:
            best = candidate
    if best is None:  # pragma: no cover - maximum >= 1
        raise FailureDistributionError("could not fit a Gaussian mixture")
    return best


def _fit_kde(X: np.ndarray, config: FailureAnalysisConfig) -> dict[str, Any]:
    n, dimension = X.shape
    std = float(np.mean(np.std(X, axis=0)))
    silverman = 1.06 * std * n ** (-1.0 / (dimension + 4.0)) if n > 1 else 0.0
    bandwidth = max(config.bandwidth_floor, silverman)
    return {
        "bandwidth": float(bandwidth),
        "centers": [[float(value) for value in row] for row in X],
    }


def _fit_neighbor(X: np.ndarray, config: FailureAnalysisConfig) -> dict[str, Any]:
    return {
        "sigma": float(config.neighbor_sigma_fraction),
        "centers": [[float(value) for value in row] for row in X],
    }


def _expert_success_map(paired: Any) -> dict[int, bool]:
    if paired is None:
        return {}
    if isinstance(paired, dict):
        return {int(seed): bool(value) for seed, value in paired.items()}
    result: dict[int, bool] = {}
    for item in paired:
        result[int(item["seed"])] = bool(item["expert_success"])
    return result


def failure_feature_vectors(
    records: list[EvalEpisodeRecord],
    config: EvaluationConfig,
    *,
    paired: Any = None,
    profile_config: RandomizationConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the normalised feature matrix for the configured failure set."""
    analysis = config.failure_analysis
    if analysis.failure_definition == "policy_only":
        if paired is None:
            raise FailureDistributionError(
                "failure_definition=policy_only requires paired evaluation data "
                "(seed -> expert_success); pass run_paired_eval output"
            )
        expert_success = _expert_success_map(paired)
        selected = [
            record
            for record in records
            if (not record.success) and expert_success.get(record.seed, False)
        ]
    else:
        selected = [record for record in records if not record.success]

    profile_config = profile_config or load_profile_config(config.settings.randomization_profile)
    vectors: list[np.ndarray] = []
    rejected: list[dict[str, Any]] = []
    for record in selected:
        try:
            sample = sample_from_dict(record.randomization)
            vector = to_feature_vector(sample, profile_config)
        except (KeyError, TypeError, ValueError) as exc:
            rejected.append({"episode_id": record.episode_id, "reason": str(exc)})
            continue
        if vector.shape != (FEATURE_DIM,) or not np.all(np.isfinite(vector)):
            rejected.append({"episode_id": record.episode_id, "reason": "invalid feature vector"})
            continue
        vectors.append(vector)
    matrix = np.asarray(vectors, dtype=np.float64).reshape(-1, FEATURE_DIM)
    report = {
        "num_selected": len(selected),
        "num_valid": len(vectors),
        "rejected": rejected,
        "failure_definition": analysis.failure_definition,
    }
    return matrix, report


def fit_failure_distribution(
    records: list[EvalEpisodeRecord],
    config: EvaluationConfig,
    *,
    source_eval_run_id: str,
    paired: Any = None,
    profile_config: RandomizationConfig | None = None,
) -> FailureDistribution:
    """Fit the distribution with the documented fallback chain."""
    analysis = config.failure_analysis
    matrix, report = failure_feature_vectors(
        records, config, paired=paired, profile_config=profile_config
    )
    profile_config = profile_config or load_profile_config(config.settings.randomization_profile)
    bounds = feature_bounds(profile_config)
    fallback_reason = ""
    unique = matrix
    if matrix.shape[0] > 0:
        unique = np.unique(matrix, axis=0)
        if unique.shape[0] < matrix.shape[0]:
            fallback_reason = (
                f"collapsed {matrix.shape[0] - unique.shape[0]} duplicate failure vectors"
            )
    n = int(unique.shape[0])

    if n >= analysis.min_failures_gmm:
        try:
            parameters = _select_gmm(unique, analysis)
            distribution_type = "gmm"
        except (FailureDistributionError, np.linalg.LinAlgError, ValueError) as exc:
            fallback_reason = f"gmm failed ({exc}); fell back to kde"
            parameters = _fit_kde(unique, analysis)
            distribution_type = "kde"
    elif n >= analysis.min_failures_kde:
        parameters = _fit_kde(unique, analysis)
        distribution_type = "kde"
    elif n >= 1:
        if n < 2:
            fallback_reason = (
                fallback_reason + "; " if fallback_reason else ""
            ) + "fewer than 2 distinct failures; covariance is undefined, using neighbour fallback"
        parameters = _fit_neighbor(unique, analysis)
        distribution_type = "neighbor"
    else:
        parameters = {}
        distribution_type = "none"
        fallback_reason = (fallback_reason + "; " if fallback_reason else "") + (
            "NO_TARGETING: zero valid failures"
        )

    return FailureDistribution(
        type=distribution_type,
        parameters=parameters,
        n_failures=n,
        feature_names=FEATURE_NAMES,
        bounds=bounds,
        seed=int(analysis.random_state),
        source_eval_run_id=source_eval_run_id,
        failure_definition=analysis.failure_definition,
        config_hash=config.config_hash(),
        fallback_reason=fallback_reason,
    )


class TargetedSampler:
    """Sample feature vectors: ``targeted_fraction`` from the model, rest uniform."""

    def __init__(
        self,
        distribution: FailureDistribution,
        config: EvaluationConfig,
        *,
        seed: int,
        profile_config: RandomizationConfig | None = None,
    ) -> None:
        self.distribution = distribution
        self.analysis = config.failure_analysis
        self.seed = int(seed)
        self.profile_config = profile_config or load_profile_config(
            config.settings.randomization_profile
        )
        self._rng = np.random.default_rng([self.seed, 0x54415247])
        self._parameters = distribution.parameters
        if distribution.type == "gmm":
            self._weights = np.asarray(self._parameters["weights"], dtype=np.float64)
            self._weights = self._weights / self._weights.sum()
            self._means = [np.asarray(m, dtype=np.float64) for m in self._parameters["means"]]
            self._covariances = [
                np.asarray(c, dtype=np.float64) for c in self._parameters["covariances"]
            ]
            self._reg = float(self._parameters.get("reg_covar", 0.0))
        elif distribution.type in {"kde", "neighbor"}:
            self._centers = np.asarray(self._parameters["centers"], dtype=np.float64)
            self._scale = float(
                self._parameters.get(
                    "bandwidth",
                    self._parameters.get("sigma", self.analysis.neighbor_sigma_fraction),
                )
            )

    def _draw_targeted(self) -> np.ndarray:
        kind = self.distribution.type
        if kind == "gmm":
            index = int(self._rng.choice(len(self._weights), p=self._weights))
            covariance = self._covariances[index] + np.eye(FEATURE_DIM) * self._reg
            return self._rng.multivariate_normal(self._means[index], covariance)
        if kind in {"kde", "neighbor"}:
            index = int(self._rng.integers(0, len(self._centers)))
            return self._centers[index] + self._rng.normal(0.0, self._scale, size=FEATURE_DIM)
        return self._rng.uniform(0.0, 1.0, size=FEATURE_DIM)

    def sample(self, count: int) -> tuple[np.ndarray, dict[str, Any]]:
        if count < 0:
            raise ValueError("count must be >= 0")
        if self.distribution.type == "none":
            targeted = 0
        else:
            targeted = int(round(count * self.analysis.targeted_fraction))
        uniform = count - targeted
        samples: list[np.ndarray] = []
        clip_fallbacks = 0
        rejection_tries = 0
        for _ in range(targeted):
            candidate = None
            for _attempt in range(self.analysis.max_rejection_tries):
                rejection_tries += 1
                trial = self._draw_targeted()
                if np.all((trial >= 0.0) & (trial <= 1.0)):
                    candidate = trial
                    break
            if candidate is None:
                candidate = np.clip(self._draw_targeted(), 0.0, 1.0)
                clip_fallbacks += 1
            samples.append(candidate)
        for _ in range(uniform):
            samples.append(self._rng.uniform(0.0, 1.0, size=FEATURE_DIM))
        matrix = np.asarray(samples, dtype=np.float64).reshape(-1, FEATURE_DIM)
        if len(matrix) > 1:
            order = self._rng.permutation(len(matrix))
            matrix = matrix[order]
        matrix = np.clip(matrix, 0.0, 1.0)
        report = {
            "count": count,
            "targeted_requested": targeted,
            "targeted_fraction_actual": (targeted / count) if count else 0.0,
            "targeted_fraction_configured": self.analysis.targeted_fraction,
            "uniform": uniform,
            "clip_fallbacks": clip_fallbacks,
            "rejection_tries": rejection_tries,
            "distribution_type": self.distribution.type,
        }
        return matrix, report

    def samples_to_randomization(
        self, features: np.ndarray, base_seed: int
    ) -> list[tuple[Any, dict[str, object]]]:
        samples = []
        for index, vector in enumerate(np.asarray(features, dtype=np.float64)):
            samples.append(
                from_feature_vector(
                    vector, self.profile_config, base_seed=int(base_seed), episode_index=index
                )
            )
        return samples


__all__ = [
    "DISTRIBUTION_TYPES",
    "FailureDistribution",
    "FailureDistributionError",
    "TargetedSampler",
    "failure_feature_vectors",
    "fit_failure_distribution",
    "load_profile_config",
]
