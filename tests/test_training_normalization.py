"""Normalization round trips, constant dimensions, and embodiment isolation."""

from __future__ import annotations

import numpy as np
import pytest

from sim2real.training.normalization import (
    FeatureStats,
    NormalizationBundle,
    NormalizationError,
    Normalizer,
    bundle_from_stats,
)

MEAN_STD_STATS = {
    "mean": np.array([1.0, -2.0, 0.5]),
    "std": np.array([0.5, 2.0, 1.5]),
}
QUANTILE_STATS = {
    "q01": np.array([-1.0, 0.0, 10.0]),
    "q99": np.array([1.0, 4.0, 14.0]),
}


def test_mean_std_round_trip():
    normalizer = Normalizer("mean_std", FeatureStats.from_mapping(MEAN_STD_STATS))
    values = np.array([[0.0, 1.0, 2.0], [3.0, -4.0, 0.5]])
    normalized = normalizer.apply(values)
    restored = normalizer.invert(normalized)
    np.testing.assert_allclose(restored, values, atol=1e-6)


def test_quantile_round_trip_and_range():
    normalizer = Normalizer("quantile", FeatureStats.from_mapping(QUANTILE_STATS))
    values = np.array([[-1.0, 0.0, 10.0], [1.0, 4.0, 14.0], [0.0, 2.0, 12.0]])
    normalized = normalizer.apply(values)
    np.testing.assert_allclose(normalized[0], [-1.0, -1.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(normalized[1], [1.0, 1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(normalizer.invert(normalized), values, atol=1e-6)


def test_constant_dimension_is_safe_at_its_value():
    stats = {"mean": np.array([0.0, 5.0]), "std": np.array([1.0, 0.0])}
    normalizer = Normalizer("mean_std", FeatureStats.from_mapping(stats))
    normalized = normalizer.apply(np.array([0.0, 5.0]))
    assert np.all(np.isfinite(normalized))
    # The constant dimension is centered at zero and returns to its only value.
    assert normalized[1] == pytest.approx(0.0)
    assert normalizer.invert(normalized)[1] == pytest.approx(5.0)


def test_zero_quantile_range_uses_epsilon_and_round_trips_constant_value():
    stats = {"q01": np.array([2.0, 1.0]), "q99": np.array([2.0, 3.0])}
    normalizer = Normalizer("quantile", FeatureStats.from_mapping(stats))
    normalized = normalizer.apply(np.array([2.0, 2.0]))
    assert normalized[0] == pytest.approx(-1.0)
    assert normalizer.invert(normalized)[0] == pytest.approx(2.0)


def test_missing_mode_statistics_raise():
    with pytest.raises(NormalizationError, match="q01 and q99"):
        Normalizer("quantile", FeatureStats(mean=np.zeros(2), std=np.ones(2))).apply(np.zeros(2))


def test_embodiment_bundle_rejects_other_embodiment():
    stats = {
        "action": {"mean": np.zeros(2), "std": np.ones(2)},
        "observation.state": {"mean": np.zeros(2), "std": np.ones(2)},
    }
    bundle = bundle_from_stats(
        stats,
        embodiment_id="panda_sim",
        modes={"action": "mean_std", "observation.state": "mean_std"},
    )
    np.testing.assert_allclose(
        bundle.apply("panda_sim", "action", np.array([1.0, 2.0])), [1.0, 2.0]
    )
    with pytest.raises(NormalizationError, match="known embodiments"):
        bundle.require("widowx_bridge", "action")


def test_identity_and_clip_policy_are_explicit():
    identity = Normalizer("identity", FeatureStats())
    values = np.array([1.0, -2.0])
    np.testing.assert_allclose(identity.apply(values), values)
    clipped = Normalizer(
        "mean_std", FeatureStats.from_mapping({"mean": np.zeros(1), "std": np.ones(1)}), clip=1.0
    )
    assert clipped.apply(np.array([5.0]))[0] == pytest.approx(1.0)


def test_duplicate_bundle_entry_rejected():
    bundle = NormalizationBundle()
    normalizer = Normalizer("identity", FeatureStats())
    bundle.add("panda_sim", "action", normalizer)
    with pytest.raises(NormalizationError, match="duplicate"):
        bundle.add("panda_sim", "action", normalizer)
