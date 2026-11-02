"""Normalization statistics and deterministic episode splitting."""

from __future__ import annotations

import numpy as np
import pytest

from sim2real.dataset_builder import (
    RunningStats,
    compute_dataset_stats,
    compute_feature_stats,
    split_episodes,
)
from sim2real.dataset_builder.stats import StatsError
from sim2real.expert.images import read_png, write_png
from tests.fixtures.dataset_builder import make_episode


def test_stats_match_numpy_for_known_array():
    values = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    stats = compute_feature_stats(values)
    np.testing.assert_allclose(stats["mean"], values.mean(axis=0))
    np.testing.assert_allclose(stats["std"], values.std(axis=0))
    np.testing.assert_allclose(stats["q01"], np.quantile(values, 0.01, axis=0))
    np.testing.assert_allclose(stats["q99"], np.quantile(values, 0.99, axis=0))
    assert int(stats["count"][0]) == 3
    assert {"q01", "q10", "q50", "q90", "q99"} <= set(stats)


def test_streaming_merge_equals_one_shot():
    values = np.arange(30, dtype=np.float64).reshape(10, 3)
    one_shot = compute_feature_stats(values)
    first = RunningStats(3)
    first.update(values[:4])
    second = RunningStats(3)
    second.update(values[4:])
    first.merge(second)
    merged = first.finalize()
    for key in ("mean", "std", "min", "max", "q01", "q99"):
        np.testing.assert_allclose(merged[key], one_shot[key])


def test_constant_dimension_has_floored_std():
    stats = compute_feature_stats(np.ones((4, 2)))
    assert np.all(stats["std"] > 0)
    assert np.all(stats["std"] == 1e-8)


def test_nan_input_raises():
    with pytest.raises(StatsError, match="NaN"):
        compute_feature_stats(np.array([[0.0], [np.nan]]))


def test_stdlib_png_round_trip(tmp_path):
    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    path = tmp_path / "image.png"
    write_png(path, image)
    np.testing.assert_array_equal(read_png(path), image)


def test_dataset_stats_accepts_episodes():
    episodes = [make_episode(frames=3), make_episode(episode_id="ep1", frames=3)]
    stats = compute_dataset_stats(episodes)
    assert int(stats["observation.state"]["count"][0]) == 6
    assert int(stats["action"]["count"][0]) == 6


def test_reservoir_quantile_error_within_documented_bound():
    rng = np.random.default_rng(0)
    values = rng.uniform(size=(4000, 1))
    exact = compute_feature_stats(values)
    approximate = compute_feature_stats(
        values, quantile_method="reservoir", reservoir_size=800, seed=7
    )
    bound = 3.0 / np.sqrt(800)
    assert abs(float(approximate["q99"][0]) - float(exact["q99"][0])) <= bound


def test_split_is_deterministic_stratified_and_disjoint():
    episodes = [
        make_episode(episode_id=f"a{index}", source="a", frames=2) for index in range(6)
    ] + [make_episode(episode_id=f"b{index}", source="b", frames=2) for index in range(6)]
    first = split_episodes(episodes, val_fraction=0.25, seed=7)
    second = split_episodes(episodes, val_fraction=0.25, seed=7)
    assert first.train == second.train
    assert first.val == second.val
    assert set(first.train).isdisjoint(first.val)
    assert set(first.train) | set(first.val) == set(range(12))
    assert first.by_source["a"]["val"] >= 1
    assert first.by_source["b"]["val"] >= 1


def test_appended_episodes_do_not_duplicate_or_leak():
    base = [make_episode(episode_id=f"a{index}", source="a", frames=2) for index in range(4)]
    first = split_episodes(base, val_fraction=0.25, seed=3)
    assert set(first.train).isdisjoint(first.val)
    extended = base + [
        make_episode(episode_id=f"a{index}", source="a", frames=2) for index in range(4, 8)
    ]
    second = split_episodes(extended, val_fraction=0.25, seed=3)
    assert set(second.train).isdisjoint(second.val)
    assert set(second.train) | set(second.val) == set(range(8))
