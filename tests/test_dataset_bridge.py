"""Bridge fixture adapter boundaries, language policy, and finite conversion."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sim2real.dataset_builder.adapters.rlds_bridge import (
    BridgeAdapterConfig,
    FixtureEpisodeSource,
    adapt_rlds_source,
    resize_nearest,
    select_episode_indices,
)
from tests.fixtures.dataset_builder import bridge_spec

FIXTURE = Path("tests/fixtures/bridge/tiny_bridge.json")


def test_fixture_fill_default_counts():
    source = FixtureEpisodeSource(FIXTURE)
    episodes, report = adapt_rlds_source(
        source,
        bridge_spec(),
        BridgeAdapterConfig(max_episodes=5, seed=0, resolution=(4, 4)),
    )
    assert report.converted == 3
    assert report.filled_default_instruction == 1
    assert report.skipped_nan == 1
    assert report.skipped_missing_image == 1
    assert {episode.task for episode in episodes} == {
        "pick up the block",
        "perform the task",
        "put it down",
    }
    assert all(episode.fps == 5.0 for episode in episodes)
    assert all(episode.license_tag == "CC-BY-4.0" for episode in episodes)


def test_fixture_skip_policy_counts_missing_language():
    source = FixtureEpisodeSource(FIXTURE)
    episodes, report = adapt_rlds_source(
        source,
        bridge_spec(),
        BridgeAdapterConfig(
            max_episodes=5,
            seed=0,
            resolution=(4, 4),
            on_missing_instruction="skip",
        ),
    )
    assert len(episodes) == 2
    assert report.skipped_missing_language == 1
    assert all(episode.task != "perform the task" for episode in episodes)


def test_max_episodes_and_seeded_subset_are_deterministic():
    source = FixtureEpisodeSource(FIXTURE)
    selected = select_episode_indices(source.num_episodes(), 2, seed=4)
    assert selected == select_episode_indices(source.num_episodes(), 2, seed=4)
    episodes, report = adapt_rlds_source(
        source,
        bridge_spec(),
        BridgeAdapterConfig(max_episodes=2, seed=4, resolution=(4, 4)),
    )
    assert report.selected == 2
    assert len(episodes) <= 2
    assert len(episodes) == report.converted


def test_resize_nearest_and_zero_fill():
    source = FixtureEpisodeSource(FIXTURE)
    image = np.zeros((3, 5, 3), dtype=np.uint8)
    image[:, :, 0] = 200
    resized = resize_nearest(image, (6, 10))
    assert resized.shape == (6, 10, 3)
    assert resized.dtype == np.uint8

    episodes, report = adapt_rlds_source(
        source,
        bridge_spec(),
        BridgeAdapterConfig(max_episodes=5, seed=0, resolution=(4, 4), missing_image="zeros"),
    )
    assert report.zero_filled_images == 4
    assert report.skipped_missing_image == 0
    assert len(episodes) == 4
