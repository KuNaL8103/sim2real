"""LeRobot writer round trip; skipped with a reason outside .venv-ml."""

from __future__ import annotations

import json

import pytest

from sim2real.dataset_builder import compute_dataset_stats, load_embodiments, split_episodes
from tests.fixtures.dataset_builder import make_episode

pytest.importorskip("lerobot", reason="optional .venv-ml LeRobot environment is absent")


@pytest.mark.ml
def test_writer_round_trip(tmp_path):
    from sim2real.dataset_builder.lerobot_writer import (
        build_lerobot_dataset,
        compare_episode,
        load_lerobot_dataset,
    )

    spec = load_embodiments("configs/dataset/embodiments.yaml")["panda_sim"]
    episodes = [
        make_episode(episode_id="a", frames=3, renderer="llvmpipe"),
        make_episode(episode_id="b", frames=3, task="place it", renderer="llvmpipe"),
    ]
    split = split_episodes(episodes, val_fraction=0.5, seed=9)
    result = build_lerobot_dataset(
        episodes,
        spec,
        tmp_path,
        "tiny_round_trip",
        representation="joint_position",
        renderer="llvmpipe",
        decimation=2,
        split_manifest=split.manifest(episodes),
    )
    dataset = load_lerobot_dataset(result.path, "tiny_round_trip")
    assert result.num_episodes == 2
    assert result.num_frames == 6
    assert len(dataset) == 6
    assert dataset.meta.total_episodes == 2
    assert dataset.meta.fps == 10
    assert dataset.meta.total_tasks == 2
    state_stats = dataset.meta.stats["observation.state"]
    assert {"mean", "std", "min", "max", "q01", "q99"} <= set(state_stats)
    ours = compute_dataset_stats(episodes)
    for feature in ("observation.state", "action"):
        for key in ("mean", "std"):
            difference = abs(ours[feature][key] - dataset.meta.stats[feature][key])
            assert float(difference.max()) < 1e-5
        # LeRobot aggregates per-episode histogram quantiles; exact quantiles
        # can differ on a two-episode fixture. Record the difference for the
        # docs rather than asserting exact identity.
        assert float(abs(ours[feature]["q01"] - dataset.meta.stats[feature]["q01"]).max()) < 0.1
        assert float(abs(ours[feature]["q99"] - dataset.meta.stats[feature]["q99"]).max()) < 0.1
    comparison = compare_episode(dataset, episodes[0])
    assert comparison["state_max_abs_diff"] < 1e-6
    assert comparison["action_max_abs_diff"] < 1e-6
    assert comparison["task"] == "pick up the red cube"
    assert comparison["observation.images.overhead_shape"] == (3, 4, 4)
    meta = json.loads((result.path / "meta" / "sim2real_build.json").read_text(encoding="utf-8"))
    assert meta["renderer"] == "llvmpipe"
    assert meta["decimation"] == 2
    assert meta["fps"] == 10
    assert meta["image_mode"] == "image"
    assert meta["resolution"] == [4, 4]
    assert meta["cameras"] == ["overhead", "wrist"]
    assert meta["replay_code_sha256"]
    split_payload = json.loads(
        (result.path / "meta" / "sim2real_split.json").read_text(encoding="utf-8")
    )
    train_indices = {row["index"] for row in split_payload["train"]}
    val_indices = {row["index"] for row in split_payload["val"]}
    assert train_indices.isdisjoint(val_indices)
