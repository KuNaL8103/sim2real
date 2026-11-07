"""Fake-adapter training loop: determinism, learning, NaN abort, exact resume."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from sim2real.training.adapters import FakePolicyAdapter
from sim2real.training.config import load_training_config, parse_training_config
from sim2real.training.trainer import Trainer, TrainerError, select_episode_ids


def _config(**training):
    base = load_training_config("configs/training/smolvla_lora.yaml")
    raw = copy.deepcopy(base.raw)
    raw["training"].update(
        {"max_steps": 4, "save_every": 0, "grad_accum": 1, "lr": 0.05, **training}
    )
    return parse_training_config(raw)


def _batches(count: int = 4, dim: int = 8, seed: int = 0, learnable: bool = True):
    rng = np.random.default_rng(seed)
    teacher = rng.normal(0.0, 0.3, size=(dim, dim))
    batches = []
    for index in range(count):
        inputs = rng.normal(0.0, 1.0, size=(4, dim))
        targets = inputs @ teacher if learnable else rng.normal(0.0, 1.0, size=(4, dim))
        batches.append({"inputs": inputs, "targets": targets, "episode_id": f"train_{index}"})
    return batches, teacher


def _metadata(teacher: np.ndarray) -> dict:
    return {
        "dataset_root": "synthetic",
        "dataset_version": "v1",
        "dataset_hash": "hash",
        "stats_file_hash": None,
        "state_dim": teacher.shape[0],
        "action_dim": teacher.shape[1],
    }


def _trainer(tmp_path, config, batches, *, seed=0, output="run"):
    adapter = FakePolicyAdapter(8, 8, seed=seed)
    split = {
        "train": [{"episode_id": f"train_{i}"} for i in range(len(batches))],
        "val": [{"episode_id": "val"}],
    }
    return Trainer(
        config,
        adapter,
        batches,
        output_dir=tmp_path / output,
        dataset_metadata=_metadata(np.zeros((8, 8))),
        split_manifest=split,
        train_episode_ids=[f"train_{i}" for i in range(len(batches))],
        val_episode_ids=["val"],
    )


def test_same_seed_gives_identical_loss_curve(tmp_path):
    config = _config()
    batches, _teacher = _batches()
    first = _trainer(tmp_path, config, batches, output="first").run(max_steps=4)
    second = _trainer(tmp_path, config, batches, output="second").run(max_steps=4)
    assert first.losses == pytest.approx(second.losses)
    assert first.losses == pytest.approx(second.losses)


def test_loss_decreases_on_a_learnable_task(tmp_path):
    config = _config()
    batches, _teacher = _batches(count=8, seed=5)
    result = _trainer(tmp_path, config, batches, seed=1).run(max_steps=80)
    assert result.final_step == 80
    assert result.losses[-1] < result.losses[0]


def test_nan_loss_aborts_with_clear_error(tmp_path):
    config = _config()
    batches, _teacher = _batches()
    trainer = _trainer(tmp_path, config, batches)
    trainer.adapter.forward_loss = lambda batch: float("nan")
    with pytest.raises(TrainerError, match="loss is nan"):
        trainer.run(max_steps=2)


def test_resume_reproduces_the_continuation_exactly(tmp_path):
    config = _config(save_every=2)
    batches, _teacher = _batches(count=8, seed=9)
    full = _trainer(tmp_path, config, batches, output="full").run(max_steps=4)
    assert full.losses

    # Train two steps, then resume from that checkpoint to step four.
    partial = _trainer(tmp_path, config, batches, output="partial").run(max_steps=2)
    resumed = _trainer(tmp_path, config, batches, output="resumed")
    continuation = resumed.run(max_steps=4, resume_from=partial.checkpoint_dir)
    assert partial.final_step == 2
    assert continuation.final_step == 4
    assert continuation.losses == pytest.approx(full.losses[2:])
    full_adapter = FakePolicyAdapter(8, 8, seed=0)
    full_adapter.load_adapter(full.checkpoint_dir + "/adapter")
    assert resumed.adapter.state_dict_hash() == full_adapter.state_dict_hash()


def test_validation_episode_in_a_batch_is_a_hard_error(tmp_path):
    config = _config()
    batches = [{"inputs": np.zeros((1, 8)), "targets": np.zeros((1, 8)), "episode_id": "val"}]
    with pytest.raises(TrainerError, match="validation episode"):
        _trainer(tmp_path, config, batches).run(max_steps=1)


def test_train_and_val_overlap_is_a_hard_error(tmp_path):
    config = _config()
    batches, _teacher = _batches()
    adapter = FakePolicyAdapter(8, 8, seed=0)
    with pytest.raises(TrainerError, match="overlap"):
        Trainer(
            config,
            adapter,
            batches,
            output_dir=tmp_path / "overlap",
            dataset_metadata=_metadata(np.zeros((8, 8))),
            split_manifest={},
            train_episode_ids=["a"],
            val_episode_ids=["a"],
        )


def test_exclude_perturbed_selects_only_requested_episodes():
    records = [
        {"episode_id": "a", "is_perturbed": False},
        {"episode_id": "b", "is_perturbed": True},
        {"episode_id": "c", "is_perturbed": False},
        {"episode_id": "val", "is_perturbed": False},
    ]
    train_ids = ["a", "b", "c"]
    assert select_episode_ids(records, train_ids=train_ids, exclude_perturbed=False) == [
        "a",
        "b",
        "c",
    ]
    assert select_episode_ids(records, train_ids=train_ids, exclude_perturbed=True) == ["a", "c"]


def test_graceful_stop_saves_a_checkpoint(tmp_path):
    config = _config(save_every=0)
    batches, _teacher = _batches(count=8)
    trainer = _trainer(tmp_path, config, batches)
    trainer.request_stop()
    result = trainer.run(max_steps=4)
    assert result.stopped is True
    assert result.final_step == 0
    assert result.checkpoint_dir is None
