"""Training config validation and horizon bookkeeping."""

from __future__ import annotations

import copy

import pytest

from sim2real.training.config import ConfigError, load_training_config, parse_training_config


@pytest.fixture()
def raw_config() -> dict:
    config = load_training_config("configs/training/smolvla_lora.yaml")
    return copy.deepcopy(config.raw)


def _parse(raw: dict):
    return parse_training_config(copy.deepcopy(raw))


def test_verified_smolvla_config_loads():
    config = load_training_config("configs/training/smolvla_lora.yaml")
    assert config.model.family == "smolvla"
    assert config.model.revision == "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6"
    assert config.lora.r == 32
    assert config.lora.alpha == 64
    assert config.data.fps == 10.0
    assert config.horizon_seconds == pytest.approx(5.0)


def test_openvla_config_is_explicitly_unvalidated():
    config = load_training_config("configs/training/openvla7b_lora.yaml")
    assert config.model.family == "openvla7b"
    assert config.model.validated is False
    assert config.model.revision is None


@pytest.mark.parametrize(
    ("mutate", "key_path"),
    [
        (lambda raw: raw["lora"].__setitem__("r", 0), "lora.r"),
        (lambda raw: raw["lora"].__setitem__("alpha", 0), "lora.alpha"),
        (lambda raw: raw["lora"].__setitem__("dropout", 1.0), "lora.dropout"),
        (lambda raw: raw["lora"].__setitem__("target_modules", []), "lora.target_modules"),
        (lambda raw: raw["model"].__setitem__("family", "unknown"), "model.family"),
        (lambda raw: raw["training"].__setitem__("max_steps", 0), "training.max_steps"),
        (lambda raw: raw["training"].__setitem__("batch_size", 0), "training.batch_size"),
        (lambda raw: raw["training"].pop("seed"), "training.seed"),
        (lambda raw: raw["training"].__setitem__("n_action_steps", 99), "training.n_action_steps"),
        (lambda raw: raw["tracking"].__setitem__("backend", "tensorboard"), "tracking.backend"),
        (lambda raw: raw["data"].__setitem__("rename_map", {}), "data.rename_map"),
        (
            lambda raw: raw["data"]["rename_map"].pop("observation.images.wrist"),
            "does not map every camera",
        ),
        (lambda raw: raw.__setitem__("unexpected_section", {}), "unknown keys"),
    ],
)
def test_invalid_config_has_key_path(raw_config, mutate, key_path):
    mutate(raw_config)
    with pytest.raises(ConfigError, match=key_path):
        _parse(raw_config)


def test_horizon_seconds_comes_from_chunk_and_fps(raw_config):
    raw_config["training"]["chunk_size"] = 25
    raw_config["training"]["n_action_steps"] = 10
    raw_config["data"]["fps"] = 5.0
    config = _parse(raw_config)
    assert config.horizon_seconds == pytest.approx(5.0)
    assert config.training.n_action_steps == 10


def test_duplicate_rename_targets_rejected(raw_config):
    raw_config["data"]["rename_map"]["observation.images.wrist"] = "observation.images.camera1"
    with pytest.raises(ConfigError, match="data.rename_map"):
        _parse(raw_config)


def test_config_hash_is_deterministic_and_value_sensitive():
    first = load_training_config("configs/training/smolvla_lora.yaml")
    second = load_training_config("configs/training/smolvla_lora.yaml")
    assert first.config_hash == second.config_hash
    changed = second.to_dict()
    changed["training"]["max_steps"] = 99
    from sim2real.training.checkpoint import config_hash

    assert config_hash(changed) != first.config_hash


def test_episode_indices_subset_is_validated_and_exported(raw_config):
    raw_config["data"]["episode_indices"] = [0, 1]
    config = _parse(raw_config)
    assert config.data.episode_indices == (0, 1)
    assert config.to_dict()["data"]["episode_indices"] == [0, 1]
    assert config.to_dict()["data"]["episode_indices"] == [0, 1]

    for invalid in ([], [0, 0], [-1], [True]):
        broken = copy.deepcopy(raw_config)
        broken["data"]["episode_indices"] = invalid
        with pytest.raises(ConfigError, match="episode_indices"):
            _parse(broken)
