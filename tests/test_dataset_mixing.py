"""Deterministic mixture sampling, caps, resumption, and validation."""

from __future__ import annotations

import pytest

from sim2real.dataset_builder import (
    MixingError,
    MixingSampler,
    MixtureConfig,
    MixtureSource,
    SourceIndex,
    indexes_from_config,
    load_mixture_config,
)
from sim2real.dataset_builder.validation import (
    DataValidationError,
    validate_mixture_compatibility,
)


def _config(**kwargs):
    sources = (
        MixtureSource("a", 0.7, "frames", action_representation="joint_position"),
        MixtureSource("b", 0.3, "frames", action_representation="joint_position"),
    )
    values = {"name": "test", "sources": sources, "seed": 5}
    values.update(kwargs)
    return MixtureConfig(**values)


def _indexes():
    return {
        "a": SourceIndex("a", tuple([10] * 20)),
        "b": SourceIndex("b", tuple([8] * 10)),
    }


def test_empirical_proportions_match_weights():
    sampler = MixingSampler(_config(), _indexes())
    draws = sampler.draw_n(100_000)
    counts = {key: 0 for key in ("a", "b")}
    for ref in draws:
        counts[ref.source_id] += 1
    assert abs(counts["a"] / 100_000 - 0.7) < 0.01
    assert abs(counts["b"] / 100_000 - 0.3) < 0.01


def test_same_seed_and_resume_are_identical():
    first = MixingSampler(_config(), _indexes())
    prefix = first.draw_n(20)
    state = first.state_dict()
    after = first.draw_n(20)
    second = MixingSampler(_config(), _indexes())
    assert second.draw_n(20) == prefix
    second.load_state_dict(state)
    assert second.draw_n(20) == after


def test_max_fraction_caps_a_source():
    config = _config(
        sources=(
            MixtureSource("a", 0.8, "frames", max_fraction=0.25),
            MixtureSource("b", 0.2, "frames"),
        )
    )
    sampler = MixingSampler(config, _indexes())
    draws = sampler.draw_n(1000)
    assert sum(ref.source_id == "a" for ref in draws) <= 260


def test_invalid_configs_raise():
    with pytest.raises(MixingError):
        MixtureConfig("bad", (MixtureSource("a", 0.0),), seed=1)
    with pytest.raises(MixingError):
        MixtureConfig("bad", (MixtureSource("a", 0.5),), seed=1)
    with pytest.raises(MixingError):
        MixingSampler(_config(), {"a": SourceIndex("a", (1,))})
    with pytest.raises(MixingError):
        MixingSampler(_config(), {"a": SourceIndex("a", (1,)), "b": SourceIndex("b", ())})
    with pytest.raises(MixingError):
        load_mixture_config("does-not-exist.yaml")


def test_example_mixture_config_loads_and_previews():
    config = load_mixture_config("configs/dataset/mixtures/example.yaml")
    sampler = MixingSampler(config, indexes_from_config(config))
    refs = sampler.draw_n(2000)
    assert {ref.source_id for ref in refs} == {"panda_sim", "widowx_bridge"}


def test_mixed_action_semantics_require_manifest():
    with pytest.raises(DataValidationError):
        validate_mixture_compatibility({"a": "joint_position", "b": "bridge_ee_delta"})
    validate_mixture_compatibility(
        {"a": "joint_position", "b": "bridge_ee_delta"},
        allow_cross_embodiment=True,
    )
