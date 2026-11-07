"""LoRA target resolution against an observed module tree (no torch)."""

from __future__ import annotations

import pytest

from sim2real.training.lora import (
    DEFAULT_SMOLVLA_TARGET_MODULES,
    LoraTargetError,
    adapted_scope,
    group_by_type,
    module_name_tree,
    resolve_target_modules,
    trainable_parameter_report,
)


def fake_smolvla_names() -> list[str]:
    names = [
        "model.state_proj",
        "model.action_in_proj",
        "model.action_out_proj",
        "model.action_time_mlp_in",
        "model.action_time_mlp_out",
        "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn.q_proj",
        "model.vlm_with_expert.vlm.model.text_model.layers.0.self_attn.q_proj",
        "model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj",
        "model.vlm_with_expert.lm_expert.layers.0.self_attn.k_proj",
        "model.vlm_with_expert.lm_expert.layers.0.self_attn.v_proj",
        "model.vlm_with_expert.lm_expert.layers.0.self_attn.o_proj",
    ]
    return module_name_tree(names)


def test_default_smolvla_pattern_resolves_only_expert_and_projections():
    resolved = resolve_target_modules(DEFAULT_SMOLVLA_TARGET_MODULES, fake_smolvla_names())
    assert "model.state_proj" in resolved
    assert "model.action_in_proj" in resolved
    assert "model.action_out_proj" in resolved
    assert "model.action_time_mlp_in" in resolved
    assert "model.action_time_mlp_out" in resolved
    assert any(name.endswith("lm_expert.layers.0.self_attn.q_proj") for name in resolved)
    assert any(name.endswith("lm_expert.layers.0.self_attn.v_proj") for name in resolved)
    # The vision encoder and VLM text stack must not be targeted.
    assert not any("vision_model" in name for name in resolved)
    assert not any("vlm.model.text_model" in name for name in resolved)
    assert not any(name.endswith("k_proj") for name in resolved)


def test_unmatched_pattern_fails_clearly():
    with pytest.raises(LoraTargetError, match="matched no module"):
        resolve_target_modules(["modules_that_do_not_exist"], fake_smolvla_names())


def test_suffix_and_exact_patterns_are_supported():
    assert resolve_target_modules(["state_proj"], fake_smolvla_names()) == ["model.state_proj"]
    assert resolve_target_modules(
        ["model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj"], fake_smolvla_names()
    ) == ["model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj"]


def test_grouping_and_scope_classification():
    named = [
        ("model.state_proj", "Linear"),
        ("model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj", "Linear"),
        ("model.vlm_with_expert.vlm.model.vision_model.encoder", "SiglipEncoder"),
    ]
    groups = group_by_type(named)
    assert groups["Linear"] == sorted([name for name, _ in named if _ == "Linear"])
    assert adapted_scope("model.vlm_with_expert.lm_expert.layers.0.q_proj") == "action_expert"
    assert adapted_scope("model.vlm_with_expert.vlm.model.vision_model.encoder") == "vision_encoder"
    assert adapted_scope("model.state_proj") == "projections"


def test_trainable_parameter_report():
    report = trainable_parameter_report({"a": (100, False), "b": (50, True), "c": (50, True)})
    assert report["total_parameters"] == 200
    assert report["trainable_parameters"] == 100
    assert report["trainable_fraction"] == pytest.approx(0.5)
