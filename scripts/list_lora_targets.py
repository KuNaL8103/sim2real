#!/usr/bin/env python3
"""List SmolVLA LoRA target modules and resolve the configured patterns.

``--fake-tree`` is CI-safe and needs no torch; it builds a realistic name tree
and proves the resolver.  Without it, the script loads the configured model from
.venv-ml, prints the observed module types, and resolves the real names.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sim2real.training.config import load_training_config  # noqa: E402
from sim2real.training.lora import (  # noqa: E402
    DEFAULT_SMOLVLA_TARGET_MODULES,
    LoraTargetError,
    adapted_scope,
    group_by_type,
    resolve_target_modules,
)


def fake_tree() -> list[tuple[str, str]]:
    modules: list[tuple[str, str]] = [
        ("model", "SmolVLAPolicy"),
        ("model.state_proj", "Linear"),
        ("model.action_in_proj", "Linear"),
        ("model.action_out_proj", "Linear"),
        ("model.action_time_mlp_in", "Linear"),
        ("model.action_time_mlp_out", "Linear"),
        ("model.vlm_with_expert", "SmolVLMWithExpert"),
        ("model.vlm_with_expert.vlm", "VLM"),
        ("model.vlm_with_expert.vlm.model", "SmolVLMForConditionalGeneration"),
        ("model.vlm_with_expert.vlm.model.vision_model", "SiglipVisionModel"),
        (
            "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn.q_proj",
            "Linear",
        ),
        (
            "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn.k_proj",
            "Linear",
        ),
        ("model.vlm_with_expert.vlm.model.text_model", "TextModel"),
        ("model.vlm_with_expert.vlm.model.text_model.layers.0.self_attn.q_proj", "Linear"),
        ("model.vlm_with_expert.vlm.model.text_model.layers.0.self_attn.k_proj", "Linear"),
        ("model.vlm_with_expert.lm_expert", "TextModel"),
        (
            "model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj",
            "Linear",
        ),
        (
            "model.vlm_with_expert.lm_expert.layers.0.self_attn.k_proj",
            "Linear",
        ),
        (
            "model.vlm_with_expert.lm_expert.layers.0.self_attn.v_proj",
            "Linear",
        ),
        ("model.vlm_with_expert.lm_expert.layers.0.self_attn.o_proj", "Linear"),
    ]
    return modules


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/smolvla_lora.yaml")
    parser.add_argument("--fake-tree", action="store_true")
    args = parser.parse_args(argv)

    config = load_training_config(args.config)
    patterns = list(config.lora.target_modules) or list(DEFAULT_SMOLVLA_TARGET_MODULES)

    if args.fake_tree:
        modules = fake_tree()
        names = [name for name, _type_name in modules]
        resolved = resolve_target_modules(patterns[:1], names)
        groups = group_by_type((name, type_name) for name, type_name in modules if name in resolved)
        print("fake module tree (resolved targets)")
        for type_name, members in groups.items():
            print(f"  {type_name}: {len(members)}")
            for member in members:
                print(f"    {member}")
        print(json.dumps({"resolved": resolved, "scope_counts": _scope_counts(resolved)}, indent=2))
        # A stale pattern must fail loudly.
        try:
            resolve_target_modules(["this_module_does_not_exist"], names)
        except LoraTargetError as exc:
            print(f"resolver-negative-check: {exc}")
        return 0

    try:
        from sim2real.training.adapters import SmolVLAAdapter

        adapter = SmolVLAAdapter(config)
        report = adapter.resolve_target_modules()
        report_data = {
            "resolved": report.resolved,
            "scope_counts": report.scope_counts,
            "parameter_report": adapter.parameter_report(),
        }
        print(json.dumps(report_data, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001 - a CLI must print the actionable cause
        print(f"error: could not inspect the real model: {exc}", file=sys.stderr)
        return 2


def _scope_counts(names: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in names:
        scope = adapted_scope(name)
        counts[scope] = counts.get(scope, 0) + 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
