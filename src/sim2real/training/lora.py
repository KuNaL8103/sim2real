"""Resolve LoRA target modules against the model's *actual* named modules.

Nothing here hardcodes ``q_proj``/``v_proj`` as if every transformer had them.
The caller supplies the observed module names (from ``named_modules()`` on the
real model, or a fake tree in tests), and every configured pattern must match at
least one observed module or resolution fails with the offending pattern.

PEFT's own convention is a regular expression matched with ``re.fullmatch``;
this resolver also accepts an exact name or a dotted suffix so configuration can
stay readable.  A pattern that matches nothing is always an error.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence

# The default verified for SmolVLA in LeRobot 0.6.1
# (``SmolVLAPolicy._get_default_peft_targets``): q/v projections of the action
# expert plus the state/action projections.  The vision encoder and the VLM
# backbone stay frozen per ``freeze_vision_encoder``/``train_expert_only``.
DEFAULT_SMOLVLA_TARGET_MODULES = (
    r"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj"
    r"|model\.(state_proj|action_in_proj|action_out_proj"
    r"|action_time_mlp_in|action_time_mlp_out))",
)


class LoraTargetError(ValueError):
    """Raised when a LoRA target pattern matches no observed module."""


def _matches(name: str, pattern: str) -> bool:
    if name == pattern or name.endswith(f".{pattern}"):
        return True
    try:
        if re.fullmatch(pattern, name):
            return True
        return bool(re.search(pattern, name))
    except re.error:
        # A plain suffix/name that is not valid regex syntax has already been
        # checked above; there is nothing else it can match.
        return False


def resolve_target_modules(
    patterns: Sequence[str],
    module_names: Iterable[str],
) -> list[str]:
    """Return the sorted unique module names matched by every pattern.

    Raises:
        LoraTargetError: a pattern is empty or matches no module.  The message
            names the pattern so a stale configuration is obvious.
    """
    names = [str(name) for name in module_names]
    if not patterns:
        raise LoraTargetError("target_modules: expected at least one pattern")
    resolved: set[str] = set()
    missing: list[str] = []
    for pattern in patterns:
        if not pattern:
            raise LoraTargetError("target_modules: empty pattern")
        matched = {name for name in names if _matches(name, pattern)}
        if not matched:
            missing.append(pattern)
        resolved.update(matched)
    if missing:
        raise LoraTargetError(
            "target_modules: patterns matched no module in this model: "
            f"{missing}; observed modules: {len(names)}"
        )
    return sorted(resolved)


def module_name_tree(paths: Sequence[str]) -> list[str]:
    """Expand dotted leaf paths into every intermediate module name.

    ``a.b.c`` yields ``a``, ``a.b`` and ``a.b.c`` (duplicates removed).  This is
    used to build a realistic fake ``named_modules()`` tree without torch.
    """
    result: set[str] = set()
    for path in paths:
        parts = [part for part in str(path).split(".") if part]
        for index in range(1, len(parts) + 1):
            result.add(".".join(parts[:index]))
    return sorted(result)


def group_by_type(named_modules: Iterable[tuple[str, str]]) -> dict[str, list[str]]:
    """Group ``(name, type_name)`` pairs by type, sorted for stable output."""
    grouped: dict[str, list[str]] = {}
    for name, type_name in named_modules:
        grouped.setdefault(str(type_name), []).append(str(name))
    return {key: sorted(value) for key, value in sorted(grouped.items())}


def trainable_parameter_report(
    parameter_counts: Mapping[str, tuple[int, bool]],
) -> dict[str, float | int]:
    """Summarise ``{name: (numel, trainable)}`` without importing torch."""
    total = 0
    trainable = 0
    for numel, is_trainable in parameter_counts.values():
        count = int(numel)
        total += count
        if is_trainable:
            trainable += count
    fraction = (trainable / total) if total else 0.0
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "trainable_fraction": fraction,
    }


def adapted_scope(module_name: str) -> str:
    """Classify a resolved module for the human-readable LoRA report."""
    if "vision_model" in module_name or "vision" in module_name:
        return "vision_encoder"
    if "lm_expert" in module_name:
        return "action_expert"
    if "vlm" in module_name or "text_model" in module_name or "language_model" in module_name:
        return "vlm_backbone"
    return "projections"


__all__ = [
    "DEFAULT_SMOLVLA_TARGET_MODULES",
    "LoraTargetError",
    "adapted_scope",
    "group_by_type",
    "module_name_tree",
    "resolve_target_modules",
    "trainable_parameter_report",
]
