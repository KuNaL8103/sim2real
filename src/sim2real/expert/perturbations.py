"""Deterministic, *intentional* demonstration perturbations.

These are deliberate variations used to generate degraded/failure demonstrations
for evaluation and later filtering. They are not domain randomization; they act
on the expert's own commands and are recorded per episode.

Every perturbation carries an ``expected_quality`` label (``clean``,
``degraded``, or ``failure``) that is used **only** for diagnostics and tests.
The runner never uses it to decide the outcome; the measured outcome is recorded.

Selection is a pure function of ``(seed, episode_index)``. The same inputs always
choose the same perturbation (or none), independent of call order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sim2real.expert.config import ExpertConfig, ExpertConfigError

VALID_QUALITIES = frozenset({"clean", "degraded", "failure"})


@dataclass(frozen=True)
class Perturbation:
    name: str
    expected_quality: str
    params: dict[str, float | int | bool]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "expected_quality": self.expected_quality,
            "params": dict(self.params),
        }


@dataclass(frozen=True)
class PerturbationSet:
    perturbed_fraction: float
    types: dict[str, Perturbation]

    @property
    def enabled_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.types))

    def get(self, name: str) -> Perturbation:
        try:
            return self.types[name]
        except KeyError as exc:
            raise ValueError(f"unknown perturbation {name!r}") from exc


def load_perturbations(config: ExpertConfig) -> PerturbationSet:
    path = Path(config.perturbations_file)
    if not path.is_file():
        raise ExpertConfigError(f"missing perturbations file: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ExpertConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ExpertConfigError(f"{path}: schema_version must be 1")
    fraction = float(data.get("perturbed_fraction", 0.0))
    if not 0.0 <= fraction <= 1.0:
        raise ExpertConfigError(f"{path}: perturbed_fraction must be in [0, 1]")
    raw_types = data.get("types")
    if not isinstance(raw_types, dict):
        raise ExpertConfigError(f"{path}: types must be a mapping")

    types: dict[str, Perturbation] = {}
    for name, entry in raw_types.items():
        if not isinstance(entry, dict):
            raise ExpertConfigError(f"{path}: type {name!r} must be a mapping")
        if not bool(entry.get("enabled", True)):
            continue
        quality = str(entry.get("expected_quality", "degraded"))
        if quality not in VALID_QUALITIES:
            raise ExpertConfigError(
                f"{path}: type {name!r} expected_quality must be one of {sorted(VALID_QUALITIES)}"
            )
        params = entry.get("params", {})
        if not isinstance(params, dict):
            raise ExpertConfigError(f"{path}: type {name!r} params must be a mapping")
        types[str(name)] = Perturbation(
            name=str(name), expected_quality=quality, params=dict(params)
        )
    if not types:
        raise ExpertConfigError(f"{path}: no enabled perturbation types")
    return PerturbationSet(perturbed_fraction=fraction, types=types)


def select_perturbation(
    perturbations: PerturbationSet,
    seed: int,
    episode_index: int,
    *,
    perturbed_fraction: float | None = None,
) -> Perturbation | None:
    """Deterministically choose a perturbation for ``(seed, episode_index)``."""
    fraction = (
        perturbations.perturbed_fraction if perturbed_fraction is None else perturbed_fraction
    )
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("perturbed_fraction must be in [0, 1]")
    if fraction == 0.0:
        return None
    sequence = np.random.SeedSequence([int(seed), int(episode_index), 0x50455254])
    rng = np.random.Generator(np.random.PCG64(sequence))
    if float(rng.random()) >= fraction:
        return None
    names = perturbations.enabled_names
    index = int(rng.integers(0, len(names)))
    return perturbations.types[names[index]]


def perturbation_phase(seed: int, episode_index: int) -> np.ndarray:
    """Deterministic per-joint phase offsets for jitter (radians)."""
    sequence = np.random.SeedSequence([int(seed), int(episode_index), 0x4A495454])
    rng = np.random.Generator(np.random.PCG64(sequence))
    return rng.uniform(0.0, 2.0 * math.pi, size=7)


def apply_joint_jitter(
    q_target: np.ndarray,
    step_index: int,
    dt_s: float,
    perturbation: Perturbation,
    phase: np.ndarray,
) -> np.ndarray:
    """Add a deterministic high-frequency sinusoid to a joint command."""
    amplitude = float(perturbation.params.get("amplitude_rad", 0.01))
    frequency = float(perturbation.params.get("frequency_hz", 4.0))
    time_s = step_index * dt_s
    offset = amplitude * np.sin(2.0 * math.pi * frequency * time_s + phase)
    return np.asarray(q_target, dtype=np.float64) + offset


def hesitation_pause_steps(perturbation: Perturbation, seed: int, episode_index: int) -> int:
    """Pause length for the hesitation perturbation (0 disables it)."""
    low = int(perturbation.params.get("pause_min_steps", 2))
    high = int(perturbation.params.get("pause_max_steps", 5))
    if high < low:
        raise ValueError("hesitation pause_max_steps must be >= pause_min_steps")
    sequence = np.random.SeedSequence([int(seed), int(episode_index), 0x48455349])
    rng = np.random.Generator(np.random.PCG64(sequence))
    return int(rng.integers(low, high + 1))


def overspeed_scale(perturbation: Perturbation) -> float:
    scale = float(perturbation.params.get("time_scale", 0.6))
    if not 0.0 < scale <= 1.0:
        raise ValueError("overspeed time_scale must be in (0, 1]")
    return scale


def detour_offset(perturbation: Perturbation) -> np.ndarray:
    offset = perturbation.params.get("offset_m", [0.0, 0.08, 0.0])
    vector = np.asarray(offset, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError("detour offset_m must have 3 entries")
    return vector


def bad_grasp_offset(perturbation: Perturbation) -> np.ndarray:
    offset = perturbation.params.get("offset_m", [0.04, 0.0, 0.0])
    vector = np.asarray(offset, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError("bad_grasp offset_m must have 3 entries")
    return vector


def low_sweep_clearance(perturbation: Perturbation) -> float:
    return float(perturbation.params.get("clearance_m", 0.03))


def describe(perturbations: PerturbationSet) -> dict[str, Any]:
    return {
        "perturbed_fraction": perturbations.perturbed_fraction,
        "types": {name: spec.to_dict() for name, spec in sorted(perturbations.types.items())},
    }
