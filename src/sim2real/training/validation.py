"""Offline action-validation metrics (L1) with interpretable baselines.

Predictions and targets are compared on held-out episodes only, with padded
chunk positions excluded.  Numbers are reported in native physical units (after
de-normalization) and optionally normalized by the dataset's action std.

Two trivial baselines make a policy number interpretable:

* ``hold_last_state`` repeats the last observed state as the action;
* ``mean_action`` repeats the training-set mean action.

With a handful of episodes this module reports counts and never claims
generalisation.  A caller cannot accidentally compare chunks with different
horizons: mismatched chunk lengths raise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

MIN_STD = 1e-8


class ValidationError(ValueError):
    """Raised for mismatched shapes or an invalid mask."""


def _as_chunks(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValidationError(f"{name}: expected (N, K, D) or (K, D)")
    if array.shape[1] < 1 or array.shape[2] < 1:
        raise ValidationError(f"{name}: K and D must be positive")
    if not np.all(np.isfinite(array)):
        raise ValidationError(f"{name}: contains NaN or infinity")
    return array


def _as_mask(mask: Any, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    array = np.asarray(mask, dtype=bool)
    if array.ndim == 1:
        array = array[None, ...]
    if array.shape != shape:
        raise ValidationError(f"mask: expected shape {shape}, got {array.shape}")
    return array


def _safe_divide(total: np.ndarray, counts: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        result = total / counts
    return np.where(counts > 0, result, np.nan)


@dataclass(frozen=True)
class L1Report:
    n_samples: int
    n_valid_steps: int
    per_dimension: np.ndarray | None
    per_horizon: np.ndarray | None
    normalized_per_dimension: np.ndarray | None
    baselines: dict[str, dict[str, np.ndarray | None]]
    horizon_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        def clean(value: Any) -> Any:
            if value is None:
                return None
            array = np.asarray(value)
            return [None if math.isnan(float(item)) else float(item) for item in array.reshape(-1)]

        return {
            "n_samples": self.n_samples,
            "n_valid_steps": self.n_valid_steps,
            "horizon_seconds": self.horizon_seconds,
            "per_dimension_l1": clean(self.per_dimension),
            "per_horizon_l1": clean(self.per_horizon),
            "normalized_per_dimension_l1": clean(self.normalized_per_dimension),
            "baselines": {
                name: {
                    "per_dimension_l1": clean(values["per_dimension"]),
                    "per_horizon_l1": clean(values["per_horizon"]),
                }
                for name, values in self.baselines.items()
            },
        }


def _per_dimension(predictions: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> np.ndarray:
    errors = np.abs(predictions - targets)
    weights = mask[..., None].astype(np.float64)
    return _safe_divide((errors * weights).sum(axis=(0, 1)), mask.sum(axis=(0, 1)))


def _per_horizon(predictions: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> np.ndarray:
    errors = np.abs(predictions - targets)
    weights = mask[..., None].astype(np.float64)
    # A valid step contributes every action dimension, so the denominator is
    # (valid steps at this horizon) x (native action dimensions).
    counts = mask.sum(axis=0) * predictions.shape[2]
    return _safe_divide((errors * weights).sum(axis=(0, 2)), counts)


def _baseline_report(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
) -> dict[str, np.ndarray | None]:
    return {
        "per_dimension": _per_dimension(predictions, targets, mask),
        "per_horizon": _per_horizon(predictions, targets, mask),
    }


def valid_mask_from_padding(action_is_pad: Any) -> np.ndarray:
    """Convert LeRobot's ``action_is_pad`` to an ``evaluate_l1`` valid mask.

    LeRobot sets ``action_is_pad=True`` for **padded** positions, while
    ``evaluate_l1`` treats ``True`` as **valid**. Keeping this inversion in one
    named helper is the regression guard for the Task 8 validation bug where the
    mask was passed through unchanged and every real step was excluded.
    """
    return ~np.asarray(action_is_pad, dtype=bool)


def hold_last_state_baseline(
    state: Any,
    *,
    chunk_size: int,
    action_dim: int,
    batch: int,
) -> np.ndarray:
    """Repeat the last state's first ``action_dim`` entries for every chunk step."""
    array = np.asarray(state, dtype=np.float64)
    if array.ndim == 1:
        rows = np.tile(array[:action_dim], (batch, 1))
    elif array.ndim == 2:
        if array.shape[0] != batch:
            raise ValidationError("state: expected one row per batch entry")
        rows = array[:, :action_dim]
    else:
        raise ValidationError("state: expected (D,) or (N, D)")
    if rows.shape[-1] < action_dim:
        raise ValidationError("state: expected at least action_dim values")
    if chunk_size < 1:
        raise ValidationError("chunk_size must be >= 1")
    return np.repeat(rows[:, None, :], chunk_size, axis=1)


def mean_action_baseline(action_mean: Any, *, chunk_size: int, batch: int) -> np.ndarray:
    array = np.asarray(action_mean, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValidationError("action_mean: expected a non-empty 1-D array")
    if batch < 1 or chunk_size < 1:
        raise ValidationError("batch and chunk_size must be >= 1")
    return np.tile(array, (batch, chunk_size, 1))


def evaluate_l1(
    predictions: Any,
    targets: Any,
    mask: Any = None,
    *,
    action_std: Any = None,
    last_state: Any = None,
    mean_action: Any = None,
    horizon_seconds: float | None = None,
) -> L1Report:
    """Evaluate L1 over non-padded positions and compute the two baselines."""
    pred = _as_chunks(predictions, "predictions")
    target = _as_chunks(targets, "targets")
    if pred.shape != target.shape:
        raise ValidationError(
            f"predictions and targets must have the same shape; got {pred.shape} and {target.shape}"
        )
    valid = _as_mask(mask, pred.shape[:2])
    n_episodes, chunk_size, action_dim = pred.shape

    if n_episodes == 0:
        return L1Report(
            n_samples=0,
            n_valid_steps=0,
            per_dimension=None,
            per_horizon=None,
            normalized_per_dimension=None,
            baselines={},
            horizon_seconds=horizon_seconds,
        )

    per_dimension = _per_dimension(pred, target, valid)
    per_horizon = _per_horizon(pred, target, valid)
    normalized = None
    if action_std is not None:
        std = np.asarray(action_std, dtype=np.float64)
        if std.shape != (action_dim,):
            raise ValidationError(f"action_std: expected shape {(action_dim,)}, got {std.shape}")
        denom = np.where(np.abs(std) < MIN_STD, np.nan, std)
        normalized = per_dimension / denom

    baselines: dict[str, dict[str, np.ndarray | None]] = {}
    if last_state is not None:
        baseline = hold_last_state_baseline(
            last_state, chunk_size=chunk_size, action_dim=action_dim, batch=n_episodes
        )
        baselines["hold_last_state"] = _baseline_report(baseline, target, valid)
    if mean_action is not None:
        baseline = mean_action_baseline(mean_action, chunk_size=chunk_size, batch=n_episodes)
        baselines["mean_action"] = _baseline_report(baseline, target, valid)

    return L1Report(
        n_samples=n_episodes,
        n_valid_steps=int(valid.sum()),
        per_dimension=per_dimension,
        per_horizon=per_horizon,
        normalized_per_dimension=normalized,
        baselines=baselines,
        horizon_seconds=horizon_seconds,
    )


__all__ = [
    "L1Report",
    "ValidationError",
    "evaluate_l1",
    "hold_last_state_baseline",
    "mean_action_baseline",
    "valid_mask_from_padding",
]
