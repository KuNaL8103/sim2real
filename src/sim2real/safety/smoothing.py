"""Exponential moving average of commanded targets.

The filter is initialised from the measured state, so the first command cannot
jump from an arbitrary previous chunk.  ``alpha = 1`` is an exact passthrough.
Validation always runs on the smoothed output, not on the raw input.
"""

from __future__ import annotations

import numpy as np

from .config import SafetyConfig, SafetyConfigError


def ema_targets(
    positions: np.ndarray,
    measured: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Return ``y_k = alpha*x_k + (1-alpha)*y_{k-1}`` with ``y_{-1}=measured``."""
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise SafetyConfigError("ema_alpha must be a number")
    value = float(alpha)
    if not np.isfinite(value) or not 0.0 < value <= 1.0:
        raise SafetyConfigError("ema_alpha must be in (0, 1]")
    targets = np.asarray(positions, dtype=np.float64)
    initial = np.asarray(measured, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != initial.shape[0]:
        raise ValueError("positions and measured must have matching feature dimensions")
    if not np.all(np.isfinite(targets)) or not np.all(np.isfinite(initial)):
        raise ValueError("EMA inputs must be finite")
    if value == 1.0:
        return np.array(targets, dtype=np.float64, copy=True)
    output = np.empty_like(targets)
    previous = initial
    for index, target in enumerate(targets):
        previous = value * target + (1.0 - value) * previous
        output[index] = previous
    return output


def ema_alpha(config: SafetyConfig) -> float:
    return config.ema_alpha


__all__ = ["ema_alpha", "ema_targets"]
