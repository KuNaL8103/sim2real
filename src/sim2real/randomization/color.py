"""Colour helpers for domain randomization.

``kelvin_to_rgb`` is an **approximate** black-body tint, not a colorimetric
conversion. It uses Tanner Helland's widely copied piecewise fit (the same
formulation used by many "colour temperature" utilities):

* ``t = kelvin / 100``
* red is 255 for ``t <= 66``, otherwise ``329.698727446 * (t - 60) ** -0.1332047592``
* green is ``99.4708025861 * ln(t) - 161.1195681661`` for ``t <= 66``,
  otherwise ``288.1221695283 * (t - 60) ** -0.0755148492``
* blue is 255 for ``t >= 66``, 0 for ``t <= 19``, otherwise
  ``138.5177312231 * ln(t - 10) - 305.0447927307``
* channels are clipped to ``0..255`` and divided by 255.

The valid range is **1000 K to 40000 K**. Outside it the fit is not meaningful,
so this module raises :class:`ValueError` rather than extrapolating. Tests only
check qualitative properties (warm light is red-biased, cool light is
blue-biased, channels stay in ``[0, 1]``, and the trend is monotone); the
values are not claimed to match a measured spectrum.
"""

from __future__ import annotations

import math

KELVIN_MIN = 1000.0
KELVIN_MAX = 40000.0

# A small, fixed palette used by the ``material_pool`` texture mode. These are
# flat RGBA values, not loaded texture assets: MuJoCo can swap a geom's RGBA at
# runtime but cannot swap texture images, so the applier reports PARTIAL.
MATERIAL_POOL: dict[str, tuple[float, float, float, float]] = {
    "red": (0.85, 0.10, 0.10, 1.0),
    "green": (0.10, 0.75, 0.15, 1.0),
    "blue": (0.10, 0.20, 0.90, 1.0),
    "yellow": (0.90, 0.80, 0.10, 1.0),
    "cyan": (0.10, 0.80, 0.85, 1.0),
    "magenta": (0.85, 0.10, 0.80, 1.0),
    "white": (0.90, 0.90, 0.90, 1.0),
    "black": (0.05, 0.05, 0.05, 1.0),
}


def kelvin_to_rgb(kelvin: float) -> tuple[float, float, float]:
    """Return an approximate ``(r, g, b)`` tint in ``[0, 1]`` for a temperature.

    Args:
        kelvin: Colour temperature in kelvin. Must be finite and in
            ``[1000, 40000]``.

    Raises:
        ValueError: If ``kelvin`` is non-finite or outside the documented range.
    """
    if not isinstance(kelvin, (int, float)) or isinstance(kelvin, bool):
        raise ValueError(f"kelvin must be a number, got {type(kelvin).__name__}")
    value = float(kelvin)
    if not math.isfinite(value):
        raise ValueError("kelvin must be finite")
    if not KELVIN_MIN <= value <= KELVIN_MAX:
        raise ValueError(f"kelvin must be in [{KELVIN_MIN:.0f}, {KELVIN_MAX:.0f}], got {value:.1f}")

    temperature = value / 100.0

    if temperature <= 66.0:
        red = 255.0
        green = 99.4708025861 * math.log(temperature) - 161.1195681661
    else:
        red = 329.698727446 * (temperature - 60.0) ** -0.1332047592
        green = 288.1221695283 * (temperature - 60.0) ** -0.0755148492

    if temperature >= 66.0:
        blue = 255.0
    elif temperature <= 19.0:
        blue = 0.0
    else:
        blue = 138.5177312231 * math.log(temperature - 10.0) - 305.0447927307

    return (
        min(1.0, max(0.0, red / 255.0)),
        min(1.0, max(0.0, green / 255.0)),
        min(1.0, max(0.0, blue / 255.0)),
    )
