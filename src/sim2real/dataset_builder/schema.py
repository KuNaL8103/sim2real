"""Framework-agnostic episode schema for dataset export.

Nothing here imports torch, LeRobot, ROS, or a renderer.  The schema is the
smallest common denominator used by the curated adapter, the RLDS adapter, the
statistics code, the splitter, and the mixing sampler.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

NUMERIC_DTYPES = frozenset({"float32", "float64", "int64", "int32", "uint8", "uint16", "bool"})
FEATURE_DTYPES = NUMERIC_DTYPES | {"string", "image", "video"}


class SchemaError(ValueError):
    """Raised when a feature or episode does not match the unified schema."""


@dataclass(frozen=True)
class UnifiedFeatureSpec:
    """One feature in a unified episode.

    ``shape`` includes the feature dimensions but not the time dimension.  For
    images the shape is ``(height, width, channels)``.  ``names`` and ``units``
    are per-dimension metadata when known.
    """

    name: str
    dtype: str
    shape: tuple[int, ...]
    names: tuple[str, ...] | None = None
    units: tuple[str, ...] | None = None
    frame: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise SchemaError("feature name must be a non-empty string")
        if self.dtype not in FEATURE_DTYPES:
            raise SchemaError(f"{self.name}: unsupported dtype {self.dtype!r}")
        if not self.shape or any(int(dim) <= 0 for dim in self.shape):
            raise SchemaError(f"{self.name}: shape must contain positive dimensions")
        if self.names is not None and len(self.names) != int(np.prod(self.shape)):
            raise SchemaError(
                f"{self.name}: names has {len(self.names)} entries for shape {self.shape}"
            )
        if self.units is not None and len(self.units) != int(np.prod(self.shape)):
            raise SchemaError(
                f"{self.name}: units has {len(self.units)} entries for shape {self.shape}"
            )

    @property
    def dim(self) -> int:
        return int(np.prod(self.shape))


@dataclass(frozen=True)
class ImageFrame:
    """An image aligned to one unified frame.

    An in-memory array is allowed for tests and fixture sources.  Production
    callers normally keep a path and a SHA-256 digest, so a large dataset does
    not have to be decoded until the writer reaches the frame.
    """

    step_index: int
    camera: str
    array: np.ndarray | None = None
    path: Path | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise SchemaError("image step_index must be >= 0")
        if not self.camera:
            raise SchemaError("image camera must be non-empty")
        if self.array is None and self.path is None:
            raise SchemaError("image must provide either an array or a path")

    def load(self) -> np.ndarray:
        """Return an HxWx3 uint8 RGB array, verifying a path digest when set."""
        if self.array is not None:
            array = np.asarray(self.array)
        else:
            assert self.path is not None
            if self.sha256 is not None:
                digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
                if digest != self.sha256:
                    raise SchemaError(
                        f"image sha256 mismatch for {self.path}: "
                        f"expected {self.sha256}, got {digest}"
                    )
            try:
                from sim2real.expert.images import read_png

                array = read_png(self.path)
            except ValueError:
                from PIL import Image  # imported lazily for non-project PNGs

                with Image.open(self.path) as handle:
                    array = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise SchemaError("image must be an HxWx3 uint8 RGB array")
        return array


@dataclass
class UnifiedEpisode:
    """One episode in the framework-neutral representation."""

    observation_state: np.ndarray
    action: np.ndarray
    images: dict[str, tuple[ImageFrame, ...]]
    timestamps: np.ndarray
    task: str
    embodiment_id: str
    source: str
    fps: float
    original_episode_id: str
    license_tag: str
    features: dict[str, UnifiedFeatureSpec] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return int(np.asarray(self.observation_state).shape[0])

    @property
    def num_images(self) -> int:
        return sum(len(frames) for frames in self.images.values())

    def frame(self, index: int) -> dict[str, Any]:
        """Return the non-image fields for one frame."""
        return {
            "observation.state": np.asarray(self.observation_state[index], dtype=np.float32),
            "action": np.asarray(self.action[index], dtype=np.float32),
        }


__all__ = [
    "FEATURE_DTYPES",
    "NUMERIC_DTYPES",
    "ImageFrame",
    "SchemaError",
    "UnifiedEpisode",
    "UnifiedFeatureSpec",
]
