"""Pure orchestration for the ROS workspace mapper.

``WorkspaceMapper`` owns the ``TSDFMap``, looks camera poses up through a
:class:`TransformProvider`, counts skips, detects camera-static-transform
changes, and produces a :class:`MapStatus`.  It imports no ROS; the ROS adapter
only decodes messages and calls this class, so the whole mapping path is
hardware-free and testable.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from sim2real import contracts

from .config import MappingConfig
from .obstacle_query import LOW_COVERAGE, NO_DATA, STALE, TF_FAILURES, MapStatus
from .transforms import (
    TransformProvider,
    camera_static_transforms,
    static_transforms_changed,
)
from .tsdf_fusion import IntegrationResult, TSDFMap

CAMERA_FRAMES = (
    contracts.CAMERA_OVERHEAD_OPTICAL_FRAME,
    contracts.CAMERA_WRIST_OPTICAL_FRAME,
)


@dataclass
class MapperCounters:
    integrated: int = 0
    skipped_no_tf: int = 0
    skipped_invalid_depth: int = 0
    resets_static_tf: int = 0
    resets_service: int = 0
    static_tf_updates: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "integrated": self.integrated,
            "skipped_no_tf": self.skipped_no_tf,
            "skipped_invalid_depth": self.skipped_invalid_depth,
            "resets_static_tf": self.resets_static_tf,
            "resets_service": self.resets_service,
            "static_tf_updates": self.static_tf_updates,
        }


class WorkspaceMapper:
    def __init__(
        self,
        config: MappingConfig,
        transform_provider: TransformProvider,
        *,
        clock: Callable[[], float] = time.monotonic,
        tsdf_map: TSDFMap | None = None,
    ) -> None:
        self.config = config
        self.transforms = transform_provider
        self.clock = clock
        self.map = tsdf_map if tsdf_map is not None else TSDFMap(config, clock=clock)
        self.counters = MapperCounters()
        self._camera_static: dict[tuple[str, str], np.ndarray] | None = None
        self._last_wall: float | None = None
        self._last_stamp: float | None = None
        self._last_tf_failure: str | None = None
        self._last_tf_log_wall = -math.inf

    # ------------------------------------------------------------------
    # Frames
    # ------------------------------------------------------------------
    def integrate_frame(
        self,
        camera: str,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        intrinsics: object,
        stamp_ns: int,
        camera_frame: str,
    ) -> IntegrationResult:
        transform = self.transforms.lookup(contracts.WORLD_FRAME, camera_frame, stamp_ns)
        if transform is None:
            self.counters.skipped_no_tf += 1
            self._last_tf_failure = f"{contracts.WORLD_FRAME} <- {camera_frame} at stamp {stamp_ns}"
            return IntegrationResult(False, "NO_TF", 0)
        result = self.map.integrate(rgb, depth_m, intrinsics, transform, stamp=stamp_ns)
        if result.accepted:
            self.counters.integrated += 1
            self._last_wall = self.clock()
            self._last_stamp = stamp_ns
        else:
            self.counters.skipped_invalid_depth += 1
        return result

    def tf_failure_log_due(self, period_s: float = 5.0) -> bool:
        """Throttle tf-failure logging; never raises and never blocks."""
        if self._last_tf_failure is None:
            return False
        now = self.clock()
        if now - self._last_tf_log_wall >= period_s:
            self._last_tf_log_wall = now
            return True
        return False

    @property
    def last_tf_failure(self) -> str | None:
        return self._last_tf_failure

    # ------------------------------------------------------------------
    # Scene-change reset
    # ------------------------------------------------------------------
    def on_static_transforms(self, transforms: Mapping[tuple[str, str], np.ndarray]) -> bool:
        """Update camera static transforms; reset on a change beyond epsilon.

        Returns ``True`` when the map was reset.  The first observation never
        resets.  Only transforms whose child is a camera optical frame are
        compared.  A reset without camera jitter does not change
        ``/tf_static``, so callers must call :meth:`reset` after
        ``/sim2real/reset_env`` as well.
        """
        current = camera_static_transforms(transforms, CAMERA_FRAMES)
        self.counters.static_tf_updates += 1
        changed = static_transforms_changed(
            self._camera_static,
            current,
            camera_frames=CAMERA_FRAMES,
            translation_epsilon_m=self.config.static_tf_translation_epsilon_m,
            rotation_epsilon_rad=self.config.static_tf_rotation_epsilon_rad,
        )
        self._camera_static = current
        if not self.config.reset_on_static_tf_change:
            return False
        if changed:
            self.map.reset()
            self.counters.resets_static_tf += 1
            self._last_wall = None
            return True
        return False

    def reset(self) -> None:
        """Explicit service reset (``/workspace/reset_map``)."""
        self.map.reset()
        self.counters.resets_service += 1
        self._last_wall = None
        self._last_stamp = None

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def status(self) -> MapStatus:
        stats = self.map.stats
        now = self.clock()
        if stats.last_integration_wall is None:
            age_s = math.inf
        else:
            age_s = max(0.0, now - stats.last_integration_wall)
        reasons: list[str] = []
        if stats.num_integrations == 0:
            reasons.append(NO_DATA)
        if math.isfinite(age_s) and age_s > self.config.map_stale_after_s:
            reasons.append(STALE)
        if stats.coverage_ratio < self.config.min_coverage_ratio:
            reasons.append(LOW_COVERAGE)
        if self.counters.skipped_no_tf > 0:
            reasons.append(TF_FAILURES)
        blocking = {NO_DATA, STALE, LOW_COVERAGE}
        valid = not any(reason in blocking for reason in reasons)
        return MapStatus(
            valid=valid,
            age_s=age_s,
            num_integrations=stats.num_integrations,
            coverage_ratio=stats.coverage_ratio,
            reasons=tuple(reasons),
        )
