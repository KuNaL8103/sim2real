"""WorkspaceMapper orchestration: per-camera integration counters."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sim2real.mapping import load_mapping_config
from sim2real.mapping.mapper import WorkspaceMapper
from sim2real.mapping.tsdf_fusion import IntegrationResult

ROOT = Path(__file__).resolve().parents[1]


class _IdentityTransforms:
    def lookup(self, target: str, source: str, stamp_ns: int) -> np.ndarray:
        return np.eye(4)


class _AcceptingMap:
    def integrate(self, rgb, depth_m, intrinsics, transform, *, stamp):
        return IntegrationResult(True, "OK", valid_pixels=1)


def test_integrations_are_counted_per_camera():
    config = load_mapping_config(ROOT / "configs" / "mapping" / "default.yaml")
    mapper = WorkspaceMapper(config, _IdentityTransforms(), tsdf_map=_AcceptingMap())
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    depth = np.ones((2, 2), dtype=np.float32)
    intrinsics = object()

    mapper.integrate_frame("overhead", image, depth, intrinsics, 1, "overhead_frame")
    mapper.integrate_frame("wrist", image, depth, intrinsics, 2, "wrist_frame")
    mapper.integrate_frame("wrist", image, depth, intrinsics, 3, "wrist_frame")

    assert mapper.counters.integrated == 3
    assert mapper.counters.integrated_by_camera == {"overhead": 1, "wrist": 2}
    assert mapper.counters.as_dict()["integrated_by_camera"] == {"overhead": 1, "wrist": 2}
