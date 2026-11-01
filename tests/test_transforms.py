"""Transform lookup, static-change detection, counters, and MapStatus."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from sim2real.mapping import (
    StaticTransformProvider,
    TSDFStats,
    WorkspaceMapper,
    static_transforms_changed,
    transform_delta,
)
from sim2real.mapping.tsdf_fusion import IntegrationResult
from tests.fixtures.mapping import make_config

OVERHEAD = "camera_overhead_optical_frame"
WRIST = "camera_wrist_optical_frame"


def _identity() -> np.ndarray:
    return np.eye(4)


def _translated(x: float) -> np.ndarray:
    t = np.eye(4)
    t[0, 3] = x
    return t


def _rotated(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    t = np.eye(4)
    t[0, 0] = c
    t[0, 1] = -s
    t[1, 0] = s
    t[1, 1] = c
    return t


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class FakeMap:
    """Minimal TSDFMap stand-in for pure orchestration tests."""

    def __init__(self, clock: FakeClock, *, accept: bool = True) -> None:
        self.clock = clock
        self.accept = accept
        self.reset_count = 0
        self.integrate_calls = 0
        self.transforms_seen: list[np.ndarray] = []
        self._stats = TSDFStats(
            num_integrations=0,
            last_stamp=None,
            coverage_ratio=0.0,
            estimated_memory_mb=1.0,
            reset_count=0,
            resets_due_to_limit=0,
            valid_pixels=0,
            last_integration_wall=None,
        )

    def reset(self) -> None:
        self.reset_count += 1
        self._stats = replace(
            self._stats,
            num_integrations=0,
            coverage_ratio=0.0,
            last_integration_wall=None,
        )

    def integrate(self, rgb, depth, intrinsics, transform, *, stamp=None):
        self.integrate_calls += 1
        self.transforms_seen.append(np.asarray(transform))
        if not self.accept:
            return IntegrationResult(False, "NO_VALID_DEPTH", 0)
        self._stats = replace(
            self._stats,
            num_integrations=self._stats.num_integrations + 1,
            last_integration_wall=self.clock(),
            last_stamp=stamp,
        )
        return IntegrationResult(True, "INTEGRATED", 10)

    @property
    def stats(self) -> TSDFStats:
        return self._stats

    def set_stats(self, **kwargs) -> None:
        self._stats = replace(self._stats, **kwargs)


def test_static_transform_change_detection_thresholds():
    previous = {("world", OVERHEAD): _identity()}
    same = {("world", OVERHEAD): _identity()}
    assert not static_transforms_changed(
        previous,
        same,
        camera_frames=(OVERHEAD, WRIST),
        translation_epsilon_m=0.005,
        rotation_epsilon_rad=0.01,
    )
    assert static_transforms_changed(
        previous,
        {("world", OVERHEAD): _translated(0.01)},
        camera_frames=(OVERHEAD, WRIST),
        translation_epsilon_m=0.005,
        rotation_epsilon_rad=0.01,
    )
    assert not static_transforms_changed(
        previous,
        {("world", OVERHEAD): _translated(0.001)},
        camera_frames=(OVERHEAD, WRIST),
        translation_epsilon_m=0.005,
        rotation_epsilon_rad=0.01,
    )
    assert static_transforms_changed(
        previous,
        {("world", OVERHEAD): _rotated(0.1)},
        camera_frames=(OVERHEAD, WRIST),
        translation_epsilon_m=0.005,
        rotation_epsilon_rad=0.01,
    )
    # A disappearing camera transform is a change.
    assert static_transforms_changed(
        previous,
        {},
        camera_frames=(OVERHEAD, WRIST),
        translation_epsilon_m=0.005,
        rotation_epsilon_rad=0.01,
    )


def test_transform_delta_reports_metres_and_radians():
    translation, rotation = transform_delta(_identity(), _translated(0.02))
    assert translation == 0.02
    assert rotation == 0.0
    translation, rotation = transform_delta(_identity(), _rotated(0.25))
    assert translation == 0.0
    assert abs(rotation - 0.25) < 1e-9


def test_static_transform_provider_lookup_and_missing():
    provider = StaticTransformProvider({("world", OVERHEAD): _identity()})
    assert provider.lookup("world", OVERHEAD) is not None
    assert provider.lookup("world", WRIST) is None
    provider.set("world", WRIST, _translated(0.1))
    assert provider.lookup("world", WRIST)[0, 3] == 0.1


def test_scene_change_resets_only_beyond_epsilon(tmp_path):
    clock = FakeClock()
    config = make_config(tmp_path)
    fake = FakeMap(clock)
    mapper = WorkspaceMapper(config, StaticTransformProvider({}), clock=clock, tsdf_map=fake)

    assert not mapper.on_static_transforms({("world", OVERHEAD): _identity()})
    assert fake.reset_count == 0
    assert not mapper.on_static_transforms({("world", OVERHEAD): _identity()})
    assert fake.reset_count == 0

    assert mapper.on_static_transforms({("world", OVERHEAD): _translated(0.05)})
    assert fake.reset_count == 1
    assert mapper.counters.resets_static_tf == 1

    assert not mapper.on_static_transforms({("world", OVERHEAD): _translated(0.051)})
    assert fake.reset_count == 1

    assert mapper.on_static_transforms({("world", OVERHEAD): _translated(0.2)})
    assert fake.reset_count == 2


def test_scene_change_reset_can_be_disabled(tmp_path):
    clock = FakeClock()
    config = make_config(tmp_path, reset_on_static_tf_change=False)
    fake = FakeMap(clock)
    mapper = WorkspaceMapper(config, StaticTransformProvider({}), clock=clock, tsdf_map=fake)
    mapper.on_static_transforms({("world", OVERHEAD): _identity()})
    assert not mapper.on_static_transforms({("world", OVERHEAD): _translated(0.5)})
    assert fake.reset_count == 0


def test_missing_transform_skips_frame_without_raising(tmp_path):
    clock = FakeClock()
    config = make_config(tmp_path)
    fake = FakeMap(clock)
    mapper = WorkspaceMapper(config, StaticTransformProvider({}), clock=clock, tsdf_map=fake)
    result = mapper.integrate_frame(
        "overhead",
        np.zeros((4, 4, 3), np.uint8),
        np.ones((4, 4), np.float32),
        np.eye(3),
        stamp_ns=1,
        camera_frame=OVERHEAD,
    )
    assert not result.accepted
    assert result.reason == "NO_TF"
    assert mapper.counters.skipped_no_tf == 1
    assert fake.integrate_calls == 0
    assert mapper.last_tf_failure is not None


def test_tf_failure_logging_is_throttled(tmp_path):
    clock = FakeClock(100.0)
    config = make_config(tmp_path)
    mapper = WorkspaceMapper(
        config, StaticTransformProvider({}), clock=clock, tsdf_map=FakeMap(clock)
    )
    mapper.integrate_frame(
        "overhead",
        np.zeros((4, 4, 3), np.uint8),
        np.ones((4, 4), np.float32),
        np.eye(3),
        1,
        OVERHEAD,
    )
    assert mapper.tf_failure_log_due(period_s=5.0)
    assert not mapper.tf_failure_log_due(period_s=5.0)
    clock.now += 5.1
    assert mapper.tf_failure_log_due(period_s=5.0)


def test_integrated_frame_updates_counters(tmp_path):
    clock = FakeClock(10.0)
    config = make_config(tmp_path)
    fake = FakeMap(clock)
    provider = StaticTransformProvider({("world", OVERHEAD): _identity()})
    mapper = WorkspaceMapper(config, provider, clock=clock, tsdf_map=fake)
    result = mapper.integrate_frame(
        "overhead",
        np.zeros((4, 4, 3), np.uint8),
        np.ones((4, 4), np.float32),
        np.eye(3),
        42,
        OVERHEAD,
    )
    assert result.accepted
    assert mapper.counters.integrated == 1
    assert fake.integrate_calls == 1
    assert np.allclose(fake.transforms_seen[0], _identity())


def test_map_status_no_data_valid_stale_and_low_coverage(tmp_path):
    clock = FakeClock(100.0)
    config = make_config(tmp_path)
    fake = FakeMap(clock)
    mapper = WorkspaceMapper(config, StaticTransformProvider({}), clock=clock, tsdf_map=fake)

    status = mapper.status()
    assert not status.valid
    assert status.has("NO_DATA")
    assert math.isinf(status.age_s)

    fake.set_stats(
        num_integrations=3,
        coverage_ratio=0.9,
        last_integration_wall=clock.now,
    )
    status = mapper.status()
    assert status.valid
    assert status.reasons == ()
    assert status.num_integrations == 3
    assert status.coverage_ratio == 0.9
    assert status.age_s == 0.0

    clock.now += 6.0
    status = mapper.status()
    assert not status.valid
    assert status.has("STALE")

    fake.set_stats(coverage_ratio=0.01)
    status = mapper.status()
    assert not status.valid
    assert status.has("LOW_COVERAGE")

    fake.set_stats(coverage_ratio=0.9, last_integration_wall=clock.now)
    mapper.counters.skipped_no_tf = 1
    status = mapper.status()
    assert status.has("TF_FAILURES")
    assert status.valid  # a skipped frame is a warning, not by itself invalid
