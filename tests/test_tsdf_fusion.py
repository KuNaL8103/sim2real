"""TSDF fusion geometry and invalid-input handling (hardware-free, Open3D)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("open3d")

from sim2real.mapping import TSDFMap  # noqa: E402
from tests.fixtures.mapping import make_config  # noqa: E402
from tests.fixtures.synthetic_rgbd import (  # noqa: E402
    SyntheticScene,
    oblique_camera,
    overhead_camera,
    render,
    unproject,
)

pytestmark = pytest.mark.o3d


def _overhead_frame():
    return render(SyntheticScene(), T_world_camera=overhead_camera())


def _oblique_frame():
    return render(SyntheticScene(), T_world_camera=oblique_camera())


def test_backprojection_matches_analytic_scene():
    frame = _overhead_frame()
    points, valid = unproject(frame.depth, frame.K, frame.T_world_camera)
    expected = frame.points_world[frame.hit_mask]
    assert valid.sum() > 100
    errors = points[frame.hit_mask] - expected
    rms = float(np.sqrt(np.mean(np.sum(errors**2, axis=1))))
    assert rms < 1.0e-3, f"back-projection RMS {rms} m is not sub-millimetre"
    assert np.max(np.abs(errors)) < 1.0e-3


def test_single_view_table_plane_within_one_voxel(tmp_path):
    config = make_config(tmp_path)
    tsdf = TSDFMap(config)
    frame = _overhead_frame()
    result = tsdf.integrate(frame.rgb, frame.depth, frame.K, frame.T_world_camera, stamp=1.0)
    assert result.accepted
    assert result.valid_pixels > 0
    cloud = tsdf.extract_point_cloud()
    table = cloud.points[
        (np.abs(cloud.points[:, 0]) < 0.15)
        & (np.abs(cloud.points[:, 1]) < 0.15)
        & (cloud.points[:, 2] < 0.015)
    ]
    assert len(table) > 50
    assert abs(float(np.median(table[:, 2]))) <= config.voxel_size_m


def test_two_view_consistency_catches_inverted_extrinsic(tmp_path):
    config = make_config(tmp_path)
    results = {}
    for name, frame in (("overhead", _overhead_frame()), ("oblique", _oblique_frame())):
        tsdf = TSDFMap(config)
        result = tsdf.integrate(frame.rgb, frame.depth, frame.K, frame.T_world_camera, stamp=1.0)
        assert result.accepted
        results[name] = tsdf.extract_point_cloud().points

    plane_z = {}
    box_z = {}
    for name, points in results.items():
        plane = points[
            (np.abs(points[:, 0]) < 0.12)
            & (np.abs(points[:, 1]) < 0.12)
            & (np.abs(points[:, 2]) < 0.01)
        ]
        box = points[
            (np.abs(points[:, 0]) < 0.015)
            & (np.abs(points[:, 1]) < 0.015)
            & (points[:, 2] > 0.03)
            & (points[:, 2] < 0.05)
        ]
        assert len(plane) > 20, name
        plane_z[name] = float(np.median(plane[:, 2]))
        # The oblique view may have fewer box-top points; require some.
        assert len(box) > 0, name
        box_z[name] = float(np.median(box[:, 2]))

    assert abs(plane_z["overhead"] - plane_z["oblique"]) <= config.voxel_size_m
    assert abs(box_z["overhead"] - box_z["oblique"]) <= 2 * config.voxel_size_m
    assert abs(box_z["overhead"] - 0.04) <= 2 * config.voxel_size_m


def test_invalid_depth_is_ignored_without_changing_counts(tmp_path):
    config = make_config(tmp_path)
    tsdf = TSDFMap(config)
    frame = _overhead_frame()
    assert tsdf.integrate(frame.rgb, frame.depth, frame.K, frame.T_world_camera).accepted
    count = tsdf.stats.num_integrations
    assert count == 1

    bad = np.full_like(frame.depth, np.nan)
    h, w = bad.shape
    bad[: h // 4] = np.inf
    bad[h // 4 : h // 2] = 0.0
    bad[h // 2 : 3 * h // 4] = -1.0
    bad[3 * h // 4 :] = config.depth_trunc_m + 1.0
    result = tsdf.integrate(frame.rgb, bad, frame.K, frame.T_world_camera)
    assert not result.accepted
    assert result.reason == "NO_VALID_DEPTH"
    assert tsdf.stats.num_integrations == count

    mixed = frame.depth.copy()
    mixed[:, : w // 2] = np.nan
    mixed_result = tsdf.integrate(frame.rgb, mixed, frame.K, frame.T_world_camera)
    assert mixed_result.accepted
    assert 0 < mixed_result.valid_pixels < frame.hit_mask.sum()
    assert tsdf.stats.num_integrations == count + 1


def test_workspace_crop_discards_points_outside_bounds(tmp_path):
    config = make_config(tmp_path)
    tsdf = TSDFMap(config)
    frame = render(SyntheticScene(table_half_extent=2.0), T_world_camera=overhead_camera())
    assert tsdf.integrate(frame.rgb, frame.depth, frame.K, frame.T_world_camera).accepted
    cloud = tsdf.extract_point_cloud()
    assert len(cloud.points) > 0
    margin = config.sdf_trunc_m + config.voxel_size_m
    lo = np.asarray(config.workspace_bounds.minimum) - margin
    hi = np.asarray(config.workspace_bounds.maximum) + margin
    assert np.all(cloud.points >= lo)
    assert np.all(cloud.points <= hi)
    assert tsdf.stats.estimated_memory_mb < config.max_memory_mb
    assert tsdf.stats.estimated_memory_mb < 100.0


def test_mesh_extraction_respects_triangle_cap(tmp_path):
    config = make_config(tmp_path)
    tsdf = TSDFMap(config)
    frame = _overhead_frame()
    assert tsdf.integrate(frame.rgb, frame.depth, frame.K, frame.T_world_camera).accepted
    mesh = tsdf.extract_mesh(max_triangles=10)
    assert len(mesh.vertices) > 0
    assert 0 < len(mesh.triangles) <= 10
    assert mesh.triangles.shape[1] == 3


def test_reset_clears_the_map(tmp_path):
    config = make_config(tmp_path)
    tsdf = TSDFMap(config)
    frame = _overhead_frame()
    assert tsdf.integrate(frame.rgb, frame.depth, frame.K, frame.T_world_camera).accepted
    assert tsdf.has_data
    tsdf.reset()
    assert not tsdf.has_data
    assert tsdf.stats.num_integrations == 0
    assert tsdf.stats.reset_count == 1
    assert len(tsdf.extract_point_cloud().points) == 0
