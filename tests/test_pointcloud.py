"""PointCloud2 and Marker field builders (pure NumPy, no ROS)."""

from __future__ import annotations

import numpy as np
import pytest

from sim2real.ros_utils.pointcloud import (
    MARKER_TRIANGLE_LIST,
    POINTFIELD_FLOAT32,
    build_pointcloud2,
    mesh_to_marker,
    parse_pointcloud2,
)


def test_pointcloud2_layout_and_round_trip():
    points = np.array([[1.0, 2.0, 3.0], [-4.0, 5.5, -6.25]], dtype=np.float64)
    colors = np.array([[255, 0, 0], [0, 128, 255]], dtype=np.uint8)
    cloud = build_pointcloud2(points, colors, frame_id="world", stamp_ns=123)
    assert cloud.width == 2
    assert cloud.height == 1
    assert cloud.point_step == 16
    assert cloud.row_step == 32
    assert len(cloud.data) == 32
    assert cloud.is_dense is True
    assert cloud.is_bigendian is False
    assert cloud.frame_id == "world"
    assert cloud.stamp_ns == 123
    assert [f.name for f in cloud.fields] == ["x", "y", "z", "rgb"]
    assert [f.offset for f in cloud.fields] == [0, 4, 8, 12]
    assert all(f.datatype == POINTFIELD_FLOAT32 for f in cloud.fields)

    parsed = parse_pointcloud2(cloud)
    assert np.allclose(parsed["x"], points[:, 0], atol=1e-6)
    assert np.allclose(parsed["y"], points[:, 1], atol=1e-6)
    assert np.allclose(parsed["z"], points[:, 2], atol=1e-6)
    assert np.array_equal(parsed["r"], colors[:, 0])
    assert np.array_equal(parsed["g"], colors[:, 1])
    assert np.array_equal(parsed["b"], colors[:, 2])


def test_pointcloud2_float_colors_and_missing_colors():
    points = np.zeros((3, 3), dtype=np.float32)
    cloud = build_pointcloud2(points, np.array([[0.0, 0.5, 1.0]] * 3))
    parsed = parse_pointcloud2(cloud)
    assert np.array_equal(parsed["r"], np.array([0, 0, 0], dtype=np.uint8))
    assert np.array_equal(parsed["g"], np.array([128, 128, 128], dtype=np.uint8))
    assert np.array_equal(parsed["b"], np.array([255, 255, 255], dtype=np.uint8))

    blank = build_pointcloud2(points, None)
    parsed = parse_pointcloud2(blank)
    assert np.all(parsed["r"] == 0)


def test_pointcloud2_empty_cloud():
    cloud = build_pointcloud2(np.zeros((0, 3)), None)
    assert cloud.width == 0
    assert cloud.row_step == 0
    assert cloud.data == b""
    parsed = parse_pointcloud2(cloud)
    assert parsed["x"].shape == (0,)


def test_pointcloud2_rejects_absolute_frame():
    with pytest.raises(ValueError):
        build_pointcloud2(np.zeros((1, 3)), None, frame_id="/world")


def test_mesh_to_marker_caps_triangles_and_uses_world_frame():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    triangles = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    marker = mesh_to_marker(vertices, triangles, max_triangles=2)
    assert marker.type == MARKER_TRIANGLE_LIST
    assert marker.frame_id == "world"
    assert marker.points.shape == (6, 3)
    assert len(marker.points) % 3 == 0
    assert marker.colors.shape == (6, 4)
    assert np.allclose(marker.colors[:, 3], 1.0)


def test_mesh_to_marker_uses_vertex_colors_and_rejects_bad_index():
    vertices = np.zeros((3, 3))
    triangles = np.array([[0, 1, 2]])
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    marker = mesh_to_marker(vertices, triangles, colors)
    assert np.allclose(marker.colors[:, 0], [1.0, 0.0, 0.0])
    assert np.allclose(marker.colors[:, 1], [0.0, 1.0, 0.0])
    assert np.allclose(marker.colors[:, 2], [0.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        mesh_to_marker(vertices, np.array([[0, 1, 9]]))
