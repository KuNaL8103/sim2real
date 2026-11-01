"""Coarse robot self-filter behaviour."""

from __future__ import annotations

import numpy as np

from sim2real.mapping import RobotSelfFilter, SelfFilterConfig


def _filter() -> RobotSelfFilter:
    return RobotSelfFilter(
        SelfFilterConfig(
            link_frames=("link_a", "link_b"),
            radii_m={"link_a": 0.1, "link_b": 0.2},
        )
    )


def test_points_inside_link_spheres_are_removed():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.15],
        ]
    )
    kept, result = _filter().filter(
        points, {"link_a": np.zeros(3), "link_b": np.array([1.0, 0.0, 0.0])}
    )
    assert result.kept == 2
    assert result.removed == 3
    assert result.missing_links == ()
    assert np.allclose(kept, np.array([[0.5, 0.0, 0.0], [0.0, 0.0, 0.15]]))


def test_missing_link_is_reported_and_not_filtered():
    points = np.array([[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    kept, result = _filter().filter(points, {"link_a": np.zeros(3)})
    assert result.missing_links == ("link_b",)
    assert result.kept == 2
    assert result.removed == 0
    assert np.allclose(kept, points)


def test_empty_input_is_supported():
    kept, result = _filter().filter(np.zeros((0, 3)), {})
    assert kept.shape == (0, 3)
    assert result.kept == 0
    assert result.removed == 0
    assert result.missing_links == ("link_a", "link_b")
