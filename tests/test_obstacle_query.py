"""Vectorized obstacle-distance queries against the analytic box."""

from __future__ import annotations

import numpy as np
from tests.fixtures.synthetic_rgbd import SyntheticScene, box_surface_points

from sim2real.mapping import MapStatus, ObstacleQuery
from sim2real.mapping.config import WorkspaceBounds

BOUNDS = WorkspaceBounds((-0.2, -0.2, -0.1), (0.2, 0.2, 0.2))
RESOLUTION = 0.02
MAX_DISTANCE = 0.5


def _surface_query() -> ObstacleQuery:
    return ObstacleQuery(
        box_surface_points(SyntheticScene(), samples=15),
        BOUNDS,
        RESOLUTION,
        MAX_DISTANCE,
        status=MapStatus(True, 0.0, 1, 0.9),
    )


def _filled_box_points(scene: SyntheticScene, samples: int = 9) -> np.ndarray:
    margins = 0.002
    axes = [
        np.linspace(scene.box_min[i] + margins, scene.box_max[i] - margins, samples)
        for i in range(3)
    ]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return grid.reshape(-1, 3)


def _analytic_distance(points: np.ndarray, scene: SyntheticScene) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    closest = np.clip(pts, scene.box_min, scene.box_max)
    outside = np.linalg.norm(pts - closest, axis=1)
    inside = np.min(np.minimum(pts - scene.box_min, scene.box_max - pts), axis=1)
    return np.where(
        np.all((pts >= scene.box_min) & (pts <= scene.box_max), axis=1), inside, outside
    )


def test_distances_match_analytic_box_within_one_query_cell():
    scene = SyntheticScene()
    query = _surface_query()
    probes = np.array(
        [
            [0.10, 0.0, 0.02],
            [0.0, 0.10, 0.02],
            [0.0, 0.0, 0.12],
            [0.12, 0.12, 0.02],
            [-0.09, 0.05, 0.01],
        ]
    )
    measured = query.distance(probes)
    analytic = _analytic_distance(probes, scene)
    assert measured.shape == (len(probes),)
    assert np.allclose(measured, analytic, atol=RESOLUTION, rtol=0.0)


def test_inside_box_is_occupied_and_surface_is_occupied():
    scene = SyntheticScene()
    filled = ObstacleQuery(_filled_box_points(scene), BOUNDS, RESOLUTION, MAX_DISTANCE)
    probes = np.array([[0.0, 0.0, 0.02], [0.019, 0.0, 0.02]])
    assert np.all(filled.is_occupied(probes))
    assert not filled.is_occupied(np.array([[0.15, 0.15, 0.15]]))[0]

    surface = _surface_query()
    surface_probe = np.array([scene.box_max[0], 0.0, 0.02])
    assert surface.is_occupied(surface_probe)[0]


def test_distant_points_are_clamped_and_flagged_outside():
    query = _surface_query()
    distant = np.array([[10.0, 10.0, 10.0], [-5.0, 0.0, 0.0]])
    measured = query.distance(distant)
    assert np.all(measured == MAX_DISTANCE)
    assert np.all(query.outside_workspace(distant))
    assert not np.any(query.is_occupied(distant))


def test_vectorized_and_empty_inputs():
    query = _surface_query()
    probes = np.zeros((7, 3))
    assert query.distance(probes).shape == (7,)
    assert query.is_occupied(probes).shape == (7,)
    assert query.outside_workspace(probes).shape == (7,)
    assert query.distance(np.zeros((0, 3))).shape == (0,)
    assert query.is_occupied(np.zeros((0, 3))).shape == (0,)
    assert query.outside_workspace(np.zeros((0, 3))).shape == (0,)


def test_status_is_exposed_to_consumers():
    status = MapStatus(False, 9.0, 0, 0.0, ("NO_DATA", "STALE"))
    query = ObstacleQuery(
        box_surface_points(SyntheticScene(), samples=5),
        BOUNDS,
        RESOLUTION,
        MAX_DISTANCE,
        status=status,
    )
    assert query.status is status
    assert not query.status.valid
    assert query.status.blocking_reasons == ("NO_DATA", "STALE")
    assert query.occupied_cells > 0


def test_empty_occupancy_is_not_reported_as_safe():
    query = ObstacleQuery(np.zeros((0, 3)), BOUNDS, RESOLUTION, MAX_DISTANCE)
    assert not query.status.valid
    assert query.status.has("NO_DATA")
    assert np.all(query.distance(np.zeros((3, 3))) == MAX_DISTANCE)
