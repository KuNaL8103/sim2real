"""Deterministic analytic RGB-D renderer for hardware-free mapping tests.

The scene is a table plane at ``table_z`` plus one axis-aligned box resting on
it.  Rays are cast through every pixel in the ROS optical convention
(x right, y down, z forward); the returned depth is the camera-space z (metres).
No simulator, camera, GUI, or GPU is involved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticIntrinsics:
    width: int = 64
    height: int = 48
    fx: float = 80.0
    fy: float = 80.0
    cx: float | None = None
    cy: float | None = None

    def __post_init__(self) -> None:
        if self.cx is None:
            object.__setattr__(self, "cx", (self.width - 1) / 2.0)
        if self.cy is None:
            object.__setattr__(self, "cy", (self.height - 1) / 2.0)

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0.0, float(self.cx)],
                [0.0, self.fy, float(self.cy)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class SyntheticScene:
    table_z: float = 0.0
    table_half_extent: float = 0.50
    box_center: tuple[float, float, float] = (0.0, 0.0, 0.02)
    box_size: tuple[float, float, float] = (0.04, 0.04, 0.04)
    table_color: tuple[float, float, float] = (0.62, 0.62, 0.60)
    box_color: tuple[float, float, float] = (0.80, 0.10, 0.10)

    @property
    def box_min(self) -> np.ndarray:
        centre = np.asarray(self.box_center, dtype=np.float64)
        half = np.asarray(self.box_size, dtype=np.float64) * 0.5
        return centre - half

    @property
    def box_max(self) -> np.ndarray:
        centre = np.asarray(self.box_center, dtype=np.float64)
        half = np.asarray(self.box_size, dtype=np.float64) * 0.5
        return centre + half


@dataclass(frozen=True)
class SyntheticFrame:
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: SyntheticIntrinsics
    T_world_camera: np.ndarray
    points_world: np.ndarray
    hit_mask: np.ndarray

    @property
    def K(self) -> np.ndarray:
        return self.intrinsics.matrix


def look_at(eye, target, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Return ``T_world_camera`` for a look-at pose in the optical convention."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm == 0.0:
        raise ValueError("eye and target must differ")
    forward /= norm
    if abs(float(np.dot(forward, up))) > 0.99:
        up = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(forward, up))) > 0.99:
            up = np.array([1.0, 0.0, 0.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.column_stack([right, down, forward])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = eye
    return transform


def overhead_camera(distance: float = 1.0) -> np.ndarray:
    return look_at((0.0, 0.0, distance), (0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0))


def oblique_camera() -> np.ndarray:
    return look_at((0.35, -0.35, 0.55), (0.0, 0.0, 0.01), up=(0.0, 0.0, 1.0))


def _plane_hits(origin: np.ndarray, directions: np.ndarray, scene: SyntheticScene):
    denom = directions[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (scene.table_z - origin[2]) / denom
    valid = (np.abs(denom) > 1e-12) & (t > 0.0)
    x = origin[0] + t * directions[:, 0]
    y = origin[1] + t * directions[:, 1]
    inside = (np.abs(x) <= scene.table_half_extent) & (np.abs(y) <= scene.table_half_extent)
    return np.where(valid & inside, t, np.inf)


def _box_hits(origin: np.ndarray, directions: np.ndarray, scene: SyntheticScene):
    safe = np.where(np.abs(directions) > 1e-12, directions, 1e-12)
    t0 = (scene.box_min - origin) / safe
    t1 = (scene.box_max - origin) / safe
    tmin = np.max(np.minimum(t0, t1), axis=1)
    tmax = np.min(np.maximum(t0, t1), axis=1)
    hit = (tmax >= np.maximum(tmin, 0.0)) & (tmin > 0.0)
    return np.where(hit, tmin, np.inf)


def render(
    scene: SyntheticScene | None = None,
    intrinsics: SyntheticIntrinsics | None = None,
    T_world_camera: np.ndarray | None = None,
    *,
    stamp_ns: int = 0,
) -> SyntheticFrame:
    """Ray-cast the analytic scene into a small RGB-D frame."""
    scene = scene or SyntheticScene()
    intrinsics = intrinsics or SyntheticIntrinsics()
    pose = overhead_camera() if T_world_camera is None else np.asarray(T_world_camera)
    height, width = intrinsics.height, intrinsics.width
    u = np.arange(width, dtype=np.float64)[None, :]
    v = np.arange(height, dtype=np.float64)[:, None]
    dirs_camera = np.stack(
        [
            (u - float(intrinsics.cx)) / intrinsics.fx * np.ones_like(v),
            (v - float(intrinsics.cy)) / intrinsics.fy * np.ones_like(u),
            np.ones_like(u * v),
        ],
        axis=-1,
    ).reshape(-1, 3)
    rotation = pose[:3, :3]
    origin = pose[:3, 3]
    directions = dirs_camera @ rotation.T
    plane_t = _plane_hits(origin, directions, scene)
    box_t = _box_hits(origin, directions, scene)
    t = np.minimum(plane_t, box_t)
    hit = np.isfinite(t) & (t > 0.0)
    pixels = origin + t[:, None] * directions

    rgb = np.zeros((height * width, 3), dtype=np.uint8)
    table = np.asarray(scene.table_color, dtype=np.float64)
    box = np.asarray(scene.box_color, dtype=np.float64)
    box_hit = np.isfinite(box_t) & (box_t <= plane_t)
    # A deterministic checker makes the table colour non-uniform for RGB tests.
    checker = np.ones(len(pixels), dtype=np.float64)
    checker[hit] = 0.85 + 0.15 * (
        (np.floor(pixels[hit, 0] * 20.0) + np.floor(pixels[hit, 1] * 20.0)) % 2.0
    )
    colors = table[None, :] * checker[:, None]
    colors = np.where(box_hit[:, None], box[None, :], colors)
    rgb[hit] = np.clip(colors[hit] * 255.0, 0, 255).astype(np.uint8)

    depth = np.where(hit, t, 0.0).astype(np.float32).reshape(height, width)
    points = np.where(hit[:, None], pixels, np.nan).reshape(height, width, 3)
    return SyntheticFrame(
        rgb=rgb.reshape(height, width, 3),
        depth=depth,
        intrinsics=intrinsics,
        T_world_camera=pose,
        points_world=points,
        hit_mask=hit.reshape(height, width),
    )


def unproject(
    depth: np.ndarray, K: np.ndarray, T_world_camera: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project a metric depth image to world points (returns points, mask)."""
    depth = np.asarray(depth, dtype=np.float64)
    height, width = depth.shape
    valid = np.isfinite(depth) & (depth > 0.0)
    u = np.arange(width, dtype=np.float64)[None, :]
    v = np.arange(height, dtype=np.float64)[:, None]
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    camera_points = np.stack([x, y, depth], axis=-1)
    rotation = np.asarray(T_world_camera)[:3, :3]
    translation = np.asarray(T_world_camera)[:3, 3]
    world = camera_points @ rotation.T + translation
    return world, valid


def box_surface_points(scene: SyntheticScene | None = None, samples: int = 12) -> np.ndarray:
    """Dense samples on the box faces, used as occupied points in query tests."""
    scene = scene or SyntheticScene()
    axes = [np.linspace(scene.box_min[i], scene.box_max[i], samples) for i in range(3)]
    points: list[tuple[float, float, float]] = []
    for fixed_axis in range(3):
        for value in (scene.box_min[fixed_axis], scene.box_max[fixed_axis]):
            other = [i for i in range(3) if i != fixed_axis]
            for a in axes[other[0]]:
                for b in axes[other[1]]:
                    point = [0.0, 0.0, 0.0]
                    point[fixed_axis] = float(value)
                    point[other[0]] = float(a)
                    point[other[1]] = float(b)
                    points.append(tuple(point))
    return np.asarray(points, dtype=np.float64)
