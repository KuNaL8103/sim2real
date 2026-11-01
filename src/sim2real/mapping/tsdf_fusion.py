"""Workspace-bounded TSDF fusion using Open3D.

Why ``UniformTSDFVolume`` and not ``ScalableTSDFVolume``
-------------------------------------------------------
Open3D 0.20.0's legacy ``ScalableTSDFVolume`` is broken for metric float32
depth: its volume-activation step calls
``CreateFromDepthImage(image.depth_, ..., 1000.0, 1000.0, stride)`` on a depth
image that ``RGBDImage.create_from_color_and_depth`` has already converted to
metres, so it activates volume units at 1/1000 of the true position while the
voxel integrator uses the true metric depth.  The result is an empty volume.
This is upstream issue <https://github.com/isl-org/Open3D/issues/7592>
(open PR #7594); Open3D 0.19.0 is unaffected but its eager ``open3d.ml``
import needs a large ML dependency set under Python 3.12.

Because our workspace is bounded, a cubic ``UniformTSDFVolume`` is a good fit
and its metric-depth path is correct.  The volume is expanded by ``sdf_trunc``
on every side so a surface on the workspace boundary (the table top) still has
a zero crossing.  There is no free-space carving beyond the truncation band and
no raycasting: this is a surface map, not an occupancy map.

Coordinate convention
---------------------
``integrate`` receives ``T_world_camera`` (the camera pose **in world**).  Open3D
expects the *extrinsic* matrix, which is camera-from-world, i.e.
``np.linalg.inv(T_world_camera)``.  The inverse is computed explicitly and is
covered by a translated-camera test.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import MappingConfig, WorkspaceBounds

try:  # open3d is an optional core dependency; the ROS adapter may run without it.
    import open3d as o3d
except (ImportError, OSError):  # pragma: no cover - exercised by the o3d marker
    o3d = None  # type: ignore[assignment]

OPEN3D_AVAILABLE = o3d is not None


class Open3DUnavailableError(RuntimeError):
    """Raised when TSDF fusion is requested without an importable open3d."""


@dataclass(frozen=True)
class IntegrationResult:
    accepted: bool
    reason: str
    valid_pixels: int = 0


@dataclass(frozen=True)
class TSDFStats:
    num_integrations: int
    last_stamp: float | None
    coverage_ratio: float
    estimated_memory_mb: float
    reset_count: int
    resets_due_to_limit: int
    valid_pixels: int
    last_integration_wall: float | None


@dataclass(frozen=True)
class PointCloudData:
    points: np.ndarray
    colors: np.ndarray | None


@dataclass(frozen=True)
class MeshData:
    vertices: np.ndarray
    triangles: np.ndarray
    colors: np.ndarray | None


def _require_open3d() -> Any:
    if o3d is None:
        raise Open3DUnavailableError(
            "open3d is not importable; install the open3d-cpu wheel (see scripts/setup_envs.sh)"
        )
    return o3d


def intrinsics_matrix(intrinsics: Any, width: int, height: int) -> tuple[np.ndarray, int, int]:
    """Return a 3x3 pinhole matrix plus image width/height.

    Accepts a 3x3 matrix, a mapping with ``fx/fy/cx/cy``, or a
    :class:`sim2real.sim.backend.CameraIntrinsics`-like object.
    """
    if hasattr(intrinsics, "fx"):
        k = np.array(
            [
                [float(intrinsics.fx), 0.0, float(intrinsics.cx)],
                [0.0, float(intrinsics.fy), float(intrinsics.cy)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        w = int(getattr(intrinsics, "width", width))
        h = int(getattr(intrinsics, "height", height))
        return k, w, h
    if isinstance(intrinsics, dict):
        k = np.array(
            [
                [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
                [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return k, int(intrinsics.get("width", width)), int(intrinsics.get("height", height))
    k = np.asarray(intrinsics, dtype=np.float64)
    if k.shape != (3, 3):
        raise ValueError("intrinsics must be a 3x3 matrix or an object with fx/fy/cx/cy")
    return k, int(width), int(height)


class TSDFMap:
    """A workspace-bounded TSDF surface map."""

    def __init__(
        self,
        config: MappingConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        _require_open3d()
        self.config = config
        self._clock = clock
        self._bounds: WorkspaceBounds = config.workspace_bounds
        self._origin = np.asarray(self._bounds.minimum, dtype=np.float64) - config.sdf_trunc_m
        self._length = config.volume_extent_m
        self._resolution = config.volume_resolution
        nx = max(1, int(math.ceil(self._bounds.extent[0] / config.query_resolution_m)))
        ny = max(1, int(math.ceil(self._bounds.extent[1] / config.query_resolution_m)))
        self._coverage = np.zeros((nx, ny), dtype=bool)
        self._volume = self._new_volume()
        self._num_integrations = 0
        self._last_stamp: float | None = None
        self._last_integration_wall: float | None = None
        self._valid_pixels = 0
        self._reset_count = 0
        self._resets_due_to_limit = 0

    # ------------------------------------------------------------------
    # Volume lifecycle
    # ------------------------------------------------------------------
    def _new_volume(self) -> Any:
        o3d_mod = _require_open3d()
        return o3d_mod.pipelines.integration.UniformTSDFVolume(
            length=float(self._length),
            resolution=int(self._resolution),
            sdf_trunc=float(self.config.sdf_trunc_m),
            color_type=o3d_mod.pipelines.integration.TSDFVolumeColorType.RGB8,
            origin=self._origin.copy(),
        )

    def _retire_volume(self) -> None:
        """Drop the volume to bound map age; coverage and counters restart."""
        self._volume = self._new_volume()
        self._coverage[:] = False
        self._num_integrations = 0
        self._last_stamp = None
        self._last_integration_wall = self._clock()
        self._resets_due_to_limit += 1

    def reset(self) -> None:
        """Clear the map and its statistics (scene-change or service reset)."""
        self._volume = self._new_volume()
        self._coverage[:] = False
        self._num_integrations = 0
        self._last_stamp = None
        self._last_integration_wall = None
        self._valid_pixels = 0
        self._reset_count += 1

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------
    def integrate(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        intrinsics: Any,
        T_world_camera: np.ndarray,
        *,
        stamp: float | None = None,
    ) -> IntegrationResult:
        """Fuse one RGB-D frame.

        Invalid depth (NaN, inf, <=0, outside ``[depth_min, depth_trunc]``) is
        set to zero.  Pixels whose back-projected world point falls outside the
        workspace are also set to zero BEFORE integration, so they never touch
        the volume.  A frame with no valid pixels is ignored without changing
        ``num_integrations``.
        """
        rgb_arr = np.asarray(rgb)
        depth = np.asarray(depth_m)
        if rgb_arr.dtype != np.uint8 or rgb_arr.ndim != 3 or rgb_arr.shape[2] != 3:
            raise ValueError("rgb must be a uint8 HxWx3 array")
        if depth.dtype != np.float32 or depth.ndim != 2:
            raise ValueError("depth_m must be a float32 HxW array")
        if rgb_arr.shape[:2] != depth.shape:
            raise ValueError(f"rgb shape {rgb_arr.shape[:2]} does not match depth {depth.shape}")
        height, width = depth.shape
        k, k_width, k_height = intrinsics_matrix(intrinsics, width, height)
        if (k_width, k_height) != (width, height):
            raise ValueError(
                f"intrinsics size {(k_width, k_height)} does not match image {(width, height)}"
            )
        transform = np.asarray(T_world_camera, dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("T_world_camera must be a finite 4x4 matrix")

        valid = (
            np.isfinite(depth)
            & (depth >= self.config.depth_min_m)
            & (depth <= self.config.depth_trunc_m)
        )
        depth_safe = np.where(valid, depth, 0.0).astype(np.float32)
        u = np.arange(width, dtype=np.float64)[None, :]
        v = np.arange(height, dtype=np.float64)[:, None]
        x = (u - k[0, 2]) / k[0, 0] * depth_safe
        y = (v - k[1, 2]) / k[1, 1] * depth_safe
        camera_points = np.stack([x, y, depth_safe], axis=-1)
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        world_points = camera_points @ rotation.T + translation
        inside = self._bounds.contains(world_points)
        mask = valid & inside
        valid_pixels = int(mask.sum())
        if valid_pixels == 0:
            return IntegrationResult(False, "NO_VALID_DEPTH", 0)

        depth_clean = np.where(mask, depth_safe, 0.0).astype(np.float32)
        o3d_mod = _require_open3d()
        color_image = o3d_mod.geometry.Image(np.ascontiguousarray(rgb_arr))
        depth_image = o3d_mod.geometry.Image(np.ascontiguousarray(depth_clean))
        rgbd = o3d_mod.geometry.RGBDImage.create_from_color_and_depth(
            color_image,
            depth_image,
            depth_scale=1.0,
            depth_trunc=float(self.config.depth_trunc_m),
            convert_rgb_to_intensity=False,
        )
        intrinsic = o3d_mod.camera.PinholeCameraIntrinsic(
            int(width),
            int(height),
            float(k[0, 0]),
            float(k[1, 1]),
            float(k[0, 2]),
            float(k[1, 2]),
        )
        # Open3D's extrinsic is camera-from-world: the inverse of the camera
        # pose in world.  Do not pass T_world_camera directly.
        extrinsic = np.linalg.inv(transform)
        self._volume.integrate(rgbd, intrinsic, np.ascontiguousarray(extrinsic))

        self._num_integrations += 1
        self._last_stamp = float(stamp) if stamp is not None else None
        self._last_integration_wall = self._clock()
        self._valid_pixels += valid_pixels
        self._mark_coverage(world_points[mask])
        if self._num_integrations >= self.config.max_integrations_before_reset:
            self._retire_volume()
        return IntegrationResult(True, "INTEGRATED", valid_pixels)

    def _mark_coverage(self, points_world: np.ndarray) -> None:
        if points_world.size == 0:
            return
        resolution = self.config.query_resolution_m
        lo = np.asarray(self._bounds.minimum, dtype=np.float64)
        ix = np.floor((points_world[:, 0] - lo[0]) / resolution).astype(np.int64)
        iy = np.floor((points_world[:, 1] - lo[1]) / resolution).astype(np.int64)
        nx, ny = self._coverage.shape
        np.clip(ix, 0, nx - 1, out=ix)
        np.clip(iy, 0, ny - 1, out=iy)
        self._coverage[ix, iy] = True

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------
    def extract_point_cloud(self) -> PointCloudData:
        pcd = self._volume.extract_point_cloud()
        points = np.asarray(pcd.points, dtype=np.float64).reshape(-1, 3)
        colors = (
            np.asarray(pcd.colors, dtype=np.float64).reshape(-1, 3) if len(pcd.colors) else None
        )
        return PointCloudData(points=points, colors=colors)

    def extract_mesh(self, max_triangles: int | None = None) -> MeshData:
        mesh = self._volume.extract_triangle_mesh()
        vertices = np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)
        triangles = np.asarray(mesh.triangles, dtype=np.int64).reshape(-1, 3)
        colors = (
            np.asarray(mesh.vertex_colors, dtype=np.float64).reshape(-1, 3)
            if len(mesh.vertex_colors)
            else None
        )
        cap = self.config.max_triangles if max_triangles is None else int(max_triangles)
        if cap > 0 and len(triangles) > cap:
            triangles = triangles[:cap]
        return MeshData(vertices=vertices, triangles=triangles, colors=colors)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    @property
    def stats(self) -> TSDFStats:
        coverage = float(self._coverage.mean()) if self._coverage.size else 0.0
        return TSDFStats(
            num_integrations=self._num_integrations,
            last_stamp=self._last_stamp,
            coverage_ratio=coverage,
            estimated_memory_mb=self.config.estimated_memory_mb,
            reset_count=self._reset_count,
            resets_due_to_limit=self._resets_due_to_limit,
            valid_pixels=self._valid_pixels,
            last_integration_wall=self._last_integration_wall,
        )

    @property
    def has_data(self) -> bool:
        return self._num_integrations > 0

    @property
    def bounds(self) -> WorkspaceBounds:
        return self._bounds
