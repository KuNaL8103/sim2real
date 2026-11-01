"""Pure-NumPy builders for ``sensor_msgs/PointCloud2`` and Marker fields.

No ``sensor_msgs_py.point_cloud2`` or ``cv_bridge``: both can drag in a
different NumPy ABI than the system NumPy used by ROS Jazzy.  These helpers
produce plain dictionaries/bytes; the ROS adapter copies them into messages.

Layout
------
``x, y, z`` are little-endian float32 at offsets 0, 4, 8.  ``rgb`` is a packed
float32 at offset 12 (the PCL/RViz convention: bits ``0x00RRGGBB`` reinterpreted
as float32).  ``point_step=16``, ``row_step=16*width``, ``height=1``, and
``is_dense=True``.  Color is optional; a missing color leaves ``rgb`` zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

POINTFIELD_INT8 = 1
POINTFIELD_UINT8 = 2
POINTFIELD_INT16 = 3
POINTFIELD_UINT16 = 4
POINTFIELD_INT32 = 5
POINTFIELD_UINT32 = 6
POINTFIELD_FLOAT32 = 7
POINTFIELD_FLOAT64 = 8

MARKER_TRIANGLE_LIST = 8
MARKER_ADD = 0

_POINT_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")])


@dataclass(frozen=True)
class PointField:
    name: str
    offset: int
    datatype: int
    count: int = 1


@dataclass(frozen=True)
class PointCloud2Data:
    fields: tuple[PointField, ...]
    width: int
    height: int
    point_step: int
    row_step: int
    is_dense: bool
    is_bigendian: bool
    frame_id: str
    stamp_ns: int
    data: bytes


@dataclass(frozen=True)
class MarkerData:
    type: int
    frame_id: str
    stamp_ns: int
    points: np.ndarray
    colors: np.ndarray
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ns: str = "workspace_tsdf"
    marker_id: int = 0
    action: int = MARKER_ADD


def _packed_rgb(colors: np.ndarray, count: int) -> np.ndarray:
    if colors is None or len(colors) == 0:
        return np.zeros(count, dtype=np.uint32)
    arr = np.asarray(colors)
    if arr.shape != (count, 3):
        raise ValueError("colors must have shape (N, 3)")
    if arr.dtype == np.uint8:
        rgb = arr.astype(np.uint32)
    else:
        rgb = np.clip(np.rint(np.asarray(arr, dtype=np.float64) * 255.0), 0, 255).astype(np.uint32)
    return (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]


def build_pointcloud2(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    *,
    frame_id: str = "world",
    stamp_ns: int = 0,
) -> PointCloud2Data:
    """Build a dense XYZ+RGB ``PointCloud2`` payload."""
    if not frame_id or frame_id.startswith("/"):
        raise ValueError("frame_id must be a non-empty relative frame name")
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    count = len(pts)
    packed = _packed_rgb(colors, count)
    structured = np.empty(count, dtype=_POINT_DTYPE)
    structured["x"] = pts[:, 0]
    structured["y"] = pts[:, 1]
    structured["z"] = pts[:, 2]
    structured["rgb"] = packed.view(np.float32)
    data = structured.tobytes()
    point_step = _POINT_DTYPE.itemsize
    return PointCloud2Data(
        fields=(
            PointField("x", 0, POINTFIELD_FLOAT32),
            PointField("y", 4, POINTFIELD_FLOAT32),
            PointField("z", 8, POINTFIELD_FLOAT32),
            PointField("rgb", 12, POINTFIELD_FLOAT32),
        ),
        width=count,
        height=1,
        point_step=point_step,
        row_step=point_step * count,
        is_dense=True,
        is_bigendian=False,
        frame_id=frame_id,
        stamp_ns=int(stamp_ns),
        data=data,
    )


def parse_pointcloud2(message: PointCloud2Data) -> dict[str, np.ndarray]:
    """Parse the payload built by :func:`build_pointcloud2` back to NumPy."""
    count = message.width * message.height
    if message.point_step != _POINT_DTYPE.itemsize:
        raise ValueError(f"unsupported point_step {message.point_step}")
    if len(message.data) != count * message.point_step:
        raise ValueError("data length does not match width * point_step")
    structured = np.frombuffer(message.data, dtype=_POINT_DTYPE, count=count)
    packed = structured["rgb"].view(np.uint32)
    return {
        "x": structured["x"].astype(np.float64),
        "y": structured["y"].astype(np.float64),
        "z": structured["z"].astype(np.float64),
        "r": ((packed >> 16) & 0xFF).astype(np.uint8),
        "g": ((packed >> 8) & 0xFF).astype(np.uint8),
        "b": (packed & 0xFF).astype(np.uint8),
    }


def _to_rgba(colors: np.ndarray | None, indices: np.ndarray) -> np.ndarray:
    count = len(indices)
    if colors is None or len(colors) == 0:
        rgba = np.tile(np.array([0.7, 0.7, 0.7, 1.0], dtype=np.float32), (count, 1))
        return rgba
    arr = np.asarray(colors)
    if arr.shape[0] <= int(indices.max(initial=0)):
        raise ValueError("colors must have one entry per mesh vertex")
    selected = arr[indices]
    if selected.dtype == np.uint8:
        rgba = selected.astype(np.float32) / 255.0
    else:
        rgba = selected.astype(np.float32)
    if rgba.shape[1] == 3:
        alpha = np.ones((len(rgba), 1), dtype=np.float32)
        rgba = np.concatenate([rgba, alpha], axis=1)
    return rgba


def mesh_to_marker(
    vertices: np.ndarray,
    triangles: np.ndarray,
    colors: np.ndarray | None = None,
    *,
    max_triangles: int = 200_000,
    frame_id: str = "world",
    stamp_ns: int = 0,
) -> MarkerData:
    """Flatten a triangle mesh into a ``TRIANGLE_LIST`` Marker payload.

    The point count is always three times the triangle count.  Triangles beyond
    ``max_triangles`` are dropped (a documented cap, not a quality claim).
    Colors are per-vertex RGB in [0, 1] or uint8; missing colors are grey.
    """
    if not frame_id or frame_id.startswith("/"):
        raise ValueError("frame_id must be a non-empty relative frame name")
    verts = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    tris = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    if max_triangles <= 0:
        raise ValueError("max_triangles must be > 0")
    tris = tris[:max_triangles]
    flat = tris.reshape(-1)
    if len(flat) and int(flat.max()) >= len(verts):
        raise ValueError("triangle index out of range")
    points = verts[flat].astype(np.float32) if len(flat) else np.zeros((0, 3), np.float32)
    rgba = _to_rgba(colors, flat) if len(flat) else np.zeros((0, 4), np.float32)
    return MarkerData(
        type=MARKER_TRIANGLE_LIST,
        frame_id=frame_id,
        stamp_ns=int(stamp_ns),
        points=points,
        colors=rgba,
    )
