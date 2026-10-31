"""Message field builders shared by ROS adapters.

No ``cv_bridge``: it links against a specific NumPy ABI, and mixing ROS apt
packages with pip wheels has broken imports here before. These helpers return
plain dictionaries; the ROS node copies them into ``sensor_msgs`` messages.
"""

from __future__ import annotations

import numpy as np

from sim2real import contracts
from sim2real.sim.backend import CameraIntrinsics


def image_fields(
    array: np.ndarray, encoding: str, frame_id: str, stamp_ns: int
) -> dict[str, object]:
    """Build ``sensor_msgs/Image`` fields for ``rgb8`` or ``32FC1``."""
    if encoding == contracts.COLOR_ENCODING:
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("rgb8 requires a uint8 HxWx3 array")
        height, width = array.shape[:2]
        step = width * 3
    elif encoding == contracts.DEPTH_ENCODING:
        if array.dtype != np.float32 or array.ndim != 2:
            raise ValueError("32FC1 requires a float32 HxW array")
        height, width = array.shape
        step = width * 4
    else:
        raise ValueError(f"unsupported encoding {encoding!r}")

    if not frame_id or frame_id.startswith("/"):
        raise ValueError("frame_id must be a non-empty relative frame name")
    contiguous = np.ascontiguousarray(array)
    data = contiguous.tobytes()
    return {
        "height": int(height),
        "width": int(width),
        "encoding": encoding,
        "is_bigendian": 0,
        "step": int(step),
        "data": data,
        "data_length": len(data),
        "frame_id": frame_id,
        "stamp_ns": int(stamp_ns),
    }


def camera_info_fields(
    intrinsics: CameraIntrinsics, frame_id: str, stamp_ns: int
) -> dict[str, object]:
    """Build ``sensor_msgs/CameraInfo`` fields with a zero-distortion pinhole."""
    if not frame_id or frame_id.startswith("/"):
        raise ValueError("frame_id must be a non-empty relative frame name")
    k = [
        intrinsics.fx,
        0.0,
        intrinsics.cx,
        0.0,
        intrinsics.fy,
        intrinsics.cy,
        0.0,
        0.0,
        1.0,
    ]
    r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    p = [
        intrinsics.fx,
        0.0,
        intrinsics.cx,
        0.0,
        0.0,
        intrinsics.fy,
        intrinsics.cy,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]
    return {
        "height": int(intrinsics.height),
        "width": int(intrinsics.width),
        "distortion_model": "plumb_bob",
        "d": [0.0, 0.0, 0.0, 0.0, 0.0],
        "k": k,
        "r": r,
        "p": p,
        "binning_x": 0,
        "binning_y": 0,
        "frame_id": frame_id,
        "stamp_ns": int(stamp_ns),
    }
