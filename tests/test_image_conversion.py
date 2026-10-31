"""Image and CameraInfo field builders (no cv_bridge)."""

from __future__ import annotations

import numpy as np
import pytest

from sim2real import contracts
from sim2real.ros_utils.image_conversion import camera_info_fields, image_fields
from sim2real.sim.backend import CameraIntrinsics


def test_rgb8_fields():
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    fields = image_fields(rgb, contracts.COLOR_ENCODING, "camera_overhead_optical_frame", 123)
    assert fields["encoding"] == "rgb8"
    assert fields["height"] == 4
    assert fields["width"] == 6
    assert fields["step"] == 6 * 3
    assert fields["is_bigendian"] == 0
    assert fields["data_length"] == 4 * 6 * 3
    assert len(fields["data"]) == fields["data_length"]
    assert fields["stamp_ns"] == 123
    assert fields["frame_id"] == "camera_overhead_optical_frame"


def test_depth_32fc1_fields_and_nan_preserved():
    depth = np.array([[1.0, np.nan], [0.0, 2.5]], dtype=np.float32)
    fields = image_fields(depth, contracts.DEPTH_ENCODING, "camera_wrist_optical_frame", 5)
    assert fields["encoding"] == "32FC1"
    assert fields["step"] == 2 * 4
    assert fields["data_length"] == 2 * 2 * 4
    decoded = np.frombuffer(fields["data"], dtype=np.float32).reshape(2, 2)
    assert np.isnan(decoded[0, 1])
    assert float(decoded[1, 1]) == pytest.approx(2.5)


def test_camera_info_layout():
    intrinsics = CameraIntrinsics(fx=100.0, fy=110.0, cx=32.0, cy=24.0, width=64, height=48)
    fields = camera_info_fields(intrinsics, "camera_overhead_optical_frame", 77)
    assert fields["distortion_model"] == "plumb_bob"
    assert fields["d"] == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert fields["k"] == [100.0, 0.0, 32.0, 0.0, 110.0, 24.0, 0.0, 0.0, 1.0]
    assert fields["r"] == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    assert fields["p"] == [
        100.0,
        0.0,
        32.0,
        0.0,
        0.0,
        110.0,
        24.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]
    assert fields["width"] == 64
    assert fields["height"] == 48


def test_rejects_absolute_frame_and_wrong_dtype():
    with pytest.raises(ValueError):
        image_fields(np.zeros((2, 2, 3), dtype=np.uint8), "rgb8", "/world", 0)
    with pytest.raises(ValueError):
        image_fields(np.zeros((2, 2), dtype=np.uint16), "32FC1", "world", 0)
    with pytest.raises(ValueError):
        image_fields(np.zeros((2, 2), dtype=np.float32), "mono8", "world", 0)
