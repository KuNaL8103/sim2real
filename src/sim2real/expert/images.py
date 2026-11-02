"""Minimal stdlib PNG writer/reader shared by replay and dataset tooling.

No image library is installed or needed for the 8-bit RGB PNGs this project
writes. The bytes are deterministic for the same RGB array, which the replay
determinism test relies on.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def write_png(path: str | Path, rgb: np.ndarray) -> None:
    """Write an HxWx3 uint8 RGB array as an 8-bit truecolour PNG."""
    target = Path(path)
    array = np.asarray(rgb)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("expected an HxWx3 uint8 RGB array")
    height, width, _ = array.shape
    raw = b"".join(b"\x00" + array[row].tobytes() for row in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    payload = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)


def read_png(path: str | Path) -> np.ndarray:
    """Read an 8-bit RGB truecolour PNG written by :func:`write_png`.

    Supports filter types 0-4 and no interlacing; this covers the project's
    deterministic PNGs without adding Pillow to the core environment.
    """
    target = Path(path)
    data = target.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"{target}: not a PNG file")
    offset = 8
    width = height = None
    compressed = bytearray()
    while offset + 8 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if kind == b"IHDR":
            width, height, depth, color, compression, filter_method, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
            if (depth, color, compression, filter_method, interlace) != (8, 2, 0, 0, 0):
                raise ValueError(f"{target}: only non-interlaced 8-bit RGB PNGs are supported")
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            break
    if width is None or height is None:
        raise ValueError(f"{target}: missing IHDR")
    raw = zlib.decompress(bytes(compressed))
    stride = width * 3
    expected = height * (stride + 1)
    if len(raw) != expected:
        raise ValueError(f"{target}: unexpected PNG data length {len(raw)} != {expected}")
    rows = np.zeros((height, stride), dtype=np.uint8)
    previous = np.zeros(stride, dtype=np.uint8)
    cursor = 0
    for row in range(height):
        filter_type = raw[cursor]
        cursor += 1
        line = np.frombuffer(raw[cursor : cursor + stride], dtype=np.uint8).copy()
        cursor += stride
        reconstructed = np.zeros(stride, dtype=np.uint8)
        if filter_type == 0:
            reconstructed = line
        elif filter_type == 1:  # Sub
            for index in range(stride):
                left = reconstructed[index - 3] if index >= 3 else 0
                reconstructed[index] = (int(line[index]) + int(left)) & 0xFF
        elif filter_type == 2:  # Up
            reconstructed = (line.astype(np.uint16) + previous).astype(np.uint8)
        elif filter_type == 3:  # Average
            reconstructed = np.zeros(stride, dtype=np.uint8)
            for index in range(stride):
                left = reconstructed[index - 3] if index >= 3 else 0
                reconstructed[index] = (
                    int(line[index]) + ((int(left) + int(previous[index])) // 2)
                ) & 0xFF
        elif filter_type == 4:  # Paeth
            reconstructed = np.zeros(stride, dtype=np.uint8)
            for index in range(stride):
                left = reconstructed[index - 3] if index >= 3 else 0
                up_left = previous[index - 3] if index >= 3 else 0
                estimate = int(left) + int(previous[index]) - int(up_left)
                pa = abs(estimate - int(left))
                pb = abs(estimate - int(previous[index]))
                pc = abs(estimate - int(up_left))
                predictor = (
                    left if pa <= pb and pa <= pc else (previous[index] if pb <= pc else up_left)
                )
                reconstructed[index] = (int(line[index]) + int(predictor)) & 0xFF
        else:
            raise ValueError(f"{target}: unsupported PNG filter {filter_type}")
        rows[row] = reconstructed
        previous = reconstructed
    return rows.reshape(height, width, 3)


__all__ = ["read_png", "write_png"]
