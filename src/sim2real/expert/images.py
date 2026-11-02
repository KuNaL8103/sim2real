"""Minimal stdlib PNG writer shared by keyframe and replay rendering.

No image library is installed or needed. The bytes are deterministic for the
same RGB array, which the replay determinism test relies on.
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
