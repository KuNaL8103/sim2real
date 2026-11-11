"""Length-prefixed JSON + raw-array protocol shared by the policy server and its
client.

Why pipes instead of a socket: the server is a short-lived child of the
simulator process, and stdin/stdout already give ordered, back-pressured,
buffered IPC with no port allocation, no filesystem socket permissions, and no
stale-socket cleanup.  Frames are 4-byte big-endian length prefixes followed by
bytes.  A packet is one JSON header frame plus zero or more raw blob frames
(for example uint8 camera images and float32 action chunks).  There is **no
pickle** anywhere on the wire.

This module is pure standard library and imports no torch, LeRobot, or ROS, so
the core evaluation environment can speak the protocol without the ML
dependencies.
"""

from __future__ import annotations

import json
import struct
from typing import Any, BinaryIO

PROTOCOL_VERSION = 1
MAX_FRAME_BYTES = 32 * 1024 * 1024


class ProtocolError(RuntimeError):
    """Raised on a malformed frame, oversized frame, or protocol mismatch."""


def write_frame(stream: BinaryIO, payload: bytes) -> None:
    if len(payload) > MAX_FRAME_BYTES:
        raise ProtocolError(f"frame of {len(payload)} bytes exceeds the {MAX_FRAME_BYTES} limit")
    stream.write(struct.pack(">I", len(payload)))
    stream.write(payload)
    stream.flush()


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise ProtocolError("unexpected end of stream")
        chunks.extend(chunk)
    return bytes(chunks)


def read_frame(stream: BinaryIO) -> bytes:
    header = _read_exact(stream, 4)
    (length,) = struct.unpack(">I", header)
    if length > MAX_FRAME_BYTES:
        raise ProtocolError(f"frame of {length} bytes exceeds the {MAX_FRAME_BYTES} limit")
    return _read_exact(stream, length)


def write_packet(
    stream: BinaryIO, header: dict[str, Any], blobs: list[bytes] | None = None
) -> None:
    payload = dict(header)
    payload["blob_lengths"] = [len(blob) for blob in (blobs or [])]
    write_frame(stream, json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8"))
    for blob in blobs or []:
        write_frame(stream, blob)


def read_packet(stream: BinaryIO) -> tuple[dict[str, Any], list[bytes]]:
    header = json.loads(read_frame(stream).decode("utf-8"))
    if not isinstance(header, dict):
        raise ProtocolError("packet header must be a JSON object")
    lengths = header.pop("blob_lengths", [])
    if not isinstance(lengths, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in lengths
    ):
        raise ProtocolError("invalid blob_lengths")
    blobs = [read_frame(stream) for _ in lengths]
    return header, blobs


__all__ = [
    "MAX_FRAME_BYTES",
    "PROTOCOL_VERSION",
    "ProtocolError",
    "read_frame",
    "read_packet",
    "write_frame",
    "write_packet",
]
