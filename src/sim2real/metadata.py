"""Validation for bounded, safe JSON metadata carried by ROS messages.

Rollout metadata is transported as a JSON string so that the wire interface
stays stable while the metadata schema evolves. Two rules matter for safety:

* the payload is bounded (default 64 KiB), and
* it must be a JSON object that round-trips without NaN/Infinity, because
  non-finite values are not portable across JSON implementations.
"""

from __future__ import annotations

import json
from typing import Any, Final

MAX_METADATA_BYTES: Final = 65536


class MetadataError(ValueError):
    """Raised when metadata is not a valid, bounded JSON object."""


def _reject_non_finite(value: str) -> Any:
    raise MetadataError(f"non-finite JSON value is not allowed: {value}")


def validate_metadata_json(s: str, max_bytes: int = MAX_METADATA_BYTES) -> dict[str, Any]:
    """Parse and validate a metadata JSON string.

    Args:
        s: JSON text. An empty string is accepted and means "no metadata".
        max_bytes: Maximum accepted UTF-8 encoded size of ``s``.

    Returns:
        The decoded mapping. ``""`` returns ``{}``.

    Raises:
        MetadataError: If ``s`` is not a string, exceeds ``max_bytes``, is not
            valid JSON, has a non-object root, or contains NaN/Infinity.
    """
    if not isinstance(s, str):
        raise MetadataError("metadata must be a JSON string encoding an object with string keys")
    if max_bytes < 0:
        raise MetadataError("max_bytes must be non-negative")

    encoded_size = len(s.encode("utf-8"))
    if encoded_size > max_bytes:
        raise MetadataError(
            f"metadata is {encoded_size} bytes, exceeding the {max_bytes}-byte limit"
        )

    if s == "":
        return {}

    try:
        decoded = json.loads(s, parse_constant=_reject_non_finite)
    except MetadataError:
        raise
    except json.JSONDecodeError as exc:
        raise MetadataError(
            f"invalid JSON at line {exc.lineno} column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(decoded, dict):
        raise MetadataError(f"metadata root must be a JSON object, got {type(decoded).__name__}")

    # JSON object keys are always strings; the check is defensive and keeps the
    # guarantee explicit if this function is extended to accept decoded input.
    if not all(isinstance(key, str) for key in decoded):
        raise MetadataError("metadata object keys must be strings")

    return decoded
