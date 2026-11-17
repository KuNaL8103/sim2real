"""Pure validation helpers for the inference observation path.

Every rejection has a stable ``reason`` code so the node can count the cause
and expose it on ``/vla/status``. The helpers do not import ROS and do not know
about the policy server; they only check plain arrays, stamps, and strings.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from sim2real import contracts

# Reason codes (stable; tests and status counters use the strings verbatim).
NO_INSTRUCTION = "NO_INSTRUCTION"
INSTRUCTION_EMPTY = "INSTRUCTION_EMPTY"
INSTRUCTION_TOO_LONG = "INSTRUCTION_TOO_LONG"
NO_STATE = "NO_STATE"
STATE_TOO_OLD = "STATE_TOO_OLD"
STATE_WRONG_SHAPE = "STATE_WRONG_SHAPE"
STATE_NONFINITE = "STATE_NONFINITE"
STATE_NAMES = "STATE_NAMES"
NO_OVERHEAD = "NO_OVERHEAD"
NO_WRIST = "NO_WRIST"
CAMERA_SKEW = "CAMERA_SKEW"
OBSERVATION_TOO_OLD = "OBSERVATION_TOO_OLD"
IMAGE_ENCODING = "IMAGE_ENCODING"
IMAGE_ENDIANNESS = "IMAGE_ENDIANNESS"
IMAGE_SHAPE = "IMAGE_SHAPE"
IMAGE_STEP = "IMAGE_STEP"
IMAGE_LENGTH = "IMAGE_LENGTH"
IMAGE_DTYPE = "IMAGE_DTYPE"
SEQUENCE_REGRESSION = "SEQUENCE_REGRESSION"
NEGATIVE_STAMP = "NEGATIVE_STAMP"


class ValidationError(ValueError):
    """A rejected observation with a stable reason code."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        message = reason if not detail else f"{reason}: {detail}"
        super().__init__(message)


def validate_instruction(text: str | None, max_chars: int) -> str:
    if text is None:
        raise ValidationError(NO_INSTRUCTION, "no instruction received")
    if not isinstance(text, str):
        raise ValidationError(INSTRUCTION_EMPTY, f"instruction is {type(text).__name__}")
    value = text.strip()
    if not value:
        raise ValidationError(INSTRUCTION_EMPTY, "instruction is empty")
    if len(value) > max_chars:
        raise ValidationError(INSTRUCTION_TOO_LONG, f"{len(value)} characters > {max_chars}")
    return value


def validate_joint_state(
    names: Sequence[str],
    positions: Sequence[float],
    *,
    expected_names: Sequence[str] = contracts.CONTROLLED_JOINT_NAMES,
) -> tuple[np.ndarray, float]:
    from sim2real.joint_state import positions_from_named_state

    try:
        joints, gripper = positions_from_named_state(
            names, positions, expected_names=expected_names
        )
    except ValueError as exc:
        raise ValidationError(STATE_NAMES, str(exc)) from exc
    state = np.concatenate([joints, [gripper]])
    if state.shape != (8,):
        raise ValidationError(STATE_WRONG_SHAPE, f"state shape {state.shape} != (8,)")
    if not np.all(np.isfinite(state)):
        raise ValidationError(STATE_NONFINITE, "state contains NaN or infinity")
    return joints, gripper


def validate_state_age(now_ns: int, state_received_ns: int, max_age_s: float) -> float:
    age_ns = int(now_ns) - int(state_received_ns)
    age_s = age_ns / 1_000_000_000.0
    if age_s > max_age_s:
        raise ValidationError(STATE_TOO_OLD, f"state age {age_s:.3f} s > {max_age_s:.3f} s")
    return age_s


def validate_camera_skew(overhead_stamp_ns: int, wrist_stamp_ns: int, max_skew_s: float) -> float:
    skew_ns = abs(int(overhead_stamp_ns) - int(wrist_stamp_ns))
    skew_s = skew_ns / 1_000_000_000.0
    if skew_s > max_skew_s:
        raise ValidationError(
            CAMERA_SKEW,
            f"camera skew {skew_s:.3f} s > {max_skew_s:.3f} s",
        )
    return skew_s


def validate_observation_age(now_ns: int, camera_received_ns: int, max_age_s: float) -> float:
    age_ns = int(now_ns) - int(camera_received_ns)
    age_s = age_ns / 1_000_000_000.0
    if age_s > max_age_s:
        raise ValidationError(
            OBSERVATION_TOO_OLD,
            f"observation age {age_s:.3f} s > {max_age_s:.3f} s",
        )
    return age_s


def validate_sequence(sequence: int, last_sequence: int | None) -> int:
    if not isinstance(sequence, int) or isinstance(sequence, bool):
        raise ValidationError(SEQUENCE_REGRESSION, "sequence must be an integer")
    if last_sequence is not None and sequence <= last_sequence:
        raise ValidationError(
            SEQUENCE_REGRESSION,
            f"sequence {sequence} <= last accepted {last_sequence}",
        )
    return sequence


def validate_stamp(stamp_ns: int, name: str) -> int:
    if not isinstance(stamp_ns, int) or isinstance(stamp_ns, bool):
        raise ValidationError(NEGATIVE_STAMP, f"{name} stamp must be an integer")
    if stamp_ns < 0:
        raise ValidationError(NEGATIVE_STAMP, f"{name} stamp {stamp_ns} < 0")
    return stamp_ns


def image_from_fields(
    *,
    encoding: str,
    height: int,
    width: int,
    step: int,
    data: bytes | bytearray | memoryview | np.ndarray,
    is_bigendian: int = 0,
) -> np.ndarray:
    """Validate ``sensor_msgs/Image`` fields and return an owned uint8 HxWx3 array.

    No ``cv_bridge``: this is the same plain-NumPy path used elsewhere in the
    repository. Only ``rgb8`` is accepted for the model's cameras.
    """
    if encoding != contracts.COLOR_ENCODING:
        raise ValidationError(IMAGE_ENCODING, f"expected rgb8, got {encoding!r}")
    if int(is_bigendian) != 0:
        raise ValidationError(IMAGE_ENDIANNESS, f"is_bigendian={is_bigendian}")
    if not isinstance(height, int) or not isinstance(width, int) or isinstance(height, bool):
        raise ValidationError(IMAGE_SHAPE, "height/width must be integers")
    if height <= 0 or width <= 0:
        raise ValidationError(IMAGE_SHAPE, f"invalid height/width {height}x{width}")
    expected_step = width * 3
    if int(step) != expected_step:
        raise ValidationError(IMAGE_STEP, f"step {step} != {expected_step} for rgb8")
    if isinstance(data, np.ndarray):
        array = np.asarray(data)
        if array.dtype != np.uint8:
            raise ValidationError(IMAGE_DTYPE, f"image dtype {array.dtype} != uint8")
        flat = array.reshape(-1)
    else:
        try:
            flat = np.frombuffer(data, dtype=np.uint8)
        except (TypeError, ValueError) as exc:
            raise ValidationError(IMAGE_DTYPE, f"image data is not bytes: {exc}") from exc
    expected_length = expected_step * height
    if int(flat.size) != expected_length:
        raise ValidationError(IMAGE_LENGTH, f"{flat.size} bytes != {expected_length} expected")
    return np.ascontiguousarray(flat.reshape(height, width, 3))


def validate_image_array(name: str, array: np.ndarray) -> np.ndarray:
    """Check an already-converted image (used by the assembler and tests)."""
    value = np.asarray(array)
    if value.dtype != np.uint8:
        raise ValidationError(IMAGE_DTYPE, f"{name} dtype {value.dtype} != uint8")
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValidationError(IMAGE_SHAPE, f"{name} shape {value.shape} != HxWx3")
    if value.shape[0] <= 0 or value.shape[1] <= 0:
        raise ValidationError(IMAGE_SHAPE, f"{name} has an empty dimension")
    if not np.isfinite(value).all():  # uint8 is always finite; kept for symmetry
        raise ValidationError(IMAGE_DTYPE, f"{name} contains non-finite values")
    return np.ascontiguousarray(value)


def finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError("VALUE", f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError("VALUE", f"{name} must be finite")
    return number


__all__ = [
    "CAMERA_SKEW",
    "IMAGE_DTYPE",
    "IMAGE_ENCODING",
    "IMAGE_ENDIANNESS",
    "IMAGE_LENGTH",
    "IMAGE_SHAPE",
    "IMAGE_STEP",
    "INSTRUCTION_EMPTY",
    "INSTRUCTION_TOO_LONG",
    "NEGATIVE_STAMP",
    "NO_INSTRUCTION",
    "NO_OVERHEAD",
    "NO_STATE",
    "NO_WRIST",
    "OBSERVATION_TOO_OLD",
    "SEQUENCE_REGRESSION",
    "STATE_NAMES",
    "STATE_NONFINITE",
    "STATE_TOO_OLD",
    "STATE_WRONG_SHAPE",
    "ValidationError",
    "finite_float",
    "image_from_fields",
    "validate_camera_skew",
    "validate_image_array",
    "validate_instruction",
    "validate_joint_state",
    "validate_observation_age",
    "validate_sequence",
    "validate_stamp",
    "validate_state_age",
]
