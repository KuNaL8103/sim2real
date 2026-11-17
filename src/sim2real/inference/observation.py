"""Assemble the policy-server observation from plain arrays and stamps.

The same 8-D layout is used by the closed-loop runner (through
``sim2real.joint_state.state_from_parts``) and by the ROS node. Images are
``uint8 HxWx3`` RGB; the model does not use depth (training was RGB only).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from sim2real import contracts
from sim2real.evaluation.policy_interface import PolicyObservation
from sim2real.inference import validation
from sim2real.inference.config import InferenceConfig
from sim2real.joint_state import state_from_parts


@dataclass(frozen=True)
class JointStateInput:
    """Plain ``sensor_msgs/JointState`` fields plus local receipt time."""

    names: Sequence[str]
    positions: Sequence[float]
    stamp_ns: int
    received_ns: int


@dataclass(frozen=True)
class ImageInput:
    """Plain ``sensor_msgs/Image`` fields plus local receipt time."""

    data: bytes | bytearray | memoryview | np.ndarray
    encoding: str
    height: int
    width: int
    step: int
    stamp_ns: int
    received_ns: int
    is_bigendian: int = 0
    frame_id: str = ""

    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        *,
        stamp_ns: int,
        received_ns: int,
        frame_id: str = "",
    ) -> ImageInput:
        value = np.ascontiguousarray(array)
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
            raise ValueError("image must be uint8 HxWx3")
        height, width = value.shape[:2]
        return cls(
            data=value.tobytes(),
            encoding=contracts.COLOR_ENCODING,
            height=height,
            width=width,
            step=width * 3,
            stamp_ns=stamp_ns,
            received_ns=received_ns,
            frame_id=frame_id,
        )

    def as_array(self, name: str) -> np.ndarray:
        array = validation.image_from_fields(
            encoding=self.encoding,
            height=self.height,
            width=self.width,
            step=self.step,
            data=self.data,
            is_bigendian=self.is_bigendian,
        )
        return validation.validate_image_array(name, array)


@dataclass(frozen=True)
class ServerObservation:
    """Validated observation ready to be sent to the policy server."""

    state: np.ndarray  # float32 (8,)
    overhead: np.ndarray  # uint8 HxWx3
    wrist: np.ndarray  # uint8 HxWx3
    instruction: str
    stamp_ns: int
    received_ns: int
    sequence: int
    state_stamp_ns: int
    overhead_stamp_ns: int
    wrist_stamp_ns: int

    def to_policy_observation(self) -> PolicyObservation:
        return PolicyObservation(
            state=self.state.astype(np.float64),
            task_instruction=self.instruction,
            stamp_ns=self.stamp_ns,
            images={"overhead": self.overhead, "wrist": self.wrist},
        )


class ObservationAssembler:
    """Validate and assemble one observation; raise ``ValidationError`` on reject."""

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config

    def assemble(
        self,
        *,
        joint_state: JointStateInput | None,
        overhead: ImageInput | None,
        wrist: ImageInput | None,
        instruction: str | None,
        now_ns: int,
        sequence: int,
        last_sequence: int | None = None,
    ) -> ServerObservation:
        cfg = self.config
        text = validation.validate_instruction(instruction, cfg.max_instruction_chars)
        validation.validate_sequence(sequence, last_sequence)

        if joint_state is None:
            raise validation.ValidationError(validation.NO_STATE, "no /joint_states received")
        if overhead is None:
            raise validation.ValidationError(validation.NO_OVERHEAD, "no overhead image")
        if wrist is None:
            raise validation.ValidationError(validation.NO_WRIST, "no wrist image")

        validation.validate_stamp(joint_state.stamp_ns, "state")
        validation.validate_stamp(overhead.stamp_ns, "overhead")
        validation.validate_stamp(wrist.stamp_ns, "wrist")
        validation.validate_stamp(now_ns, "now")
        validation.validate_state_age(now_ns, joint_state.received_ns, cfg.max_state_age_s)
        validation.validate_camera_skew(overhead.stamp_ns, wrist.stamp_ns, cfg.max_camera_skew_s)
        camera_received_ns = min(overhead.received_ns, wrist.received_ns)
        validation.validate_observation_age(now_ns, camera_received_ns, cfg.max_observation_age_s)

        joints, gripper = validation.validate_joint_state(joint_state.names, joint_state.positions)
        state = state_from_parts(joints, gripper).astype(np.float32)
        overhead_array = overhead.as_array("overhead")
        wrist_array = wrist.as_array("wrist")

        stamp_ns = max(joint_state.stamp_ns, overhead.stamp_ns, wrist.stamp_ns)
        return ServerObservation(
            state=np.ascontiguousarray(state),
            overhead=overhead_array,
            wrist=wrist_array,
            instruction=text,
            stamp_ns=int(stamp_ns),
            received_ns=int(now_ns),
            sequence=int(sequence),
            state_stamp_ns=int(joint_state.stamp_ns),
            overhead_stamp_ns=int(overhead.stamp_ns),
            wrist_stamp_ns=int(wrist.stamp_ns),
        )


__all__ = [
    "ImageInput",
    "JointStateInput",
    "ObservationAssembler",
    "ServerObservation",
]
