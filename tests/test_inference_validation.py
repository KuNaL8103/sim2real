"""Every validation rule rejects with the right stable reason code."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from sim2real import contracts
from sim2real.inference import validation
from sim2real.inference.config import InferenceConfig
from sim2real.inference.observation import (
    ImageInput,
    JointStateInput,
    ObservationAssembler,
)


def _inputs():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    stamp_ns = 5_000_000_000
    received_ns = 10_000_000_000
    return {
        "joint_state": JointStateInput(
            names=contracts.CONTROLLED_JOINT_NAMES,
            positions=[0.0] * 7 + [0.02],
            stamp_ns=stamp_ns,
            received_ns=received_ns - 1_000_000,
        ),
        "overhead": ImageInput.from_array(
            image, stamp_ns=stamp_ns, received_ns=received_ns - 1_000_000
        ),
        "wrist": ImageInput.from_array(
            image, stamp_ns=stamp_ns, received_ns=received_ns - 1_000_000
        ),
        "instruction": "pick up the red cube",
        "now_ns": received_ns,
        "sequence": 10,
    }


def _assemble(config: InferenceConfig | None = None, **overrides):
    values = _inputs()
    values.update(overrides)
    assembler = ObservationAssembler(config or InferenceConfig())
    return assembler.assemble(**values)


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"instruction": None}, validation.NO_INSTRUCTION),
        ({"instruction": "   "}, validation.INSTRUCTION_EMPTY),
        ({"instruction": "x" * 513}, validation.INSTRUCTION_TOO_LONG),
        ({"joint_state": None}, validation.NO_STATE),
        ({"overhead": None}, validation.NO_OVERHEAD),
        ({"wrist": None}, validation.NO_WRIST),
    ],
)
def test_missing_or_invalid_instruction_and_inputs(overrides, reason):
    with pytest.raises(validation.ValidationError) as error:
        _assemble(**overrides)
    assert error.value.reason == reason


def test_state_too_old_is_rejected():
    values = _inputs()
    values["joint_state"] = dataclasses.replace(
        values["joint_state"], received_ns=values["now_ns"] - 2_000_000_000
    )
    with pytest.raises(validation.ValidationError) as error:
        _assemble(**values)
    assert error.value.reason == validation.STATE_TOO_OLD


def test_observation_too_old_is_rejected():
    values = _inputs()
    values["overhead"] = dataclasses.replace(
        values["overhead"], received_ns=values["now_ns"] - 2_000_000_000
    )
    with pytest.raises(validation.ValidationError) as error:
        _assemble(**values)
    assert error.value.reason == validation.OBSERVATION_TOO_OLD


def test_camera_skew_too_large_is_rejected():
    values = _inputs()
    values["wrist"] = dataclasses.replace(
        values["wrist"], stamp_ns=values["overhead"].stamp_ns + 100_000_000
    )
    with pytest.raises(validation.ValidationError) as error:
        _assemble(**values)
    assert error.value.reason == validation.CAMERA_SKEW


def test_nan_state_is_rejected():
    values = _inputs()
    positions = list(values["joint_state"].positions)
    positions[0] = float("nan")
    values["joint_state"] = dataclasses.replace(values["joint_state"], positions=positions)
    with pytest.raises(validation.ValidationError) as error:
        _assemble(**values)
    assert error.value.reason == validation.STATE_NONFINITE


def test_wrong_joint_names_are_rejected():
    values = _inputs()
    values["joint_state"] = dataclasses.replace(
        values["joint_state"], names=contracts.CONTROLLED_JOINT_NAMES[:-1]
    )
    with pytest.raises(validation.ValidationError) as error:
        _assemble(**values)
    assert error.value.reason == validation.STATE_NAMES


@pytest.mark.parametrize(
    "changes, reason",
    [
        ({"encoding": "bgr8"}, validation.IMAGE_ENCODING),
        ({"is_bigendian": 1}, validation.IMAGE_ENDIANNESS),
        ({"step": 13}, validation.IMAGE_STEP),
        ({"data": b"\x00\x01\x02"}, validation.IMAGE_LENGTH),
        ({"height": 0}, validation.IMAGE_SHAPE),
    ],
)
def test_malformed_image_fields_are_rejected(changes, reason):
    values = _inputs()
    values["overhead"] = dataclasses.replace(values["overhead"], **changes)
    with pytest.raises(validation.ValidationError) as error:
        _assemble(**values)
    assert error.value.reason == reason


def test_image_array_dtype_is_rejected():
    image = np.zeros((4, 4, 3), dtype=np.float32)
    with pytest.raises(validation.ValidationError) as error:
        validation.validate_image_array("overhead", image)
    assert error.value.reason == validation.IMAGE_DTYPE


def test_sequence_regression_is_rejected():
    with pytest.raises(validation.ValidationError) as error:
        _assemble(last_sequence=10)
    assert error.value.reason == validation.SEQUENCE_REGRESSION


def test_negative_stamp_is_rejected():
    values = _inputs()
    values["overhead"] = dataclasses.replace(values["overhead"], stamp_ns=-1)
    with pytest.raises(validation.ValidationError) as error:
        _assemble(**values)
    assert error.value.reason == validation.NEGATIVE_STAMP


def test_oversized_instruction_uses_configured_bound():
    config = InferenceConfig.from_values({"max_instruction_chars": 4})
    with pytest.raises(validation.ValidationError) as error:
        _assemble(config, instruction="12345")
    assert error.value.reason == validation.INSTRUCTION_TOO_LONG
