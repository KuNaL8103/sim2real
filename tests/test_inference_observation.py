"""Observation assembly and parity with the closed-loop runner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sim2real import contracts
from sim2real.evaluation.episode_runner import _policy_observation
from sim2real.inference.config import InferenceConfig
from sim2real.inference.observation import (
    ImageInput,
    JointStateInput,
    ObservationAssembler,
)
from sim2real.sim.backend import Observation, RobotState

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "lerobot" / "train_v1"


def _images() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    return {
        "overhead": rng.integers(0, 256, size=(12, 10, 3), dtype=np.uint8),
        "wrist": rng.integers(0, 256, size=(12, 10, 3), dtype=np.uint8),
    }


def _joint_state(stamp_ns: int, received_ns: int) -> JointStateInput:
    positions = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, 0.03]
    return JointStateInput(
        names=contracts.CONTROLLED_JOINT_NAMES,
        positions=positions,
        stamp_ns=stamp_ns,
        received_ns=received_ns,
    )


def test_observation_layout_and_finger_value():
    images = _images()
    stamp_ns = 5_000_000_000
    now_ns = 10_000_000_000
    assembler = ObservationAssembler(InferenceConfig())
    observation = assembler.assemble(
        joint_state=_joint_state(stamp_ns, now_ns - 1_000_000),
        overhead=ImageInput.from_array(
            images["overhead"], stamp_ns=stamp_ns, received_ns=now_ns - 1_000_000
        ),
        wrist=ImageInput.from_array(
            images["wrist"], stamp_ns=stamp_ns, received_ns=now_ns - 1_000_000
        ),
        instruction="  pick up the red cube  ",
        now_ns=now_ns,
        sequence=1,
    )
    assert observation.state.dtype == np.float32
    assert observation.state.shape == (8,)
    np.testing.assert_allclose(
        observation.state, [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, 0.03], rtol=0, atol=1e-7
    )
    # The gripper is the single panda_finger_joint1 value, not a sum of fingers.
    assert observation.state[7] == pytest.approx(0.03)
    assert observation.instruction == "pick up the red cube"
    assert observation.overhead.tobytes() == images["overhead"].tobytes()
    assert observation.wrist.tobytes() == images["wrist"].tobytes()
    assert observation.stamp_ns == stamp_ns


def test_runner_parity_images_bytes_and_state_within_1e6():
    images = _images()
    joint_positions = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7])
    gripper = 0.03
    stamp_ns = 5_000_000_000
    now_ns = 10_000_000_000
    backend_observation = Observation(
        robot_state=RobotState(
            joint_positions=joint_positions,
            joint_velocities=np.zeros(7),
            gripper_position=gripper,
            ee_pose=np.eye(4),
            stamp_ns=stamp_ns,
        ),
        cameras={},
        object_poses={},
    )
    runner = _policy_observation(backend_observation, images, "pick up the red cube")
    assembler = ObservationAssembler(InferenceConfig())
    node = assembler.assemble(
        joint_state=_joint_state(stamp_ns, now_ns - 1_000_000),
        overhead=ImageInput.from_array(
            images["overhead"], stamp_ns=stamp_ns, received_ns=now_ns - 1_000_000
        ),
        wrist=ImageInput.from_array(
            images["wrist"], stamp_ns=stamp_ns, received_ns=now_ns - 1_000_000
        ),
        instruction="pick up the red cube",
        now_ns=now_ns,
        sequence=1,
    ).to_policy_observation()
    assert node.images["overhead"].tobytes() == runner.images["overhead"].tobytes()
    assert node.images["wrist"].tobytes() == runner.images["wrist"].tobytes()
    assert node.state.shape == runner.state.shape == (8,)
    np.testing.assert_allclose(node.state, runner.state, rtol=0, atol=1e-6)
    assert node.stamp_ns == runner.stamp_ns
    assert node.task_instruction == runner.task_instruction


def test_real_dataset_frame_parity_when_available():
    """Optional integration: use episode 0 frames if decoded images exist."""
    image_dir = DATASET / "images" / "observation.images.overhead"
    files = sorted(image_dir.glob("*.png")) if image_dir.is_dir() else []
    if not files:
        pytest.skip(
            "data/lerobot/train_v1 has no decoded overhead PNG frames "
            "(the checked-out copy stores metadata only)"
        )
    parquet = DATASET / "data" / "chunk-000" / "file-000.parquet"
    if not parquet.is_file():
        pytest.skip("data/lerobot/train_v1 has no data parquet")
    try:
        import pyarrow.parquet as pq
    except ImportError:
        pytest.skip("pyarrow is not installed in the core environment")

    from sim2real.expert.images import read_png

    table = pq.read_table(parquet, columns=["episode_index", "observation.state"])
    states = table.column("observation.state").to_pylist()
    episodes = table.column("episode_index").to_pylist()
    index = next((i for i, episode in enumerate(episodes) if episode == 0), None)
    if index is None:
        pytest.skip("episode 0 not found in the dataset parquet")
    overhead = read_png(files[0])
    wrist_dir = DATASET / "images" / "observation.images.wrist"
    wrist_files = sorted(wrist_dir.glob("*.png")) if wrist_dir.is_dir() else []
    if not wrist_files:
        pytest.skip("episode 0 wrist frames are absent")
    wrist = read_png(wrist_files[0])
    state = np.asarray(states[index][:8], dtype=np.float64)
    stamp_ns = 5_000_000_000
    now_ns = 10_000_000_000
    assembler = ObservationAssembler(InferenceConfig())
    observation = assembler.assemble(
        joint_state=JointStateInput(
            names=contracts.CONTROLLED_JOINT_NAMES,
            positions=[*state[:7], state[7]],
            stamp_ns=stamp_ns,
            received_ns=now_ns - 1_000_000,
        ),
        overhead=ImageInput.from_array(overhead, stamp_ns=stamp_ns, received_ns=now_ns - 1_000_000),
        wrist=ImageInput.from_array(wrist, stamp_ns=stamp_ns, received_ns=now_ns - 1_000_000),
        instruction="pick up the red cube and place it in the left zone",
        now_ns=now_ns,
        sequence=1,
    )
    assert observation.overhead.tobytes() == overhead.tobytes()
    assert observation.wrist.tobytes() == wrist.tobytes()
    np.testing.assert_allclose(observation.state, state.astype(np.float32), rtol=0, atol=1e-6)
