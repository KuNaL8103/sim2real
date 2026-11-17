"""Chunk validation, JointTrajectory mapping, pass-through, and latency trim."""

from __future__ import annotations

import numpy as np
import pytest

from sim2real import contracts
from sim2real.inference.chunk import (
    ChunkError,
    finger_out_of_range,
    make_chunk,
    select_actions,
    trim_for_latency,
)

STAMP_NS = 7_000_000_000


def _actions(rows: int = 50) -> np.ndarray:
    base = np.arange(8, dtype=np.float64)
    return np.stack([base + index * 0.01 for index in range(rows)])


def test_chunk_selects_first_n_and_uses_k_dt_convention():
    actions = _actions()
    chunk = make_chunk(
        actions,
        stamp_ns=STAMP_NS,
        step_period_s=0.1,
        n_action_steps=10,
        max_chunk_length=100,
    )
    assert chunk.point_count == 10
    assert chunk.joint_names == contracts.CONTROLLED_JOINT_NAMES
    assert chunk.stamp_ns == STAMP_NS
    assert chunk.frame_id == ""
    np.testing.assert_array_equal(chunk.positions, actions[:10])
    assert chunk.time_from_start_ns == tuple(index * 100_000_000 for index in range(10))


def test_select_actions_returns_exactly_n_rows():
    selected = select_actions(_actions(), 3)
    assert selected.shape == (3, 8)
    np.testing.assert_array_equal(selected, _actions()[:3])


@pytest.mark.parametrize(
    "actions, match",
    [
        (np.zeros((4, 7)), "columns"),
        (np.zeros((0, 8)), "at least one row"),
        (np.zeros((4, 8, 1)), "2-D"),
        (np.full((4, 8), np.nan), "NaN"),
        (np.full((101, 8), 0.0), "max_chunk_length"),
    ],
)
def test_invalid_chunks_are_rejected(actions, match):
    with pytest.raises(ChunkError, match=match):
        select_actions(actions, 1, max_chunk_length=100)


def test_chunk_shorter_than_horizon_is_rejected():
    with pytest.raises(ChunkError, match="n_action_steps"):
        make_chunk(
            np.zeros((4, 8)),
            stamp_ns=STAMP_NS,
            step_period_s=0.1,
            n_action_steps=10,
        )


def test_invalid_period_and_stamp_are_rejected():
    with pytest.raises(ChunkError, match="step_period_s"):
        make_chunk(np.zeros((4, 8)), stamp_ns=STAMP_NS, step_period_s=0.0, n_action_steps=1)
    with pytest.raises(ChunkError, match="stamp_ns"):
        make_chunk(np.zeros((4, 8)), stamp_ns=-1, step_period_s=0.1, n_action_steps=1)


def test_pass_through_values_are_unchanged():
    actions = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.01],
            [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 0.02],
        ]
    )
    chunk = make_chunk(actions, stamp_ns=STAMP_NS, step_period_s=0.1, n_action_steps=2)
    np.testing.assert_array_equal(chunk.positions, actions)
    assert chunk.positions[0, 7] == 0.01
    assert chunk.positions[1, 7] == 0.02


def test_trim_boundaries_exact():
    chunk = make_chunk(_actions(10), stamp_ns=STAMP_NS, step_period_s=0.1, n_action_steps=10)
    # Nothing elapsed.
    assert trim_for_latency(chunk, STAMP_NS) is chunk
    # Exactly the first point time: point 0 has passed, point 1 is kept.
    trimmed = trim_for_latency(chunk, STAMP_NS + 100_000_000)
    assert trimmed is not None
    assert trimmed.point_count == 9
    assert trimmed.time_from_start_ns[0] == 100_000_000
    assert trimmed.stamp_ns == STAMP_NS
    # Just after the start: point 0 has passed, point 1 is still kept.
    trimmed = trim_for_latency(chunk, STAMP_NS + 99_999_999)
    assert trimmed is not None
    assert trimmed.point_count == 9
    assert trimmed.time_from_start_ns[0] == 100_000_000
    # Partial.
    partial = trim_for_latency(chunk, STAMP_NS + 250_000_000)
    assert partial is not None
    assert partial.time_from_start_ns[0] == 300_000_000
    assert partial.point_count == 7
    # All elapsed.
    assert trim_for_latency(chunk, STAMP_NS + 1_000_000_000) is None


def test_finger_out_of_range_is_flagged_not_clipped():
    actions = _actions(2)
    actions[0, 7] = -0.01
    actions[1, 7] = 0.06
    chunk = make_chunk(actions, stamp_ns=STAMP_NS, step_period_s=0.1, n_action_steps=2)
    assert finger_out_of_range(chunk) == (0, 1)
    assert chunk.positions[0, 7] == -0.01
    assert chunk.positions[1, 7] == 0.06
