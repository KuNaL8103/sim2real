"""Action chunk encoding, padding, masks, and horizon bookkeeping."""

from __future__ import annotations

import numpy as np
import pytest

from sim2real.training.actions import (
    ActionChunkError,
    chunk_offsets,
    decode_chunk,
    encode_chunk,
    horizon_seconds,
)


def episode(steps: int = 10, dim: int = 8) -> np.ndarray:
    return np.arange(steps * dim, dtype=np.float64).reshape(steps, dim) / 10.0


def test_chunk_pads_past_episode_end_and_marks_mask():
    actions = episode(10)
    chunk = encode_chunk(
        actions,
        actions[3],
        start=3,
        chunk_size=15,
        fps=10.0,
        max_action_dim=32,
        max_state_dim=32,
    )
    assert chunk.actions.shape == (15, 32)
    assert chunk.state.shape == (32,)
    assert chunk.action_is_pad.sum() == 8
    assert not chunk.action_is_pad[:7].any()
    assert chunk.action_is_pad[7:].all()
    # Positions 10..14 repeat the last recorded action.
    np.testing.assert_allclose(chunk.actions[7, :8], actions[-1].astype(np.float32))
    assert np.all(chunk.actions[:, 8:] == 0)
    assert chunk.horizon_seconds == pytest.approx(1.5)


def test_short_episode_is_entirely_padded_after_its_last_step():
    actions = episode(3)
    chunk = encode_chunk(
        actions,
        actions[0],
        start=0,
        chunk_size=5,
        fps=5.0,
        max_action_dim=8,
        max_state_dim=8,
    )
    assert chunk.valid_steps == 3
    assert chunk.action_is_pad.tolist() == [False, False, False, True, True]
    np.testing.assert_allclose(chunk.actions[2, :8], actions[-1].astype(np.float32))


def test_decode_recovers_native_dimension_exactly():
    actions = episode(6, dim=3)
    chunk = encode_chunk(
        actions,
        actions[0],
        start=1,
        chunk_size=4,
        fps=10.0,
        max_action_dim=32,
        max_state_dim=32,
    )
    decoded = decode_chunk(chunk)
    assert decoded.shape == (4, 3)
    np.testing.assert_allclose(decoded[0], actions[1].astype(np.float32))
    np.testing.assert_allclose(decoded[3], actions[4].astype(np.float32))


def test_horizon_seconds_tracks_fps_not_episode_length():
    slow = encode_chunk(
        episode(60),
        episode(1)[0],
        start=0,
        chunk_size=50,
        fps=5.0,
        max_action_dim=8,
        max_state_dim=8,
    )
    fast = encode_chunk(
        episode(60),
        episode(1)[0],
        start=0,
        chunk_size=50,
        fps=10.0,
        max_action_dim=8,
        max_state_dim=8,
    )
    assert slow.chunk_size == fast.chunk_size == 50
    assert slow.horizon_seconds == pytest.approx(10.0)
    assert fast.horizon_seconds == pytest.approx(5.0)
    assert horizon_seconds(25, 10.0) == pytest.approx(2.5)
    np.testing.assert_allclose(chunk_offsets(fast)[:3], [0.0, 0.1, 0.2])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start": 99, "chunk_size": 5, "fps": 10.0, "max_action_dim": 8, "max_state_dim": 8},
        {"start": 0, "chunk_size": 0, "fps": 10.0, "max_action_dim": 8, "max_state_dim": 8},
        {"start": 0, "chunk_size": 5, "fps": 0.0, "max_action_dim": 8, "max_state_dim": 8},
        {"start": 0, "chunk_size": 5, "fps": 10.0, "max_action_dim": 4, "max_state_dim": 8},
    ],
)
def test_invalid_chunk_requests_raise(kwargs):
    with pytest.raises(ActionChunkError):
        encode_chunk(episode(10), episode(10)[0], **kwargs)


def test_non_finite_actions_rejected():
    actions = episode(10)
    actions[2, 3] = np.nan
    with pytest.raises(ActionChunkError, match="finite"):
        encode_chunk(
            actions,
            actions[0],
            start=0,
            chunk_size=5,
            fps=10.0,
            max_action_dim=8,
            max_state_dim=8,
        )
