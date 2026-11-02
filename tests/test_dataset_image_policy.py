"""One-renderer, imageless fallback, fidelity, and mixture policy."""

from __future__ import annotations

import pytest

from sim2real.dataset_builder import DataValidationError, validate_image_policy
from tests.fixtures.dataset_builder import make_episode


def test_mixed_renderers_are_refused():
    episodes = [
        make_episode(episode_id="a", renderer="llvmpipe"),
        make_episode(episode_id="b", renderer="d3d12"),
    ]
    with pytest.raises(DataValidationError, match="mixes renderers"):
        validate_image_policy(episodes, renderer="llvmpipe", decimation=2, fps=10.0)


def test_imageless_error_and_explicit_fallback():
    episode = make_episode(with_images=False)
    with pytest.raises(DataValidationError, match="no images"):
        validate_image_policy([episode], renderer="llvmpipe", decimation=2, fps=10.0)
    summary = validate_image_policy(
        [episode], renderer="llvmpipe", decimation=2, fps=10.0, allow_imageless=True
    )
    assert summary["mode"] == "none"
    assert summary["imageless_episodes"] == 1


def test_replay_fidelity_tolerance_rejects_episode():
    episode = make_episode(metadata={"replay_max_joint_error_rad": 0.01})
    with pytest.raises(DataValidationError, match="replay joint error"):
        validate_image_policy([episode], renderer="llvmpipe", decimation=2, fps=10.0)
    episode = make_episode(metadata={"replay_max_object_error_m": 0.01})
    with pytest.raises(DataValidationError, match="replay object error"):
        validate_image_policy([episode], renderer="llvmpipe", decimation=2, fps=10.0)


def test_policy_records_renderer_decimation_fps_cameras():
    summary = validate_image_policy([make_episode()], renderer="llvmpipe", decimation=2, fps=10.0)
    assert summary["renderer"] == "llvmpipe"
    assert summary["decimation"] == 2
    assert summary["fps"] == 10.0
    assert summary["cameras"] == ["overhead", "wrist"]
