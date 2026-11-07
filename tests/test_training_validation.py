"""Offline L1 validation on known predictions, masks, and baselines."""

from __future__ import annotations

import numpy as np
import pytest

from sim2real.training.validation import (
    ValidationError,
    evaluate_l1,
    hold_last_state_baseline,
    mean_action_baseline,
    valid_mask_from_padding,
)


def test_action_is_pad_true_means_padded_not_valid():
    """Regression guard for the Task 8 inverted validation mask."""
    padded = np.array([[True, False, False]])
    valid = valid_mask_from_padding(padded)
    assert valid.tolist() == [[False, True, True]]

    predictions = np.zeros((1, 3, 1))
    targets = np.array([[[9.0], [1.0], [3.0]]])
    report = evaluate_l1(predictions, targets, valid)
    assert report.n_valid_steps == 2
    assert np.isnan(report.per_horizon[0])  # padded position is excluded
    assert report.per_horizon[1] == pytest.approx(1.0)
    assert report.per_horizon[2] == pytest.approx(3.0)


def test_known_l1_excludes_padded_positions():
    predictions = np.array([[[0.0, 0.0], [2.0, 2.0]]])
    targets = np.array([[[1.0, 0.0], [0.0, 0.0]]])
    mask = np.array([[True, False]])
    report = evaluate_l1(
        predictions,
        targets,
        mask,
        action_std=np.array([1.0, 2.0]),
        last_state=np.array([1.0, 0.0]),
        mean_action=np.array([0.0, 0.0]),
        horizon_seconds=5.0,
    )
    assert report.n_samples == 1
    assert report.n_valid_steps == 1
    np.testing.assert_allclose(report.per_dimension, [1.0, 0.0])
    assert report.per_horizon[0] == pytest.approx(0.5)
    assert np.isnan(report.per_horizon[1])  # every sample is padded at k=1
    np.testing.assert_allclose(report.normalized_per_dimension, [1.0, 0.0])
    assert report.horizon_seconds == 5.0

    hold = report.baselines["hold_last_state"]
    np.testing.assert_allclose(hold["per_dimension"], [0.0, 0.0])
    mean = report.baselines["mean_action"]
    np.testing.assert_allclose(mean["per_dimension"], [1.0, 0.0])
    assert mean["per_horizon"][0] == pytest.approx(0.5)


def test_baselines_are_computed_with_the_same_mask():
    states = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    hold = hold_last_state_baseline(states, chunk_size=3, action_dim=2, batch=2)
    assert hold.shape == (2, 3, 2)
    np.testing.assert_allclose(hold[0, 0], [1.0, 2.0])
    np.testing.assert_allclose(hold[1, 2], [4.0, 5.0])
    mean = mean_action_baseline(np.array([0.5, -0.5]), chunk_size=2, batch=3)
    assert mean.shape == (3, 2, 2)
    np.testing.assert_allclose(mean[:, 1, :], [[0.5, -0.5]] * 3)


def test_empty_validation_set_is_handled():
    report = evaluate_l1(
        np.zeros((0, 4, 3)),
        np.zeros((0, 4, 3)),
        np.zeros((0, 4), dtype=bool),
        horizon_seconds=0.4,
    )
    assert report.n_samples == 0
    assert report.n_valid_steps == 0
    assert report.per_dimension is None
    assert report.per_horizon is None
    assert report.to_dict()["per_dimension_l1"] is None


def test_different_chunk_lengths_are_never_compared():
    with pytest.raises(ValidationError, match="same shape"):
        evaluate_l1(np.zeros((1, 4, 3)), np.zeros((1, 5, 3)))


def test_mask_shape_is_checked():
    with pytest.raises(ValidationError, match="mask"):
        evaluate_l1(np.zeros((1, 4, 3)), np.zeros((1, 4, 3)), np.ones((1, 3), dtype=bool))


def test_known_mask_keeps_only_requested_horizons():
    predictions = np.zeros((2, 3, 1))
    targets = np.array([[[1.0], [2.0], [3.0]], [[1.0], [2.0], [3.0]]])
    mask = np.array([[True, False, True], [True, False, False]])
    report = evaluate_l1(predictions, targets, mask)
    # k=0 has two samples, k=1 has none, k=2 has one.
    assert report.n_valid_steps == 3
    assert report.per_horizon[0] == pytest.approx(1.0)
    assert np.isnan(report.per_horizon[1])
    assert report.per_horizon[2] == pytest.approx(3.0)
    assert report.per_dimension[0] == pytest.approx((1.0 + 3.0 + 1.0) / 3.0)
