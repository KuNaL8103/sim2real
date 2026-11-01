"""Metric formula and edge-case tests (hardware-free)."""

from __future__ import annotations

import numpy as np
import pytest

from sim2real import contracts
from sim2real.filtering.config import ValidationConfig, load_filter_config
from sim2real.filtering.decision import decide
from sim2real.filtering.metrics import (
    DegenerateTrajectoryError,
    collision_run_count,
    compute_metrics,
    dimensionless_squared_jerk,
    limit_margins,
    path_lengths,
    sparc,
)
from sim2real.filtering.synthetic import make_episode, min_jerk_positions

CONFIG = load_filter_config("configs/filtering/default.yaml")
VALIDATION = ValidationConfig(
    min_steps=20,
    dt_jitter_tolerance_s=0.005,
    min_path_length_m=0.01,
    derivative_method="savgol",
    savgol_window=9,
    savgol_polyorder=5,
)


def _trapezoid_positions(duration_s: float, distance_m: float, dt: float = 0.05):
    count = int(round(duration_s / dt)) + 1
    times = np.linspace(0.0, duration_s, count)
    u = times / duration_s
    speed = np.where(u < 0.2, u / 0.2, np.where(u < 0.8, 1.0, (1.0 - u) / 0.2))
    distance = np.concatenate([[0.0], np.cumsum((speed[1:] + speed[:-1]) * 0.5 * dt)])
    distance *= distance_m / distance[-1]
    positions = np.zeros((count, 3))
    positions[:, 0] = distance
    return times, positions


def test_minimum_jerk_metric_is_about_720():
    # Analytic value is exactly 720 for any distance/duration. Observed with
    # savgol(order 5, window 9) at 20 Hz: 724.5 (T=2), 721.1 (T=4), 738.0 (T=1).
    # The 5% tolerance covers the 20 Hz discretisation and end effects.
    for duration, distance in ((2.0, 0.4), (4.0, 0.8)):
        times, positions = min_jerk_positions(duration, distance)
        value = dimensionless_squared_jerk(positions, times, VALIDATION)
        assert value == pytest.approx(720.0, rel=0.05)


def test_dimensionless_jerk_is_scale_invariant():
    values = []
    for duration, distance in ((2.0, 0.4), (4.0, 0.8), (2.0, 0.2)):
        times, positions = min_jerk_positions(duration, distance)
        values.append(dimensionless_squared_jerk(positions, times, VALIDATION))
    assert max(values) == pytest.approx(min(values), rel=0.05)


def test_jerk_orders_min_jerk_below_trapezoid_below_jerky():
    times, smooth = min_jerk_positions(4.0, 0.5)
    _, trapezoid = _trapezoid_positions(4.0, 0.5)
    jerky = smooth.copy()
    jerky[:, 0] += 0.003 * np.sin(2 * np.pi * 6.0 * times)
    values = [
        dimensionless_squared_jerk(points, times, VALIDATION)
        for points in (smooth, trapezoid, jerky)
    ]
    assert values[0] < values[1] < values[2], values


def test_degenerate_trajectory_is_rejected():
    times = np.arange(40) * 0.05
    with pytest.raises(DegenerateTrajectoryError):
        dimensionless_squared_jerk(np.zeros((40, 3)), times, VALIDATION)
    with pytest.raises(DegenerateTrajectoryError):
        dimensionless_squared_jerk(np.zeros((40, 3)) + [0, 0, 0], np.zeros(40), VALIDATION)

    stationary = make_episode(positions=np.zeros((40, 3)))
    result = decide(stationary, CONFIG)
    assert result.decision == contracts.DECISION_REJECT
    assert [reason.code for reason in result.reasons] == ["DEGENERATE_TRAJECTORY"]


def test_sparc_smooth_better_than_jerky_and_scale_invariant():
    times, smooth = min_jerk_positions(2.0, 0.4)
    jerky = smooth.copy()
    jerky[:, 0] += 0.004 * np.sin(2 * np.pi * 5.0 * times)
    smooth_value, cutoff, rate = sparc(smooth, times, VALIDATION, CONFIG.sparc)
    jerky_value, _, _ = sparc(jerky, times, VALIDATION, CONFIG.sparc)
    assert smooth_value > jerky_value
    assert rate == pytest.approx(20.0)
    assert cutoff < rate / 2.0
    assert CONFIG.sparc.effective_cutoff_hz(rate) == pytest.approx(8.0)

    other_times, other = min_jerk_positions(4.0, 0.8)
    other_value, _, _ = sparc(other, other_times, VALIDATION, CONFIG.sparc)
    assert smooth_value == pytest.approx(other_value, abs=1e-3)


def test_sparc_all_zero_speed_is_explicit():
    times = np.arange(40) * 0.05
    with pytest.raises(DegenerateTrajectoryError):
        sparc(np.zeros((40, 3)), times, VALIDATION, CONFIG.sparc)


def test_path_length_matches_analytic_polyline():
    line = np.column_stack([np.linspace(0, 1, 11), np.zeros(11), np.zeros(11)])
    joint_line = np.zeros((11, 7))
    assert path_lengths(line, joint_line)[0] == pytest.approx(1.0)

    polyline = np.asarray([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=float)
    assert path_lengths(polyline, np.zeros((3, 7)))[0] == pytest.approx(2.0)


def test_limit_margins():
    times, _positions = min_jerk_positions(2.0, 0.4)
    limits = CONFIG.limits
    midpoint = np.asarray(
        [(low + high) / 2.0 for low, high in limits.joint_limits], dtype=np.float64
    )
    joints = np.tile(midpoint, (len(times), 1))
    velocities = np.zeros_like(joints)

    # Mid-range in every joint -> margin 1.
    position_margin, velocity_margin = limit_margins(joints, velocities, limits)
    assert position_margin == pytest.approx(1.0)
    assert velocity_margin == pytest.approx(1.0)

    at_limit = joints.copy()
    at_limit[:, 0] = limits.joint_limits[0][0]
    assert limit_margins(at_limit, velocities, limits)[0] == pytest.approx(0.0)

    beyond = joints.copy()
    beyond[:, 0] = limits.joint_limits[0][0] - 0.1
    assert limit_margins(beyond, velocities, limits)[0] < 0.0

    half_speed = velocities.copy()
    half_speed[:, 0] = limits.velocity_limits[0] / 2.0
    assert limit_margins(joints, half_speed, limits)[1] == pytest.approx(0.5)


def test_collision_runs_count_maximal_runs_not_steps():
    assert collision_run_count([0, 0, 0]) == 0
    assert collision_run_count([1, 1, 1]) == 1
    assert collision_run_count([0, 1, 1, 0, 1, 1, 1, 0]) == 2
    assert collision_run_count([1, 0, 1, 0, 1]) == 3


def test_compute_metrics_uses_executed_state_not_commands():
    times, positions = min_jerk_positions(2.0, 0.4)
    record = make_episode(positions=positions, actions=np.zeros((len(times), 8)))
    metrics = compute_metrics(record, CONFIG)
    assert metrics.path_length_m == pytest.approx(0.4, abs=0.01)
    assert metrics.dimensionless_squared_jerk > 0
    assert metrics.task_success is True


def test_savgol_window_suppresses_noise_without_masking_real_jerk():
    # Empirical trade-off at 20 Hz with 1 mm Gaussian position noise:
    #   window 5/poly 3: clean 697.4, noisy 6.50e6
    #   window 9/poly 5: clean 721.1, noisy 3.15e6
    #   window 17/poly 5: clean 721.1, noisy 5.58e4, but a 6 Hz real oscillation
    #   drops from 5.54e6 (window 9) to 1.29e4 (window 17), i.e. over-smoothed.
    times, smooth = min_jerk_positions(4.0, 0.5)
    noise = np.random.default_rng(0).normal(0, 0.001, smooth.shape)
    small = ValidationConfig(
        min_steps=20, derivative_method="savgol", savgol_window=5, savgol_polyorder=3
    )
    chosen = ValidationConfig(
        min_steps=20, derivative_method="savgol", savgol_window=9, savgol_polyorder=5
    )
    wide = ValidationConfig(
        min_steps=20, derivative_method="savgol", savgol_window=17, savgol_polyorder=5
    )
    noisy_small = dimensionless_squared_jerk(smooth + noise, times, small)
    noisy_chosen = dimensionless_squared_jerk(smooth + noise, times, chosen)
    assert noisy_chosen < noisy_small

    jerky = smooth.copy()
    jerky[:, 0] += 0.003 * np.sin(2 * np.pi * 6.0 * times)
    chosen_jerky = dimensionless_squared_jerk(jerky, times, chosen)
    wide_jerky = dimensionless_squared_jerk(jerky, times, wide)
    assert chosen_jerky > dimensionless_squared_jerk(smooth, times, chosen) * 100
    assert wide_jerky < chosen_jerky / 10
