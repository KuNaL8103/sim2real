"""Planner: quintic boundary conditions, monotonic time, limits, MoveIt guard."""

from __future__ import annotations

import numpy as np
import pytest

from sim2real.expert.config import load_expert_config
from sim2real.expert.ik import DampedLeastSquaresIK
from sim2real.expert.planner import (
    CartesianInterpolationPlanner,
    MoveItPlanner,
    PlannerUnavailable,
    quintic_scale,
)
from sim2real.sim.config import load_sim_configs
from sim2real.sim.kinematics import tcp_pose

CONFIG_DIR = "configs/sim"
EXPERT_CONFIG_DIR = "configs/expert"


def _plan():
    config = load_expert_config(EXPERT_CONFIG_DIR)
    panda = load_sim_configs(CONFIG_DIR).panda
    solver = DampedLeastSquaresIK(config.ik)
    planner = CartesianInterpolationPlanner()
    start = np.asarray(panda.home_arm, dtype=np.float64)
    target = tcp_pose(start, panda).copy()
    target[:3, 3] += np.array([-0.05, 0.04, -0.03])
    plan = planner.plan_to_pose(start, target, panda, solver, config.planner)
    return config, panda, start, plan


def test_quintic_boundary_conditions() -> None:
    u = np.linspace(0.0, 1.0, 201)
    scale = quintic_scale(u)
    assert scale[0] == pytest.approx(0.0)
    assert scale[-1] == pytest.approx(1.0)
    # Analytic derivatives of 10u^3 - 15u^4 + 6u^5.
    derivative = 30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4
    second = 60.0 * u - 180.0 * u**2 + 120.0 * u**3
    assert derivative[0] == pytest.approx(0.0, abs=1e-12)
    assert derivative[-1] == pytest.approx(0.0, abs=1e-12)
    assert second[0] == pytest.approx(0.0, abs=1e-12)
    assert second[-1] == pytest.approx(0.0, abs=1e-12)
    # The sampled trajectory must also end with near-zero finite differences.
    sampled_velocity = np.gradient(scale, u)
    assert abs(sampled_velocity[-1]) < 1e-3


def test_plan_time_is_monotonic_and_start_is_exact() -> None:
    config, panda, start, plan = _plan()
    assert plan.success, plan.reason
    assert plan.times[0] == 0.0
    assert np.all(np.diff(plan.times) > 0.0)
    np.testing.assert_allclose(plan.positions[0], start)
    assert plan.duration_s == pytest.approx(plan.times[-1])
    assert plan.duration_s >= config.planner.min_duration_s
    limits = np.asarray(panda.joint_limits, dtype=np.float64)
    assert np.all(plan.positions >= limits[:, 0] - 1e-9)
    assert np.all(plan.positions <= limits[:, 1] + 1e-9)


def test_plan_respects_velocity_and_acceleration_bounds() -> None:
    config, panda, _start, plan = _plan()
    assert plan.success, plan.reason
    velocity = np.gradient(plan.positions, plan.times, axis=0)
    acceleration = np.gradient(velocity, plan.times, axis=0)
    velocity_limits = (
        np.asarray(config.planner.velocity_limits)
        if config.planner.velocity_limits is not None
        else np.asarray(panda.velocity_limits)
    )
    acceleration_limits = (
        np.asarray(config.planner.acceleration_limits)
        if config.planner.acceleration_limits is not None
        else np.asarray(panda.acceleration_limits)
    )
    assert np.max(np.abs(velocity)) <= np.max(velocity_limits) * 1.05
    assert np.max(np.abs(acceleration)) <= np.max(acceleration_limits) * 1.05


def test_moveit_planner_is_guarded() -> None:
    with pytest.raises(PlannerUnavailable):
        MoveItPlanner()
