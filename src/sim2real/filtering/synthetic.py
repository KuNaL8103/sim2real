"""Small analytic episode factory used by tests and diagnostic scripts.

This module is not used by the ROS adapter at runtime. It builds deterministic
``EpisodeRecord`` values so metric and decision behaviour can be tested without a
simulator.
"""

from __future__ import annotations

import numpy as np

from sim2real import contracts
from sim2real.expert.rollout import EpisodeRecord, StepRecord

MID_RANGE_ARM = (0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, 0.0)


def min_jerk_positions(
    duration_s: float,
    distance_m: float,
    *,
    dt: float = 0.05,
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(times_s, positions)`` for a 1-D minimum-jerk reach."""
    count = int(round(duration_s / dt)) + 1
    times = np.linspace(0.0, duration_s, count)
    u = times / duration_s
    profile = 10 * u**3 - 15 * u**4 + 6 * u**5
    positions = np.zeros((count, 3), dtype=np.float64)
    positions[:, axis] = distance_m * profile
    return times, positions


def make_episode(
    episode_id: str = "ep-0000",
    positions: np.ndarray | None = None,
    *,
    dt: float = 0.05,
    timestamps_s: np.ndarray | None = None,
    outcome: int = contracts.OUTCOME_SUCCESS,
    collisions: list[bool] | np.ndarray | None = None,
    joint_positions: np.ndarray | None = None,
    joint_velocities: np.ndarray | None = None,
    actions: np.ndarray | None = None,
    gripper: float = 0.04,
    is_perturbed: bool = False,
    perturbation_name: str | None = None,
    expected_quality: str = "clean",
    seed: int = 0,
    backend: str = "synthetic",
    run_id: str = "run",
) -> EpisodeRecord:
    """Build a well-formed synthetic episode. Shapes are validated by the filter."""
    if positions is None:
        _, positions = min_jerk_positions(4.0, 0.5, dt=dt)
    points = np.asarray(positions, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(-1, 3)
    count = len(points)
    times = (
        np.asarray(timestamps_s, dtype=np.float64)
        if timestamps_s is not None
        else np.arange(count, dtype=np.float64) * dt
    )
    if joint_positions is None:
        joints = np.tile(np.asarray(MID_RANGE_ARM, dtype=np.float64), (count, 1))
    else:
        joints = np.asarray(joint_positions, dtype=np.float64)
        if joints.ndim == 1:
            joints = np.tile(joints, (count, 1))
    if joint_velocities is None:
        speeds = np.zeros_like(joints)
    else:
        speeds = np.asarray(joint_velocities, dtype=np.float64)
        if speeds.ndim == 1:
            speeds = np.tile(speeds, (count, 1))
    if actions is None:
        commands = np.column_stack([joints, np.full(count, gripper)])
    else:
        commands = np.asarray(actions, dtype=np.float64)
    flags = (
        np.zeros(count, dtype=bool) if collisions is None else np.asarray(collisions, dtype=bool)
    )
    steps: list[StepRecord] = []
    for index in range(count):
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = points[index]
        steps.append(
            StepRecord(
                stamp_ns=int(round(times[index] * 1e9)),
                step_index=index,
                joint_positions=joints[index].copy(),
                joint_velocities=speeds[index].copy(),
                ee_pose=pose,
                action=commands[index].copy(),
                action_space=contracts.ACTION_JOINT_POSITION_TARGET,
                gripper_position=float(commands[index][-1]),
                unintended_collision=bool(flags[index]),
                state="DONE" if index == count - 1 else "TRANSIT",
            )
        )
    return EpisodeRecord(
        episode_id=episode_id,
        run_id=run_id,
        episode_index=0,
        seed=seed,
        randomization_seed=seed,
        backend=backend,
        task_instruction="pick up the red cube and place it in the left zone",
        task_object_id="cube_red",
        task_zone_id="zone_left",
        control_rate_hz=1.0 / dt,
        outcome=outcome,
        failure_reason="" if outcome == contracts.OUTCOME_SUCCESS else "synthetic failure",
        is_perturbed=is_perturbed,
        perturbation_name=perturbation_name,
        expected_quality=expected_quality,
        steps=steps,
    )


__all__ = ["MID_RANGE_ARM", "make_episode", "min_jerk_positions"]
