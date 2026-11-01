"""In-process rollout runner for the scripted pick-and-place expert.

The runner owns the episode lifecycle: deterministic reset through the same
Task 3 path (``reset_environment``), planning, execution, failure handling, and
recording. It never drops or crashes on a failed episode.

Rendering is **off** by default: this module calls ``backend.step`` and
``backend.get_observation`` only. If a ``RenderCoordinator`` exists the caller is
responsible for quiescing it around the reset; the in-process runner creates
none, so no render can overlap a reset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from sim2real import contracts
from sim2real.expert.config import ExpertConfig, load_expert_config
from sim2real.expert.ik import DampedLeastSquaresIK, IKSolver
from sim2real.expert.instruction import TaskSpec
from sim2real.expert.perturbations import (
    Perturbation,
    apply_joint_jitter,
    bad_grasp_offset,
    detour_offset,
    hesitation_pause_steps,
    low_sweep_clearance,
    overspeed_scale,
    perturbation_phase,
)
from sim2real.expert.planner import (
    CartesianInterpolationPlanner,
    JointPlan,
    Planner,
)
from sim2real.expert.state_machine import (
    ExpertState,
    PickPlaceStateMachine,
)
from sim2real.sim.camera import make_pose
from sim2real.sim.kinematics import hand_pose, tcp_pose
from sim2real.sim.reset_handler import reset_environment

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class StepRecord:
    stamp_ns: int
    step_index: int
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    ee_pose: np.ndarray
    action: np.ndarray
    action_space: int
    gripper_position: float
    unintended_collision: bool
    state: str

    def to_dict(self) -> dict[str, object]:
        return {
            "stamp_ns": int(self.stamp_ns),
            "step_index": int(self.step_index),
            "joint_positions": [float(v) for v in self.joint_positions],
            "joint_velocities": [float(v) for v in self.joint_velocities],
            "ee_pose": [[float(v) for v in row] for row in self.ee_pose],
            "action": [float(v) for v in self.action],
            "action_space": int(self.action_space),
            "gripper_position": float(self.gripper_position),
            "unintended_collision": bool(self.unintended_collision),
            "state": self.state,
        }


@dataclass
class EpisodeRecord:
    episode_id: str
    run_id: str
    episode_index: int
    seed: int
    randomization_seed: int
    backend: str
    task_instruction: str
    task_object_id: str
    task_zone_id: str
    control_rate_hz: float
    outcome: int
    failure_reason: str
    is_perturbed: bool
    perturbation_name: str | None
    expected_quality: str
    steps: list[StepRecord] = field(default_factory=list)
    transitions: list[dict[str, object]] = field(default_factory=list)
    randomization_report: dict[str, object] = field(default_factory=dict)
    success_metrics: dict[str, object] = field(default_factory=dict)
    started_at: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def start_stamp_ns(self) -> int:
        return int(self.steps[0].stamp_ns) if self.steps else 0

    @property
    def succeeded(self) -> bool:
        return self.outcome == contracts.OUTCOME_SUCCESS

    def metadata_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": contracts.SCHEMA_VERSION,
            "run_id": self.run_id,
            "episode_index": int(self.episode_index),
            "seed": int(self.seed),
            "randomization_seed": int(self.randomization_seed),
            "expected_quality": self.expected_quality,
            "perturbation_name": self.perturbation_name,
            "task_object_id": self.task_object_id,
            "task_zone_id": self.task_zone_id,
            "success_metrics": self.success_metrics,
            "transitions": self.transitions,
            "randomization": self.randomization_report,
            "num_steps": len(self.steps),
        }
        payload.update(self.metadata)
        return payload


def derive_episode_seed(seed: int, episode_index: int) -> int:
    """Per-episode randomization seed derived from ``(run seed, episode index)``."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(episode_index, bool) or not isinstance(episode_index, int) or episode_index < 0:
        raise ValueError("episode_index must be a non-negative integer")
    state = np.random.SeedSequence([seed, episode_index, 0x52414E44]).generate_state(
        1, dtype=np.uint32
    )
    return int(state[0])


def _unintended_collision(collision: Any, target_object: str) -> bool:
    details = getattr(collision, "details", ()) or ()
    count = int(getattr(collision, "num_unintended_contacts", 0) or 0)
    if count == 0 and not details:
        return False
    for detail in details:
        parts = str(detail).split("<->")
        if target_object in parts and any("finger" in part for part in parts):
            continue
        return True
    return count > 0


def _grasp_confirmed(backend: Any, target_object: str, observation: Any) -> bool:
    held = getattr(backend, "held_object", None)
    if held is not None:
        return held == target_object
    collision = backend.collision_report()
    for detail in getattr(collision, "details", ()) or ():
        parts = str(detail).split("<->")
        if target_object in parts and any("finger" in part for part in parts):
            return True
    return False


def _object_position(observation: Any, name: str) -> np.ndarray:
    return np.asarray(observation.object_poses[name][:3, 3], dtype=np.float64)


def _success_check(
    backend: Any,
    task: TaskSpec,
    observation: Any,
    object_history: list[np.ndarray],
    config: ExpertConfig,
) -> tuple[bool, str, dict[str, object]]:
    scene = backend.configs.scene
    zone = scene.zones[task.zone_id]
    cube = scene.cubes[task.object_id]
    position = _object_position(observation, task.object_id)
    zone_position = np.asarray(zone.position, dtype=np.float64)
    xy_error = float(np.linalg.norm(position[:2] - zone_position[:2]))
    allowed_radius = zone.radius * config.success.success_zone_scale
    table_top = scene.table_position[2] + scene.table_size[2] / 2.0
    expected_z = table_top + cube.size / 2.0
    height_error = float(abs(position[2] - expected_z))
    gripper_open = float(observation.robot_state.gripper_position) >= (
        config.motion.gripper_open - 0.005
    )
    history = object_history[-config.success.stable_steps :]
    stable = True
    stability_range = 0.0
    if len(history) >= config.success.stable_steps:
        stacked = np.asarray(history)
        stability_range = float(np.max(stacked.max(axis=0) - stacked.min(axis=0)))
        stable = stability_range <= config.success.stability_tolerance_m
    else:
        stable = False

    reasons: list[str] = []
    if xy_error > allowed_radius + config.success.zone_radius_tolerance_m:
        reasons.append(
            f"cube is {xy_error:.4f} m from zone centre (allowed {allowed_radius:.4f} m)"
        )
    if height_error > config.success.rest_height_tolerance_m:
        reasons.append(
            f"cube rest height error {height_error:.4f} m > "
            f"{config.success.rest_height_tolerance_m:.4f} m"
        )
    if config.success.require_gripper_open and not gripper_open:
        reasons.append("gripper is still closed")
    if not stable:
        reasons.append(
            f"cube not stable for {config.success.stable_steps} steps "
            f"(range {stability_range:.5f} m)"
        )

    metrics: dict[str, object] = {
        "xy_error_m": xy_error,
        "allowed_radius_m": float(allowed_radius),
        "height_error_m": height_error,
        "gripper_open": bool(gripper_open),
        "stable_steps": int(min(len(history), config.success.stable_steps)),
        "stability_range_m": stability_range,
        "final_object_position": [float(v) for v in position],
        "zone_position": [float(v) for v in zone_position],
    }
    return (not reasons, "; ".join(reasons), metrics)


class _EpisodeRunner:
    def __init__(
        self,
        backend: Any,
        task: TaskSpec,
        record: EpisodeRecord,
        config: ExpertConfig,
        ik_solver: IKSolver,
        planner: Planner,
        perturbation: Perturbation | None,
        max_steps: int,
    ) -> None:
        self.backend = backend
        self.task = task
        self.record = record
        self.config = config
        self.ik_solver = ik_solver
        self.planner = planner
        self.perturbation = perturbation
        self.max_steps = max_steps
        self.panda = backend.configs.panda
        self.machine = PickPlaceStateMachine.from_config(config)
        self.dt = 1.0 / float(backend.control_rate_hz)
        self.step_index = 0
        self.object_history: list[np.ndarray] = []
        self._plan: JointPlan | None = None
        self._plan_cursor = 0
        self._last_command: np.ndarray | None = None
        self._hesitation_remaining = 0
        self._jitter_phase = perturbation_phase(record.seed, record.episode_index)
        self._success = False
        self._success_reason = ""
        self._success_metrics: dict[str, object] = {}
        self._initial_cube_pose: np.ndarray | None = None
        self._targets_map: dict[str, np.ndarray] | None = None

    # -- geometry ---------------------------------------------------------
    def _compute_targets(self, observation: Any) -> dict[str, np.ndarray]:
        scene = self.backend.configs.scene
        cube = observation.object_poses[self.task.object_id]
        self._initial_cube_pose = np.asarray(cube, dtype=np.float64).copy()
        zone = scene.zones[self.task.zone_id]
        cube_spec = scene.cubes[self.task.object_id]
        table_top = scene.table_position[2] + scene.table_size[2] / 2.0
        motion = self.config.motion

        orientation = hand_pose(self.panda.home_arm, self.panda)[:3, :3]
        grasp_position = np.asarray(cube[:3, 3], dtype=np.float64).copy()
        grasp_position[2] += motion.grasp_depth_m
        if self.perturbation and self.perturbation.name == "bad_grasp":
            grasp_position += bad_grasp_offset(self.perturbation)

        def pose(position: np.ndarray) -> np.ndarray:
            return make_pose(np.asarray(position, dtype=np.float64), orientation)

        clearance = motion.transit_clearance_m
        if self.perturbation and self.perturbation.name == "low_sweep":
            clearance = min(clearance, low_sweep_clearance(self.perturbation))
        place_z = table_top + cube_spec.size / 2.0 + motion.place_depth_m
        zone_xy = np.asarray(zone.position, dtype=np.float64)
        self._targets_map = {
            "APPROACH_PREGRASP": pose(grasp_position + [0.0, 0.0, motion.pregrasp_height_m]),
            "DESCEND": pose(grasp_position),
            "LIFT": pose(grasp_position + [0.0, 0.0, motion.lift_height_m]),
            "TRANSIT": pose([zone_xy[0], zone_xy[1], grasp_position[2] + clearance]),
            "PLACE_DESCEND": pose([zone_xy[0], zone_xy[1], place_z]),
            "RETREAT": pose([zone_xy[0], zone_xy[1], place_z + motion.retreat_height_m]),
        }
        return self._targets_map

    def _targets(self) -> dict[str, np.ndarray]:
        assert self._targets_map is not None
        return self._targets_map

    def _plan_scale(self) -> float:
        if self.perturbation and self.perturbation.name == "overspeed":
            return overspeed_scale(self.perturbation)
        return 1.0

    def _make_plan(self, observation: Any) -> None:
        targets = self._targets()
        target = targets[self.machine.state.value]
        q = np.asarray(observation.robot_state.joint_positions, dtype=np.float64)
        planner_config = self.config.planner
        if (
            self.machine.state is ExpertState.TRANSIT
            and self.perturbation
            and self.perturbation.name == "detour"
        ):
            offset = detour_offset(self.perturbation)
            detour = target.copy()
            detour[:3, 3] += offset
            self._plan = self.planner.plan_waypoints(  # type: ignore[attr-defined]
                q,
                (detour, target),
                self.panda,
                self.ik_solver,
                planner_config,
                duration_scale=self._plan_scale(),
            )
            return
        self._plan = self.planner.plan_to_pose(
            q,
            target,
            self.panda,
            self.ik_solver,
            planner_config,
            duration_scale=self._plan_scale(),
        )

    # -- execution --------------------------------------------------------
    def _gripper_command(self) -> float:
        motion = self.config.motion
        if self.machine.state in {ExpertState.GRASP, ExpertState.LIFT, ExpertState.TRANSIT}:
            if (
                self.perturbation
                and self.perturbation.name == "bad_grasp"
                and self.machine.state is ExpertState.LIFT
            ):
                return motion.gripper_open
            return motion.gripper_closed
        if self.machine.state in {ExpertState.PLACE_DESCEND}:
            return motion.gripper_closed
        return motion.gripper_open

    def _record_step(self, observation: Any, action: np.ndarray, collision: Any) -> None:
        self.object_history.append(_object_position(observation, self.task.object_id))
        self.record.steps.append(
            StepRecord(
                stamp_ns=int(observation.stamp_ns),
                step_index=self.step_index,
                joint_positions=np.asarray(
                    observation.robot_state.joint_positions, dtype=np.float64
                ).copy(),
                joint_velocities=np.asarray(
                    observation.robot_state.joint_velocities, dtype=np.float64
                ).copy(),
                ee_pose=np.asarray(observation.robot_state.ee_pose, dtype=np.float64).copy(),
                action=np.asarray(action, dtype=np.float64).copy(),
                action_space=contracts.ACTION_JOINT_POSITION_TARGET,
                gripper_position=float(observation.robot_state.gripper_position),
                unintended_collision=_unintended_collision(collision, self.task.object_id),
                state=self.machine.state.value,
            )
        )

    def _step(self, command: np.ndarray, gripper: float) -> Any:
        result = self.backend.step(command, gripper)
        observation = result.observation
        self._record_step(observation, np.append(command, gripper), result.collision)
        self.step_index += 1
        self.machine.tick(self.dt)
        return observation

    def _apply_hesitation(self, command: np.ndarray, gripper: float) -> Any | None:
        if self.perturbation is None or self.perturbation.name != "hesitation":
            return None
        if self._hesitation_remaining == 0:
            self._hesitation_remaining = hesitation_pause_steps(
                self.perturbation, self.record.seed, self.record.episode_index
            )
        held = self._last_command if self._last_command is not None else command
        observation = None
        while self._hesitation_remaining > 0 and not self.machine.terminal:
            observation = self._step(held.copy(), gripper)
            self._hesitation_remaining -= 1
            if self.step_index >= self.max_steps:
                break
        return observation

    def _pose_error(self, observation: Any, target_pose: np.ndarray) -> float:
        current = tcp_pose(observation.robot_state.joint_positions, self.panda)
        return float(np.linalg.norm(current[:3, 3] - target_pose[:3, 3]))

    def _advance_motion(self, observation: Any, target_pose: np.ndarray) -> Any:
        assert self._plan is not None
        if self.machine.state is ExpertState.IDLE:
            return observation
        if self.step_index >= self.max_steps:
            return observation
        command = self._plan.positions[self._plan_cursor]
        is_final = self._plan_cursor >= len(self._plan.positions) - 1
        if self.perturbation and self.perturbation.name == "jitter" and not is_final:
            command = apply_joint_jitter(
                command,
                self.step_index,
                self.dt,
                self.perturbation,
                self._jitter_phase,
            )
        gripper = self._gripper_command()
        observation = self._step(command.copy(), gripper)
        self._last_command = command.copy()
        self._plan_cursor += 1
        if self._plan_cursor >= len(self._plan.positions):
            final_command = self._plan.positions[-1]
            tolerance = self.config.motion.reached_tolerance_m
            for _ in range(self.config.motion.settle_steps):
                if self.machine.terminal or self.step_index >= self.max_steps:
                    break
                observation = self._step(final_command.copy(), gripper)
                if self._pose_error(observation, target_pose) <= tolerance:
                    break
            paused = self._apply_hesitation(command, gripper)
            if paused is not None:
                observation = paused
            if self.machine.terminal:
                return observation
            error = self._pose_error(observation, target_pose)
            extra: dict[str, Any] = {}
            if self.machine.state is ExpertState.RETREAT:
                extra = {
                    "success": self._success,
                    "success_reason": self._success_reason,
                }
            context = self.machine.context(
                pose_error_m=error,
                position_tolerance_m=tolerance,
                **extra,
            )
            self.machine.fire("reached", context)
            self._plan = None
            self._plan_cursor = 0
        return observation

    def _run_hold(self, observation: Any, gripper: float, steps: int) -> Any:
        q = np.asarray(observation.robot_state.joint_positions, dtype=np.float64)
        for _ in range(steps):
            if self.machine.terminal or self.step_index >= self.max_steps:
                break
            observation = self._step(q.copy(), gripper)
        return observation

    def run(self) -> EpisodeRecord:
        observation = self.backend.get_observation()
        targets = self._compute_targets(observation)
        # Record the post-reset observation first, so even a planning failure
        # before the first motion leaves a valid, convertible record.
        initial_q = np.asarray(observation.robot_state.joint_positions, dtype=np.float64)
        self._record_step(
            observation,
            np.append(initial_q, float(observation.robot_state.gripper_position)),
            self.backend.collision_report(),
        )
        self.step_index = 1
        self.machine.fire("start", self.machine.context())

        while not self.machine.terminal and self.step_index < self.max_steps:
            state = self.machine.state
            if state is ExpertState.GRASP:
                observation = self._run_hold(
                    observation,
                    self.config.motion.gripper_closed,
                    self.config.motion.grasp_hold_steps,
                )
                if not self.machine.terminal:
                    confirmed = _grasp_confirmed(self.backend, self.task.object_id, observation)
                    self.machine.fire("grasped", self.machine.context(grasp_confirmed=confirmed))
                continue
            if state is ExpertState.RELEASE:
                observation = self._run_hold(
                    observation,
                    self.config.motion.gripper_open,
                    self.config.motion.release_hold_steps,
                )
                if not self.machine.terminal:
                    open_now = float(observation.robot_state.gripper_position) >= (
                        self.config.motion.gripper_open - 0.005
                    )
                    self.machine.fire("released", self.machine.context(gripper_open=open_now))
                continue
            if state is ExpertState.RETREAT and self._plan is None:
                success, reason, metrics = _success_check(
                    self.backend,
                    self.task,
                    observation,
                    self.object_history,
                    self.config,
                )
                self._success = success
                self._success_reason = reason
                self._success_metrics = metrics
            if self.machine.needs_motion:
                if self._plan is None:
                    self._make_plan(observation)
                    assert self._plan is not None
                    if not self._plan.success:
                        self.machine.fail(self._plan.reason)
                        break
                    self._plan_cursor = 0
                target_key = self.machine.state.value
                observation = self._advance_motion(observation, targets[target_key])
            else:
                break

        if self.step_index >= self.max_steps and not self.machine.terminal:
            self.record.failure_reason = (
                f"max_steps={self.max_steps} reached in {self.machine.state.value}"
            )
            self.record.outcome = contracts.OUTCOME_TIMEOUT
        elif self.machine.state is ExpertState.DONE:
            self.record.outcome = contracts.OUTCOME_SUCCESS
            self.record.success_metrics = self._success_metrics
        elif self.machine.state is ExpertState.TIMEOUT:
            self.record.outcome = contracts.OUTCOME_TIMEOUT
            self.record.failure_reason = self.machine.failure_reason
        elif self.machine.state is ExpertState.FAILED:
            self.record.outcome = contracts.OUTCOME_FAILURE
            self.record.failure_reason = self.machine.failure_reason
        else:
            self.record.outcome = contracts.OUTCOME_FAILURE
            self.record.failure_reason = (
                f"episode ended in {self.machine.state.value} without success"
            )

        self.record.transitions = [entry.to_dict() for entry in self.machine.log]
        if self._success_metrics:
            self.record.success_metrics = self._success_metrics
        if not self.record.success_metrics:
            self.record.success_metrics = {
                "xy_error_m": None,
                "reason": self.record.failure_reason,
            }
        return self.record


def run_episode(
    backend: Any,
    task: TaskSpec,
    seed: int,
    *,
    instruction: str = "",
    episode_index: int = 0,
    run_id: str = "run",
    randomization_profile: str | None = None,
    randomization_dir: str | None = None,
    perturbation: Perturbation | None = None,
    config: ExpertConfig | None = None,
    ik_solver: IKSolver | None = None,
    planner: Planner | None = None,
    max_steps: int | None = None,
) -> EpisodeRecord:
    """Run and record one episode. Never raises for a normal failure."""
    settings = config or load_expert_config()
    profile = randomization_profile or settings.default_profile
    randomization_seed = derive_episode_seed(seed, episode_index)
    backend_name = getattr(backend, "name", type(backend).__name__)
    episode_id = f"{run_id}_ep{episode_index:04d}"
    record = EpisodeRecord(
        episode_id=episode_id,
        run_id=run_id,
        episode_index=episode_index,
        seed=seed,
        randomization_seed=randomization_seed,
        backend=str(backend_name),
        task_instruction=instruction,
        task_object_id=task.object_id,
        task_zone_id=task.zone_id,
        control_rate_hz=float(backend.control_rate_hz),
        outcome=contracts.OUTCOME_UNKNOWN,
        failure_reason="",
        is_perturbed=perturbation is not None,
        perturbation_name=perturbation.name if perturbation is not None else None,
        expected_quality=(perturbation.expected_quality if perturbation is not None else "clean"),
        started_at=datetime.now(UTC).isoformat(),
    )
    steps = settings.max_steps if max_steps is None else int(max_steps)
    if steps <= 0:
        raise ValueError("max_steps must be positive")

    try:
        if randomization_dir is None:
            config_dir = getattr(backend, "config_dir", None)
            if config_dir is not None:
                randomization_dir = str(Path(config_dir).parent / "randomization")
        outcome = reset_environment(
            backend,
            randomization_seed,
            randomization_enabled=profile != "none",
            randomization_profile=profile,
            task_instruction=instruction,
            randomization_dir=randomization_dir,
            episode_index=episode_index,
        )
        record.metadata = {
            "requested_profile": profile,
            "reset_message": outcome.message,
            "reset_success": bool(outcome.success),
        }
        if not outcome.success:
            record.outcome = contracts.OUTCOME_FAILURE
            record.failure_reason = f"reset failed: {outcome.message}"
            return record
        try:
            import json

            record.randomization_report = json.loads(outcome.applied_config_json)
        except (TypeError, ValueError):  # pragma: no cover - reset validates JSON
            record.randomization_report = {"raw": outcome.applied_config_json}

        runner = _EpisodeRunner(
            backend=backend,
            task=task,
            record=record,
            config=settings,
            ik_solver=ik_solver or DampedLeastSquaresIK(settings.ik),
            planner=planner or CartesianInterpolationPlanner(),
            perturbation=perturbation,
            max_steps=steps,
        )
        return runner.run()
    except Exception as exc:  # noqa: BLE001 - a bad episode must not stop the batch
        LOGGER.exception("episode %s aborted by backend/runner exception", record.episode_id)
        record.outcome = contracts.OUTCOME_ABORTED
        record.failure_reason = f"{type(exc).__name__}: {exc}"
        return record


def run_episodes(
    backend: Any,
    task: TaskSpec,
    seeds: list[int] | tuple[int, ...],
    *,
    instruction: str = "",
    run_id: str = "run",
    randomization_profile: str | None = None,
    randomization_dir: str | None = None,
    perturbations: Any | None = None,
    perturbed_fraction: float | None = None,
    config: ExpertConfig | None = None,
    max_steps: int | None = None,
) -> list[EpisodeRecord]:
    """Run a batch; each episode continues even if the previous one failed."""
    settings = config or load_expert_config()
    records: list[EpisodeRecord] = []
    for episode_index, seed in enumerate(seeds):
        perturbation = None
        if perturbations is not None:
            from sim2real.expert.perturbations import select_perturbation

            perturbation = select_perturbation(
                perturbations,
                int(seed),
                episode_index,
                perturbed_fraction=perturbed_fraction,
            )
        records.append(
            run_episode(
                backend,
                task,
                int(seed),
                instruction=instruction,
                episode_index=episode_index,
                run_id=run_id,
                randomization_profile=randomization_profile,
                randomization_dir=randomization_dir,
                perturbation=perturbation,
                config=settings,
                max_steps=max_steps,
            )
        )
    return records


__all__ = [
    "EpisodeRecord",
    "StepRecord",
    "derive_episode_seed",
    "run_episode",
    "run_episodes",
]
