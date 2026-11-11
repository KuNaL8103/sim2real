"""Closed-loop evaluation runner.

Timing contract (also in ``docs/EVALUATION.md``): the policy period is the
dataset period (0.1 s at 10 Hz); the backend runs at 20 Hz, so every policy
action is held for two control steps.  A chunk is re-planned every
``eval.n_action_steps`` policy steps (default 10 = 1.0 s, **not** the training
value 50).

The runner owns reset, execution, termination and recording.  A policy never
steps the backend.  Every outcome is recorded; a raising, NaN, or wrong-shape
policy is a ``POLICY_ERROR`` and the batch continues.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from sim2real.evaluation.config import EvaluationConfig
from sim2real.evaluation.policy_interface import (
    ActionChunk,
    Policy,
    PolicyError,
    PolicyObservation,
    _state_from_observation,
)
from sim2real.evaluation.records import (
    EvalEpisodeRecord,
    EvalRun,
    EvalStep,
    aggregate_records,
    derive_eval_seeds,
    metrics_for_record,
    task_completion,
    wilson_interval,
)
from sim2real.expert import derive_episode_seed, load_expert_config, parse_instruction
from sim2real.expert.rollout import _success_check
from sim2real.sim.reset_handler import reset_environment

EVAL_CAMERAS: tuple[str, ...] = ("overhead", "wrist")


def _collision_run_count(flags: Any) -> int:
    from sim2real.filtering.metrics import collision_run_count

    return collision_run_count(flags)


def renderer_label(backend: Any, requested: str = "llvmpipe") -> str:
    """Describe the renderer actually selected, including opt-in D3D12.

    The dataset was rendered with llvmpipe.  If the opt-in D3D12 environment is
    present it is recorded explicitly (and therefore visible in the manifest);
    it is never used silently.
    """
    name = str(getattr(backend, "name", type(backend).__name__))
    if name == "fake":
        return "fake-analytic"
    gallium = os.environ.get("GALLIUM_DRIVER", "").lower()
    if gallium == "d3d12":
        adapter = os.environ.get("MESA_D3D12_DEFAULT_ADAPTER_NAME", "")
        suffix = f":{adapter}" if adapter else ""
        return f"d3d12{suffix}-OPT_IN"
    return requested


def _flatten_sample(sample: dict[str, Any], cube: str) -> dict[str, Any]:
    poses = sample.get("object_pose", {}).get("poses", {})
    pose = poses.get(cube)
    if pose is None:
        raise PolicyError(f"randomization sample has no pose for {cube}")
    x, y = float(pose[0][3]), float(pose[1][3])
    yaw = float(np.arctan2(float(pose[1][0]), float(pose[0][0])))
    physics = sample.get("physics", {})
    friction_map = physics.get("friction") or {}
    mass_map = physics.get("object_mass_kg") or {}
    friction = float(friction_map.get(cube, [0.0, 0.0, 0.0])[0]) if friction_map else 0.0
    mass = float(mass_map.get(cube, 0.0)) if mass_map else 0.0
    lighting = sample.get("lighting", {})
    camera = sample.get("camera", {}).get("translation_jitter", {})
    jitter = np.asarray(camera.get("overhead", [0.0, 0.0, 0.0]), dtype=np.float64)
    return {
        "initial_object_pose": (x, y, yaw),
        "friction_sliding": friction,
        "object_mass": mass,
        "light_intensity": float(lighting.get("intensity_scale", 1.0)),
        "color_temperature_k": float(lighting.get("color_temperature_k", 6500.0)),
        "camera_jitter_norm": float(np.linalg.norm(jitter)),
    }


def _policy_observation(
    observation: Any, images: dict[str, np.ndarray], task_instruction: str
) -> PolicyObservation:
    return PolicyObservation(
        state=_state_from_observation(observation),
        task_instruction=task_instruction,
        stamp_ns=int(observation.stamp_ns),
        images=images,
        object_poses=dict(observation.object_poses),
    )


def _physics_instability(backend: Any, observation: Any, expert_config: Any) -> str | None:
    """Same checks as the Task 4 expert guard, without changing expert code."""
    robot = observation.robot_state
    q = np.asarray(robot.joint_positions, dtype=np.float64)
    qd = np.asarray(robot.joint_velocities, dtype=np.float64)
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(qd))):
        return "PHYSICS_INSTABILITY: non-finite joint state"
    data = getattr(backend, "data", None)
    qacc = getattr(data, "qacc", None) if data is not None else None
    if qacc is not None:
        qacc = np.asarray(qacc, dtype=np.float64)
        limit = expert_config.physics_guard.max_abs_qacc
        if not np.all(np.isfinite(qacc)) or float(np.max(np.abs(qacc))) > limit:
            return f"PHYSICS_INSTABILITY: qacc exceeds {limit:.1f} rad/s^2"
    scene = backend.configs.scene
    margin = expert_config.physics_guard.workspace_margin_m
    low = np.asarray(scene.workspace_min, dtype=np.float64) - margin
    high = np.asarray(scene.workspace_max, dtype=np.float64) + margin
    for name, pose in observation.object_poses.items():
        position = np.asarray(pose, dtype=np.float64)[:3, 3]
        if not np.all(np.isfinite(position)) or np.any(position < low) or np.any(position > high):
            return (
                f"PHYSICS_INSTABILITY: {name} at "
                f"{np.round(position, 3).tolist()} outside workspace margin"
            )
    return None


def _error_record(
    *,
    episode_id: str,
    eval_run_id: str,
    seed: int,
    episode_index: int,
    settings: Any,
    config_hash: str,
    git_commit: str,
    git_dirty: bool,
    renderer: str,
    policy_id: str,
    checkpoint_hash: str,
    reason: str,
    task_instruction: str,
) -> EvalEpisodeRecord:
    return EvalEpisodeRecord(
        episode_id=episode_id,
        eval_run_id=eval_run_id,
        seed=int(seed),
        episode_index=int(episode_index),
        randomization_profile=settings.randomization_profile,
        randomization={},
        initial_object_pose=(0.0, 0.0, 0.0),
        friction_sliding=0.0,
        object_mass=0.0,
        light_intensity=1.0,
        color_temperature_k=6500.0,
        camera_jitter_norm=0.0,
        task_instruction=task_instruction,
        success=False,
        task_completion=0.0,
        collision_count=0,
        termination_reason=reason,
        policy_id=policy_id,
        checkpoint_hash=checkpoint_hash,
        config_hash=config_hash,
        git_commit=git_commit,
        git_dirty=bool(git_dirty),
        n_action_steps=settings.n_action_steps,
        renderer=renderer,
        failure={"reason": reason},
    )


def run_eval_episode(
    backend: Any,
    policy: Policy,
    *,
    seed: int,
    episode_index: int,
    config: EvaluationConfig,
    eval_run_id: str,
    config_hash: str,
    git_commit: str = "unknown",
    git_dirty: bool = True,
    randomization_dir: str | None = None,
    filter_config: Any,
    expert_config: Any | None = None,
) -> EvalEpisodeRecord:
    """Run one closed-loop episode and always return a validated record."""
    settings = config.settings
    expert = expert_config or load_expert_config("configs/expert")
    renderer = renderer_label(backend, settings.renderer)
    episode_id = f"{eval_run_id}_ep{episode_index:04d}"
    policy_id = str(getattr(policy, "name", type(policy).__name__))
    checkpoint_hash = str(getattr(policy, "info", {}).get("checkpoint_hash", "none"))
    randomization_seed = derive_episode_seed(int(seed), int(episode_index))
    set_index = getattr(policy, "set_episode_index", None)
    if set_index is not None:
        set_index(int(episode_index))
    set_actions = getattr(policy, "set_n_action_steps", None)
    if set_actions is not None:
        set_actions(int(settings.n_action_steps))

    base = {
        "episode_id": episode_id,
        "eval_run_id": eval_run_id,
        "seed": int(seed),
        "episode_index": int(episode_index),
        "settings": settings,
        "config_hash": config_hash,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "renderer": renderer,
        "policy_id": policy_id,
        "checkpoint_hash": checkpoint_hash,
        "task_instruction": settings.task_instruction,
    }

    try:
        task = parse_instruction(
            settings.task_instruction, backend.configs.scene, expert.instructions_file
        )
        outcome = reset_environment(
            backend,
            randomization_seed,
            randomization_enabled=settings.randomization_profile != "none",
            randomization_profile=settings.randomization_profile,
            task_instruction=settings.task_instruction,
            randomization_dir=randomization_dir,
            episode_index=int(episode_index),
        )
        if not outcome.success:
            return _error_record(reason="BACKEND_ERROR", **base)
        payload = json.loads(outcome.applied_config_json)
        sample = payload["sample"]
        cube = next(iter(backend.configs.scene.cubes))
        values = _flatten_sample(sample, cube)
        observation = backend.get_observation()
        set_context = getattr(policy, "set_reset_context", None)
        if set_context is not None:
            set_context(sample, observation)
        policy.reset(settings.task_instruction, int(seed))
    except Exception as exc:  # noqa: BLE001 - reset/policy must not crash a batch
        record = _error_record(reason="POLICY_ERROR", **base)
        record.failure = {"reason": f"{type(exc).__name__}: {exc}"}
        return record

    zone = backend.configs.scene.zones[task.zone_id]
    zone_xy = np.asarray(zone.position, dtype=np.float64)[:2]
    initial_xy = np.asarray(observation.object_poses[cube], dtype=np.float64)[:2, 3]
    object_history: list[np.ndarray] = [np.asarray(observation.object_poses[cube])]
    steps: list[EvalStep] = []
    collision_flags: list[bool] = []
    last_policy_info: dict[str, Any] = dict(getattr(policy, "info", {}))
    reason: str | None = None
    failure: dict[str, Any] = {}
    success_metrics: dict[str, Any] | None = None
    control_step = 0
    chunk: ActionChunk | None = None
    chunk_cursor = 0
    chunk_actions: np.ndarray | None = None

    try:
        while reason is None and control_step < settings.max_control_steps:
            if chunk_actions is None or chunk_cursor >= len(chunk_actions):
                images: dict[str, np.ndarray] = {}
                if getattr(policy, "needs_images", False) or settings.render_images:
                    for camera in EVAL_CAMERAS:
                        frame = backend.render(camera)
                        images[camera] = np.asarray(frame.rgb, dtype=np.uint8)
                policy_observation = _policy_observation(
                    backend.get_observation(), images, settings.task_instruction
                )
                try:
                    chunk = policy.act(policy_observation)
                except Exception as exc:  # noqa: BLE001 - policy failure is an outcome
                    reason = "POLICY_ERROR"
                    failure = {"reason": f"{type(exc).__name__}: {exc}"}
                    break
                if not isinstance(chunk, ActionChunk):
                    reason = "POLICY_ERROR"
                    failure = {"reason": f"policy returned {type(chunk).__name__}, not ActionChunk"}
                    break
                last_policy_info = dict(getattr(policy, "info", {}))
                chunk_actions = chunk.actions[: settings.n_action_steps]
                chunk_cursor = 0
                if len(chunk_actions) == 0:
                    reason = "POLICY_ERROR"
                    failure = {"reason": "policy returned an empty chunk"}
                    break

            action = np.asarray(chunk_actions[chunk_cursor], dtype=np.float64)
            chunk_cursor += 1
            if action.shape != (8,) or not np.all(np.isfinite(action)):
                reason = "POLICY_ERROR"
                failure = {"reason": f"invalid action {action.tolist()}"}
                break
            joint_target = action[:7]
            gripper_target = float(action[7])
            limits = np.asarray(backend.joint_limits, dtype=np.float64)
            outside = (joint_target < limits[:, 0]) | (joint_target > limits[:, 1])
            if np.any(outside):
                if settings.terminate_on_limit:
                    reason = "LIMIT_VIOLATION"
                    failure = {"reason": f"joint target {joint_target.tolist()} outside limits"}
                    break
                joint_target = np.clip(joint_target, limits[:, 0], limits[:, 1])

            for _ in range(settings.hold_steps):
                if control_step >= settings.max_control_steps:
                    reason = "TIMEOUT"
                    break
                try:
                    result = backend.step(joint_target, gripper_target)
                except Exception as exc:  # noqa: BLE001 - backend failure is an outcome
                    reason = "BACKEND_ERROR"
                    failure = {"reason": f"{type(exc).__name__}: {exc}"}
                    break
                observation = result.observation
                object_history.append(np.asarray(observation.object_poses[cube]))
                collision = bool(result.collision.num_unintended_contacts > 0)
                collision_flags.append(collision)
                steps.append(
                    EvalStep(
                        stamp_ns=int(observation.stamp_ns),
                        step_index=control_step,
                        joint_positions=observation.robot_state.joint_positions,
                        joint_velocities=observation.robot_state.joint_velocities,
                        ee_pose=observation.robot_state.ee_pose,
                        action=np.clip(
                            np.concatenate([joint_target, [gripper_target]]), -1.0e6, 1.0e6
                        ),
                        gripper_position=float(observation.robot_state.gripper_position),
                        unintended_collision=collision,
                    )
                )
                control_step += 1
                instability = _physics_instability(backend, observation, expert)
                if instability is not None:
                    reason = "PHYSICS_INSTABILITY"
                    failure = {"reason": instability}
                    break
                if settings.collision_terminate and collision:
                    reason = "COLLISION"
                    failure = {"reason": "unintended collision"}
                    break
                success, _success_reason, success_metrics = _success_check(
                    backend, task, observation, object_history, expert
                )
                if success:
                    reason = "SUCCESS"
                    failure = {}
                    break
        if reason is None:
            reason = "TIMEOUT"
    except Exception as exc:  # noqa: BLE001 - one bad episode must not stop the batch
        reason = "BACKEND_ERROR"
        failure = {"reason": f"{type(exc).__name__}: {exc}"}

    final_pose = np.asarray(observation.object_poses[cube], dtype=np.float64)
    record = EvalEpisodeRecord(
        episode_id=episode_id,
        eval_run_id=eval_run_id,
        seed=int(seed),
        episode_index=int(episode_index),
        randomization_profile=settings.randomization_profile,
        randomization=sample,
        initial_object_pose=tuple(values["initial_object_pose"]),
        friction_sliding=float(values["friction_sliding"]),
        object_mass=float(values["object_mass"]),
        light_intensity=float(values["light_intensity"]),
        color_temperature_k=float(values["color_temperature_k"]),
        camera_jitter_norm=float(values["camera_jitter_norm"]),
        task_instruction=settings.task_instruction,
        success=reason == "SUCCESS",
        task_completion=task_completion(initial_xy, final_pose[:2, 3], zone_xy),
        collision_count=_collision_run_count(collision_flags),
        termination_reason=reason,
        policy_id=policy_id,
        checkpoint_hash=checkpoint_hash,
        config_hash=config_hash,
        git_commit=git_commit,
        git_dirty=bool(git_dirty),
        n_action_steps=settings.n_action_steps,
        renderer=renderer,
        failure={},
        steps=steps,
    )
    record.metrics = metrics_for_record(record, filter_config)
    record.metrics_error = (
        ""
        if not record.metrics.get("degenerate")
        else str(record.metrics.get("reason", "DEGENERATE"))
    )
    if reason == "SUCCESS":
        if success_metrics is not None:
            record.failure = {"success_metrics": success_metrics}
    elif failure:
        record.failure = {
            **failure,
            "state_at_termination": {
                "joint_positions": [float(v) for v in observation.robot_state.joint_positions],
                "gripper_position": float(observation.robot_state.gripper_position),
                "object_poses": {
                    name: [[float(v) for v in row] for row in pose]
                    for name, pose in observation.object_poses.items()
                },
            },
            "last_policy_info": last_policy_info,
        }
    return record


@dataclass
class EvalResult:
    run: EvalRun
    records: list[EvalEpisodeRecord]

    def summary(self) -> dict[str, Any]:
        return {"run": self.run.to_dict(), "records": [r.to_dict() for r in self.records]}


def run_eval(
    backend: Any,
    policy: Policy,
    config: EvaluationConfig,
    *,
    num_episodes: int | None = None,
    eval_seed: int | None = None,
    run_id: str | None = None,
    config_hash: str | None = None,
    git_commit: str = "unknown",
    git_dirty: bool = True,
    randomization_dir: str | None = None,
    filter_config: Any,
    expert_config: Any | None = None,
) -> EvalResult:
    settings = config.settings
    count = settings.num_episodes if num_episodes is None else int(num_episodes)
    seed = settings.eval_seed if eval_seed is None else int(eval_seed)
    if count < 0:
        raise ValueError("num_episodes must be >= 0")
    seeds = derive_eval_seeds(seed, count)
    run_id = run_id or f"eval_{seed}_{len(seeds)}"
    config_hash = config_hash or config.config_hash()
    expert = expert_config or load_expert_config("configs/expert")
    renderer = renderer_label(backend, settings.renderer)
    records: list[EvalEpisodeRecord] = []
    for index, episode_seed in enumerate(seeds):
        try:
            record = run_eval_episode(
                backend,
                policy,
                seed=episode_seed,
                episode_index=index,
                config=config,
                eval_run_id=run_id,
                config_hash=config_hash,
                git_commit=git_commit,
                git_dirty=git_dirty,
                randomization_dir=randomization_dir,
                filter_config=filter_config,
                expert_config=expert,
            )
        except Exception as exc:  # noqa: BLE001 - never lose a batch to one record
            record = _error_record(
                episode_id=f"{run_id}_ep{index:04d}",
                eval_run_id=run_id,
                seed=episode_seed,
                episode_index=index,
                settings=settings,
                config_hash=config_hash,
                git_commit=git_commit,
                git_dirty=git_dirty,
                renderer=renderer,
                policy_id=str(getattr(policy, "name", type(policy).__name__)),
                checkpoint_hash=str(getattr(policy, "info", {}).get("checkpoint_hash", "none")),
                reason="BACKEND_ERROR",
                task_instruction=settings.task_instruction,
            )
            record.failure = {"reason": f"{type(exc).__name__}: {exc}"}
        records.append(record)
    run = EvalRun(
        run_id=run_id,
        config_hash=config_hash,
        eval_seed=seed,
        seeds=seeds,
        policy_id=str(getattr(policy, "name", type(policy).__name__)),
        policy_info=dict(getattr(policy, "info", {})),
        randomization_profile=settings.randomization_profile,
        n_action_steps=settings.n_action_steps,
        renderer=renderer,
    )
    run.aggregates = aggregate_records(records)
    return EvalResult(run=run, records=records)


def run_paired_eval(
    *,
    expert_backend_factory: Any,
    backend: Any,
    expert_policy: Policy,
    candidate_policy: Policy,
    config: EvaluationConfig,
    num_episodes: int | None = None,
    eval_seed: int | None = None,
    run_id: str = "paired_eval",
    config_hash: str | None = None,
    git_commit: str = "unknown",
    git_dirty: bool = True,
    randomization_dir: str | None = None,
    filter_config: Any,
    expert_config: Any | None = None,
) -> dict[str, Any]:
    """Evaluate the SAME seeds with the expert and the candidate policy."""
    del expert_backend_factory  # the expert policy owns its own backend
    settings = config.settings
    count = settings.num_episodes if num_episodes is None else int(num_episodes)
    seed = settings.eval_seed if eval_seed is None else int(eval_seed)
    seeds = derive_eval_seeds(seed, count)
    config_hash = config_hash or config.config_hash()
    expert = expert_config or load_expert_config("configs/expert")
    pairs: list[dict[str, Any]] = []
    expert_records: list[EvalEpisodeRecord] = []
    policy_records: list[EvalEpisodeRecord] = []
    for index, episode_seed in enumerate(seeds):
        ep_id = f"{run_id}_expert_ep{index:04d}"
        expert_record = run_eval_episode(
            backend,
            expert_policy,
            seed=episode_seed,
            episode_index=index,
            config=config,
            eval_run_id=ep_id,
            config_hash=config_hash,
            git_commit=git_commit,
            git_dirty=git_dirty,
            randomization_dir=randomization_dir,
            filter_config=filter_config,
            expert_config=expert,
        )
        policy_record = run_eval_episode(
            backend,
            candidate_policy,
            seed=episode_seed,
            episode_index=index,
            config=config,
            eval_run_id=f"{run_id}_policy_ep{index:04d}",
            config_hash=config_hash,
            git_commit=git_commit,
            git_dirty=git_dirty,
            randomization_dir=randomization_dir,
            filter_config=filter_config,
            expert_config=expert,
        )
        expert_records.append(expert_record)
        policy_records.append(policy_record)
        pairs.append(
            {
                "seed": int(episode_seed),
                "episode_index": index,
                "expert_success": bool(expert_record.success),
                "policy_success": bool(policy_record.success),
                "expert_termination_reason": expert_record.termination_reason,
                "policy_termination_reason": policy_record.termination_reason,
                "policy_only_failure": bool(expert_record.success and not policy_record.success),
            }
        )
    expert_success = sum(1 for p in pairs if p["expert_success"])
    policy_success = sum(1 for p in pairs if p["policy_success"])
    on_expert_success = [p for p in pairs if p["expert_success"]]
    policy_on_expert = sum(1 for p in on_expert_success if p["policy_success"])
    only_failures = sum(1 for p in pairs if p["policy_only_failure"])
    expert_low, expert_high = wilson_interval(expert_success, len(pairs))
    policy_low, policy_high = wilson_interval(policy_success, len(pairs))
    conditional_low, conditional_high = wilson_interval(policy_on_expert, len(on_expert_success))
    return {
        "seeds": [int(s) for s in seeds],
        "pairs": pairs,
        "num_pairs": len(pairs),
        "expert": {
            "num_success": expert_success,
            "success_rate": expert_success / len(pairs) if pairs else 0.0,
            "wilson_95": [expert_low, expert_high],
            "aggregates": aggregate_records(expert_records),
        },
        "policy": {
            "num_success": policy_success,
            "success_rate": policy_success / len(pairs) if pairs else 0.0,
            "wilson_95": [policy_low, policy_high],
            "aggregates": aggregate_records(policy_records),
        },
        "policy_success_given_expert_success": {
            "num_success": policy_on_expert,
            "num_eligible": len(on_expert_success),
            "rate": (policy_on_expert / len(on_expert_success)) if on_expert_success else 0.0,
            "wilson_95": [conditional_low, conditional_high],
        },
        "policy_only_failures": only_failures,
        "expert_records": expert_records,
        "policy_records": policy_records,
    }


__all__ = [
    "EVAL_CAMERAS",
    "EvalResult",
    "renderer_label",
    "run_eval",
    "run_eval_episode",
    "run_paired_eval",
]
