"""The runner aborts a physically impossible state with PHYSICS_INSTABILITY."""

from __future__ import annotations

from dataclasses import replace

from sim2real import contracts
from sim2real.expert import load_expert_config, parse_instruction, run_episode
from sim2real.sim.fake_backend import FakeBackend


class TeleportingFakeBackend(FakeBackend):
    """Teleport the target cube outside the workspace on the first step."""

    def __init__(self, config_dir: str) -> None:
        super().__init__(config_dir)
        self._teleported = False

    def step(self, joint_target, gripper_target):
        result = super().step(joint_target, gripper_target)
        if not self._teleported:
            self._teleported = True
            poses = dict(result.observation.object_poses)
            pose = poses["cube_red"].copy()
            pose[0, 3] = 10.0
            poses["cube_red"] = pose
            observation = replace(result.observation, object_poses=poses)
            return replace(result, observation=observation)
        return result


def _task(backend):
    config = load_expert_config("configs/expert")
    return config, parse_instruction(
        "pick up the red cube and place it in the left zone",
        backend.configs.scene,
        config.instructions_file,
    )


def test_physics_guard_aborts_a_teleported_cube():
    backend = TeleportingFakeBackend("configs/sim")
    try:
        config, task = _task(backend)
        record = run_episode(
            backend,
            task,
            seed=0,
            randomization_profile="none",
            randomization_dir="configs/randomization",
            config=config,
        )
        assert record.outcome == contracts.OUTCOME_ABORTED
        assert "PHYSICS_INSTABILITY" in record.failure_reason
    finally:
        backend.close()


def test_physics_guard_does_not_abort_a_nominal_fake_episode():
    backend = FakeBackend("configs/sim")
    try:
        config, task = _task(backend)
        record = run_episode(
            backend,
            task,
            seed=0,
            randomization_profile="none",
            randomization_dir="configs/randomization",
            config=config,
        )
        assert record.outcome != contracts.OUTCOME_ABORTED
        assert "PHYSICS_INSTABILITY" not in record.failure_reason
    finally:
        backend.close()
