"""MuJoCo contact-flag behaviour: nominal approach clean, table contact flagged."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from sim2real.expert.config import load_expert_config  # noqa: E402
from sim2real.expert.ik import DampedLeastSquaresIK  # noqa: E402
from sim2real.sim.kinematics import tcp_pose  # noqa: E402
from sim2real.sim.mujoco_backend import MuJoCoBackend  # noqa: E402

pytestmark = pytest.mark.sim


def test_nominal_approach_clean_then_table_contact_flagged():
    backend = MuJoCoBackend("configs/sim")
    try:
        backend.set_collision_context("cube_red")
        observation = backend.reset(seed=0)
        q0 = np.asarray(observation.robot_state.joint_positions, dtype=np.float64)
        panda = backend.configs.panda
        target = tcp_pose(q0, panda).copy()
        target[2, 3] = 0.345  # below the table top (0.42), so the fingers hit it
        ik = DampedLeastSquaresIK(load_expert_config().ik)
        result = ik.solve(target, q0, panda)
        assert result.success, f"IK failed: {result.position_error}"

        nominal_max = 0
        late_unintended = 0
        for step in range(70):
            alpha = min(1.0, (step + 1) / 45.0)
            command = q0 + (np.asarray(result.q) - q0) * alpha
            backend.step(command, 0.04)
            count = backend.collision_report().num_unintended_contacts
            if step <= 20:
                nominal_max = max(nominal_max, count)
            if step >= 55:
                late_unintended = max(late_unintended, count)

        assert nominal_max == 0, "the nominal approach flagged an unintended contact"
        assert late_unintended > 0, "driving into the table did not flag a contact"
        details = " ".join(backend.collision_report().details)
        assert "table" in details and "finger" in details
    finally:
        backend.close()
