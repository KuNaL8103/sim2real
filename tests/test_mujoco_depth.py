"""MuJoCo depth is metric metres, checked against a known plane distance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.sim, pytest.mark.render]

CONFIG_DIR = Path("configs/sim")
REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS = REPO_ROOT / "assets" / "mujoco" / "franka_emika_panda"


def _backend():
    pytest.importorskip("mujoco")
    if not (ASSETS / "panda.xml").is_file():
        pytest.skip("Menagerie assets missing; run scripts/fetch_assets.sh")
    from sim2real.sim import create_backend

    return create_backend("mujoco", CONFIG_DIR)


def test_overhead_depth_equals_known_plane_distance():
    backend = _backend()
    try:
        backend.reset(seed=0)
        frame = backend.render("overhead")
        # The image centre is occupied by the arm at the home pose, so sample a
        # clear table point and project it to pixels with the same intrinsics.
        from sim2real.sim.camera import project_point

        expected = 1.10 - 0.42
        u, v, _ = project_point(
            np.array([0.45, -0.30, 0.42]),
            backend.camera_pose_world("overhead"),
            backend.get_camera_info("overhead"),
        )
        assert np.isfinite(u) and np.isfinite(v)
        centre = frame.depth[int(round(v)), int(round(u))]
        assert np.isfinite(centre)
        assert float(centre) == pytest.approx(expected, abs=0.01)
        # If depth were millimetres this would be ~680, not ~0.68.
        assert 0.5 < float(centre) < 0.9
    finally:
        backend.close()


def test_wrist_depth_is_positive_and_close():
    backend = _backend()
    try:
        backend.reset(seed=0)
        frame = backend.render("wrist")
        valid = frame.depth[np.isfinite(frame.depth)]
        assert valid.size > 0
        assert float(valid.max()) < 1.0
        assert float(valid.min()) > 0.0
    finally:
        backend.close()
