"""ROS test: a slow renderer must serve both cameras fairly.

CPU-only: the fake backend's render delay is a pure sleep. Written as
``unittest`` because colcon runs ``python -m unittest`` for ``ament_python``.
Skipped with a reason when ROS is unavailable; a skip is never a pass.
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    import rclpy

    from sim2real.sim import create_backend
    from sim2real_sim.sim_node import SimNode

    IMPORT_ERROR: str | None = None
except ImportError as exc:  # pragma: no cover - depends on the environment
    IMPORT_ERROR = f"ROS prerequisites missing: {exc}"

pytestmark = pytest.mark.ros


@unittest.skipIf(IMPORT_ERROR is not None, IMPORT_ERROR or "")
class TestCameraSchedule(unittest.TestCase):
    def test_slow_renderer_serves_both_cameras_fairly(self) -> None:
        rclpy.init()
        node = None
        try:
            backend = create_backend("fake", str(REPO_ROOT / "configs" / "sim"))
            backend.render_delay_s = 0.3
            node = SimNode(injected_backend=backend)
            ticks = 24
            for _ in range(ticks):
                node._control_tick()
                node._camera_tick()
                time.sleep(0.1)
            deadline = time.monotonic() + 2.0
            while node._render_coordinator.render_in_flight and time.monotonic() < deadline:
                time.sleep(0.02)
            time.sleep(0.2)
            stats = node._render_coordinator.stats()
            served = stats["published_by_camera"]
            self.assertGreater(served.get("overhead", 0), 0, stats)
            self.assertGreater(served.get("wrist", 0), 0, stats)
            # The renderer serves about one camera per 0.1 s tick; over 24 ticks
            # the two counts may differ by at most one completed pair.
            self.assertLessEqual(
                abs(served.get("overhead", 0) - served.get("wrist", 0)),
                2,
                stats,
            )
            self.assertEqual(stats["failed"], 0, stats)
        finally:
            if node is not None:
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
