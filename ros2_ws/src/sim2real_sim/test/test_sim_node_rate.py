"""ROS rate test: a slow renderer must not starve /joint_states.

Uses the fake backend with an injected 300 ms render delay, so it is
hardware-independent. No MuJoCo timing threshold is asserted.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import unittest
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
    from rclpy.node import Node
    from sensor_msgs.msg import JointState as JointStateMsg
    from sim2real_sim.sim_node import SimNode

    from sim2real import contracts
    from sim2real.sim import create_backend

    IMPORT_ERROR: str | None = None
except ImportError as exc:  # pragma: no cover
    IMPORT_ERROR = f"ROS prerequisites missing: {exc}"

pytestmark = pytest.mark.ros


@unittest.skipIf(IMPORT_ERROR is not None, IMPORT_ERROR or "")
class TestSimNodeRate(unittest.TestCase):
    def test_joint_state_rate_survives_slow_rendering(self) -> None:
        backend = create_backend("fake", str(REPO_ROOT / "configs" / "sim"))
        backend.render_delay_s = 0.3
        backend.render_delay_only = True
        rclpy.init()
        node = SimNode(injected_backend=backend)
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        # Measure from a separate node/executor: sharing the sim node's executor
        # makes the probe compete with timers and can drop callbacks on CI.
        probe = Node("rate_probe")
        probe_executor = SingleThreadedExecutor()
        probe_executor.add_node(probe)
        counter = {"joint": 0}
        probe.create_subscription(
            JointStateMsg,
            contracts.JOINT_STATES_TOPIC,
            lambda _message: counter.__setitem__("joint", counter["joint"] + 1),
            10,
        )
        thread = threading.Thread(target=executor.spin, daemon=True)
        probe_thread = threading.Thread(target=probe_executor.spin, daemon=True)
        try:
            probe_thread.start()
            thread.start()
            # Wait until DDS discovery is complete and the subscription is
            # actually receiving; otherwise a CI startup delay is counted as a
            # control-rate deficit.
            deadline = time.monotonic() + 10.0
            while counter["joint"] < 5 and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertGreaterEqual(counter["joint"], 5, "no /joint_states messages during warmup")
            counter["joint"] = 0
            window_s = 3.0
            start = time.monotonic()
            time.sleep(window_s)
            elapsed = time.monotonic() - start
            rate = counter["joint"] / elapsed

            # Shared CI runners are noisier than a local 8-vCPU WSL machine.
            # The 60% CI floor still catches the starvation regressions that
            # motivated this test (4.98-6.5 Hz); local runs keep the stricter
            # 80% floor. This is a CI-only tolerance, not a local weakening.
            ci = os.environ.get("CI", "").lower() in {"1", "true", "yes"}
            floor = 0.6 if ci else 0.8
            self.assertGreaterEqual(
                rate,
                floor * node._control_rate,
                f"/joint_states rate {rate:.2f} Hz fell below {floor:.0%} of "
                f"{node._control_rate} Hz",
            )
            stats = node._render_coordinator.stats()
            print(
                f"joint_rate_hz={rate:.2f} elapsed_s={elapsed:.2f} "
                f"published={stats['published']} skipped_busy={stats['skipped_busy']}",
                flush=True,
            )
            self.assertGreater(
                stats["skipped_busy"],
                0,
                "slow renders were not skipped/counted; camera callbacks may be serialized",
            )
            self.assertLessEqual(stats["published"], 10)
        finally:
            probe_executor.shutdown()
            probe_thread.join(timeout=2.0)
            probe.destroy_node()
            executor.shutdown()
            thread.join(timeout=2.0)
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
