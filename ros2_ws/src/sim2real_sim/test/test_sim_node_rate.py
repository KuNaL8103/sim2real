"""ROS rate test: a slow renderer must not starve /joint_states.

Uses the fake backend with an injected 300 ms render delay, so it is
hardware-independent. No MuJoCo timing threshold is asserted.
"""

from __future__ import annotations

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

    from sim2real import contracts
    from sim2real.sim import create_backend
    from sim2real_sim.sim_node import SimNode

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
            # A 5 s window averages short scheduler bursts. The regression this
            # test exists to catch is the documented 1.87-3.28 Hz blocking
            # renderer and the 4.98-6.5 Hz shared-runner starvation; an 8 Hz
            # floor (40% of 20 Hz) still catches all of them. The previous 3 s
            # window intermittently measured 8.33 Hz under load even though the
            # control path was healthy.
            window_s = 5.0
            start = time.monotonic()
            time.sleep(window_s)
            elapsed = time.monotonic() - start
            rate = counter["joint"] / elapsed

            floor = 0.4
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
            # The 5 s window can legitimately fit more slow renders than the
            # old 3 s window (the injected delay is 0.3 s); the real assertion
            # is that most submissions were skipped, not rendered serially.
            self.assertLessEqual(stats["published"], 20)
        finally:
            probe_executor.shutdown()
            probe_thread.join(timeout=2.0)
            probe.destroy_node()
            executor.shutdown()
            thread.join(timeout=2.0)
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
