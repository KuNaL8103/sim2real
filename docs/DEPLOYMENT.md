# Deployment

Two paths are documented here:

1. **Mock-hardware bring-up** — runnable on this machine, no robot, no GPU. It is
   plumbing evidence only.
2. **Physical Franka bring-up** — a **documented-only** procedure. It was **not
   run**, no robot is connected, and it requires the operator's explicit written
   confirmation before any step.

Nothing in this file is a functional-safety claim. A software `/estop` is not a
safety-rated emergency stop, and mock hardware models no dynamics, latency,
torque, velocity enforcement, or collisions.

## 1. Mock-hardware bring-up (runnable here)

### What is real and what is mocked

| Piece | Real or mock | Notes |
| --- | --- | --- |
| ROS 2 graph, QoS, services | real | rclpy, DDS, message types |
| `safety_watchdog_node` | real | validation, EMA, retime, hold, e-stop latch |
| `vla_inference_node` (`policy_backend:=fake`) | fake policy | observation assembly and chunk publication are real |
| `controller_manager` + `joint_trajectory_controller` | real ROS packages | `mock_components/GenericSystem` mirrors positions to states |
| `fake_observation_publisher` | fake | synthetic 8x8 images, synthetic table cloud, fixed instruction |
| `sim_node` | **not started** | `joint_state_broadcaster` is the single `/joint_states` owner |

`mock_components` has `calculate_dynamics=false`: it does not model dynamics,
latency, torque, velocity enforcement, or collisions.

### Commands

Build once:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --parallel-workers 2 --base-paths ros2_ws/src
source ros2_ws/install/setup.bash
```

Terminal A — integrated mock-hardware graph (control stack + safety + fake
inference + fake observations):

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
ros2 launch sim2real_control pipeline_mock_hw.launch.py
```

Terminal B — inspection:

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
ros2 topic info /joint_trajectory_controller/joint_trajectory -v
ros2 topic hz /joint_trajectory_controller/joint_trajectory
ros2 topic echo /safety/status
ros2 topic echo /safety/diagnostics --once
ros2 service call /estop std_srvs/srv/Trigger {}
ros2 service call /estop_reset std_srvs/srv/Trigger {}   # immediate: expect refusal
# after at least 2 s of fresh inputs:
ros2 service call /estop_reset std_srvs/srv/Trigger {}   # expect success
```

Expected: the controller topic has **Publisher count 1**, node
`safety_watchdog_node`; commands are forwarded only while inputs are fresh and
the map is valid; `/estop` latches and stops forwarding; an immediate reset is
refused and a reset after the hold succeeds. The reset commands no motion.

The launch refuses `start_sim:=true`: `sim_node` and `joint_state_broadcaster`
would both publish `/joint_states`.

### Simulation graph without a controller

`pipeline_sim.launch.py` starts `sim_node` + mapper + fake inference + safety
and **no controller**. `sim_node` has no command subscription, so this launch
does not close a loop; see `docs/ARCHITECTURE.md` for the documented gap.

## 2. Physical Franka bring-up (DOCUMENTED ONLY — NOT RUN)

**Status: NOT RUN. No physical robot is connected to this repository. Do not
execute this procedure without an explicit written go-ahead and a hazard review.
It is a checklist, not validated instructions.**

### 2.1 Prerequisites (fetched 2026-11-18)

Sources:

- FCI minimum system/network requirements:
  <https://frankarobotics.github.io/docs/doc/libfranka/docs/system_requirements.html>
- franka_ros2 Jazzy compatibility matrix:
  <https://frankarobotics.github.io/docs/doc/franka_ros2_jazzy/franka_ros2/doc/compatibility_matrix.html>
- FCI overview (FCI activation on system >= 4.2.0, Desk -> Activate FCI):
  <https://frankarobotics.github.io/docs/>

| Requirement | Official statement | Verified here? |
| --- | --- | --- |
| OS kernel | Linux with a **PREEMPT_RT** patched kernel | **NO** — this WSL2 kernel is not PREEMPT_RT |
| Network card | 100BASE-TX | **NO** |
| Network path | Workstation connected to the **Control** LAN port (shop-floor network), **not** the Arm port; direct connection preferred, no switches/relays | **NO** |
| Cycle budget | RTT + control-loop execution + robot processing **< 1 ms**; >20 consecutive dropped packets stop the robot | **NOT MEASURED** |
| ROS 2 | Jazzy on Ubuntu 24.04 recommended | Jazzy is installed here |
| franka_ros2 | Jazzy: `franka_ros2 >= v3.4.0`, `libfranka >= 0.20.4`, `franka_description >= 2.8.0` (latest tested row) | **NO** — not installed |
| FCI | Enabled on the robot (system >= 4.2.0: Desk -> Activate FCI) | **NO** |
| Robot model | Confirm FER/Panda vs FR3 and the exact system version | **NO** |
| libfranka <-> robot system pairing | Use the official libfranka compatibility matrix | **UNVERIFIED** — matrix not fetched; check <https://frankarobotics.github.io/docs/doc/libfranka/docs/compatibility_matrix.html> |
| Hardware e-stop | Required in the safety-rated circuit | **NO** |

The repository model is the **FER/Panda** Menagerie `panda.xml`, not FR3. The
official `franka_description` must be used on hardware and the joint/frame names
mapped deliberately; do not assume the mock URDF is compatible.

### 2.2 Replacing the mock stack (staged, each step gated)

1. **Hazard review.** Workspace limits, payload, speed, separation distances,
   and a documented safe state. No motion before this is signed off.
2. **Hardware e-stop.** Wire and test the safety-rated e-stop independently of
   ROS. The software `/estop` is not a substitute.
3. **Install the official description and driver** on a PREEMPT_RT host that
   meets the network requirements. Do not reuse `panda_mock.urdf`.
4. **Single owner.** Disable `sim_node` and the mock broadcaster. The real
   `franka_hardware` / `joint_state_broadcaster` must be the only
   `/joint_states` publisher.
5. **Split the gripper.** `/vla/raw_action` carries 8 values
   (`panda_joint1..7` + `panda_finger_joint1`). On a real Franka the gripper is a
   **separate action server**; the safety node's 8th value must be routed there,
   not sent to the arm's `JointTrajectoryController`. **Not implemented.**
6. **Strict safety profile.** Use `configs/safety/default.yaml`
   (`obstacle_mode: strict`). `configs/safety/sim_mock.yaml` is simulation/mock
   only and must not be used on hardware.
7. **Slow first motions.** Configure a low velocity/acceleration scaling
   (for example 0.05-0.10 of nominal) in the controller, keep the workspace
   small, and stand at the hardware e-stop. Verify `/safety/status` and the
   measured state before enabling motion.
8. **Staged checks**, each stopping on any anomaly:
   - controller activates with no trajectory;
   - a single 1-point hold at the measured state;
   - a 3-point chunk at the measured state plus 0.01 rad;
   - a short approach/retreat with no object;
   - the scripted expert at reduced speed, empty workspace;
   - only then the pick-and-place task.
9. **Communication-loss behaviour.** Kill the safety node, the policy server,
   and the map publisher one at a time; the robot must reach the documented safe
   state. This has **not** been verified on hardware.
10. **Record** every run: git commit, config hashes, controller limits, measured
    rates, and any deviation. Nothing from a mock run transfers as evidence.

### 2.3 What is explicitly not covered

- torque/effort enforcement, whole-arm/self-collision checking, singularity
  handling, payload estimation, fieldbus timing, encoder/calibration faults;
- a safety-rated e-stop or safety PLC;
- gripper integration (separate action server);
- MoveIt 2 (not installed, adapter unexercised);
- Isaac Sim (UNAVAILABLE on this machine).

## 3. Container image

`Dockerfile` is a CPU-only development image (`ros:jazzy-ros-base`, non-root
user, pinned `uv`). It has **not** been built or pulled; Docker Desktop's WSL
integration for Ubuntu-24.04 is disabled. Building requires approval.

## 4. Environment setup

```bash
# Core/CI environment (no ROS)
uv sync --group sim

# Core + ROS test environments
./scripts/setup_envs.sh

# Optional ML environment (opt-in; never CI)
ML_TORCH_BACKEND=cu126 ./scripts/setup_envs.sh --with-ml
```

See `docs/ARCHITECTURE.md` for why there are three environments.
