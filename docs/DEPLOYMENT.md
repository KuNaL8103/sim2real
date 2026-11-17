# Deployment

## Status

**Not yet implemented.** There is no deployment procedure, service unit,
container image publication, or hardware bring-up. The Dockerfile in this
repository is a CPU-only development image and has **not** been built or pulled.

Task 11 adds a **mock-hardware** control stack
(`ros2_ws/src/sim2real_control/launch/control_mock.launch.py`) and a software
safety watchdog. That is plumbing evidence only. Mock hardware does not model
dynamics, latency, torque, velocity enforcement, or collisions, and it says
nothing about physical safety.

## Tiers

| Tier | Target | Status |
| --- | --- | --- |
| dev/test | CPU-only, fake backend | **available**; deterministic `FakeBackend` is the CI default |
| high-fidelity | MuJoCo, optional Isaac Sim | **MuJoCo headless available**; Isaac Sim **unavailable on this machine** per published requirements (see `ARCHITECTURE.md`) |
| physical | Franka Panda via ROS 2 / ros2_control | out of scope |

## Physical bring-up

**Physical bring-up is a separate step requiring the user's explicit
confirmation.** It is not part of Task 11 and was not run. No physical Franka is
connected to this repository.

The full deployment guide is Task 12. Before any physical bring-up, at minimum:

1. a hardware e-stop in the safety-rated circuit (a software `/estop` is not a
   safety-rated emergency stop);
2. a hazard analysis and workspace/limit review against the actual robot model;
3. a separate gripper action path (on a real Franka the gripper is not an 8th
   JointTrajectoryController joint);
4. verified controller/hardware limits, torque/velocity enforcement, and
   communication-loss behaviour;
5. a documented safe-state, recovery, and operator procedure.

Do not connect this stack to a physical robot without explicit approval.

## Container image

`Dockerfile`:

- Base: `ros:jazzy-ros-base` (tag verified against Docker Hub on 2026-10-30).
- Non-root user `sim2real` (UID 1000).
- Installs `colcon`, `rosdep`, `curl`, `git`, and a pinned `uv` (0.13.6).
- No GPU runtime, no CUDA, no datasets, no model weights.
- Build approval is required; image size is a large download.

### Host constraints for Docker

- Docker Desktop's WSL integration for **Ubuntu-24.04 is currently disabled**.
  Enabling it is required before `docker` can run in this distro.
- The WSL VM is configured with `memory=6GB` (5.8 GiB visible), 2 processors,
  and 8 GB swap. Building or running images competes with ROS 2, RViz, and
  MuJoCo for that RAM.
- Do not build or pull the image without explicit approval; no build has been
  attempted in this project.

## Environment setup

```bash
# Core/CI environment (no ROS)
uv sync

# Core + ROS test environments
./scripts/setup_envs.sh
```

`.venv-ml` (torch, LeRobot, PEFT) is documented only and is not created in
Task 1.

## CI

`.github/workflows/ci.yml` defines a CPU-only core job and a ROS 2 Jazzy job.
The workflow has not run yet. Do not treat CI as passing until the Actions tab
shows a green run after the first push.

## Not yet defined

- Model artifact storage and versioning.
- Deployment target and rollout procedure.
- Real robot hardware interface and real ros2_control configuration (Task 11
  provides only `mock_components/GenericSystem`).
- Reproducibility pins for GPU environments.
- Full physical bring-up guide and validation matrix (Task 12).
