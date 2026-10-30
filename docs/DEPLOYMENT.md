# Deployment

## Status

**Not yet implemented.** There is no deployment procedure, service unit,
container image publication, or hardware bring-up. The Dockerfile in this
repository is a CPU-only development image and has **not** been built or pulled.

## Tiers

| Tier | Target | Status |
| --- | --- | --- |
| dev/test | CPU-only, fake backend | scaffolded; fake backend is Task 2 |
| high-fidelity | MuJoCo, optional Isaac Sim | not implemented |
| physical | Franka Panda via ROS 2 / ros2_control | out of scope |

## Container image

`Dockerfile`:

- Base: `ros:jazzy-ros-base` (tag verified against Docker Hub on 2026-10-30).
- Non-root user `sim2real` (UID 1000).
- Installs `colcon`, `rosdep`, `curl`, `git`, and a pinned `uv` (0.13.6).
- No GPU runtime, no CUDA, no datasets, no model weights.
- Build approval is required; image size is a large download.

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
- ROS 2 launch composition.
- Hardware interface and ros2_control configuration.
- Reproducibility pins for GPU environments.
