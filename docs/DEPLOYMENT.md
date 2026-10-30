# Deployment

## Status

**Not yet implemented.** There is no deployment procedure, service unit,
container image publication, or hardware bring-up. The Dockerfile in this
repository is a CPU-only development image and has **not** been built or pulled.

## Tiers

| Tier | Target | Status |
| --- | --- | --- |
| dev/test | CPU-only, fake backend | scaffolded; fake backend is Task 2 |
| high-fidelity | MuJoCo, optional Isaac Sim | MuJoCo pending Task 2; Isaac Sim **unavailable on this machine** per published requirements (see `ARCHITECTURE.md`) |
| physical | Franka Panda via ROS 2 / ros2_control | out of scope |

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
- ROS 2 launch composition.
- Hardware interface and ros2_control configuration.
- Reproducibility pins for GPU environments.
