# sim2real

End-to-end sim2real robotics data pipeline and VLA fine-tuning framework for a
Franka Panda, built around ROS 2 Jazzy, MuJoCo (Isaac Sim optional), LeRobot,
and SmolVLA + LoRA.

**Task 1 status: scaffold only.** No simulator, dataset, training, mapping,
inference, or safety code exists yet.

## What exists today

- Pure-Python core library (`src/sim2real`) with contract constants, bounded
  JSON metadata validation, and status parsing.
- ROS 2 Jazzy workspace with `sim2real_msgs` (six interfaces) and four empty
  node-adapter packages.
- Documentation, config stubs, scripts, CPU-only Dockerfile, CI workflow.
- Hardware-free tests.

## Quick start

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest -ra
```

ROS 2 (only if Jazzy is installed):

```bash
source /opt/ros/jazzy/setup.bash
cd ros2_ws
colcon build --symlink-install
colcon test
colcon test-result --verbose
```

See `CLAUDE.md` for verified environment details and canonical commands, and
`docs/ARCHITECTURE.md` for the design and contract table.

## Layout

```
src/sim2real/        pure-Python core (no rclpy)
ros2_ws/src/         ROS 2 interface + adapter packages
configs/             per-subsystem YAML stubs (schema_version: 1)
docs/                architecture, datasets, safety, deployment, master spec
scripts/             environment check, setup, test entry points
tests/               hardware-free tests
data/                ignored; holds datasets/rollouts locally
```

## Safety

This project has no physical robot and no implemented safety system. Simulator
success is not a safety case. See `docs/SAFETY.md`.

## License

Apache-2.0 (provisional; confirm before distribution).
