# sim2real

A robotics data pipeline that turns a simulated Franka Panda pick-and-place task
into a versioned training dataset, fine-tunes a vision-language-action (VLA)
policy offline, evaluates it in closed loop, and feeds failures back into data
generation. A ROS 2 safety watchdog sits between raw policy output and any
controller.

**In plain English:** the pipeline works end to end on a CPU and in headless
simulation, and the trained policy is **not** good at the task yet. The
deliverable is the pipeline, its tests, and its honest validation — not a
working robot policy.

## What works

- Deterministic fake backend and a headless MuJoCo Franka tabletop with two
  cameras (256x256 RGB-D), seeded domain randomization, and a scripted expert.
- Atomic rollout staging, trajectory filtering with measured quality metrics, a
  versioned `curated_v1` writer, and a two-layer LeRobot / Open X dataset builder
  with statistics, split, and mixing.
- Offline SmolVLA + PEFT LoRA fine-tuning with provenance checkpoints and
  offline L1 validation (CPU fake path and a measured GPU smoke run).
- Closed-loop evaluation with paired expert comparison, failure analysis
  (GMM/KDE/neighbour), failure-targeted rollout generation, and a guarded,
  append-only flywheel ledger.
- A ROS 2 VLA inference bridge and a safety watchdog: raw `/vla/raw_action` is
  validated, smoothed, retimed, and only then published to the controller;
  latched `/estop`, `/estop_reset`, `/safety/status`, `/safety/diagnostics`.
- An explicit pipeline orchestrator (`GENERATE -> FILTER -> BUILD -> TRAIN ->
  EVALUATE -> ANALYZE_FAILURES -> TARGETED_GENERATE`) with deterministic stage
  ids, resume, quarantine, hard cycle/time/write/GPU bounds, approval gates, and
  full lineage manifests.
- A mock-hardware ros2_control path and an integrated launch for
  cameras → inference → safety → controller.

## What does not work

- **The trained policy does not complete the task.** The pre-registered overfit
  rule is unmet (first-step arm L1 0.0333 rad vs a 0.010 bar; gripper
  0.00688 m vs 0.005; closed loop 0/6 on training seeds). The local GPU is
  throttled at 210 MHz. This is reported as a failure, not hidden.
- **No live closed loop through MuJoCo.** `sim_node` has no command
  subscription; closed-loop evaluation uses the in-process runner. The mock
  hardware stack has no cameras. See `docs/ARCHITECTURE.md`.
- **No physical robot, no physical safety, no functional-safety certification.**
  Mock hardware models no dynamics, latency, torque, velocity enforcement, or
  collisions. A software e-stop is not a safety-rated emergency stop.
- **Isaac Sim is unavailable on this machine** (6 GB VRAM vs a 16 GB minimum;
  requirements-based, not tested).
- MoveIt 2 is not installed; its adapter is a refusing stub. OctoMap is
  deferred. Real Bridge data and video encoding were never exercised.
- Latency runs 3 and 4 are NOT RUN. Remote-run provenance is user-reported.

## Quick start (no GPU, no robot)

```bash
# Core environment, lint, and the ROS-free test suite
uv sync --group sim
env -u PYTHONPATH -u AMENT_PREFIX_PATH uv run ruff check .
env -u PYTHONPATH -u AMENT_PREFIX_PATH uv run ruff format --check .
env -u PYTHONPATH -u AMENT_PREFIX_PATH uv run pytest -ra
```

Plan the whole pipeline without writing anything (the default mode):

```bash
uv run python scripts/pipeline_orchestrator.py --config configs/pipeline/default.yaml
```

Run one fake-backend cycle on the CPU (real scripts, fake TRAIN only; seeds
20950-20959):

```bash
uv run python scripts/pipeline_orchestrator.py --config configs/pipeline/fake_e2e.yaml \
    --root /tmp/sim2real/pipeline_e2e --execute --cycles 1
```

Lineage for a checkpoint:

```bash
uv run python scripts/pipeline_orchestrator.py --config configs/pipeline/default.yaml \
    --lineage <checkpoint-dir-or-metadata-hash>
```

Headless MuJoCo smoke (physics only):

```bash
uv run python scripts/collect_rollouts.py --backend mujoco --num 1 --seed 21000 \
    --out /tmp/sim2real/mujoco_smoke
```

ROS 2 (only if Jazzy is installed):

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --parallel-workers 2 --base-paths ros2_ws/src
source ros2_ws/install/setup.bash
colcon test && colcon test-result --verbose
.venv-ros/bin/python -m pytest -ra
```

Integrated launches:

```bash
# Simulation graph (no controller): sim + mapper + fake inference + safety
ros2 launch sim2real_control pipeline_sim.launch.py backend:=fake

# Mock-hardware graph: ros2_control + safety + fake inference + fake observations
ros2 launch sim2real_control pipeline_mock_hw.launch.py
```

## Repository layout

```
src/sim2real/        pure-Python core (no rclpy): sim, randomization, expert,
                     filtering, data, mapping, dataset_builder, training,
                     evaluation, inference, safety, orchestrator
ros2_ws/src/         thin ROS 2 adapters: sim2real_msgs, sim2real_sim,
                     sim2real_perception, sim2real_vla, sim2real_control
configs/             versioned YAML per subsystem (schema_version: 1)
scripts/             entry points: collection, filtering, datasets, training,
                     evaluation, policy server, orchestrator, verification
tests/               hardware-free tests and analytic fixtures
docs/                architecture, validation, deployment, recovery, safety, ...
data/                ignored; staging, curated datasets, evaluations, flywheel
outputs/             ignored; checkpoints and training artifacts
```

## Three environments and why

| Environment | Purpose | ROS? |
| --- | --- | --- |
| `.venv` | core library, lint, CPU tests, CI | no |
| `.venv-ros` | ROS adapter tests and integration (`--system-site-packages`) | yes |
| `.venv-ml` | offline VLA fine-tuning (torch, LeRobot, PEFT); opt-in, never CI | no |

ROS Jazzy binds to the system Python and system NumPy. `.venv-ros` must see the
system `rclpy` and must not install its own NumPy. `.venv-ml` holds the ML stack
and must not contain ROS or simulator packages. Never mix them.

## Validation

`docs/VALIDATION.md` is the matrix: each level says what it proves, what it does
**not** prove, its status, and the command/date/commit evidence. A skipped test
is not a pass. Integration levels that were not run are marked NOT RUN.

## Safety disclaimer

This project has no physical robot and no safety-rated system. Simulator
success, mock-hardware success, and unit tests are not evidence that a real
Franka will move or stop safely. A software e-stop is not a substitute for a
hardware e-stop in the safety-rated circuit. See `docs/SAFETY.md` and
`docs/DEPLOYMENT.md`.

## Documentation

- `docs/ARCHITECTURE.md` — packages, data flow, contracts/QoS, decision log.
- `docs/VALIDATION.md` — validation matrix and open items.
- `docs/DEPLOYMENT.md` — mock bring-up and documented-only physical bring-up.
- `docs/RECOVERY.md` — failure-recovery procedures.
- `docs/SAFETY.md` — safety design, limits, measured false positives.
- `docs/EVALUATION.md`, `docs/TRAINING.md`, `docs/DATASETS.md`,
  `docs/FILTER_METRICS.md`, `docs/VLA_INFERENCE.md`, `docs/REMOTE_TRAINING.md`.
- `docs/MANUAL_CHECKS.md` — pending manual checkpoints for the operator.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
