# CLAUDE.md

Project instructions and verified status for AI-assisted work on `sim2real`.
Update the **Current status** block at the end of every task. Never claim a
check passed without output from the current session.

## Project summary

End-to-end sim2real robotics data pipeline and VLA fine-tuning framework:

- Franka Panda, MuJoCo first, Isaac Sim optional and gated.
- ROS 2 Jazzy on Ubuntu 24.04 under WSL2.
- LeRobot datasets, SmolVLA + LoRA fine-tuning.
- Pure-Python core algorithms, thin ROS 2 adapters, CPU-testable CI.
- No physical robot.

Task 1 created scaffolding, interfaces, documentation, CI, and hardware-free
tests. No simulator, dataset, training, mapping, inference, or safety logic is
implemented.

## Verified environment

Recorded from commands run in this session on 2026-10-30. This is a factual
preflight, not a claim that the environment is fully validated.

| Item | Verified value |
| --- | --- |
| Kernel | `6.18.54.1-microsoft-standard-WSL2` |
| Distro | Ubuntu 24.04.4 LTS (noble), `WSL_DISTRO_NAME=Ubuntu-24.04` |
| Working directory | `/home/kunal/Python/sim2real` (Linux filesystem; **not** `~/projects/sim2real`) |
| Python | 3.12.3 (`/usr/bin/python3`) |
| ROS 2 | Jazzy at `/opt/ros/jazzy`; `rclpy` imports with system Python |
| ROS CLI | `ros2` present; `colcon` 0.21.0; `rosdep` present |
| uv | 0.13.6, installed this session at `~/.local/bin/uv` (approved) |
| git | 2.43.0; user `KuNaL8103 <kunaldabbe2003@gmail.com>` |
| CPU / RAM | 2 vCPU; 5.8 GiB RAM total, 4.2 GiB available |
| Disk | 1007G total, 930G available on `/` |
| GPU | NVIDIA GeForce RTX 4050 Laptop, 6141 MiB; driver 551.76 reports CUDA 12.4 capability |
| CUDA toolkit | Not installed and not claimed; `nvidia-smi` works through the Windows driver |
| Docker | Windows Docker CLI path is visible, but WSL integration is **disabled** (`docker` unusable here) |
| GitHub CLI | `gh` not installed (not needed for push via git credentials) |
| VS Code WSL check | **PENDING** — user has not yet supplied the integrated-terminal output |

**VS Code WSL check: PENDING.** Do not write "VS Code WSL verified" or
"environment ready" until the user confirms:

```bash
echo $WSL_DISTRO_NAME ; uname -r ; pwd
```

expected `Ubuntu-24.04`, a kernel containing `microsoft-standard-WSL2`, and a
`/home` path, with the VS Code status bar showing `WSL: Ubuntu-24.04`.

**Master spec SHA-256:** pending. `docs/MASTER_SPEC.md` is a placeholder.

## Canonical commands

```bash
# Core / CI
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest -ra

# One-shot core verification
./scripts/run_core_tests.sh

# Create .venv and .venv-ros
./scripts/setup_envs.sh

# ROS 2 build + test + .venv-ros tests
./scripts/build_and_test_ros.sh
```

ROS commands must run from a shell where `/opt/ros/jazzy/setup.bash` is sourced.
Never run ROS nodes with `.venv/bin/python`.

## Conventions

- Python 3.12 only (`>=3.12,<3.13`).
- Line length 100; ruff rules `E,F,I,UP,B,SIM`; formatting is enforced.
- All algorithms in `src/sim2real`; **no `rclpy` imports there**.
- ROS 2 packages are thin adapters: plumbing and message conversion only.
- Message fields `lower_snake_case`; IDL constants `UPPER_CASE`.
- Contract names live in `src/sim2real/contracts.py` and are mirrored in
  `docs/ARCHITECTURE.md`; update both plus tests together.
- ROS 2 frame IDs are relative (no leading `/`).
- Tests must be hardware-free by default. ROS-marked tests skip with a reason
  when `rclpy` is unavailable; a skip is never a pass.
- Do not commit data, weights, caches, bags, or secrets.
- Do not force-push. Do not discard unrelated changes.

## Two-environment rule

| Environment | Purpose | ROS? |
| --- | --- | --- |
| `.venv` | core library, lint, CPU tests, CI | no |
| `.venv-ros` | ROS adapter tests and integration | yes, via `--system-site-packages` |
| `.venv-ml` | torch / LeRobot / PEFT (Task 8) | no; not created yet |

`.venv-ros` must see the system `rclpy` and system NumPy 1.26.4. Do not install
NumPy there; a venv copy would shadow the system version. `.venv` may use a
newer NumPy because ROS is never imported there. Never mix the environments.

## Safety rules

- **Mock hardware is not physical safety.** Simulator success proves nothing
  about a physical robot.
- No physical robot is connected to this project.
- Raw VLA output must never reach a controller without the safety adapter.
- `NaN` means unknown, never safe.
- E-stop is latched; recovery is explicit (`/estop_reset`, Task 11).
- `/joint_states` has exactly one publisher: sim node **or** ros2_control.
- Safety limits belong in configuration, not scattered literals.
- Do not claim functional safety or standards compliance.

## Decision log

Detailed rationale is in `docs/ARCHITECTURE.md`. Summary:

1. Core algorithms in `src/sim2real`; ROS packages are adapters.
2. Exactly one `/joint_states` owner.
3. `/vla/raw_action` is a `JointTrajectory` of absolute joint targets with
   `time_from_start=(k+1)*dt` and `header.stamp` = newest observation.
4. `/workspace/tsdf_mesh` is a `Marker` of type `TRIANGLE_LIST`.
5. Namespaced `/sim2real/*` services; `/estop` is global and latched.
6. `curated_v1` is planned for Task 5, not defined in Task 1.
7. Images stay on disk; `RolloutEpisode` carries metadata/paths.
8. Separate `.venv` and `.venv-ros`.
9. `metadata_json` is bounded to 64 KiB and rejects NaN/Infinity.
10. Apache-2.0 is provisional.
11. Development path is `/home/kunal/Python/sim2real` (user-selected).

## Task checklist

- [x] Task 1 — preflight, scaffold, docs, CI, ROS workspace, `sim2real_msgs`, contracts, first tests
- [ ] Task 2 — `SimBackend`, `FakeBackend`, MuJoCo Franka scene, sim ROS node
- [ ] Task 3 — domain randomization
- [ ] Task 4 — scripted expert and rollout generation
- [ ] Task 5 — trajectory filtering, metrics, `curated_v1`
- [ ] Task 6 — LeRobot export and Open-X dataset mixing
- [ ] Task 7 — RGB-D mapping, point cloud, TSDF
- [ ] Task 8 — SmolVLA + LoRA fine-tuning
- [ ] Task 9 — closed-loop evaluation
- [ ] Task 10 — ROS 2 inference bridge
- [ ] Task 11 — safety watchdog and e-stop
- [ ] Task 12 — orchestration

## Current status

- Last verified test run: passed=113 failed=0 skipped=0 errors=0 | date=2026-10-30 | base_commit=ce2eaa6
- Task 1 status: complete with pending checks: VS Code WSL check, spec copy, CI run

Run breakdown:

- Core-only (`.venv`, no ROS): **107 passed, 0 failed, 1 skipped, 0 errors**. The skip is
  the ROS round-trip test, skipped with an explicit reason (`rclpy` not importable).
- ROS (`.venv-ros` after `colcon build`): **113 passed, 0 failed, 0 skipped, 0 errors**.
- `colcon test` / `colcon test-result --verbose`: **7 entries, 0 failures, 0 errors**
  (1 CTest wrapper + 6 interface round-trips). The four `ament_python` import checks
  also passed in the colcon console; they are not counted in the xunit total.
- `ros2 interface list` shows all six `sim2real_msgs` interfaces; each
  `ros2 interface show` exits successfully.
- Initial push to `origin/main` succeeded; `main` tracks `origin/main` at `ce2eaa6`.
  The GitHub Actions run has **not** been verified by this session.

There is no previous test count to compare against; this is the first task.

## Known limitations

- No simulator, dataset, training, mapping, inference, or safety code exists.
- `docs/MASTER_SPEC.md` is a placeholder; no spec hash recorded.
- Docker image not pulled or built; Docker WSL integration is disabled.
- CI workflow not executed; first run depends on the push and the Actions tab.
- `.venv-ml` is documented only and not created.
- License is provisional (Apache-2.0).
- `hydra-core`/`omegaconf` are installed for later tasks but unused in Task 1.
- Path differs from the requested `~/projects/sim2real` (user-selected).

## Pending manual checks

- [ ] VS Code WSL verification (`echo $WSL_DISTRO_NAME ; uname -r ; pwd` from the integrated terminal).
- [ ] Copy the master spec to `docs/MASTER_SPEC.md` and record its SHA-256.
- [ ] Check the GitHub Actions tab after the first push.
- [ ] Approve and build the Docker image separately if desired.
