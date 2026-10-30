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

Recorded from the Task 1 preflight and this close-out session on 2026-10-30.
This is a factual record, not a claim that the environment is fully validated.

| Item | Verified value |
| --- | --- |
| Kernel | `6.18.54.1-microsoft-standard-WSL2` |
| Distro | Ubuntu 24.04.4 LTS (noble), `WSL_DISTRO_NAME=Ubuntu-24.04` |
| Repository root | Discovered with `git rev-parse --show-toplevel`; the verified checkout is `/home/kunal/Python/sim2real` on the Linux filesystem. No fixed path is assumed by docs or scripts. |
| Python | 3.12.3 (`/usr/bin/python3`) |
| ROS 2 | Jazzy at `/opt/ros/jazzy`; `rclpy` imports with system Python |
| ROS CLI | `ros2` present; `colcon` 0.21.0; `rosdep` present |
| ROS GUI tools | `rviz2`, `robot_state_publisher`, `tf2_tools` present |
| uv | 0.13.6, installed during Task 1 at `~/.local/bin/uv` (approved) |
| git | 2.43.0; user `KuNaL8103 <kunaldabbe2003@gmail.com>` |
| CPU / RAM | 2 vCPU; 5.8 GiB RAM total, 4.1 GiB available at close-out |
| Disk | 1007G total, 930G available on `/` |
| GPU | NVIDIA GeForce RTX 4050 Laptop, 6141 MiB; driver 551.76 reports CUDA 12.4 capability |
| CUDA toolkit | Not installed and not claimed; `nvidia-smi` works through the Windows driver |
| WSL memory config | `/mnt/c/Users/*/.wslconfig` present: `memory=6GB`, `swap=8GB`, `processors=2`, `localhostForwarding=true`, `guiApplications=true` |
| WSL GPU bridge | `/dev/dxg` present; `/usr/lib/wsl/lib` includes `libcuda.so.1`, `libd3d12.so`, `libnvidia-ml.so.1` |
| Display / graphics | `DISPLAY=:0`, `WAYLAND_DISPLAY=wayland-0`; `libegl1`, `libgl1`, Mesa 25.2.8, and `mesa-utils` installed |
| Docker | Windows Docker CLI path is visible, but WSL integration is **disabled** (`docker` unusable here) |
| GitHub CLI | `gh` not installed (not needed for push via git credentials) |
| VS Code WSL check | **VERIFIED 2026-10-30** from user-supplied integrated-terminal output: `WSL_DISTRO_NAME=Ubuntu-24.04`, kernel `6.18.54.1-microsoft-standard-WSL2`, `pwd=/home/kunal/Python/sim2real`; `ms-vscode-remote.remote-wsl` is installed. Status-bar text was not included in that output. |

**VS Code WSL check: VERIFIED** by the user's terminal output on 2026-10-30.
The status bar itself was not captured, but the integrated terminal is the WSL
Ubuntu-24.04 environment.

**Master spec SHA-256 (committed copy):** `3ec2f13b8abc8f61ee65a9b66683346f8557fe85d3ae1959d6cab13986b3d1ad`
(717 lines, 43,406 bytes, LF-only; the CR-stripped hash is identical). The
original Windows file's hash is **UNVERIFIED** until the user runs `Get-FileHash`
and pastes it.

### Hardware limits for later tiers

Official Isaac Sim 6.2.0 minimums: GeForce RTX 4080, **16 GB VRAM**, **32 GB
RAM**, 4 CPU cores, 50 GB SSD, Linux driver 595.58.03.
Source: <https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html>
(fetched 2026-10-30).

This machine has an RTX 4050 Laptop with 6 GB VRAM, 5.8 GiB WSL RAM, 2 vCPUs,
and driver 551.76. Therefore:

- **High-fidelity Isaac Sim tier: UNAVAILABLE on this machine** (based on
  published requirements, not tested). No install is authorized; the guarded
  adapter and docs remain.
- **SmolVLA fine-tuning feasibility: UNKNOWN** until measured in Task 8. The
  6 GB GPU may be insufficient; cloud/Colab-style GPU runs are an alternative,
  with checkpoints and provenance metadata brought back into the repo.
- **MuJoCo rendering + ROS on 5.8 GiB RAM is a risk.** Task 2 must keep camera
  resolutions small and measure memory rather than assuming it fits.

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
10. Apache-2.0 is provisional (no LICENSE file yet).
11. Repository root is resolved dynamically (`git rev-parse --show-toplevel`);
    no fixed path is assumed. The verified checkout is `/home/kunal/Python/sim2real`.
12. `Last verified test run:` records the **core** suite; the optional
    `Last verified ROS test run:` line records ROS separately.

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

- Last verified test run: passed=109 failed=0 skipped=1 errors=0 | date=2026-10-30 | base_commit=3cbd491
- Last verified ROS test run: passed=115 failed=0 skipped=0 errors=0 | date=2026-10-30 | base_commit=3cbd491
- Task 1 status: complete with pending checks: CI run (user confirmation), spec original hash, license choice, action pinning, WSL config tuning

The first line records the **core** suite (`.venv`, no ROS dependency). The
second line is optional and records the ROS-enabled suite separately.

Run breakdown:

- Core-only: **109 passed, 0 failed, 1 skipped, 0 errors**. The skip is the
  ROS round-trip module, which now skips when either `rclpy` or `sim2real_msgs`
  is unavailable. This makes plain `uv run pytest -ra` safe in a ROS-sourced
  shell as well.
- ROS (`.venv-ros` after `colcon build`): **115 passed, 0 failed, 0 skipped,
  0 errors**.
- `colcon test` / `colcon test-result --verbose`: **7 entries, 0 failures,
  0 errors** (1 CTest wrapper + 6 interface round-trips). The four
  `ament_python` import checks also passed in the colcon console; they are not
  counted in the xunit total.
- `ros2 interface list` shows all six `sim2real_msgs` interfaces; each
  `ros2 interface show` exits successfully.
- The previous baseline line recorded `passed=113`, which was the ROS count,
  not the core count. The convention is now explicit and split into two lines.
- CI run `41053040940` for `3cbd491`: **both jobs reported `success`** via the
  GitHub API. CI remains marked pending until the user confirms the Actions tab
  for the close-out commit.

## Known limitations

- No simulator, dataset, training, mapping, inference, or safety code exists.
- **Isaac Sim 6.2.0 high-fidelity tier is UNAVAILABLE on this machine** based
  on published minimums (16 GB VRAM, 32 GB RAM, RTX 4080, 4 cores, driver
  595.58.03) versus the measured RTX 4050 6 GB / 5.8 GiB RAM / 2 vCPU /
  driver 551.76. Not tested; no install authorized.
- **SmolVLA fine-tuning feasibility is UNKNOWN** until Task 8 measures it.
  Cloud/Colab GPU runs are an alternative path with provenance metadata.
- **MuJoCo rendering + ROS on 5.8 GiB RAM is a risk.** Task 2 must use small
  camera resolutions and measure memory.
- Docker image not pulled or built; Docker Desktop's WSL integration for
  Ubuntu-24.04 is disabled.
- The first CI run failed twice (`setup-uv@v10` had no floating tag; rosdep had
  no rule for `ament_python`). Both are fixed; the 109/115 close-out run is not
  yet confirmed in the Actions tab.
- `.venv-ml` is documented only and not created.
- License is provisional Apache-2.0; there is no `LICENSE` file and the user
  has not chosen a license.
- `hydra-core`/`omegaconf` are installed for later tasks but unused in Task 1.
- CI actions are pinned to tags (`actions/checkout@v7`,
  `astral-sh/setup-uv@v10.3.3`), not commit SHAs. Pinning policy is pending.
- The original Windows copy of the master spec has not been hash-compared.

## Pending manual checks

- [x] VS Code WSL verification — confirmed 2026-10-30 from user terminal output.
- [x] Copy the master spec to `docs/MASTER_SPEC.md` and hash the committed copy.
- [ ] Run `Get-FileHash` on the original Windows `SIM2REAL_MASTER_SPEC.md` and
  paste the result so provenance can be marked MATCH or MISMATCH.
- [ ] Confirm the GitHub Actions tab for the close-out commit (API evidence only
  until then).
- [ ] Choose the license: Apache-2.0, MIT, or undecided.
- [ ] Decide whether CI actions should be pinned to commit SHAs.
- [ ] Paste Windows total physical memory and confirm the `.wslconfig` proposal
  (current: `memory=6GB`, `swap=8GB`, `processors=2`).
- [ ] Approve and build the Docker image separately; enable Docker Desktop WSL
  integration first.
