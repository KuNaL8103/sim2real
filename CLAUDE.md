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
| CPU / RAM | 2 vCPU, 5.8 GiB RAM total, 8.0 GiB swap visible in WSL (host: 8 cores / 16 logical, 15.3 GB). Requested `memory=10GB`, `swap=4GB`, `processors=8` were **not active** on 2026-10-31. |
| Disk | 1007G total, 930G available on `/` |
| GPU | NVIDIA GeForce RTX 4050 Laptop, 6141 MiB; driver 551.76 reports CUDA 12.4 capability |
| CUDA toolkit | Not installed and not claimed; `nvidia-smi` works through the Windows driver |
| WSL memory config | `/mnt/c/Users/kunal/.wslconfig` still contained `memory=6GB`, `swap=8GB`, `processors=2`, `localhostForwarding=true`, `guiApplications=true` on 2026-10-31. The intended `memory=10GB`, `swap=4GB`, `processors=8` values were not written to that file, so `wsl --shutdown` re-read the old limits. |
| WSL GPU bridge | `/dev/dxg` present; `/usr/lib/wsl/lib` includes `libcuda.so.1`, `libd3d12.so`, `libnvidia-ml.so.1` |
| Display / graphics | `DISPLAY=:0`, `WAYLAND_DISPLAY=wayland-0`, X display reachable; `glxinfo -B` reports Mesa llvmpipe (LLVM 20.1.2), **"Accelerated: no"**, OpenGL 4.5. `MUJOCO_GL=egl` selects llvmpipe. A single `GALLIUM_DRIVER=d3d12` experiment selected `D3D12 (AMD Radeon(TM) Graphics)`, the host iGPU, not the RTX 4050. |
| MuJoCo | 3.17.0 in `.venv` (sim group, NumPy 2.5.4) and `.venv-ros` (`--no-deps`, system NumPy 1.26.4, plus glfw/PyOpenGL/absl-py/etils). MuJoCo Menagerie Panda pinned at `71f066ad0be9cd271f7ed58c030243ef157af9f4` (Apache-2.0). |
| Docker | Windows Docker CLI path is visible, but WSL integration is **disabled** (`docker` unusable here) |
| GitHub CLI | `gh` not installed (not needed for push via git credentials) |
| VS Code WSL check | **VERIFIED 2026-10-30** from user-supplied integrated-terminal output: `WSL_DISTRO_NAME=Ubuntu-24.04`, kernel `6.18.54.1-microsoft-standard-WSL2`, `pwd=/home/kunal/Python/sim2real`; `ms-vscode-remote.remote-wsl` is installed. Status-bar text was not included in that output. |

**VS Code WSL check: VERIFIED** by the user's terminal output on 2026-10-30.
The status bar itself was not captured, but the integrated terminal is the WSL
Ubuntu-24.04 environment.

**Master spec SHA-256 (committed copy):** `3ec2f13b8abc8f61ee65a9b66683346f8557fe85d3ae1959d6cab13986b3d1ad`
(717 lines, 43,406 bytes, LF-only). The original Windows file's CR-stripped
hash is identical, so **the repo copy equals the original with line endings
normalized to LF (verified 2026-10-31)**. The original PowerShell hash
`A14F622074DF8DACE7A655D4EF858FEDFEA67573E25B43E98840FFCDE8ADA0A3` is the CRLF
form.

### Hardware limits for later tiers

Official Isaac Sim 6.2.0 minimums: GeForce RTX 4080, **16 GB VRAM**, **32 GB
RAM**, 4 CPU cores, 50 GB SSD, Linux driver 595.58.03; Python 3.12; Ubuntu
22.04/24.04 or Windows 11; ROS 2 Jazzy recommended on Ubuntu 24.04.
Sources (fetched 2026-10-31):
<https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html>,
<https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/install_python.html>,
<https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/install_ros.html>.

This machine has an RTX 4050 Laptop with 6 GB VRAM, 5.8 GiB WSL RAM (host
15.3 GB), 2 vCPUs in WSL (host 8 cores / 16 logical), and driver 551.76.
Therefore:

- **High-fidelity Isaac Sim tier: UNAVAILABLE on this machine** (based on
  published requirements, not tested). No install is authorized; the guarded
  adapter and docs remain. WSL2 is documented only for external ROS nodes and
  is deprecated; "Isaac Sim runs inside WSL2" is **NOT ESTABLISHED**.
- **SmolVLA fine-tuning feasibility: UNKNOWN** until measured in Task 8. The
  6 GB GPU may be insufficient; cloud/Colab-style GPU runs are an alternative,
  with checkpoints and provenance metadata brought back into the repo.
- **Task 2 measured MuJoCo limits:** 256x256 cameras, 20 Hz control request,
  10 Hz camera request. Actual with software GL: 0.9 ms/physics step, 283.6 ms
  overhead render, 212.2 ms wrist render, 846.9 MiB Python smoke peak RSS,
  894 MiB ROS-node peak RSS, **1.87 Hz `/joint_states`** and **1.66 Hz**
  overhead images. The configured rates are not achieved under llvmpipe.

### Remaining Task 2 runtime findings

- Fake backend runtime: `/joint_states` **20.1 Hz**, all camera topics present,
  camera triplet stamps identical, reset service succeeds.
- MuJoCo runtime: topics, TF, camera info, depth `32FC1`, and reset service
  work; control is starved by the single-threaded render callback.
- Optional `GALLIUM_DRIVER=d3d12` reduced render time to 90.1/76.4 ms on the
  host AMD iGPU, but it is **not** the default and is not the RTX 4050.
- RViz/WSLg: **confirmed by the user on 2026-10-31** after camera QoS changed
  to reliable. Slow rendering is expected and is not a failure.

## Canonical commands

```bash
# Core / CI
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest -ra

# Simulator dependencies and pinned assets (approved in Task 2)
uv sync --group sim
./scripts/fetch_assets.sh

# One-shot core verification
./scripts/run_core_tests.sh

# Create .venv and .venv-ros
./scripts/setup_envs.sh

# ROS 2 build + test + .venv-ros tests (defaults to one colcon worker)
./scripts/build_and_test_ros.sh
```

CI installs the `sim` group, fetches the Menagerie meshes, and sets
`SIM2REAL_SKIP_RENDER=1` so only physics tests run there.

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
10. Apache-2.0; `LICENSE` added 2026-10-31 from the official apache.org text.
11. Repository root is resolved dynamically (`git rev-parse --show-toplevel`);
    no fixed path is assumed. The verified checkout is `/home/kunal/Python/sim2real`.
12. `Last verified test run:` records the **core** suite; the optional
    `Last verified ROS test run:` line records ROS separately.
13. `FakeBackend` is the CI default; MuJoCo is optional via the `sim` group.
14. Absolute joint-position targets only in Task 2; command subscription is
    deferred to Task 11 (safety filtering first).
15. 256x256 cameras at 10 Hz requested, 20 Hz control, 2 ms physics; sim clock
    starts at zero and `/clock` is not published.
16. Camera topics use reliable keep-last 5 so RViz Camera displays connect;
    best-effort subscribers remain compatible.
17. Menagerie Panda is pinned at `71f066ad…`; meshes are git-ignored and
    restored by `scripts/fetch_assets.sh`.

## Task checklist

- [x] Task 1 — preflight, scaffold, docs, CI, ROS workspace, `sim2real_msgs`, contracts, first tests
- [x] Task 2 — `SimBackend`, `FakeBackend`, MuJoCo Franka scene, sim ROS node
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

- Last verified test run: passed=149 failed=0 skipped=2 errors=0 | date=2026-10-31 | base_commit=830ba19
- Last verified ROS test run: passed=156 failed=0 skipped=0 errors=0 | date=2026-10-31 | base_commit=830ba19
- Task 2 status: complete with pending checks: Isaac Sim UNAVAILABLE (not installed, not run), WSL limits still 5.8 GiB / 2 vCPU, CI result after this push (user confirms Actions tab)
- Task 1 status: complete

The first line records the **core** suite (`.venv`, no ROS dependency). The
second line records the ROS-enabled suite separately.

Run breakdown (2026-10-31, base `830ba19`):

- Core-only: **149 passed, 0 failed, 2 skipped, 0 errors**. Skips are the msgs
  round-trip (ROS not sourced) and the ros-marked `sim_node` test (no `rclpy`).
- ROS (`.venv-ros` after `colcon build --parallel-workers 1`): **156 passed, 0
  failed, 0 skipped, 0 errors**. This includes 6 interface round-trips and the
  `sim_node` integration test.
- `colcon test` / `colcon test-result --verbose`: **7 entries, 0 failures, 0
  errors** (1 CTest wrapper + 6 interface round-trips). The two `sim2real_sim`
  unittest cases pass in the console but are not counted in the xunit total.
- `ruff check` and `ruff format --check`: clean.
- Runtime fake: `/joint_states` **20.1 Hz**; camera triplet has identical
  stamps; depth is `32FC1`; TF `world -> panda_hand` correct; reset seed 3
  succeeds.
- Runtime MuJoCo: topics, TF, camera info, reset succeed; **1.87 Hz**
  `/joint_states` and **1.66 Hz** overhead images under llvmpipe; ROS-node peak
  RSS **915,260 KiB (~894 MiB)**; Python smoke peak RSS **846.9 MiB**.
- Previous push `28d50e1`: **CI green confirmed by the user in the GitHub
  Actions tab on 2026-10-31** (no run id invented here).
- License decision: **Apache-2.0**; official text saved as `LICENSE` and
  committed in `69e1e13`.
- Spec provenance: repo copy equals the original after CRLF->LF normalization;
  both hashes are `3ec2f13b8abc8f61ee65a9b66683346f8557fe85d3ae1959d6cab13986b3d1ad`.
- RViz/WSLg: **confirmed by the user on 2026-10-31** after switching camera
  topics to reliable QoS; rendering is slow under software GL.

## Known limitations

- No dataset, training, mapping, inference, or safety code exists yet.
- **Isaac Sim 6.2.0 high-fidelity tier is UNAVAILABLE on this machine** based
  on published minimums (16 GB VRAM, 32 GB RAM, RTX 4080, 4 cores, driver
  595.58.03) versus the measured RTX 4050 6 GB / 5.8 GiB RAM / 2 vCPU /
  driver 551.76. Not tested; no install authorized. WSL2 simulator support is
  **NOT ESTABLISHED** (docs show WSL2 only for external ROS nodes, deprecated).
- **The intended WSL limits are not active.** `.wslconfig` still contains
  `memory=6GB`, `swap=8GB`, `processors=2`; `free -h` shows 5.8 GiB and `nproc`
  shows 2. Task 2 used one colcon worker and 256x256 cameras.
- **MuJoCo cannot sustain the configured rates under software GL.** Measured
  1.87 Hz `/joint_states` and 1.66 Hz images; the single-threaded render
  callback starves the control timer. A rendering thread, lower resolution, or
  lower camera rate is future work.
- **Camera topics use reliable keep-last 5**, not best-effort sensor-data QoS,
  because RViz Camera displays request reliable. Best-effort consumers still
  connect.
- **No joint-command subscription.** The node holds its target; command intake
  and safety filtering are Task 11.
- **Reset is nominal only.** `randomization_enabled` and object-pose overrides
  are recorded as ignored; Task 3 implements them.
- **SmolVLA fine-tuning feasibility is UNKNOWN** until Task 8 measures it.
  Cloud/Colab GPU runs are an alternative path with provenance metadata.
- RViz is confirmed but slow under llvmpipe. The optional
  `GALLIUM_DRIVER=d3d12` experiment used the host AMD iGPU, not the RTX 4050,
  and is not a default.
- Docker image not pulled or built; Docker Desktop's WSL integration for
  Ubuntu-24.04 is disabled.
- `.venv-ml` is documented only and not created.
- `hydra-core`/`omegaconf` are installed for later tasks but unused.
- CI actions are pinned to tags (`actions/checkout@v7`,
  `astral-sh/setup-uv@v10.3.3`), not commit SHAs. Pinning policy is pending.

## Pending manual checks

- [x] VS Code WSL verification — confirmed 2026-10-30 from user terminal output.
- [x] Copy the master spec to `docs/MASTER_SPEC.md` and hash both copies.
- [x] Confirm the GitHub Actions tab for the `28d50e1` push — user confirmed
  2026-10-31.
- [x] Choose the license — Apache-2.0; `LICENSE` added.
- [x] RViz/WSLg manual checkpoint — **RViz confirmed** by the user 2026-10-31.
- [ ] Write `memory=10GB`, `swap=4GB`, `processors=8` to the actual
  `%UserProfile%\.wslconfig`, run `wsl --shutdown`, and verify `free -h` and
  `nproc` before using more than one colcon worker.
- [ ] Confirm the GitHub Actions tab for the Task 2 push (API evidence only
  until then).
- [ ] Decide whether CI actions should be pinned to commit SHAs.
- [ ] Approve and build the Docker image separately; enable Docker Desktop WSL
  integration first.
