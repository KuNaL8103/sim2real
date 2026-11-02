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

Tasks 1-5 implement scaffolding, contracts, fake and MuJoCo simulator backends,
seeded domain randomization, a language-conditioned scripted pick-and-place
expert with atomic rollout staging, trajectory filtering with quality metrics,
and versioned `curated_v1` dataset output (LeRobot-style, not yet loader-validated).
Training, LeRobot export, mapping, inference, and safety logic are not
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
| CPU / RAM | 8 vCPU, 9.7 GiB RAM total, 4.0 GiB swap visible in WSL (host: 8 cores / 16 logical, 15.3 GB). Active after the 2026-10-31 `.wslconfig` rewrite and `wsl --shutdown`. |
| Disk | 1007G total, 930G available on `/` |
| GPU | NVIDIA GeForce RTX 4050 Laptop, 6141 MiB; driver 551.76 reports CUDA 12.4 capability |
| CUDA toolkit | Not installed and not claimed; `nvidia-smi` works through the Windows driver |
| WSL memory config | `%UserProfile%\.wslconfig` now active: `memory=10GB`, `swap=4GB`, `processors=8`; verified by `free -h` (9.7 GiB / 4.0 GiB) and `nproc` (8) on 2026-10-31. |
| WSL GPU bridge | `/dev/dxg` present; `/usr/lib/wsl/lib` includes `libcuda.so.1`, `libd3d12.so`, `libnvidia-ml.so.1` |
| Display / graphics | `DISPLAY=:0`, `WAYLAND_DISPLAY=wayland-0`, X display reachable; `glxinfo -B` reports Mesa llvmpipe (LLVM 20.1.2), **"Accelerated: no"**, OpenGL 4.5. Default `MUJOCO_GL=egl` selects llvmpipe. Opt-in `GALLIUM_DRIVER=d3d12` selected the AMD iGPU, and adding `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` selected `D3D12 (NVIDIA GeForce RTX 4050 Laptop GPU)`. |
| MuJoCo | 3.17.0 in `.venv` (sim group, NumPy 2.5.4) and `.venv-ros` (`--no-deps`, system NumPy 1.26.4, plus glfw/PyOpenGL/absl-py/etils). MuJoCo Menagerie Panda pinned at `71f066ad0be9cd271f7ed58c030243ef157af9f4` (Apache-2.0). |
| Open3D | 0.20.0 CPU wheel (`open3d-cpu`, 105 MB; the CUDA wheel is 400.8 MB) in `.venv` and in `.venv-ros` with `--no-deps`. Imports and integrates a synthetic frame headlessly under system NumPy 1.26.4; `plotly 7.1.0` and `dash 4.5.0` are needed by Open3D's eager visualization import and do not pull NumPy. The legacy `ScalableTSDFVolume` is broken upstream (#7592); mapping uses a bounded `UniformTSDFVolume`. |
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

This machine has an RTX 4050 Laptop with 6 GB VRAM, 9.7 GiB WSL RAM (host
15.3 GB), 8 vCPUs in WSL (host 8 cores / 16 logical), and driver 551.76.
Therefore:

- **High-fidelity Isaac Sim tier: UNAVAILABLE on this machine** (based on
  published requirements, not tested). No install is authorized; the guarded
  adapter and docs remain. WSL2 is documented only for external ROS nodes and
  is deprecated; "Isaac Sim runs inside WSL2" is **NOT ESTABLISHED**.
- **SmolVLA fine-tuning feasibility: UNKNOWN** until measured in Task 8. The
  6 GB GPU may be insufficient; cloud/Colab-style GPU runs are an alternative,
  with checkpoints and provenance metadata brought back into the repo.
- **Task 2b measured MuJoCo limits (8 vCPU / 9.7 GiB):** with images off,
  `/joint_states` and `/tf` run at **~20 Hz**. With images on and software
  rendering, `/joint_states` and `/tf` run at **20.00/20.01 Hz**, while the
  overhead image is **3.86 Hz**; ROS-node peak RSS **646,152 KiB**. The camera
  request remains 10 Hz but software GL cannot sustain it.
- **Renderer benchmark (3 frames/camera):** default llvmpipe
  **178.0/137.3 ms**, `GALLIUM_DRIVER=d3d12` AMD iGPU **71.2/66.2 ms**,
  `d3d12 + MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` RTX 4050 **83.0/80.4 ms**.
  The D3D12 paths are **OPT-IN** and never used by CI.

### Remaining Task 2b runtime findings

- Fake backend runtime: `/joint_states` **20.00 Hz**, `/tf` 19.99 Hz, overhead
  image **10.0 Hz**, peak RSS 113,100 KiB.
- MuJoCo with `publish_images:=false`: `/joint_states` **19.99 Hz**, `/tf`
  **20.00 Hz**, peak RSS 340,604 KiB.
- Before the fix, MuJoCo with images on was **3.28 Hz** `/joint_states` in this
  session (1.87 Hz in Task 2). After the async fix it is **20.00 Hz**.
- EGL contexts are thread-bound; all rendering now runs on one dedicated worker
  thread to avoid `EGL_BAD_ACCESS`.
- Camera topics remain reliable keep-last 5. A best-effort `ros2 topic echo`
  receives them, but reverting to best-effort would reintroduce the RViz Camera
  display reliability mismatch.
- RViz/WSLg: confirmed by the user on 2026-10-31; a re-check is required because
  the executor/threading changed.

## Canonical commands

```bash
# Core / CI: the ROS-free run is the official core baseline.
# Unsetting PYTHONPATH/AMENT_PREFIX_PATH reproduces CI. A module-level ROS name
# once broke CI collection while local runs passed because system rclpy was
# visible.
uv sync --group sim
env -u PYTHONPATH -u AMENT_PREFIX_PATH uv run ruff check .
env -u PYTHONPATH -u AMENT_PREFIX_PATH uv run ruff format --check .
env -u PYTHONPATH -u AMENT_PREFIX_PATH uv run pytest -ra

# Simulator dependencies and pinned assets (approved in Task 2)
uv sync --group sim
./scripts/fetch_assets.sh

# Opt-in render benchmark (never CI; no default changes)
uv run --group sim python scripts/bench_render.py --frames 3

# One-shot core verification
./scripts/run_core_tests.sh

# CI parity (core + ROS job, with scipy/open3d shadowed for the ROS half)
./scripts/ci_parity.sh

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
    restored by `scripts/fetch_assets.sh`, which now verifies the SHA and
    Apache-2.0 LICENSE.
18. State/TF publication is decoupled from rendering with snapshots and a
    single render worker thread; busy ticks are skipped and counted.
19. Reset quiesces rendering before touching the scene; Task 3 model changes
    must use the same path.
20. Camera QoS stays reliable keep-last 5; D3D12 renderer settings are opt-in.
21. Filter metrics use recorded executed state only; actions and perturbation
    labels never influence a decision.
22. Soft thresholds are per-backend and chosen only from clean successful
    tuning episodes (p95 x 1.2 / p5 - 0.3); the hold-out set is reporting-only.
23. `curated_v1` writes are atomic, never overwrite a version, and create a new
    minor version for appended accepted episodes.
24. Filter topics use reliable keep-last depth 50; the collector's staging
    publisher remains transient-local depth 10.

## Task checklist

- [x] Task 1 — preflight, scaffold, docs, CI, ROS workspace, `sim2real_msgs`, contracts, first tests
- [x] Task 2 — `SimBackend`, `FakeBackend`, MuJoCo Franka scene, sim ROS node
- [x] Task 3 — seeded domain randomization, profiles, capability-gated applier, reset integration
- [x] Task 4 — scripted expert, FK/IK/planner, perturbations, staging, rollout collector
- [x] Task 5 — trajectory filtering, metrics, `curated_v1`
- [x] Task 6 — RGB-D workspace mapping, point cloud, TSDF, obstacle query
- [x] Task 6b — collision classifier, release/physics guards, filter re-derivation, replay prototype, limit verification
- [ ] Task 7 — LeRobot v3 export, Open-X adapter, normalization stats, mixing sampler
- [ ] Task 8 — SmolVLA + LoRA fine-tuning
- [ ] Task 9 — closed-loop evaluation
- [ ] Task 10 — ROS 2 inference bridge
- [ ] Task 11 — safety watchdog and e-stop
- [ ] Task 12 — orchestration

## Current status

- Last verified test run: passed=333 failed=0 skipped=7 errors=0 | date=2026-11-02 | base_commit=8207142
- Last verified ROS test run: passed=345 failed=0 skipped=0 errors=0 | date=2026-11-02 | base_commit=8207142
- Task 6b status: complete with pending checks: RViz visual confirmation PENDING (no report in this session); CI for this push pending user confirmation in the Actions tab; expert improvement is small and not material (26/40 dev baseline -> 28/40 final dev, 36/60 one-time confirmation); Open3D 0.20.0 `ScalableTSDFVolume` remains broken upstream (#7592) and the mapper uses a bounded `UniformTSDFVolume`; OctoMap deferred; opencode permission rules are unenforced
- Task 6 status: complete with pending checks: **CI user-confirmed green for the Task 6 push (f8c77f2) and the CI fix (8207142), 2026-11-01** (no run id recorded); RViz mapping check was visually confirmed for the fake backend and the lean mapper config, but this session has no new user report; MuJoCo runtime measured but no rate/quality threshold asserted; OctoMap deferred; Open3D 0.20.0 `ScalableTSDFVolume` is broken upstream (#7592) and the mapper uses a bounded `UniformTSDFVolume`; opencode permission rules are unenforced in this setup
- Task 5 status: complete with pending checks: keyframe/Parquet manual inspection PENDING (user); CI for the Task 5 push and the follow-up import-guard fix is **user-confirmed green, 2026-11-01** (no run id recorded); MuJoCo expert success rate and the 40-episode failure census are measured but not tuned; collision flag over-reports intended grasps (Task 4 bug, expert code not changed); opencode permission rules are unenforced in this setup and the v2 migration is rolled back
- Task 4 status: complete with pending checks: MoveIt not installed/not exercised; MuJoCo success rate measured (7/10 clean, 9/20 with 30% perturbation) and not tuned; viewer check PENDING; **CI user-confirmed green for the Task 4 push, 2026-11-01** (no run id recorded); opencode.json v2 migration attempted and ROLLED BACK as UNVERIFIED (reload succeeded, but `git push --dry-run origin main` produced no observable prompt in this session; backup restored)
- Task 3 status: complete with pending checks: CI for this push (user confirms Actions tab), RViz manual checkpoint (optional, not confirmed), Isaac randomization unverified. A Task 3 MuJoCo bug was found and fixed in Task 4: `mj_setConst` reset `MjData` to `qpos0`, so the physics group homed the arm and discarded sampled object poses.
- Task 2 status: complete with Task 2b hardening; pending checks: Isaac Sim UNAVAILABLE (not installed, not run), CI result after this push (user confirms Actions tab)
- Task 1 status: complete

The first line records the **core** suite (`.venv`, ROS-free,
`env -u PYTHONPATH -u AMENT_PREFIX_PATH`). The second line records the
ROS-enabled suite separately. Task 6b tests ran on the working tree based on
`8207142`.

Run breakdown (2026-11-02, Task 6b):

- Core ROS-free: **333 passed, 0 failed, 7 skipped, 0 errors** (was 306/7
  before Task 6b; 27 new tests).
- ROS (`.venv-ros` after `colcon build --symlink-install --parallel-workers 2`):
  **345 passed, 0 failed, 0 skipped, 0 errors** (was 318).
- `colcon test-result --verbose`: **7 tests, 0 errors, 0 failures**. The new
  `scripts/ci_parity.sh` passed end-to-end: `uv sync --locked --group sim`,
  asset fetch, ruff, ROS-free pytest, and colcon build/test under system Python
  with scipy/open3d shadowed by ImportError stubs.
- Ruff is deterministic: `[tool.ruff.lint.isort]` sets
  `known-first-party = ["sim2real", "tests", "sim2real_*"]`; six existing ROS
  test files only needed import reordering.
- Collision classifier (`configs/sim/collision.yaml`,
  `sim.collision.classify_contacts`): only `dist < 0` counts; allowed
  finger<->target-cube, cube<->table, cube<->zone; robot self-contacts are
  reported as `self` and not counted; unknown names are flagged conservatively.
  Clean seeds 4000-4039 (baseline expert): **26/40 success**; successful clean
  episodes had **0.577 steps flagged on average** (max 19) and **0.308 runs**
  (max 1). Categories: cube-cube 46, cube-hand 25, table-finger 11,
  finger/non-target-cube 15. Perturbations 4100-4129: jitter 17/30 (flag mean
  2.23, runs 0.63), hesitation 18/30 (9.30, 2.00), detour 17/30 (2.23, 0.53),
  overspeed 8/30 (1.27, 0.27), bad_grasp 0/30 (1.03, 0.17), low_sweep 23/30
  (6.10, 1.23). The old "any contact" bug is fixed; the remaining flags are
  real classified contacts (cube-cube, hand-cube, finger-table,
  finger-non-target), not the intended grasp.
- Expert attempt: baseline **26/40** (Wilson [0.495, 0.779]); a wait-until-open
  release guard plus a 10-step pre-release settle raised it to **28/40**
  (Wilson [0.547, 0.819]) and fixed the release failures at seeds 4015/4033.
  Full grasp-offset compensation was **17/40**, xy-only minus **25/40**, xy-only
  plus **26/40**; all three were reverted. The one-time confirmation on seeds
  4040-4099 (never used to choose changes) was **36/60** (60%, Wilson
  [0.474, 0.714]); failed cube x mean **0.4356** vs success **0.4308**, so no
  strong x bias. The improvement is small and **not material**; off-zone and
  cube-instability failures remain. The physics guard aborts with
  `PHYSICS_INSTABILITY` on non-finite/huge qacc or an object outside the
  workspace plus 0.25 m; it is tested with a teleporting fake and did not fire
  in the 60 confirmation episodes.
- Filter re-derivation: 100 MuJoCo tuning episodes seeds 1000-1099 (40%
  perturbed): 60 success, clean-success **n=47**; jerk p95 7,197,954.18 x 1.2 =
  **8,637,545.0**; SPARC p5 -4.232499 - 0.3 = **-4.53250**; path p95
  1.092156 x 1.2 = **1.31059**; collision runs 0:3, 1:30, 2:12, 3:2, p95=2 ->
  ceil(2.4)=**3**. `configs/filtering/mujoco.yaml` updated; filter code
  unchanged. Fresh hold-out seeds 6000-6059: MuJoCo **29 accepted / 60**,
  clean-success **24/25** (1 false reject), bad_grasp 0/6, detour 2/3,
  jitter 2/5, low_sweep 0/4, overspeed 1/4. Fake **35/60**, clean-success
  **31/38** (7 false rejects; the fake default thresholds were not re-derived,
  per the scope rule).
- Replay: `scripts/bench_render.py` alone measured llvmpipe **196.5 / 138.7 ms**
  and D3D12/NVIDIA **54.7 / 56.4 ms** per camera frame. Three MuJoCo prototype
  episodes 7000-7002: **548 frames, 0.100 GiB**, default renderer
  **35.8 / 31.9 / 30.7 s** per episode, NVIDIA **17.5 / 16.2 / 17.2 s**;
  max joint replay error **0.0 rad**, max final object error **0.0 m**. Two
  replays of one episode gave identical hashes for **38/38** frames at
  decimation 5. `scripts/render_replay.py` writes the shared PNG format, never
  overwrites, updates the sidecar/manifest with path+sha256, and refuses
  >1 GiB / 10 min without `--i-approved`.
- Limits: official Franka FER table
  <https://frankarobotics.github.io/docs/robot_specifications.html> matches
  `configs/sim/panda.yaml` exactly for position, velocity, and acceleration
  (per-joint lists). Jerk (7500/3750/5000/6250/7500/10000/10000 rad/s^3) and
  effort (87/87/87/87/12/12/12 Nm) are **absent** from the YAML. No YAML or
  limit code was changed; add them before Task 11.
- Open3D issue #7592 (fetched): **0.20.0 produces empty geometry from metric
  float32 depth; 0.19.0 works**. The report is macOS arm64, CPU, NumPy 1.26.4,
  with PR #7594 open. Our x86_64 0.20.0 reproduces it. Decision log row 43
  records the `UniformTSDFVolume` deviation and the bounded-workspace memory
  budget.
- RViz: no new user report in this session. Task 6 established that the full
  `sim2real.rviz` Camera/Image displays plus MuJoCo EGL can drop the X11
  connection; the lean `mapper.rviz` plus the 8 s delay survived 165 s.

Run breakdown (2026-11-01, Task 6):

- Core-only (ROS-free): **306 passed, 0 failed, 7 skipped, 0 errors** (was
  257/6 before Task 6). The extra skip is the new ros-marked
  `test_workspace_mapper` module.
- ROS (`.venv-ros` after `colcon build --symlink-install --parallel-workers 2`):
  **318 passed, 0 failed, 0 skipped, 0 errors** (was 268).
- `colcon test` / `colcon test-result --verbose`: **7 tests, 0 errors, 0
  failures, 0 skipped**.
- `ruff check` and `ruff format --check`: clean.
- Open3D: `open3d-cpu==0.20.0`, a **105 MB** CPU-only wheel (the CUDA wheel is
  400.8 MB). It imports and integrates a synthetic frame **headlessly** under
  the sourced ROS environment with system NumPy **1.26.4**; `.venv` uses NumPy
  **2.5.4**. `plotly 7.1.0` and `dash 4.5.0` satisfy Open3D's eager
  `open3d.visualization` import and do not pull NumPy.
- **Open3D 0.20.0 `ScalableTSDFVolume` is broken for metric float32 depth**
  (upstream issue #7592, open PR #7594): the activation step re-scales an
  already-metric depth image by 1000, so volume units are opened at 1/1000 of
  the true position while the integrator uses true metric depth; the extracted
  cloud and mesh are empty. Reproduced with the official
  `SampleRedwoodRGBDImages` and with a synthetic frame. Open3D 0.19.0 works but
  its eager `open3d.ml` import needs a large ML dependency set under Python
  3.12. The mapper therefore uses a workspace-bounded `UniformTSDFVolume`; this
  is a documented deviation, not a claim that the scalable volume works.
- New core tests: **49** (mapping config/fusion/query/filter/transforms/
  pointcloud plus collection-safety and setup-script checks). Coverage includes
  back-projection RMS **< 1e-3 m**, single-view table z within one voxel,
  two-view consistency, invalid depth, workspace crop, analytic box distances,
  self-filter, TF failure, `MapStatus`, PointCloud2 round-trip, Marker cap, and
  scene-change reset.
- Fake runtime: `/workspace/pointcloud` **2.000 Hz** (window 20) and
  **1.701 Hz** (window 9) on the reset run; `frame_id=world`,
  `point_step=16`; cloud width **56** before reset and **2623 after**
  `/sim2real/reset_env` (seed 7, `default`); mesh Marker type 8
  (`TRIANGLE_LIST`), frame `world`; QoS RELIABLE/VOLATILE;
  `/workspace/reset_map` returned success. The mapper logged
  `camera static transform changed beyond epsilon; map reset` with
  `resets_static_tf=1`; no `TF_OLD_DATA` after the buffer clear.
- MuJoCo headless runtime: overhead color **4.365 Hz** (window 35) this session
  (the Task 2b baseline was 2.67-3.86 Hz; rates are measured, not asserted);
  pointcloud width **211**; integration mean **10.7-10.9 ms/frame** over
  120-180 frames; mapper `VmRSS`/`VmHWM` **202,636 KiB**; sim node
  `VmRSS`/`VmHWM` **646,564 KiB**. No rendering benchmark ran concurrently with
  colcon.
- The mapper clears its tf2 buffer on a map reset and re-adds the cached static
  transforms, because the sim clock restarts at 0 and tf2 otherwise rejects the
  new dynamic TF as `TF_OLD_DATA`.
- **RViz + MuJoCo on WSLg:** two separate issues. (1) Starting RViz in parallel
  with MuJoCo's EGL context can drop the X11 connection; both launch files now
  delay RViz by 8 s. (2) The full RViz config's four Camera/Image displays add
  GL load and crashed WSLg at 0.7-150 s under MuJoCo. The mapper now uses
  `sim2real_perception/rviz/mapper.rviz` (Grid, TF, PointCloud2, Marker only),
  which survived a **165 s** MuJoCo run with no X11 error and a 217-point cloud.
  The user's fake-backend screenshot confirms the colored cloud/TF rendering.
  For the two-terminal workflow, run the sim with `use_rviz:=false` and start
  RViz with the mapper config.

Run breakdown (2026-11-01, Task 5):

- Core-only: **257 passed, 0 failed, 6 skipped, 0 errors** (was 232/5 before
  Task 5). The extra skip is the new ros-marked `trajectory_filter` module.
- ROS (`.venv-ros` after `colcon build --symlink-install --parallel-workers 2`):
  **268 passed, 0 failed, 0 skipped, 0 errors** (was 242).
- `colcon test` / `colcon test-result --verbose`: **7 tests, 0 errors, 0
  failures, 0 skipped**; the six `sim2real_sim` unittest cases (including the new
  filter round-trip) pass in the console but are not counted in the xunit total.
- `ruff check` and `ruff format --check`: clean.
- scipy **1.16.3** and pyarrow **21.0.0** are committed to the core dev group and
  installed into `.venv-ros` with `--no-deps`; both import and round-trip
  Parquet under sourced ROS with system NumPy **1.26.4**. The newer resolved
  versions (scipy 1.18.1, pyarrow 26.0.0) were tested and **fail** on NumPy 1.26
  (scipy raises `numpy.long` missing; pyarrow requires NumPy >= 2.0).
- Threshold provenance: tuning seeds **1000-1099**, 100 episodes per backend,
  40% perturbed; hold-out seeds **2000-2059**, 60 per backend. Rule: clean +
  successful + non-perturbed p95 x 1.2 for upper bounds, p5 - 0.3 for SPARC.
  Fake (n=67): `max_jerk=14611600`, `min_sparc=-5.05682`, `max_path=1.3032`,
  `max_collisions=0`. MuJoCo (n=40): `max_jerk=4062840`,
  `min_sparc=-4.16341`, `max_path=1.15026`, `max_collisions=3`.
- Hold-out acceptance (60 per backend): fake clean **31/33** (2 clean-success
  false rejects), bad_grasp 0/3, jitter 0/2, hesitation 2/4, detour 6/6,
  low_sweep 8/8, overspeed 4/4. MuJoCo clean **9/11 successful** (2 false
  rejects; 9/33 of all clean), bad_grasp 0/3, jitter 0/2, hesitation 2/4,
  detour 3/6, low_sweep 6/8, overspeed 0/4.
- Minimum-jerk check at 20 Hz with savgol(9,5): **724.5 (T=2 s), 721.1 (T=4 s),
  738.0 (T=1 s)** vs the analytic 720; test tolerance 5%.
- MuJoCo clean failure census (seeds 3000-3039, no rendering): **22 success /
  18 failure**. Final states: RETREAT 13, RELEASE 5. Reasons: gripper did not
  release 5; cube not stable 6; cube off-zone 7; one episode lost the cube
  (rest-height error 2.735 m). Failed initial cube x mean **0.4780** vs success
  **0.4356** (y 0.0193 vs -0.0492, yaw -0.4107 vs -0.3006).
- Keyframes: `log/task5_keyframes/` (git-ignored) has **24 PNGs** for seeds
  3004, 3008, 3036 at APPROACH_PREGRASP/GRASP/LIFT/RELEASE, overhead + wrist;
  replay max joint error **0.0 rad**. Manual inspection is PENDING.

Run breakdown (2026-11-01, Task 4, retained for history):

- Core-only: **232 passed, 0 failed, 5 skipped, 0 errors**. Skips are the msgs
  round-trip and the four ros-marked `sim_node`/collector modules (no `rclpy`).
- ROS (`.venv-ros` after `colcon build --symlink-install --parallel-workers 2`):
  **242 passed, 0 failed, 0 skipped, 0 errors**.
- `colcon test` / `colcon test-result --verbose`: **7 tests, 0 errors, 0
  failures, 0 skipped**; the five `sim2real_sim` unittest cases (including the
  new collector service round-trip) pass in the console but are not counted in
  the xunit total.
- `ruff check` and `ruff format --check`: clean.
- Task 4 FK cross-check: **3.45e-8** maximum discrepancy vs MuJoCo at five
  random configurations (test tolerance `1e-6`).
- Throughput: fake `step()` **2,578 control steps/s**; MuJoCo `backend.step()`
  **1,344 control steps/s = 33,611 physics steps/s** (25 substeps/control step);
  MuJoCo raw `mj_step` **45,927 physics steps/s**; MuJoCo expert pipeline
  **598 control steps/s / 14,961 physics steps/s**.
- MuJoCo clean smoke, 10 seeded episodes (`default` profile): **7/10 success**.
- CLI fake, 20 episodes, 30% perturbed: **18 success / 2 failure** (both intended
  `bad_grasp`), 1,538 steps in 1.73 s, **0.733 MiB**.
- CLI MuJoCo, 20 episodes, 30% perturbed: **9 success / 11 failure**; clean 7/13,
  `bad_grasp` 0/2 (intended), `hesitation` 1/1, `jitter` 1/1, `detour` 0/1,
  `low_sweep` 0/1; 2.66 s, **0.727 MiB**.
- Housekeeping (a): after a jittered `reset_env`, a fresh
  `ros2 topic echo --once /tf_static` received **all three** static transforms
  (world→panda_link0, world→overhead, hand→wrist). The publisher is reliable,
  transient-local, depth 1, and republishes one TFMessage with the complete set;
  no fix was needed.
- Housekeeping (b): CI detection is `os.environ["CI"] in {1,true,yes}` in
  `test_sim_node_rate.py`; the 60% floor and its rationale are documented in that
  test's comments.
- ROS runtime: 3/3 episodes generated (1 perturbed); `/sim2real/staging_episode`
  echo received a complete episode. The staging publisher is reliable,
  transient-local, depth 10.

Task 3 breakdown retained below for history.

Run breakdown (2026-10-31, Task 3):

- Core-only: **184 passed, 0 failed, 4 skipped, 0 errors**. Skips are the msgs
  round-trip and the three ros-marked `sim_node` modules (no `rclpy`).
- ROS (`.venv-ros` after `colcon build --symlink-install --parallel-workers 2`):
  **193 passed, 0 failed, 0 skipped, 0 errors**.
- `colcon test` / `colcon test-result --verbose`: **7 entries, 0 failures, 0
  errors**; the four `sim2real_sim` unittest cases (including the new
  reset-service randomization test) pass in the console but are not counted in
  the xunit total.
- `ruff check` and `ruff format --check`: clean.
- Task 3 runtime, `backend:=fake`: seed 7 twice -> byte-identical
  `applied_config_json`; seed 8 differs; unknown profile -> `success=false`;
  `/tf_static` overhead translation changed from **(0.4645, -0.0132, 1.0951)**
  to **(0.4374, 0.0143, 1.1041)**; `/joint_states` **19.90 Hz** across a reset.
- Task 3 runtime, `backend:=mujoco` headless: same determinism and unknown-profile
  results; report `object_pose/texture/lighting/camera/physics = APPLIED`,
  `distractors = SKIPPED`; `/joint_states` **19.70 Hz** across a reset.
- Task 3 visual check: `log/task3_preview/` (git-ignored) contains overhead PNGs
  for nominal and seeds 1-4. Mean RGB and cube positions vary per seed; this is
  an approximate visual check, not a perception test.

Task 2b breakdown retained below for history.

Run breakdown (2026-10-31, Task 2b):

- Core-only: **159 passed, 0 failed, 3 skipped, 0 errors**. Skips are the msgs
  round-trip and the two ros-marked `sim_node` modules (no `rclpy`).
- ROS (`.venv-ros` after `colcon build --symlink-install --parallel-workers 2`):
  **167 passed, 0 failed, 0 skipped, 0 errors**.
- `colcon test` / `colcon test-result --verbose`: **7 entries, 0 failures, 0
  errors**; the three `sim2real_sim` unittest cases pass in the console but are
  not counted in the xunit total.
- `ruff check` and `ruff format --check`: clean.
- Runtime matrix (after fix): fake/on **20.00 Hz** joint, **19.99 Hz** TF,
  **10.0 Hz** images, peak 113,100 KiB; mujoco/off **19.99/20.00 Hz**, peak
  340,604 KiB; mujoco/on **20.00/20.01 Hz** joint/TF, **3.86 Hz** images,
  peak 646,152 KiB.
- Before fix, mujoco/on was **3.28 Hz** joint and **1.89 Hz** TF in this session
  (1.87 Hz joint in Task 2). Rendering no longer starves the state path.
- Renderer benchmark: llvmpipe **178.0/137.3 ms**; `d3d12` AMD iGPU
  **71.2/66.2 ms**; `d3d12 + MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` RTX 4050
  **83.0/80.4 ms**. RGB MAD vs llvmpipe **0.08/0.12**, depth MAD **1.16 mm**.
  Opt-in only; CI never uses them.
- Camera QoS: reliable keep-last 5 retained. `ros2 topic echo
  --qos-reliability best_effort --once` receives images, but reverting the
  publisher would reintroduce the RViz Camera reliability mismatch.
- `scripts/fetch_assets.sh`: idempotent, SHA-pinned, and now verifies the
  Apache-2.0 LICENSE before copying. Meshes remain git-ignored.
- `scripts/setup_envs.sh`: now installs MuJoCo 3.17.0 + rendering deps into
  `.venv-ros` with `--no-deps`; reproduced in a scratch `/tmp` venv, then deleted.
- Four Task 2b CI attempts failed the rate test: **6.5 Hz** with blocking render, **15.0 Hz** when startup/discovery was counted, **4.98 Hz** when the fake renderer still did a CPU-heavy raycast, and **14.998 Hz** when the probe subscription shared the sim node's executor. Rendering is fully asynchronous; the test waits for a live subscription, measures a 3 s window, uses a delay-only fake renderer, and measures from an independent probe node/executor. Local 1-2 CPU and colcon runs: **20.00 Hz**. The new run is **pending user confirmation** (no run id invented).
- License and spec provenance remain as recorded for Task 2.

## Known limitations

- **Task 6 mapping is a surface map, not a free-space map.** The TSDF only
  updates voxels within `sdf_trunc_m` of a measured surface; there is no
  carving. A moved object leaves a ghost until the map is reset. "Unknown" is
  expressed only through `MapStatus` (`NO_DATA`/`STALE`/`LOW_COVERAGE`); an
  invalid status is unsafe for Task 11.
- **The robot self-filter is a coarse sphere per link.** A grasped object near
  `panda_hand` is removed with the hand, and geometry between link frames is not
  covered.
- **Open3D 0.20.0's `ScalableTSDFVolume` is broken upstream (#7592).** The
  mapper uses a workspace-bounded `UniformTSDFVolume`. Open3D 0.19.0 works but
  its eager `open3d.ml` import needs a large ML dependency set under Python
  3.12. OctoMap is deferred.
- **`open3d-cpu` adds about 105 MB** (plus plotly/dash) to `.venv` and
  `.venv-ros`; it is installed `--no-deps` in `.venv-ros` so system NumPy
  1.26.4 stays visible.
- **Task 6b follow-up results (2026-11-02):** the collision flag is fixed
  (pure contact classifier; only actual penetration counts; allowed
  finger-target, cube-table, cube-zone; robot self-contacts reported not
  counted). A wait-until-open release guard and a 10-step pre-release settle
  raised the clean development success from **26/40 to 28/40**; the one-time
  confirmation was **36/60** (Wilson [0.474, 0.714]). Grasp-offset compensation
  was tried (17/40, 25/40, 26/40) and reverted. The expert still fails mostly
  off-zone or on cube instability; the improvement is **not material**.
- **Replay rendering is a prototype, not wired into curation.** `render_replay.py`
  writes per-step PNGs and records path+sha256 in the staging sidecar/manifest,
  but `curated_v1` still records `images: none`; integrating replay images into
  the dataset writer is Task 7. Depth is off; images are PNG frames, not video.
- **`configs/sim/panda.yaml` position/velocity/acceleration limits VERIFIED**
  against the official Franka FER table
  (<https://frankarobotics.github.io/docs/robot_specifications.html>): all three
  per-joint lists match exactly. **Jerk and effort limits are absent** from the
  YAML (official: jerk 7500/3750/5000/6250/7500/10000/10000 rad/s^3; effort
  87/87/87/87/12/12/12 Nm); add them before the Task 11 safety work. The expert
  planner uses velocity/acceleration; the filter uses joint/velocity margins.
- **The collision flag now reports real classified contacts.** Clean successful
  MuJoCo tuning episodes still have 0-3 unintended runs because cube-cube,
  hand-cube, finger-table, and finger-non-target contacts occur; these are no
  longer the intended grasp. `max_collisions: 3` is derived as
  ceil(p95(2) x 1.2), not the old "any contact" workaround.
- **The scripted MuJoCo expert remains modest.** Baseline 26/40 (Wilson
  [0.495, 0.779]) and final 28/40 (Wilson [0.547, 0.819]) on the development
  seeds; confirmation 36/60. Main failures are off-zone placement and cube
  instability. No test asserts a success threshold.
- **Task 5 filtering and `curated_v1` exist; training, LeRobot export,
  inference, and safety do not.** Mapping exists (Task 6). Staging remains raw
  and unfiltered. Curated output is a filtered subset, not yet validated against
  the LeRobot loader.
- **`curated_v1` has no images.** Staging captures none, so `info.json` records
  `images: none` and the dataset card says the data cannot train a VLA until
  RGB-D frames are rendered (Task 7 prerequisite).
- **Metrics use 20 Hz derivatives and Savitzky-Golay smoothing (window 9,
  polyorder 5).** Window 9 halves the 1 mm-noise-driven J_d versus window 5 but
  does not remove it; window 17 suppresses noise further yet attenuates a real
  6 Hz oscillation by >400x. Treat J_d as comparative.
- **Dimensionless squared jerk depends on dwell time (`T^5`).** Grasp/release
  pauses raise it, so thresholds are empirical and per-backend, not literature
  values.
- **Accepted MuJoCo data reflects a modest success rate; there is no strong
  initial-x bias in the new seeds.** The expert succeeded 22/40 on clean seeds
  3000-3039 (Task 5) and 26/40 on 4000-4039 (Task 6b); filtering cannot recover
  the failures. Failed cube x mean 0.4356 vs success 0.4308 in the confirmation.
- **The fake backend reports zero joint velocities**, so fake velocity-limit
  margins are always 1.0.
- **Threshold samples are small** (40-67 clean tuning episodes per backend); the
  docs report counts alongside percentages.
- **MoveIt 2 is not installed and was not exercised.** `MoveItPlanner` is a
  guarded stub that raises `PlannerUnavailable`; no MoveIt behaviour is claimed.
- **The scripted MuJoCo expert remains modest.** 26/40 clean development
  baseline, 28/40 after the release guard, and 36/60 on the one-time
  confirmation. The main failures are off-centre placement and cube
  instability, not a single deterministic bug. Yaw-aligning the gripper was
  tried in Task 4 and reverted. No test asserts a success threshold.
- **`FakeBackend`'s grasp is a kinematic test double, not physics.** A closed
  gripper within 5 cm of the TCP attaches the cube; opening above 0.02 m releases
  it at table height. It has no contact forces.
- **Perturbations are open-loop command variations.** They are not a learned
  failure distribution; `FAILURE_TARGETED` returns `success=false` until Task 9.
- **Images are not captured during rollouts.** `render_replay.py` replays a
  staged episode and writes PNGs with manifest references; integrating replay
  images into `curated_v1`/LeRobot is Task 7.
- **The collector runs one batch at a time** and rejects a concurrent call.
- **The collector is not a `/joint_states` publisher.** `sim_node` remains the
  only `/joint_states` owner when it runs.
- **Task 3 regression fixed in Task 4:** MuJoCo `mj_setConst` resets `MjData` to
  `qpos0`; the physics randomization group now saves/restores `qpos`/`qvel`.
  Before the fix, the `default` profile silently homed the arm and discarded
  sampled object poses.
- **`/sim2real/staging_episode` uses transient-local depth 10** so late joiners
  receive recent completed episodes. Episodes carry no images.
- **The reset service still uses `episode_index=0`.** The expert runner derives
  its own randomization seed from `(run seed, episode index)` and does not rely
  on the service index.
- **Isaac Sim 6.2.0 high-fidelity tier is UNAVAILABLE on this machine** based
  on published minimums (16 GB VRAM, 32 GB RAM, RTX 4080, 4 cores, driver
  595.58.03) versus the measured RTX 4050 6 GB / 9.7 GiB RAM / 8 vCPU /
  driver 551.76. Not tested; no install authorized. WSL2 simulator support is
  **NOT ESTABLISHED** (docs show WSL2 only for external ROS nodes, deprecated).
- **Camera images are slower than requested under software GL.** State/TF now
  run at ~20 Hz, but overhead images are **3.86 Hz** against a 10 Hz request.
  llvmpipe render time is 178/137 ms per camera (colour+depth). Lower
  resolution, a lower camera rate, or the opt-in D3D12 path are the options.
- **MuJoCo EGL contexts are thread-bound.** All rendering must stay on the
  single `RenderCoordinator` worker thread; do not render from executor
  callbacks directly.
- **Camera topics use reliable keep-last 5**, not best-effort sensor-data QoS,
  because RViz Camera displays request reliable. Best-effort consumers still
  connect.
- **D3D12 renderer settings are opt-in only.** `GALLIUM_DRIVER=d3d12` with
  `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` selected the RTX 4050 and rendered
  faster, but it is not a default and is never used in CI.
- **No joint-command subscription.** The node holds its target; command intake
  and safety filtering are Task 11.
- **Randomization is reset-scoped, not continuous.** Task 3 samples once per
  reset inside the render-quiesce path. There is no per-step randomization.
- **`material_pool` is PARTIAL in MuJoCo.** `geom_rgba` is a flat colour
  approximation; runtime texture-image swaps are not supported. The fake backend
  also approximates the pool with flat colours.
- **Distractors are SKIPPED.** The scene has no pooled parked bodies, and Task 3
  does not create bodies at runtime.
- **Isaac randomization is UNVERIFIED.** The guarded adapter cannot be
  constructed; the applier refuses to claim an application.
- **The reset service uses `episode_index=0`.** The sampler is a pure function of
  `(seed, episode_index)`; the request seed overrides the profile's default seed.
- **Camera jitter is applied in the camera's MuJoCo local frame** (before the
  optical conversion). The fake and MuJoCo backends use the same convention.
- **`/tf_static` uses a plain transient-local publisher**, not
  `tf2_ros.StaticTransformBroadcaster`, because the latter deduplicates by child
  frame ID and would silently ignore a jittered camera extrinsic.
- **The ROS rate test uses a 60% CI floor** (12 Hz) and keeps the stricter 80%
  local floor. Shared CI runners are noisier; the 60% floor still catches the
  4.98-6.5 Hz starvation regressions that motivated the test.
- **OpenCode permission rules are unenforced in this setup; the v2 migration
  was rolled back.** `opencode.json` remains the V1 `permission.bash` schema.
  The installed runtime is OpenCode v2.2.5. On 2026-11-01 the approved native-v2 `permissions`
  array was written, `opencode reload` reported "Configuration reloaded", but
  `git push --dry-run origin main` produced **no observable prompt** in this
  session. Per the agreed test rule the backup was restored byte-for-byte
  (SHA-256 `59cb4e46...`) and the migration is **UNVERIFIED**; it was not
  committed. A user-visible prompt test is required before retrying.
- **`render_only_when_subscribed` defaults to false.** When enabled, a camera
  with no subscribers is skipped and counted.
- **SmolVLA fine-tuning feasibility is UNKNOWN** until Task 8 measures it.
  Cloud/Colab GPU runs are an alternative path with provenance metadata.
- RViz was re-confirmed after the Task 2b threading change on 2026-10-31.
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
- [x] Write `memory=10GB`, `swap=4GB`, `processors=8` to the actual
  `%UserProfile%\.wslconfig`, run `wsl --shutdown`, and verify `free -h` and
  `nproc` — **done 2026-10-31**: 9.7 GiB RAM, 4.0 GiB swap, `nproc=8`.
- [x] RViz re-check after the Task 2b threading change — **RViz confirmed**
  by the user 2026-10-31.
- [ ] Confirm the GitHub Actions tab for the Task 2b push (API evidence only
  until then).
- [ ] Confirm the GitHub Actions tab for the Task 3 push (user confirms; no run
  id invented).
- [ ] Confirm the GitHub Actions tab for the Task 4 push (user confirms; no run
  id invented).
- [ ] Optional Task 4 viewer checkpoint: watch one MuJoCo rollout and confirm
  approach/grasp/lift/transport/release. **PENDING**; does not block the commit.
- [ ] Optional Task 5 artifact inspection: open one curated Parquet file and
  `data/rejected/*/rejections.jsonl`, plus the keyframe PNGs under
  `log/task5_keyframes/`. **PENDING**; does not block the commit.
- [x] Confirm the GitHub Actions tab for the Task 5 push and the follow-up
  import-guard fix — **user-confirmed green, 2026-11-01** (no run id invented).
- [ ] RViz mapping check (Task 6): fixed frame `world`, colored point cloud
  showing the table and three cubes, mesh Marker over them, wrist views filling
  in more detail, and the robot links absent from the obstacle query.
  **Fake-backend cloud visually confirmed by the user's screenshot; MuJoCo
  RViz uses the lean `mapper.rviz` and survived 165 s with no X11 error; final
  user visual confirmation of the MuJoCo cube/mesh layout is still PENDING.**
- [x] Confirm the GitHub Actions tab for the Task 6 push (f8c77f2) and the CI
  fix (8207142) — **user-confirmed green, 2026-11-01** (no run id invented).
- [x] Verify `configs/sim/panda.yaml` position/velocity/acceleration limits
  against the official Franka FER table — **verified, all three per-joint lists
  match** (2026-11-02).
- [ ] Add Franka jerk and effort limits (official FER values) to
  `configs/sim/panda.yaml` before the Task 11 safety work; they are currently
  absent.
- [x] Task 6b follow-up: collision flag fixed by the contact classifier; expert
  success census examined; replay rendering designed and prototyped.
- [ ] RViz visual confirmation of the Task 6b MuJoCo scene/mesh layout (no report
  in this session; the lean mapper config and 8 s delay are established).
- [ ] Confirm the GitHub Actions tab for the Task 6b push (user confirms; no run
  id invented).
- [ ] Optional RViz Task 3 checkpoint: call `reset_env` with different seeds and
  confirm cubes/lighting/camera TF change. Not confirmed; does not block the
  commit.
- [ ] Retry the native OpenCode v2 `permissions` rewrite only after a
  user-visible prompt test; the 2026-11-01 attempt was rolled back as UNVERIFIED
  (`opencode reload` succeeded, no observable prompt on
  `git push --dry-run origin main`).
- [ ] Decide whether CI actions should be pinned to commit SHAs.
- [ ] Approve and build the Docker image separately; enable Docker Desktop WSL
  integration first.
