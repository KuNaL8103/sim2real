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

Tasks 1-7 implement scaffolding, contracts, fake and MuJoCo simulator backends,
seeded domain randomization, a language-conditioned scripted pick-and-place
expert with atomic rollout staging, trajectory filtering with quality metrics,
versioned `curated_v1` output, RGB-D mapping, and a two-layer LeRobot v3 /
Open X dataset builder. Task 8 adds an offline SmolVLA + PEFT LoRA fine-tuning
pipeline with a measured two-step GPU smoke run and offline L1 validation.
Task 9 adds closed-loop policy evaluation with reserved seeds, paired expert
comparison, rich records, failure-mode sampling, failure-targeted rollout
generation, and flywheel guards. Closed-loop inference (Task 10) and safety
logic are not implemented; Bridge real data and video encoding are not exercised.

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
- **SmolVLA fine-tuning feasibility is MEASURED for a two-step smoke run, not a
  full run.** The 6 GB GPU fits batch 1 / chunk 50 with **1,117 MiB** peak VRAM
  and **~3.7 GiB** host RSS (2026-11-07). A longer run or a larger batch was not
  measured; cloud/GPU runs remain an option with checkpoints and provenance
  metadata brought back into the repo.
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

# Optional ML environment and tests (never created by CI)
./scripts/setup_envs.sh --with-ml
scripts/run_ml_tests.sh                  # markers: ml and not gpu
scripts/run_ml_tests.sh --gpu            # only after explicit approval

# Task 8 CLI (core-only fake path first)
uv run python scripts/train_vla.py --config configs/training/smolvla_lora.yaml --fake --smoke
uv run python scripts/validate_checkpoint.py --checkpoint outputs/task8/fake_smoke/checkpoint-00000005 --fake
uv run python scripts/list_lora_targets.py --fake-tree
.venv-ml/bin/python scripts/train_vla.py --config configs/training/smolvla_lora.yaml --smoke --max-steps 2

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
| `.venv-ml` | offline VLA fine-tuning: torch 2.11.0+cu126, LeRobot 0.6.1, PEFT, accelerate, transformers (Task 8); opt-in only | no; created 2026-11-02, GPU wheel added 2026-11-07 |

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
- [x] Task 7 — LeRobot v3 export, Open-X adapter, normalization stats, mixing sampler (complete with pending checks above)
- [x] Task 8 — SmolVLA + LoRA fine-tuning (complete with pending checks above)
- [x] Task 9 — closed-loop evaluation (complete with pending checks: manual flywheel confirmation PENDING; CI pending user confirmation)
- [x] Task 9b — real-checkpoint verification (complete: policy server, 30-episode paired run, failure fit, sklearn cross-check)
- [ ] Task 10 — ROS 2 inference bridge
- [ ] Task 11 — safety watchdog and e-stop
- [ ] Task 12 — orchestration

## Current status

- Last verified test run: passed=494 failed=0 skipped=11 errors=0 | date=2026-11-11 | base_commit=1b9fa1d
- Last verified ROS test run: passed=506 failed=0 skipped=4 errors=0 | date=2026-11-11 | base_commit=1b9fa1d
- Task 9 status: complete with pending checks: manual flywheel confirmation PENDING; CI pending user confirmation. Timing contract: dataset/policy period 0.1 s, 2 control steps held, `n_action_steps` default **10** (1.0 s, not the training 50); fake sweep 5/10/25 -> 13/10/8 of 30. Failure chain: GMM >=30 (BIC <=5 components), KDE >=10, neighbour 1-9, none = NO_TARGETING; `failure_definition=policy_only` by default. Guards: `max_cycles=10`, `max_total_rollouts=200`, `max_rollouts_per_request=32`, `min_failures_to_trigger=3`, `min_success_improvement=0.05`, `max_consecutive_no_improvement=3`, `cooldown_s=0`, idempotent eval ids, `STOP` file. Policy server: stdin/stdout length-prefixed JSON + raw arrays, no pickle; torch stays in `.venv-ml`. Seeds: eval 20000-20999 only, targeted 30000-30999 only.
- Task 9b status: complete. Real policy server verified: hash `b87f5094…22e3`, chunk 50, fps 10, cameras overhead/wrist -> camera1/camera2, state/action dim 8; 50-call median **1.331 s**, p95 **2.237 s**, max **3.247 s**, warm-up 4.229 s, first load **39.8 s**, peak VRAM **1,077 MiB**, server `VmHWM` **3,841,184 KiB**. Chunk (50,8), finite; 7 arm dims inside Panda limits, gripper dim outside [0,0.04] ([-0.010464, 0.063410], simulator clips). Wrong hash refused; killed-server and timeout paths handled. Cost: overhead **0.193 s**/frame, wrist **0.154 s**/frame, 2-episode combined 179.1 s -> 44.8 min estimate; approved 30-episode run took **26.5 min**. 30 seeds 20000-20029, n_action_steps 10, llvmpipe: expert **24/30** (Wilson [0.6269, 0.9050], exactly the earlier baseline), policy **1/30** (0.0333, [0.00591, 0.16671]), policy on expert-success **1/24** ([0.00739, 0.20242]), policy-only failures **23**, terminations SUCCESS 1 / TIMEOUT 29, mean completion **0.01686**; ZeroPolicy **0/30** ([0.0, 0.1135]). Failure fit on 23 policy-only failures = **KDE**, id `64f51eb905e0e941`; no targeted generation. scikit-learn 1.9.1 installed in core `.venv` only (not CI): cross-check selects 3 components in both, matched means differ **8.3e-17**, BIC differs **0.76%**. No policy-quality claim.
- Task 8b status: complete: real 1,000-step LoRA run on train_v1 (batch 16, fp32, lr 1e-4, seed 0, save_every 250) stopped at 1,000 of the approved 2,000-step bound by user instruction; checkpoint `outputs/task8/train_v1_lora/checkpoint-00001000` (`git_dirty=false`, metadata sha256 `b87f5094d414d97590166ba8691989dd930b21b4c166ddf5de4a4f28234422e3`); wall 28:53, peak VRAM 4,377 MiB, peak RSS 6,245,720 KiB; loss 3.544756 -> 0.320808 (50-step mean 1.833 -> 0.247); offline L1 beats both baselines on all 8 dims (hold-last is better only at horizon k=0,1); **CI user-confirmed green, 2026-11-07** (no run ids); **Task 8 smoke inspection PENDING** (loss log, checkpoint folder and metadata were never confirmed by the user); checkpoint metadata confirms `chunk_size=50`, `n_action_steps=50`, `fps=10.0`, rename map overhead/wrist -> camera1/camera2, stats hash `a25f62abd51f0d9c1cbe5aee240273a186e45e711562de51f77f2b919af21dd5`, and a **constant** learning rate (`training.lr=1e-4`; no scheduler in `trainer.py`), so stopping at 1,000 of the 2,000-step config bound is a step-bound truncation, not a mid-schedule decay
- Task 8 status: complete with pending checks: real SmolVLA smoke ran **2 steps** on the RTX 4050 (losses 2.277515 -> 3.060153, peak VRAM **1,117 MiB**, peak host RSS **~3.7 GiB**, checkpoint `outputs/task8/smolvla_smoke/checkpoint-00000002`); user **"smoke confirmed" reply PENDING**; **CI user-confirmed green, 2026-11-07** (no run ids); OpenVLA-7B not validated; bf16/fp16, gradient checkpointing and cu128 not run; no policy-quality claim; RViz mapping and Task 7 viewer checks PENDING
- Task 7 status: complete with pending checks: LeRobot 0.6.1 / format v3.0 used; `.venv-ml` exists (CPU torch 2.11.0+cpu); renderer llvmpipe; decimation 2 / 10 Hz; 200 generated, 131 successful, 123 accepted (114/117 clean); real images only for a 2-episode/92-frame demo; no real Bridge data downloaded; Bridge license CC BY 4.0 and slug `bridge` (not official `bridge_orig`); viewer check PENDING; video/MP4 path not exercised; `.venv-ml` full-suite pytest is path-limited because that minimal env lacks scipy/open3d; **CI user-confirmed green, 2026-11-02** (no run id recorded)
- Task 6b status: complete with pending checks: **CI user-confirmed green for the 6b push, 2026-11-02** (no run id recorded); RViz visual confirmation PENDING (no report in this session); expert improvement is small and not material (26/40 dev baseline -> 28/40 final dev, 36/60 one-time confirmation); Open3D 0.20.0 `ScalableTSDFVolume` remains broken upstream (#7592) and the mapper uses a bounded `UniformTSDFVolume`; OctoMap deferred; opencode permission rules are unenforced
- Task 6 status: complete with pending checks: **CI user-confirmed green for the Task 6 push (f8c77f2) and the CI fix (8207142), 2026-11-01** (no run id recorded); RViz mapping check was visually confirmed for the fake backend and the lean mapper config, but this session has no new user report; MuJoCo runtime measured but no rate/quality threshold asserted; OctoMap deferred; Open3D 0.20.0 `ScalableTSDFVolume` is broken upstream (#7592) and the mapper uses a bounded `UniformTSDFVolume`; opencode permission rules are unenforced in this setup
- Task 5 status: complete with pending checks: keyframe/Parquet manual inspection PENDING (user); CI for the Task 5 push and the follow-up import-guard fix is **user-confirmed green, 2026-11-01** (no run id recorded); MuJoCo expert success rate and the 40-episode failure census are measured but not tuned; collision flag over-reports intended grasps (Task 4 bug, expert code not changed); opencode permission rules are unenforced in this setup and the v2 migration is rolled back
- Task 4 status: complete with pending checks: MoveIt not installed/not exercised; MuJoCo success rate measured (7/10 clean, 9/20 with 30% perturbation) and not tuned; viewer check PENDING; **CI user-confirmed green for the Task 4 push, 2026-11-01** (no run id recorded); opencode.json v2 migration attempted and ROLLED BACK as UNVERIFIED (reload succeeded, but `git push --dry-run origin main` produced no observable prompt in this session; backup restored)
- Task 3 status: complete with pending checks: CI for this push (user confirms Actions tab), RViz manual checkpoint (optional, not confirmed), Isaac randomization unverified. A Task 3 MuJoCo bug was found and fixed in Task 4: `mj_setConst` reset `MjData` to `qpos0`, so the physics group homed the arm and discarded sampled object poses.
- Task 2 status: complete with Task 2b hardening; pending checks: Isaac Sim UNAVAILABLE (not installed, not run), CI result after this push (user confirms Actions tab)
- Task 1 status: complete

The first line records the **core** suite (`.venv`, ROS-free,
`env -u PYTHONPATH -u AMENT_PREFIX_PATH`). The second line records the
ROS-enabled suite separately. Task 9b suites ran on the tree committed as
`1b9fa1d`; the docs-only commit follows it.

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

Run breakdown (2026-11-07, Task 8):

- Core ROS-free: **441 passed, 0 failed, 10 skipped, 0 errors** (was
  368/8 before Task 8). 73 new passing tests; the two extra skips are the new
  `ml` and `gpu` training test modules in the core environment.
- ROS (`.venv-ros` after `colcon build --symlink-install --parallel-workers 2`):
  **453 passed, 0 failed, 3 skipped, 0 errors** (was 380/1). `colcon
  test-result --verbose`: **7 tests, 0 errors, 0 failures, 0 skipped**.
- Ruff check and format check are clean. `scripts/ci_parity.sh` **passed**
  end-to-end (core + ROS parity halves).
- ML: `scripts/run_ml_tests.sh` reported **4 passed, 1 deselected** (`gpu`).
  The tiny torch LoRA test measured a trainable fraction of **0.108108**; the
  normalization parity test measured a maximum difference of **8.0e-08** versus
  LeRobot's `NormalizerProcessorStep`.
- Housekeeping (a): Task 7 CI is **user-confirmed green, 2026-11-02** (no run id
  recorded). RViz mapping and the Task 7 viewer check remain **PENDING**.
- Housekeeping (b): `scripts/run_ml_tests.sh` added (path-selected ML tests,
  `PYTHONPATH=src`, markers `ml and not gpu`, clear failure if `.venv-ml` is
  missing); a text test covers it and the opt-in CUDA setup lines.
- Housekeeping (c): accepted episodes in `data/curated/train_v1` are **123 =
  114 clean + 9 perturbed** (detour **4**, overspeed **3**, jitter **2**). The
  first 40 accepted episodes contain **36 clean + 4 perturbed** (detour 3,
  jitter 1).
- Training dataset: **40 episodes, 1,849 frames, 10 fps, 92 MB**, episode split
  **32 train / 8 val** (seed 123), **3,698 llvmpipe PNGs (103 MB)**, decimation
  2, cameras overhead/wrist. The 38 newly rendered episodes produced **3,514
  frames in 650.3 s (10.84 min)** with max joint **0.0 rad** and max object
  **0.0 m** replay error. Subset accepted initial red-cube x: min **0.3630 m**,
  median **0.4141 m**, max **0.5104 m**; full-run values remain
  **0.3601 / 0.4351 / 0.5371 m**.
- Stack: LeRobot **0.6.1**, torch **2.11.0+cu126** (CUDA runtime 12.6, driver
  551.76), PEFT **0.21.2**, accelerate **1.15.0**, transformers **5.5.4**,
  num2words **0.5.14**; `.venv-ml` is **6.7 GB** and the HF cache is **2.9 GB**.
  The cu126 scratch-venv 2048x2048 CUDA matmul passed (sum **141957.375**,
  40.1 MiB). cu124 has no CPython 3.12 torch satisfying LeRobot
  `>=2.7,<2.12`; cu128 was not tested.
- LoRA: the verified regex resolved to **37** modules (**32** action-expert
  q/v projections + **5** state/action projections); **1,485,312 trainable of
  451,531,488 total (0.329%)**; **450,046,176** base parameters frozen.
- Real smoke run: **2 steps**, batch 1, chunk 50, 2 cameras. Losses
  **2.277515411376953 -> 3.060152530670166**; step time **5.06 s** (warm-up)
  then **0.66 s** (steady-state 0.43-0.66 s across runs); peak VRAM **1,117.21 MiB**; peak host RSS **3,807 MiB**
  (`resource`) / **3,906 MiB** (`/usr/bin/time`); checkpoint
  `outputs/task8/smolvla_smoke/checkpoint-00000002`; adapter
  `adapter_model.safetensors` **5,952,536 B**. Two seeded runs produced
  identical losses and peak VRAM.
- Offline L1 on the smoke checkpoint: **8 held-out episodes, 366 windows,
  8,561 valid steps**. Per-dimension L1 **0.0634 / 0.0973 / 0.1566 / 0.1931 /
  0.0692 / 0.2032 / 0.1980 / 0.0194**; horizon L1 starts at **0.0580** and
  grows to about **0.13**; normalized per-dimension 0.69-1.42. The
  `hold_last_state` and `mean_action` baselines are recorded in
  `docs/TRAINING.md`. After two steps the policy is comparable to, not better
  than, the baselines; **no quality claim**.
- Decisions A-C are recorded in `docs/TRAINING.md` and the decision log. What
  was **NOT** run: OpenVLA-7B, bf16/fp16, gradient checkpointing, cu128, a
  remote GPU, video/MP4, real Bridge data, policy quality, RViz/viewer checks;
  wandb/mlflow were not installed or exercised (only the `none` tracker).

Run breakdown (2026-11-07, Task 8b):

- Approved configuration: `configs/training/smolvla_lora_train_v1.yaml`,
  batch **16**, lr **1e-4**, precision **fp32** (bf16 was not verified), seed
  **0**, `save_every` **250**, chunk 50 / `n_action_steps` 50. The committed
  config's `max_steps` is **2000**; the run was stopped at **1,000** by explicit
  user instruction, so `checkpoint_metadata.json` records `step=1000` while the
  config copy still shows the approved bound.
- Batch measurement on the real model (20 full-data steps each): batch 4 ->
  **1,733 MiB** VRAM, median **0.374 s**; batch 8 -> **2,547 MiB**, median
  **0.492 s**; batch 16 -> **4,134 MiB**, median **1.176 s**. Batch 16 left
  **2,007 MiB** free and was chosen. The realistic full-data 20-step batch-16
  run measured median **1.176 s**, mean **1.741 s**, peak RSS **6,096 MiB**.
- Training: **1,000 steps in 28:53.06** (median step **1.505 s**, mean
  **1.474 s**; first step 5.234 s). Loss **3.544756 -> 0.320808**, min
  **0.141453** at step 978, max 5.049912; the mean per 50-step block fell from
  **1.833490** (steps 1-50) to **0.247039** (steps 951-1000). No NaN or abort.
- GPU start 54 °C, 2130/7000 MHz, 12.66 W; end 73 °C, 2130/7000 MHz, 16.20 W.
  During the run the sampled max temperature was **88 °C**, VRAM **4,377 MiB**,
  SM clock median **1,185 MHz** (max sample 2,670 MHz), power median **24.0 W**
  (max 79.6 W). This is consistent with power/thermal limiting; the exact
  throttle reason was not instrumented.
- Peak host RSS **6,245,720 KiB (5.96 GiB)** from `/usr/bin/time`; the 10 s
  sampler saw 5,285 MiB.
- Checkpoint `outputs/task8/train_v1_lora/checkpoint-00001000`: `git_commit`
  `bf74c84`, **`git_dirty=false`**, metadata 8,491 B (SHA-256
  `b87f5094d414d97590166ba8691989dd930b21b4c166ddf5de4a4f28234422e3`), adapter
  **5,952,536 B**, whole checkpoint **5.70 MiB**. Checkpoints at 250/500/750/1000.
- Offline L1 on the same 8 val episodes (366 windows, 8,561 valid steps):
  trained per-dimension **0.020793 / 0.027214 / 0.031092 / 0.078327 / 0.012209 /
  0.057156 / 0.041395 / 0.007821**; smoke **0.063401 / 0.097294 / 0.156631 /
  0.193076 / 0.069209 / 0.203211 / 0.197957 / 0.019423**. Trained beats
  `hold_last_state` and `mean_action` on **all 8 dimensions**; the per-dimension
  margins over hold-last are **0.0548 / 0.0796 / 0.1226 / 0.1586 / 0.0466 /
  0.1002 / 0.1679 / 0.0145**. Horizon-wise (47 valid positions), trained mean
  L1 **0.032904** vs hold-last **0.159288** and mean-action **0.104959**;
  trained beats hold-last at **k=2..46 (45 positions)** but loses at **k=0 and
  k=1**; it beats mean-action at every valid horizon. These are 8-episode
  offline numbers, not a generalisation or closed-loop claim.
- Verification after the run: core ROS-free **442 passed / 10 skipped**, ROS
  `.venv-ros` **454 passed / 3 skipped**, `colcon test-result` **7 tests / 0
  failures**, `scripts/ci_parity.sh` passed, ruff clean, ML **4 passed / 1
  deselected**.
- Not run: the remaining 1,000 steps of the approved bound, bf16/fp16,
  gradient checkpointing, cu128, a remote GPU, closed-loop evaluation, policy
  quality. CI for this push is pending user confirmation.

Run breakdown (2026-11-11, Task 9):

- Pre-edit baselines (verified before any change): core ROS-free **442 passed /
  10 skipped**; `.venv-ros` **454 passed / 3 skipped**; `colcon test-result`
  **7 tests / 0 failures**; `scripts/run_ml_tests.sh` **4 passed / 1 deselected**;
  `sim2real.training` imports with torch absent; `scripts/ci_parity.sh` passed.
  All matched the recorded Task 8b numbers.
- Post-change core ROS-free: **493 passed / 11 skipped** (51 new passing tests;
  the extra skip is the new ros-marked closed-loop node test). `.venv-ros`
  **506 passed / 3 skipped**; `colcon test-result --verbose` **7 tests /
  0 failures / 0 skipped**; `ruff check` and `ruff format --check` clean; ML
  **4 passed / 1 deselected**.
- Timing contract verified: each policy action held for exactly **2** control
  steps; `n_action_steps=10` re-plans every 10 policy steps (20 control ticks).
  Fake `noisy_expert` sweep (30 seeds 20000-20029, expert 30/30 each):
  n=5 **13/30** (Wilson [0.274, 0.608]), n=10 **10/30** ([0.192, 0.512]),
  n=25 **8/30** ([0.142, 0.445]).
- Fake development run: 60 seeds 20000-20059, `noisy_expert`, n_action_steps 10
  -> policy **18/60** (0.30, Wilson [0.199, 0.425]), expert **60/60**
  ([0.940, 1.000]), **42 policy-only failures**, mean completion 0.305.
- Failure fit on those 60 records: **GMM**, 42 failures, id
  `1ac33171af13bd1c`; 7 targeted / 3 uniform draws, 0 clip fallbacks, 8 rejection
  tries. Targeted generation (fake, 10 rollouts, seed 30000) -> **10/10**
  expert success, all seeds 30000-30999, distribution id in every staged
  sidecar.
- MuJoCo baseline (physics only, **no rendering**, 30 seeds 20000-20029):
  expert **24/30** (0.80, Wilson [0.627, 0.905]); `ZeroPolicy` **0/30**
  ([0.000, 0.138], 24 policy-only failures); `NoisyExpertPolicy` (sigma 0.01,
  dropout 0.05) **2/30** ([0.018, 0.213]), **2/24** on expert-success seeds
  ([0.023, 0.258]), 22 policy-only failures; mean completion 0.056.
- Live ROS runtime (fake, no rendering): `closed_loop_eval_node` 20 episodes ->
  service `success=True`, **4/20** policy success, summary published on
  `/sim2real/eval_summary`; `expert_collector_node` FAILURE_TARGETED with 3
  rollouts -> `success=True`, **3/3** generated, expert yield 1.00, distribution
  `1ac33171af13bd1c`, seeds [30723, 30177, 30086].
- Not run: real-checkpoint closed-loop evaluation/T5 (**NOT RUN**, cost approval
  required); real SmolVLA server mode (fake-model protocol only); no
  scikit-learn install (NumPy GMM fallback); manual flywheel inspection PENDING.

Run breakdown (2026-11-11, Task 9b):

- Pre-edit baselines re-verified before any change: core ROS-free **493 passed /
  11 skipped**; `.venv-ros` **506 passed / 3 skipped**; `colcon test-result`
  **7 tests / 0 failures**; `scripts/run_ml_tests.sh` **4 passed / 1 deselected**.
  All matched the Task 9 record.
- Test-claim audit: no policy-quality threshold assertion exists. Relaxed
  `tests/test_evaluation_policies.py` (`expert num_success >= 1`,
  `policy num_success == 0`, zero policy-on-expert success) to definitional
  identities, and replaced two arbitrary Wilson thresholds in
  `tests/test_evaluation_records.py` with interval-membership checks. The ROS
  delta (+52) is the single extra ROS-marked test
  `test_closed_loop_eval.py::test_run_eval_publishes_a_summary` (skipped in core).
- scikit-learn: dry-run 4 packages (~9.1 MiB download). Installed `1.9.1`
  (`joblib 1.6.0`, `threadpoolctl 3.8.0`, `cloudpickle 3.1.2`) into core `.venv`
  only (`sklearn` 33 MiB on disk); not in `pyproject.toml`/`uv.lock`. One
  optional cross-check test on a seeded 3-cluster fixture: both select
  **3 components**, matched means differ **8.3e-17** (tolerance 0.05), BIC
  differs **0.76%** (tolerance 2%). CI skips it when sklearn is absent.
- Policy server (real checkpoint, GPU): handshake hash `b87f5094…22e3`, chunk
  **50**, fps **10**, cameras overhead/wrist -> camera1/camera2, state/action
  dim **8**; first load **39.8 s**. 50 calls: median **1.331 s**, mean 1.457 s,
  p95 **2.237 s**, max **3.247 s**, warm-up 4.229 s; peak VRAM **1,077 MiB**;
  server peak RSS **2,455,672 KiB**, `VmHWM` **3,841,184 KiB**. Chunk **(50, 8)**,
  wire float32/client float64, finite; 7 arm dims inside Panda limits, gripper
  dim **[-0.010464, 0.063410]** against [0, 0.04] (simulator clips). Wrong hash
  refused; killed server and 1 ms timeout detected/handled. Fixed the server so
  stdout is protocol-only, added an explicit `ready` frame after model load, and
  reuse a live server across resets.
- Cost (live llvmpipe, n_action_steps 10): overhead **0.193 s/frame** mean,
  wrist **0.154 s/frame** mean; policy median **1.295 s/call**; 2-episode combined
  **179.1 s** -> **44.8 min** estimate. The estimate exceeded 10 min, approval was
  requested and given; the 30-episode run took **1,592.6 s = 26.5 min**.
- 30-episode paired run (seeds 20000-20029, n_action_steps 10, renderer
  **llvmpipe**): expert **24/30** = 0.800, Wilson [0.6269, 0.9050] (exactly the
  earlier baseline); policy **1/30** = 0.0333, Wilson [0.00591, 0.16671]; policy
  on expert-success **1/24** = 0.0417, Wilson [0.00739, 0.20242]; policy-only
  failures **23**; policy terminations SUCCESS 1 / TIMEOUT 29; mean completion
  **0.01686**; 871 policy calls, median 1.323 s, p95 1.480 s, max 12.409 s. The
  single success was episode index 10 (seed 132083934). ZeroPolicy context on the
  same seeds: **0/30**, Wilson [0.0, 0.1135], all TIMEOUT, mean completion
  0.00173.
- Failure analysis: 23 policy-only failures -> **KDE**, id
  `64f51eb905e0e941`; no targeted rollouts generated.
- Not run: rendered frames for one success/failure episode (optional).
- Final Task 9b suites (tree `1b9fa1d`): core ROS-free **494 passed / 11
  skipped** (optional sklearn cross-check passing); `.venv-ros` **506 passed /
  4 skipped** (the cross-check skips because sklearn is core-`.venv`-only);
  `colcon test-result --verbose` **7 tests / 0 failures**; `scripts/run_ml_tests.sh`
  **4 passed / 1 deselected**; `scripts/ci_parity.sh` **passed** (its
  `uv sync --locked --group sim` removes core-only sklearn, so the cross-check
  skips there; it was reinstalled afterwards for the recorded core run). Ruff
  check and format check clean. CI for this push is pending user confirmation.

Run breakdown (2026-11-02, Task 7):

- Core ROS-free: **368 passed, 0 failed, 8 skipped, 0 errors** (was 333/7
  before Task 7; 35 new passing tests plus the `ml` skip).
- ROS (`.venv-ros` after `colcon build --symlink-install --parallel-workers 2`):
  **380 passed, 0 failed, 1 skipped, 0 errors** (was 345/0). The skip is the
  `ml` writer test because `.venv-ros` has no LeRobot.
- `colcon test-result --verbose`: **7 tests, 0 errors, 0 failures**. Ruff check
  and format check are clean.
- `.venv-ml`: Python 3.12, **LeRobot 0.6.1 / format v3.0**, torch
  **2.11.0+cpu**, torchvision 0.26.0+cpu, torchcodec 0.11.1+cpu, av 15.1.0,
  datasets 4.8.5; **1.5 GiB** installed. The ml-marked writer test passed with
  `PYTHONPATH=src .venv-ml/bin/python -m pytest -m ml -ra
  tests/test_dataset_writer_ml.py` (**1 passed**). The bare `-m ml` command
  collects unrelated modules that need scipy/open3d, so it is path-limited in
  this minimal env.
- Primary sources: LeRobot stable **0.6.1**, `CODEBASE_VERSION = "v3.0"`;
  official creation API `LeRobotDataset.create/add_frame/save_episode/finalize`;
  image/video dtypes; `meta/info.json`, `meta/stats.json`,
  `meta/tasks.parquet`, `meta/episodes/...parquet`; native q01/q10/q50/q90/q99.
  Open X GCS lists **`bridge`**, not `bridge_orig`; the spreadsheet reports
  25,460 episodes / 387.49 GB / 5 Hz / 4 RGB + 1 depth + 1 wrist. BridgeData V2
  is **CC BY 4.0**. URLs and the action-frame caveat are in `docs/DATASETS.md`.
- Training-data generation: **200 generated** (seeds 10000-10199, MuJoCo,
  `default` profile, 0.25 perturbation fraction), **131 successful**, **123
  accepted** with `configs/filtering/mujoco.yaml`; clean-success accepted
  **114/117**. Accepted initial red-cube x: min **0.3601 m**, median
  **0.4351 m**, max **0.5371 m** (skew toward easier low-x poses).
- Replay: first two accepted episodes at decimation 2, **184 PNG frames**,
  **0.034 GiB**, max joint error **0.0 rad**, max object error **0.0 m**.
  Built `data/lerobot/train_v1_imagedemo`: **2 episodes, 92 frames, 10 fps,
  4.7 MiB**, image mode, q01/q99 present.
- Determinism scope: the earlier 38/38 was one episode at decimation 5, not
  every frame. Extended to **182/182 frames of episode 7000 at decimation 1**,
  two llvmpipe replays, identical SHA-256, 0.0 rad / 0.0 m error.
- Bridge fixture: **5 available / 3 converted**, one language fill, one NaN
  drop, one missing-wrist drop; built `data/lerobot/bridge_fixture` at 5 fps,
  12 frames. No TFDS download and no TensorFlow installed.
- Housekeeping collision check: 40 clean MuJoCo episodes seeds 10000-10039
  succeeded **31/40**; successful collision runs **1:22, 2:6, 3:3** (mean
  1.39, max 3), closer to the 1000-1099 tuning distribution (0:3, 1:30, 2:12,
  3:2) than the old 4000-4039 0.308 figure. No initial overlap: minimum pair
  centre distance **0.07055 m**, above the 0.04 m edge, 0.0566 m face diagonal,
  and the configured 0.07 m separation; zero contacts at reset. No config
  change proposed.
- `scripts/ci_parity.sh`: **passed** end-to-end. Its core half reported
  **366 passed / 10 skipped** with `SIM2REAL_SKIP_RENDER=1` (two render skips
  beyond the normal eight); the ROS half reported **7 tests, 0 errors, 0
  failures** with scipy/open3d shadowed. The official core baseline remains
  368/8.
- Fake filter re-derivation: current-code 100 tuning seeds 1000-1099 (33
  actually perturbed) gave clean-success n=67; J_d p95 2.193e7 x 1.2 =
  **2.6311413e7**, SPARC p5 -5.10065 - 0.3 = **-5.400654**, path p95 1.1847 x
  1.2 = **1.421672**, collisions 0. Fresh 8000-8059 hold-out (60 episodes, 35
  clean) accepted **44/60**, clean-success **33/35** (2 false rejects, 5.7%),
  versus 7/38 before. `configs/filtering/default.yaml` updated; docs mark old
  fake numbers superseded.

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

- **Task 9b is one bounded real-checkpoint run.** 30 seeds 20000-20029, one task
  instruction, `n_action_steps=10`, 1,000 training steps on 40 episodes with
  accepted-data skew: expert **24/30**, policy **1/30** (Wilson
  [0.00591, 0.16671]), policy on expert-success **1/24**, policy-only failures
  **23**. No policy-quality or generalisation claim beyond these counts.
- **The real policy's gripper dimension is outside the physical range.** Over 50
  chunks it ranged [-0.010464, 0.063410] against [0, 0.04]; the simulator clips
  it. The seven arm dimensions were inside the Panda limits.
- **Real policy-server cost.** First process load **39.8 s**; 50-call median
  **1.331 s**, p95 **2.237 s**; the 30-episode paired run took **26.5 min** with
  llvmpipe. A live server is now reused across resets (not reloaded per
  episode).
- **scikit-learn is a core `.venv`-only optional cross-check.** Installed
  `1.9.1` (~33 MiB, plus joblib/threadpoolctl/cloudpickle) after approval; it is
  not in `pyproject.toml`/`uv.lock`, so CI skips the cross-check with a reason.
  Cross-check: both select 3 components, matched means differ **8.3e-17**, BIC
  differs **0.76%** (tolerances 0.05 and 2%).
- **The flywheel manual check is PENDING.** `data/flywheel/ledger.jsonl` is
  append-only and guard-tested, but the user has not inspected a live ledger yet.
  Task 9 never retrains; `REQUEST_RETRAIN` is a reserved decision for Task 12.
- **Fake-backend evaluation numbers do not transfer.** The kinematic fake expert
  succeeded 60/60 on seeds 20000-20059 and targeted rollouts yielded 10/10; the
  MuJoCo expert succeeded 24/30 on 20000-20029, so the documented ~40% MuJoCo
  failure rate still governs real targeted yield.
- **Rendered episode frames were not saved for Task 9b** (optional item not run).
- **The evaluator re-renders only when a policy needs images.** llvmpipe is the
  default and is recorded in every record; the opt-in D3D12 path is recorded as
  `...-OPT_IN` and must not be selected silently.

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
- **Task 7 accepted data skews toward easy poses.** The 200-episode MuJoCo run
  accepted 123 (114/117 clean successes); accepted initial cube x ranged
  0.3601-0.5371 m with median 0.4351 m. Larger-x poses fail more often, so the
  dataset is not a uniform sample of the randomization profile.
- **The default Panda dataset is 10 Hz, not 20 Hz.** Decimation 2 keeps state,
  action, and images aligned, but a 50-step Task 8 chunk spans 5 s, longer than
  a typical 4.5 s episode; padding/masking is mandatory.
- **Images are optional only by explicit fallback.** Missing replay images are
  an error unless `allow_imageless: true`; that path produces an `images: none`
  dataset that cannot train a VLA. The real demo has images for only 2 episodes.
- **One renderer per dataset.** Replay records `llvmpipe` by default and refuses
  mixed renderer labels; the opt-in D3D12/NVIDIA path remains a benchmark
  setting, never a CI/default. Depth is off and the video/MP4 path was not
  exercised; the demo uses PNG images via the official `image` feature.
- **Bridge is auxiliary and fixture-first.** The official Open X slug is
  `bridge`, not `bridge_orig`; no real Bridge/TFDS data was downloaded or
  committed. BridgeData V2 is CC BY 4.0. Its rpy state is relative to the
  controller default rotation, and the action rotation is about the EEF position
  in world axes; a base-vs-world transform is not independently verified.
- **Stats cross-check is not identical to LeRobot's.** Our mean/std agreed to
  ~1e-6 on the demo, but exact q01/q99 differed from LeRobot's per-episode
  histogram aggregation by up to 0.064. LeRobot `meta/stats.json` is
  authoritative for training; image stats here are per-channel frame means, not
  full pixel histograms.
- **The `.venv-ml` full-suite pytest is path-limited.** That env lacks
  scipy/open3d/mujoco, so the bare `-m ml` selector collects unrelated modules.
  Use `scripts/run_ml_tests.sh`, which selects the ML test files by path and
  sets `PYTHONPATH=src`. The GPU smoke remains opt-in (`--gpu` and
  `SIM2REAL_RUN_GPU_SMOKE=1`).
- **Replay rendering is a prototype, not wired into curation.** `render_replay.py`
  writes per-step PNGs and records path+sha256/renderer/fidelity in the staging
  sidecar/manifest, but `curated_v1` still records `images: none`; the Task 7
  builder reads the staging sidecar instead. Depth is off; images are PNG
  frames, not video.
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
- **Task 5 filtering, Task 6 mapping, and Task 7 LeRobot export exist; training,
  inference, and safety do not.** Staging remains raw and unfiltered. Curated
  output is a filtered subset; Task 7 validates the exported LeRobot loader
  round trip, but only on a 2-episode real image demo.
- **`curated_v1` itself still has no images.** Staging captures none, so its
  `info.json` records `images: none`. The Task 7 builder reads replay PNG
  references from the staging sidecar; curated episodes without replay images
  are rejected unless the imageless fallback is explicitly enabled.
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
- **SmolVLA fine-tuning fits a two-step smoke run on this GPU, not a full
  run.** Measured peak VRAM **1,117 MiB** and host RSS **~3.7 GiB** at batch 1 /
  chunk 50 (2026-11-07). A longer run, larger batch, bf16/fp16, and gradient
  checkpointing were not measured. Remote GPU runs remain an option; see
  `docs/REMOTE_TRAINING.md` (documentation only).
- **Task 8 is offline training and offline L1 only.** It does not run a policy
  in the simulator, does not claim policy quality, and does not implement
  closed-loop evaluation (Task 9) or safety (Task 11).
- **The base SmolVLA expects three cameras; this dataset supplies two.**
  `overhead -> camera1`, `wrist -> camera2`; the third slot is absent because
  `empty_cameras: 0`. This is a pretraining-distribution deviation.
- **The smoke checkpoint was written from a dirty tree** (`git_dirty: true` in
  `checkpoint_metadata.json`) because the Task 8 code was uncommitted at
  creation. That is accurate provenance, not a failure.
- **OpenVLA-7B is an unvalidated config.** `OpenVLAAdapter` raises
  `AdapterUnavailable`; the license (Llama-derived) and VRAM were not verified
  and no weights were downloaded.
- **`collision_count` is not used by training and is not a safety signal.**
  The accepted-data caveats (small sample, accepted-pose skew, 9 perturbed
  accepted episodes) still apply.
- **Task 8b is a 1,000-step offline run, not the full approved 2,000-step
  bound.** The run was stopped at checkpoint 1,000 by user instruction. The
  committed config still records the approved 2,000-step bound; the checkpoint
  metadata records the actual step. Do not describe this as a completed
  2,000-step run.
- **The 8b offline L1 comparison uses only 8 validation episodes (366 windows).**
  The trained checkpoint beats both trivial baselines per dimension, but
  `hold_last_state` is still better at chunk positions k=0 and k=1. There is no
  closed-loop or generalisation result; Task 9 must measure that.
- **The 8b GPU reached 88 °C** with a median training SM clock of 1,185 MHz
  (max sample 2,670 MHz) and a 55 W board cap. This is consistent with
  power/thermal limiting, but the exact throttle reason was not instrumented.
- **The accepted-data caveats still apply to the 8b run:** 40 episodes, a skew
  toward easier cube poses, and perturbed-but-accepted episodes.
- RViz was re-confirmed after the Task 2b threading change on 2026-10-31.
- Docker image not pulled or built; Docker Desktop's WSL integration for
  Ubuntu-24.04 is disabled.
- `.venv-ml` is now **6.7 GB** (torch 2.11.0+cu126, LeRobot 0.6.1, PEFT,
  accelerate, transformers) and is opt-in via
  `scripts/setup_envs.sh --with-ml`; CI never creates or installs it. The HF
  cache is **2.9 GB**.
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
- [x] Confirm the GitHub Actions tab for the Task 6b push — **user-confirmed
  green, 2026-11-02** (no run id invented).
- [x] Confirm the GitHub Actions tab for the Task 7 push — **user-confirmed
  green, 2026-11-02** (no run id invented).
- [ ] Optional Task 7 viewer check: run the LeRobot dataset visualizer on
  `data/lerobot/train_v1_imagedemo` and open replayed PNGs in VS Code. **PENDING**;
  does not block the commit.
- [ ] Task 8 manual checkpoint: inspect `outputs/task8/smolvla_smoke/losses.jsonl`,
  `outputs/task8/smolvla_smoke/checkpoint-00000002/`, and its
  `checkpoint_metadata.json`, then reply **"smoke confirmed"** or
  **"smoke pending"**. The optional rendered PNGs are under
  `data/lerobot/train_v1/` and `data/staging/train_v1/.../images/`.
- [ ] Task 8b manual checkpoint: inspect
  `outputs/task8/train_v1_lora/losses.jsonl`,
  `outputs/task8/train_v1_lora/checkpoint-00001000/`, and its
  `checkpoint_metadata.json`; compare it with
  `outputs/task8/analysis/val_trained.log` and `val_smoke.log`. **PENDING**.
- [ ] Confirm the GitHub Actions tab for the Task 8b push (user confirms; no run
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
