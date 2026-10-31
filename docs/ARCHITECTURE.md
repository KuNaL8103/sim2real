# Architecture

Status: **Task 2 simulator backends.** The core library, fake backend, headless
MuJoCo Franka tabletop backend, guarded Isaac adapter, and ROS 2 `sim_node` are
implemented and tested. Randomization, datasets, training, mapping, inference,
and safety are **not** implemented. Isaac Sim is **UNAVAILABLE on this machine**
(requirements-based, not tested).

## Scope of Task 2

- `SimBackend` contract plus deterministic `FakeBackend` (default CI backend).
- Headless MuJoCo 3.17.0 Franka Panda tabletop with two 256x256 RGB-D cameras.
- Guarded `IsaacBackend` stub; no Isaac install and no behaviour claims.
- ROS 2 `sim_node`: `/joint_states`, TF, RGB-D, camera info, and nominal
  `/sim2real/reset_env`.
- Hardware-free tests; MuJoCo physics tests run with the `sim` group, rendering
  tests are marked `render` and skip in CI.
- Joint-command subscription is **deferred to Task 11**. The node holds the
  current target and steps the backend; it does not accept external commands.

## Scope of Task 1

- Pure-Python core library (`src/sim2real`) with contract constants, metadata
  validation, and status parsing.
- ROS 2 Jazzy workspace (`ros2_ws`) with one interface package and four empty
  node-adapter skeletons.
- Documentation, configuration stubs, scripts, CPU-only Dockerfile, and CI.
- Hardware-free tests. No GPU, simulator, dataset, or model is required.

## Deployment tiers

| Tier | Purpose | Backend | Status |
| --- | --- | --- | --- |
| dev/test | CPU-only algorithms, contracts, CI | fake backend | **available** (deterministic, NumPy only) |
| high-fidelity | MuJoCo first, Isaac Sim optional | MuJoCo / Isaac Sim | **MuJoCo headless available**; **Isaac Sim UNAVAILABLE on this machine** (published requirements) |
| physical | real Franka Panda | ROS 2 + ros2_control | out of scope; no hardware |

The project has no physical robot. Simulator results are not a safety case.

## Hardware reality (measured 2026-10-31)

Official Isaac Sim 6.2.0 requirements:
<https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html>

| Resource | Isaac Sim 6.2.0 minimum | This machine | Result |
| --- | --- | --- | --- |
| GPU | GeForce RTX 4080 | RTX 4050 Laptop | below |
| VRAM | 16 GB | 6141 MiB (6 GB) | below |
| RAM | 32 GB | 5.8 GiB visible in WSL (host: 15.3 GB) | below |
| CPU cores | 4 | 2 vCPU in WSL (host: 8 cores / 16 logical) | below in the current WSL config |
| Storage | 50 GB SSD | about 930 GB free | meets |
| Driver | Linux 595.58.03 | 551.76 | below tested version |

WSL limits requested in `%UserProfile%\.wslconfig` (`memory=10GB`, `swap=4GB`,
`processors=8`) were **not active** on 2026-10-31: `free -h` still showed
5.8 GiB/8.0 GiB swap and `nproc` reported 2, and the file on disk still
contained the old `memory=6GB`, `swap=8GB`, `processors=2`. Task 2 therefore ran
with **one colcon worker**, 256x256 cameras, and no simultaneous heavy jobs.

Graphics:

- `glxinfo -B`: Mesa llvmpipe (LLVM 20.1.2), **"Accelerated: no"**, OpenGL 4.5.
- `MUJOCO_GL=egl` (default in the backend): initialises successfully and selects
  **Mesa llvmpipe**, i.e. software rendering.
- One read-only experiment with `GALLIUM_DRIVER=d3d12` reported
  `D3D12 (AMD Radeon(TM) Graphics)` (host integrated GPU through WSL), **not** the
  RTX 4050. This is recorded as an experiment, not a supported default.

Consequences:

- **Isaac Sim high-fidelity tier: UNAVAILABLE on this machine**, based on
  published requirements rather than a local test. No install is authorized;
  the guarded adapter and documentation remain.
- **SmolVLA fine-tuning feasibility: UNKNOWN** until Task 8 measures it. Cloud
  or Colab-style GPU runs are an alternative, with checkpoints and provenance
  metadata brought back into the repository.
- **MuJoCo rendering plus ROS on 5.8 GiB RAM is constrained.** Task 2 uses
  256x256 at 10 Hz, one colcon worker, and measured peak RSS; see below.

## Core library vs ROS adapters

**Every algorithm lives in `src/sim2real/`.** It must remain importable without
ROS, a simulator, a GPU, or network access. This keeps algorithms CPU-testable
and reusable from scripts, notebooks, and CI.

```
src/sim2real/            pure Python, no rclpy
ros2_ws/src/*            thin adapters: subscriptions, publishers, services,
                         parameter plumbing, and message conversion only
```

ROS adapter rules:

- No algorithm or threshold constants duplicated in ROS packages.
- Message conversion belongs in the adapter; the core library receives plain
  Python types.
- Core modules must never `import rclpy`. A test enforces the source-text rule.

## Package responsibilities

| Package | Build type | Responsibility | Task 1 state |
| --- | --- | --- | --- |
| `sim2real_msgs` | `ament_cmake` | Messages/services for episodes, metrics, safety, reset, rollout generation | interfaces defined; builds if ROS deps present |
| `sim2real_sim` | `ament_python` | Simulator backend adapter, `/joint_states` owner, reset service | **implemented** (fake, MuJoCo, guarded Isaac) |
| `sim2real_perception` | `ament_python` | RGB-D mapping, point cloud, TSDF mesh publication | skeleton only |
| `sim2real_vla` | `ament_python` | Task instruction intake, policy inference, action chunk publication | skeleton only |
| `sim2real_control` | `ament_python` | Safety watchdog, trajectory dispatch, e-stop handling | skeleton only |

## Data flow (planned)

```
scripted expert (Task 4)
    -> /sim2real/staging_episode  (RolloutEpisode)
filter (Task 5)
    -> /sim2real/filtered_episode (RolloutEpisode)
    -> /sim2real/filter_metrics   (FilterMetrics)
export (Task 6)
    -> curated_v1 on disk under data/  (ignored by git)
training (Task 8)
    -> LoRA adapter on disk            (ignored by git)
inference (Tasks 9-10)
    -> /vla/raw_action                 (JointTrajectory)
safety/control (Task 11)
    -> /joint_trajectory_controller/joint_trajectory
    -> /safety/status
perception (Task 7)
    -> /workspace/pointcloud, /workspace/tsdf_mesh
```

Only the names and message types below are fixed. Rates and QoS are deliberately
left as `TBD` where no decision has been made.

## ROS 2 contract table

| name | type | producer | consumers | rate | QoS | safety note |
| --- | --- | --- | --- | --- | --- | --- |
| `/joint_states` | `sensor_msgs/msg/JointState` | sim node **or** ros2_control (exactly one) | recorder, VLA adapter, safety watchdog | 20 Hz (configurable) | reliable, keep-last depth 10 | Exactly one publisher at any time. Never run the sim node and ros2_control together; duplicate publishers would corrupt state estimation. |
| `/tf` | `tf2_msgs/msg/TFMessage` | sim node | perception, VLA adapter, safety watchdog | 20 Hz, same stamp as joint states | reliable, keep-last depth 10 | Dynamic transforms share the sim clock with `/joint_states`. |
| `/tf_static` | `tf2_msgs/msg/TFMessage` | sim node / robot_state_publisher | perception, VLA adapter | on startup | reliable, transient-local, depth 1 | Static camera extrinsics; a stale value silently corrupts mapping. |
| `/camera/overhead/color/image_raw` | `sensor_msgs/msg/Image` | sim node | recorder, perception | 10 Hz | reliable, keep-last 5 (RViz-compatible; best-effort subscribers also connect) | Encoding `rgb8`. Images stay on disk, not inside `RolloutEpisode`. |
| `/camera/overhead/depth/image_raw` | `sensor_msgs/msg/Image` | sim node | recorder, perception | 10 Hz | reliable, keep-last 5 | Encoding `32FC1` in metres. Never reinterpret as millimetres. |
| `/camera/overhead/camera_info` | `sensor_msgs/msg/CameraInfo` | sim node | perception | 10 Hz, identical stamp to images | reliable, keep-last 5 | Must match the depth image resolution and intrinsics. |
| `/camera/wrist/color/image_raw` | `sensor_msgs/msg/Image` | sim node | recorder, perception | 10 Hz | reliable, keep-last 5 | Encoding `rgb8`. |
| `/camera/wrist/depth/image_raw` | `sensor_msgs/msg/Image` | sim node | recorder, perception | 10 Hz | reliable, keep-last 5 | Encoding `32FC1` in metres. |
| `/camera/wrist/camera_info` | `sensor_msgs/msg/CameraInfo` | sim node | perception | 10 Hz, identical stamp to images | reliable, keep-last 5 | Must match the wrist depth image. |
| `/task_instruction` | `std_msgs/msg/String` | operator/UI, evaluation harness | VLA adapter | TBD (task 8) | TBD (task 8) | Instruction changes must not race an in-flight action chunk. |
| `/sim2real/staging_episode` | `sim2real_msgs/msg/RolloutEpisode` | scripted expert / sim adapter | filter, recorder | TBD (task 4) | TBD (task 4) | Unfiltered episodes are not training data. |
| `/sim2real/filtered_episode` | `sim2real_msgs/msg/RolloutEpisode` | filter | dataset exporter, recorder | TBD (task 5) | TBD (task 5) | Only accepted episodes may be exported. |
| `/sim2real/filter_metrics` | `sim2real_msgs/msg/FilterMetrics` | filter | recorder, evaluation | TBD (task 5) | TBD (task 5) | Metrics explain a decision; they do not override safety stops. |
| `/workspace/pointcloud` | `sensor_msgs/msg/PointCloud2` | perception | VLA adapter, safety watchdog | TBD (task 7) | TBD (task 7) | Frame must be documented in the message header; no implicit frame. |
| `/workspace/tsdf_mesh` | `visualization_msgs/msg/Marker` | perception | RViz, safety watchdog | TBD (task 7) | TBD (task 7) | `TRIANGLE_LIST` marker; not a collision mesh. |
| `/vla/raw_action` | `trajectory_msgs/msg/JointTrajectory` | VLA adapter | safety/control adapter | TBD (task 8) | TBD (task 8) | `joint_names` = `panda_joint1..7` + `panda_finger_joint1`; `points[k].positions` = absolute targets for chunk step `k`; `time_from_start` = `(k+1)*dt`; `header.stamp` = newest observation stamp. |
| `/joint_trajectory_controller/joint_trajectory` | `trajectory_msgs/msg/JointTrajectory` | safety/control adapter | ros2_control trajectory controller | TBD (task 11) | TBD (task 11) | Only the safety adapter may publish here; raw VLA output must never reach the controller. |
| `/safety/status` | `sim2real_msgs/msg/SafetyStatus` | safety watchdog | operator/UI, recorder, supervisor | TBD (task 11) | TBD (task 11) | `NaN` means unknown, never safe. `is_safe=false` must stop dispatch. |
| `/estop` | `std_srvs/srv/Trigger` | operator/UI | safety watchdog | TBD (task 11) | latched, depth TBD (task 11) | Latched stop. Recovery requires `/estop_reset` (Task 11); no automatic clearing. |
| `/estop_reset` | `std_srvs/srv/Trigger` | operator/UI | safety watchdog | TBD (task 11) | TBD (task 11) | Defined in Task 11. Must only clear the latch after an explicit operator action. |
| `/sim2real/reset_env` | `sim2real_msgs/srv/ResetEnv` | evaluation harness / operator | sim node | on request | reliable | Reset returns the applied seed; a failed reset is not treated as success. Task 2 applies a nominal reset only and reports randomization fields as ignored. |
| `/sim2real/generate_rollouts` | `sim2real_msgs/srv/GenerateRollouts` | dataset tooling | expert/sim adapter | TBD (task 4) | TBD (task 4) | Bounded batch sizes; failures are reported, not silently retried forever. |

## Frames and conventions

| frame | parent | convention | notes |
| --- | --- | --- | --- |
| `world` | none (root) | z up | Single simulation world frame. |
| `panda_link0` | `world` | robot base | Identity mount in Task 2; static transform. |
| `panda_link1`..`panda_link7` | previous link | robot body frames | Dynamic TF from backend forward kinematics (not in the core contract tuple). |
| `panda_hand` | `panda_link7` | EE frame | Used by `TrajectoryStep.ee_frame_id`. |
| `camera_overhead_optical_frame` | `world` | optical: x right, y down, z forward | Static overhead extrinsic. |
| `camera_wrist_optical_frame` | `panda_hand` | optical: x right, y down, z forward | Static relative to the hand; moves with the arm. |

Additional rules:

- ROS 2 frame IDs are relative, without a leading `/`.
- `/joint_states` has exactly **one** publisher: the sim node **or**
  ros2_control, never both.
- Optical frames use REP-103/REP-105 conventions: x right, y down, z forward.
  This differs from the ROS body convention (x forward, y left, z up).
- `/vla/raw_action` joint order is fixed by `contracts.CONTROLLED_JOINT_NAMES`.

## Simulator backend interface (Task 2)

`src/sim2real/sim/backend.py` is the single contract:

- Dataclasses: `RobotState`, `CameraIntrinsics`, `CameraFrame`, `Observation`,
  `StepResult`, `CollisionInfo`, `BackendCapabilities`, `ResetConfig`.
- Abstract `SimBackend`: `reset(seed, config)`, `step(joint_target,
  gripper_target)`, `get_observation()`, `render(camera)`,
  `get_camera_info(camera)`, `camera_pose_world(camera)`,
  `collision_report()`, `capabilities()`, `close()`.
- Only **absolute joint-position targets** are accepted in Task 2. There is no
  torque, velocity, or delta interface.
- `control_rate_hz`, `joint_names`, and `joint_limits` are attributes.
- `CameraFrame.depth` is float32 **metres**; `NaN` or `0` is invalid. Rendering
  uses the ROS optical convention (x right, y down, z forward), converted from
  MuJoCo's (-Z forward, +Y up) by `diag(1, -1, -1)`.

Backends:

| Backend | Purpose | Dependencies |
| --- | --- | --- |
| `FakeBackend` | deterministic CI default; seeded object poses and analytic ray-cast RGB-D | NumPy + PyYAML only |
| `MuJoCoBackend` | headless Franka tabletop, physics, RGB-D, metric depth | MuJoCo 3.17.0 (`sim` group) |
| `IsaacBackend` | guarded stub; construction raises `BackendUnavailableError` | none (Isaac not installed) |

The MuJoCo scene is `assets/mujoco/scene_tabletop.xml` (table, three cubes, two
target zones, overhead camera) and includes the unmodified Menagerie
`panda.xml`. The wrist camera is attached to the hand at compile time with
`MjSpec`. Canonical joint/frame names are mapped in `configs/sim/panda.yaml`;
the model's `joint1..joint7` / `link0..hand` names are not renamed.

### Backend capability matrix

| Capability | fake | mujoco | isaac |
| --- | --- | --- | --- |
| RGB-D | yes (analytic) | yes (`mujoco.Renderer`) | **unverified** |
| lighting | no | yes (two lights) | **unverified** |
| materials | flat colours | yes | **unverified** |
| camera extrinsics | yes (config + simple FK) | yes (compiled model) | **unverified** |
| physics mass | no | yes | **unverified** |
| physics friction | no | yes | **unverified** |
| joint damping | no (instant targets) | yes | **unverified** |
| distractors | no (Task 3) | no (Task 3) | **unverified** |

Every flag carries an explicit reason string in `BackendCapabilities.reasons`.
"Unverified" is not a claim that Isaac works; the adapter cannot be constructed.

## Time, stamps, and synchronization

- The backend owns an explicit **simulation clock**: it starts at `0` on reset
  and advances by `1 / control_rate_hz` per step. It is not wall time.
- `/clock` is **not** published. Downstream nodes must treat stamps as
  sim-relative and avoid comparing them to `node.get_clock().now()`.
- The color image, depth image, and `CameraInfo` published in one camera
  callback share **identical** `header.stamp` and optical `frame_id`.
- Dynamic TF uses the same stamp as the `/joint_states` message from that step.
- Recommended downstream sync: `message_filters.ApproximateTimeSynchronizer`
  with `queue_size=10`, `slop=0.05` s. In simulation the stamps are exact, so an
  exact synchronizer also works; the slop only matters for recorded/replayed
  data.

## TF ownership and URDF

- The sim node is the single `/joint_states` owner while it runs. Never run the
  sim node and ros2_control together.
- Static TF: `world -> panda_link0` (identity), `world ->
  camera_overhead_optical_frame`, and `panda_hand ->
  camera_wrist_optical_frame`.
- Dynamic TF: `panda_link0 -> panda_link1 -> ... -> panda_link7 -> panda_hand`
  from backend forward kinematics (`tf_source=backend_fk`, the only implemented
  source).
- A URDF and `robot_state_publisher` are **optional** for TF. RViz's RobotModel
  display is the exception: it needs a URDF, which this launch file does not
  provide. The TF, Image, and Camera displays work without one.

## Deferred: joint-command subscription

No joint-command topic is subscribed in Task 2. The node holds its current
target and steps the backend. Command intake, safety filtering, and e-stop are
Task 11; publishing raw VLA output directly to a controller remains forbidden.

## Measured runtime (2026-10-31, 2 vCPU / 5.8 GiB WSL2)

MuJoCo **3.17.0**, NumPy 2.5.4 in `.venv` and system NumPy 1.26.4 in
`.venv-ros`.

| Measurement | Result |
| --- | --- |
| MuJoCo backend construction | 0.70 s |
| Physics step (20 steps) | 0.9 ms/step |
| `render("overhead")`, 256x256 color+depth, llvmpipe | 283.6 ms |
| `render("wrist")`, 256x256 color+depth, llvmpipe | 212.2 ms |
| Python smoke peak RSS (`ru_maxrss`) | 846.9 MiB |
| MuJoCo ROS node peak RSS (`/usr/bin/time -v`) | 915,260 KiB (~894 MiB) |
| Fake ROS node `/joint_states` | 20.1 Hz |
| MuJoCo ROS node `/joint_states` | **1.87 Hz** |
| MuJoCo ROS node overhead image | **1.66 Hz** |
| Optional `GALLIUM_DRIVER=d3d12` render | 90.1 ms overhead / 76.4 ms wrist |
| EGL renderer actually selected | Mesa **llvmpipe**; D3D12 experiment selected AMD Radeon (iGPU), not the RTX 4050 |

The configured 20 Hz control / 10 Hz camera rates are **not achieved with
software OpenGL**: a single-threaded executor spends most of its time in
`mujoco.Renderer`, so the control timer is starved. This is a measured
limitation, not a claim that the rates are met. The fake backend sustains the
configured 20 Hz because it has no GL rendering. Options for Task 3+ are a
smaller resolution, a lower camera rate, or a rendering thread/snapshot; none
is implemented yet. The `sim` marker keeps physics tests independent of GL; the
`render` marker keeps offscreen rendering out of CI.


## Environment separation

| Environment | Created by | Contains | Must not contain |
| --- | --- | --- | --- |
| `.venv` | `uv sync` | core library + dev tools | ROS, torch, LeRobot |
| `.venv-ros` | `scripts/setup_envs.sh` | system Python 3.12 + ROS Jazzy via `--system-site-packages`, editable core package, pytest/pyyaml/ruff | numpy/torch installs that shadow system packages |
| `.venv-ml` | documented only, Task 8 | torch, LeRobot, PEFT | ROS |

Why the split:

- ROS Jazzy's `rclpy` and generated message modules are built against the
  **system** Python 3.12 and system NumPy (verified here: NumPy 1.26.4).
- A uv-managed `.venv` may install NumPy 2.x. That is fine for the core
  library because ROS is never imported there.
- `.venv-ros` must see `rclpy` from `/opt/ros/jazzy`. Installing NumPy into
  `.venv-ros` would shadow the system NumPy and can break generated message
  conversion through ABI/API differences. The setup script therefore installs
  only pytest/pyyaml/ruff plus an editable, dependency-free core package there.
- ROS ships pytest plugins (`launch_testing`, `launch_ros`) that re-import every
  collected file and break a multi-package layout. Both are disabled in
  `pyproject.toml` with `-p no:...`; they are not needed for these tests.
- Never run ROS nodes with `.venv/bin/python`. Never run core-only tests with
  `.venv-ml` if it changes NumPy semantics.

## Decision log

| # | Date | Decision | Rationale |
| --- | --- | --- | --- |
| 1 | 2026-10-30 | All algorithms live in `src/sim2real`; ROS packages are adapters | CPU-testable, no ROS in CI core job |
| 2 | 2026-10-30 | `/joint_states` has exactly one owner: sim node OR ros2_control | Duplicate publishers silently corrupt state |
| 3 | 2026-10-30 | `/vla/raw_action` is a `JointTrajectory` with absolute joint targets, `time_from_start=(k+1)*dt`, `header.stamp` = newest observation | One representation for chunked policies and controllers |
| 4 | 2026-10-30 | `/workspace/tsdf_mesh` uses `visualization_msgs/msg/Marker` with `TRIANGLE_LIST` | RViz-friendly, no custom mesh message |
| 5 | 2026-10-30 | Services are namespaced `/sim2real/*`; `/estop` stays global and latched | Clear ownership; e-stop must not be namespaced away |
| 6 | 2026-10-30 | `curated_v1` dataset format is planned for Task 5, not defined now | Avoid inventing a schema before filtering exists |
| 7 | 2026-10-30 | Images stay on disk; `RolloutEpisode` carries metadata/paths only | Message size and replay performance |
| 8 | 2026-10-30 | Separate `.venv` and `.venv-ros`; never mix | ROS ABI and NumPy version constraints |
| 9 | 2026-10-30 | `metadata_json` is bounded (64 KiB) and rejects NaN/Infinity | Portable, safe transport |
| 10 | 2026-10-30 | License is Apache-2.0; `LICENSE` added 2026-10-31 | User chose Apache-2.0; official text fetched from apache.org |
| 11 | 2026-10-30 | Repository root is discovered dynamically (`git rev-parse --show-toplevel`); no fixed path is assumed | Docs and scripts must work from any checkout location; the verified checkout is `/home/kunal/Python/sim2real` |
| 12 | 2026-10-30 | Core and ROS test baselines use separate status lines | The core suite can skip ROS tests; mixing counts hid the core result (the old line recorded 113, a ROS count) |
| 13 | 2026-10-30 | Isaac Sim 6.2.0 tier marked UNAVAILABLE on this machine | Published minimums (16 GB VRAM, 32 GB RAM, 4 cores, RTX 4080) exceed measured hardware; not tested |
| 14 | 2026-10-30 | SmolVLA fine-tuning feasibility deferred to Task 8 | 6 GB VRAM may be insufficient; must be measured, with cloud GPU as a fallback |
| 15 | 2026-10-30 | Remove `<buildtool_depend>ament_python</buildtool_depend>`; rely on `<export><build_type>ament_python</build_type>` | The official Jazzy template does not declare that buildtool dependency, and rosdep has no rule for it |
| 16 | 2026-10-31 | `FakeBackend` is the default CI backend; MuJoCo is optional via the `sim` group | CI must stay CPU-only and deterministic; physics tests run with `uv sync --group sim` |
| 17 | 2026-10-31 | Cameras are 256x256 at 10 Hz; control is 20 Hz with 2 ms physics | WSL2 has 5.8 GiB RAM and software OpenGL; larger frames risk memory and latency |
| 18 | 2026-10-31 | Sim clock starts at zero and `/clock` is not published; camera triplets share one stamp | Deterministic recording; downstream uses ApproximateTimeSynchronizer (queue 10, slop 0.05 s) |
| 19 | 2026-10-31 | Menagerie meshes are git-ignored and restored by `scripts/fetch_assets.sh` | The pinned folder is 36.6 MB; only small XML, license, and attribution are committed |
| 20 | 2026-10-31 | Joint-command subscription deferred to Task 11 | Safety filtering and e-stop must exist before external commands reach the backend |
| 21 | 2026-10-31 | Camera topics use reliable keep-last 5, not best-effort sensor-data QoS | RViz Camera displays request reliable; a reliable publisher still serves best-effort consumers |

## Interface changelog

| Version | Date | Change | Reason |
| --- | --- | --- | --- |
| v0 | 2026-10-30 | Initial six interfaces: `TrajectoryStep`, `RolloutEpisode`, `FilterMetrics`, `SafetyStatus`, `ResetEnv`, `GenerateRollouts` | Task 1 proposal accepted without field/type refinements |

Wire types in `ros2_ws/src/sim2real_msgs` are the source of truth for ROS.
`src/sim2real/contracts.py` mirrors names and enum values for the pure-Python
core. Any change must update both, this table, and `tests/test_contracts.py`.

## Deviations from the Task 1 request

1. **Repository path is not assumed.** Scripts resolve their own root, and
   readers should use `git rev-parse --show-toplevel`. The Task 1 checkout was
   `/home/kunal/Python/sim2real` on the Linux filesystem (not `/mnt/c`); that is
   a verified fact, not a required location.
2. **`uv` installed during Task 1** from the official installer into
   `~/.local/bin` (approved by the user). No sudo was used.
3. **`src/sim2real/status.py` added and extended.** It parses the mandatory
   core status line and the optional ROS status line; the core line format is
   unchanged.
4. **Docker image not built or pulled.** The tag `ros:jazzy-ros-base` was
   verified against Docker Hub (HTTP 200) and the Dockerfile was reviewed by
   reading it. Building requires separate approval and Docker Desktop WSL
   integration.
5. **CI action refs:** `actions/checkout@v7` (floating major exists) and
   `astral-sh/setup-uv@v10.3.3`. `setup-uv` publishes version tags but **no
   floating `v10` tag**, which caused the first CI failure; it is now pinned to
   the exact verified release. Commit-SHA pinning is an open decision.
6. **`docs/MASTER_SPEC.md` provenance verified.** The committed copy and the
   original Windows file share SHA-256
   `3ec2f13b8abc8f61ee65a9b66683346f8557fe85d3ae1959d6cab13986b3d1ad` after
   normalizing CRLF to LF (both files are 717 lines / 43,406 bytes).
7. **`LICENSE` added.** Apache-2.0 text fetched from
   <https://www.apache.org/licenses/LICENSE-2.0.txt> (11,358 bytes) and stored
   verbatim; `pyproject.toml`, every `package.xml`, and the README agree.
8. **ROS build verified locally and in CI**: `colcon build --symlink-install
   --parallel-workers 1` and `colcon test` passed on 2026-10-31 with ROS Jazzy.
   Task 2 adds the `sim` group to the core CI job and fetches Menagerie assets;
   rendering tests skip in CI via `SIM2REAL_SKIP_RENDER=1`.
9. **`.venv-ros` includes an editable install of the core package** so the same
   suite can run with ROS available. It installs no dependencies (`--no-deps`),
   preserving the system NumPy.
10. **`ament_python` buildtool dependency removed.** The official
    `ros2 pkg create --build-type ament_python` template declares only
    `<export><build_type>ament_python</build_type>`. Removing our extra
    `<buildtool_depend>` let `rosdep install --from-paths ros2_ws/src
    --ignore-src --simulate` succeed **without** `--skip-keys`; `colcon build`
    and `colcon test` still pass.
