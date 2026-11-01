# Architecture

Status: **Task 4.** Simulator backends, `sim_node`, seeded domain randomization,
and a language-conditioned scripted pick-and-place expert with a rollout
collector are implemented. Task 2b decouples `/joint_states`/TF publication from
camera rendering; Task 3 adds validated randomization profiles and reset-service
integration. Task 4 adds a pure-Panda FK single source, bounded IK/planner
abstractions, an explicit state machine, intentional perturbations, an atomic
staging buffer, and `expert_collector_node` (`UNIFORM` only). Datasets, training,
mapping, inference, and safety remain unimplemented. Isaac Sim is **UNAVAILABLE
on this machine** (requirements-based, not tested); MoveIt 2 is **not installed**
and its planner adapter is an unexercised guard.

## Scope of Task 4

- `src/sim2real/expert/`: instruction parser, IK, planner, state machine,
  perturbations, rollout runner, staging, and pure message conversion.
- One pure-NumPy Panda FK (`src/sim2real/sim/kinematics.py`) shared by the fake
  backend's TF and the expert; cross-validated against the MuJoCo model.
- `FakeBackend` kinematic grasp test double (not physics).
- `configs/expert/{default,instructions,perturbations}.yaml`.
- `expert_collector_node`, `scripts/collect_rollouts.py`,
  `scripts/summarize_staging.py`, and `launch/expert_collector.launch.py`.
- Hardware-free tests plus optional MuJoCo and ROS round-trip tests.
- Fix: MuJoCo `mj_setConst` resets `MjData` to `qpos0`; the physics
  randomization group now saves/restores `qpos`/`qvel` so it cannot silently
  home the arm and discard sampled object poses.

## Scope of Task 3

- `src/sim2real/randomization/`: validated YAML config, deterministic sampler,
  backend-aware applier, approximate colour-temperature helper.
- Profiles: `configs/randomization/{none,light,default,heavy}.yaml`.
- Per-group `BackendCapabilities` flags for object pose and texture.
- `SimBackend.apply_randomization` with `APPLIED` / `PARTIAL` / `SKIPPED`
  reporting; Isaac refuses.
- `/sim2real/reset_env` applies a profile inside `begin_reset()` /
  `end_reset()`, republishes jittered static camera TF, and rejects stale
  snapshots.
- Hardware-free tests plus one ros-marked service round-trip test.

## Scope of Task 2b

- Publish state/TF at the control rate independently of rendering.
- Render from a private snapshot; one render in flight; skip and count busy ticks.
- Reset waits for the in-flight render and blocks new renders.
- Optional `render_only_when_subscribed`.
- Opt-in renderer benchmarks; no default changes.
- Verification/hardening of `setup_envs.sh`, CI asset options, QoS, asset fetch,
  and `opencode.json`.

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
| RAM | 32 GB | 9.7 GiB visible in WSL (host: 15.3 GB) | below |
| CPU cores | 4 | 8 vCPU in WSL (host: 8 cores / 16 logical) | meets |
| Storage | 50 GB SSD | about 930 GB free | meets |
| Driver | Linux 595.58.03 | 551.76 | below tested version |

WSL limits `memory=10GB`, `swap=4GB`, `processors=8` are **active as of
2026-10-31 Task 2b**: `free -h` reports 9.7 GiB RAM / 4.0 GiB swap and `nproc`
reports 8. Task 2b used **2 colcon workers**; Task 2 measurements were taken with
the older 5.8 GiB / 2 vCPU limits.

Graphics:

- `glxinfo -B`: Mesa llvmpipe (LLVM 20.1.2), **"Accelerated: no"**, OpenGL 4.5.
- `MUJOCO_GL=egl` (default in the backend): initialises successfully and selects
  **Mesa llvmpipe**, i.e. software rendering.
- One read-only experiment with `GALLIUM_DRIVER=d3d12` reported
  `D3D12 (AMD Radeon(TM) Graphics)` (host integrated GPU through WSL). Adding
  `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` reported
  `D3D12 (NVIDIA GeForce RTX 4050 Laptop GPU)`. Both are **opt-in** benchmark
  settings, not defaults, and are never used by CI.

Consequences:

- **Isaac Sim high-fidelity tier: UNAVAILABLE on this machine**, based on
  published requirements rather than a local test. No install is authorized;
  the guarded adapter and documentation remain.
- **SmolVLA fine-tuning feasibility: UNKNOWN** until Task 8 measures it. Cloud
  or Colab-style GPU runs are an alternative, with checkpoints and provenance
  metadata brought back into the repository.
- **MuJoCo rendering plus ROS is still constrained by software GL.** Task 2b
  uses 256x256 cameras, two colcon workers with the active 9.7 GiB / 8 vCPU
  limits, and measured peak RSS; state/TF reach 20 Hz, images 2.67 Hz. See below.

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
| `/sim2real/staging_episode` | `sim2real_msgs/msg/RolloutEpisode` | expert collector / sim adapter | filter, recorder | per completed episode | reliable, transient-local, depth 10 | Unfiltered episodes are not training data. No images are embedded; late joiners receive recent completed episodes. |
| `/sim2real/filtered_episode` | `sim2real_msgs/msg/RolloutEpisode` | filter | dataset exporter, recorder | TBD (task 5) | TBD (task 5) | Only accepted episodes may be exported. |
| `/sim2real/filter_metrics` | `sim2real_msgs/msg/FilterMetrics` | filter | recorder, evaluation | TBD (task 5) | TBD (task 5) | Metrics explain a decision; they do not override safety stops. |
| `/workspace/pointcloud` | `sensor_msgs/msg/PointCloud2` | perception | VLA adapter, safety watchdog | TBD (task 7) | TBD (task 7) | Frame must be documented in the message header; no implicit frame. |
| `/workspace/tsdf_mesh` | `visualization_msgs/msg/Marker` | perception | RViz, safety watchdog | TBD (task 7) | TBD (task 7) | `TRIANGLE_LIST` marker; not a collision mesh. |
| `/vla/raw_action` | `trajectory_msgs/msg/JointTrajectory` | VLA adapter | safety/control adapter | TBD (task 8) | TBD (task 8) | `joint_names` = `panda_joint1..7` + `panda_finger_joint1`; `points[k].positions` = absolute targets for chunk step `k`; `time_from_start` = `(k+1)*dt`; `header.stamp` = newest observation stamp. |
| `/joint_trajectory_controller/joint_trajectory` | `trajectory_msgs/msg/JointTrajectory` | safety/control adapter | ros2_control trajectory controller | TBD (task 11) | TBD (task 11) | Only the safety adapter may publish here; raw VLA output must never reach the controller. |
| `/safety/status` | `sim2real_msgs/msg/SafetyStatus` | safety watchdog | operator/UI, recorder, supervisor | TBD (task 11) | TBD (task 11) | `NaN` means unknown, never safe. `is_safe=false` must stop dispatch. |
| `/estop` | `std_srvs/srv/Trigger` | operator/UI | safety watchdog | TBD (task 11) | latched, depth TBD (task 11) | Latched stop. Recovery requires `/estop_reset` (Task 11); no automatic clearing. |
| `/estop_reset` | `std_srvs/srv/Trigger` | operator/UI | safety watchdog | TBD (task 11) | TBD (task 11) | Defined in Task 11. Must only clear the latch after an explicit operator action. |
| `/sim2real/reset_env` | `sim2real_msgs/srv/ResetEnv` | evaluation harness / operator | sim node | on request | reliable | Reset returns the applied seed; a failed reset is not treated as success. Task 3 applies a validated profile inside the render quiesce and returns the sample plus an applied/skipped report. Unknown profiles or invalid overrides fail without mutating the scene. |
| `/sim2real/generate_rollouts` | `sim2real_msgs/srv/GenerateRollouts` | dataset tooling / operator | expert collector | on request | reliable | `num_rollouts` must be `1..max_rollouts_per_call`; `SAMPLING_UNIFORM` only (`FAILURE_TARGETED` returns `success=false` until Task 9). A batch runs on one worker thread and a concurrent call is rejected. |

QoS is now explicit in code (`src/sim2real/sim/qos.py`) and covered by a test
that parses this table. `/tf` uses depth 10 (tf2_ros defaults to 100). Camera
topics remain **reliable keep-last 5**: `ros2 topic echo
--qos-reliability best_effort --once` receives them because a reliable publisher
serves best-effort subscribers, but reverting to best-effort would reintroduce
the RViz Camera-display reliability mismatch observed in Task 2. The RViz Image
and Camera displays are already configured Best Effort; that alone was not
sufficient.

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
  `StepResult`, `CollisionInfo`, `BackendCapabilities`, `ResetConfig`,
  `RenderSnapshot` (with a `generation` field for stale-snapshot rejection).
- Abstract `SimBackend`: `reset(seed, config)`, `step(joint_target,
  gripper_target)`, `get_observation()`, `render(camera)`,
  `get_camera_info(camera)`, `camera_pose_world(camera)`,
  `collision_report()`, `capabilities()`, `close()`.
- Non-abstract `SimBackend.apply_randomization(sample)` defaults to returning no
  executed groups, which the applier reports as `SKIPPED`. Backends that support
  randomization override it and return per-group `GroupStatus` values.
- Only **absolute joint-position targets** are accepted. There is no torque,
  velocity, or delta interface.
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
| lighting | no | yes (three lights) | **unverified** |
| materials | flat colours | yes | **unverified** |
| camera extrinsics | yes (config + simple FK) | yes (compiled model) | **unverified** |
| physics mass | no | yes | **unverified** |
| physics friction | no | yes | **unverified** |
| joint damping | no (instant targets) | yes | **unverified** |
| distractors | no (Task 3: SKIPPED) | no (Task 3: SKIPPED) | **unverified** |
| object pose | yes (stored observation) | yes (free-joint qpos) | **unverified** |
| texture / colour | yes (flat RGBA) | yes (`geom_rgba`; `material_pool` PARTIAL) | **unverified** |

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

## Publishing/rendering decoupling (Task 2b)

Problem measured in Task 2: with `publish_images:=true`, MuJoCo's render callback
ran on the same single-threaded executor and starved the control timer
(`/joint_states` 1.87–3.28 Hz). With images off, it already reached 20 Hz.

Design (smallest change that meets the requirements):

1. **Two timer callback groups.** The control timer is in a
   `MutuallyExclusiveCallbackGroup`; the camera timer is in a
   `ReentrantCallbackGroup`. `main()` uses `MultiThreadedExecutor(num_threads=2)`.
2. **Snapshot under a short lock.** The control callback holds a backend lock
   only for `step()` + `capture_snapshot()`. `MuJoCoBackend.capture_snapshot()`
   copies `MjData` with `mj_copyData`; the fake backend copies its observation
   and object poses. The lock is released before publishing or rendering.
3. **One render thread, asynchronous submit.** `RenderCoordinator` owns a single
   worker thread because MuJoCo's EGL context is thread-bound. A camera callback
   submits a snapshot and returns immediately; the worker renders and publishes.
   If a render is already in flight the submission is rejected and counted as
   `skipped_busy`. There is no queue.
4. **Stamp/TF ordering.** The control tick publishes `/joint_states` and dynamic
   `/tf`, then exposes the snapshot to the camera group. A camera frame uses the
   snapshot's stamp, and the colour/depth/`CameraInfo` triplet shares it. A tf2
   lookup at that stamp finds a published sample; tf2 interpolates between
   samples and would fail on extrapolation into the future.
5. **Reset quiesce.** The reset service calls `begin_reset()`, which blocks new
   renders, invalidates older snapshot generations, and waits for the in-flight
   render. Task 3 applies model parameters and object poses inside this same
   window, then republishes static camera TF before `end_reset()`.
6. **Optional `render_only_when_subscribed`** (default `false`) skips a camera
   whose three publishers have no subscribers and counts the skip.
7. **Counters at shutdown:** `published`, `skipped_busy`, `skipped_reset`,
   `failed`, `skipped_unsubscribed` are printed directly (not via rosout, which
   may already be invalid on SIGINT). The async submit was added after CI showed
   that a blocking camera callback could still starve control on a smaller
   runner; it is the only path used by the ROS timer. The rate test waits for a
   live subscription and measures a 3 s window, excluding discovery startup. It
   uses `FakeBackend.render_delay_only` so the injected delay is a pure sleep,
   not a CPU-heavy raycast that would compete with the control thread. The rate
   probe runs on its own node/executor so it does not compete with the sim
   node's timers for callback threads.
8. **GIL caveat.** MuJoCo's C renderer releases the GIL, but Python callbacks and
   message construction still contend for it. This design improves control
   latency; it is not a hard real-time guarantee.

`publish_images:=false` does not create the camera timer. The control path still
captures a snapshot (cheap) so the node's structure stays identical.

## Domain randomization (Task 3)

### Module layout

```
src/sim2real/randomization/
  config.py      validated YAML schema, scene-derived defaults, profile loading
  sampler.py     pure (seed, episode_index) sampler and sample dataclasses
  applier.py     capability gating and APPLIED / PARTIAL / SKIPPED report
  color.py       approximate kelvin_to_rgb and the flat material pool
  __init__.py    public surface
configs/randomization/{none,light,default,heavy}.yaml
```

The package has no `rclpy` dependency. The ROS node calls it only inside the
reset-quiesce window.

### Config schema and validation

A profile is a YAML mapping with `schema_version: 1` and a top-level
`randomization` mapping. Every group has an optional `enabled` flag (default
`true`):

| group | fields |
| --- | --- |
| `object_pose` | `translation_bounds_m` (x and y ranges), `yaw_range_rad`, `min_separation_m` |
| `texture` | `mode` (`color_jitter` or `material_pool`), `rgba_range` or `pool` |
| `lighting` | `intensity_range`, `direction_jitter_rad`, `color_temperature_range_k` |
| `camera` | `translation_jitter_m` (3), `rotation_jitter_rad` (3), `apply_to` |
| `physics` | `object_mass_kg`, `friction.{sliding,torsional,rolling}`, `joint_damping_scale` |
| `distractors` | `enabled`, `count_range`, `pool` |

Validation is strict and path-qualified, for example
`randomization.physics.object_mass_kg: expected [lo,hi] with 0 < lo <= hi`.
It rejects unknown keys at every level, non-finite values, wrong types,
`lo > hi`, non-positive mass/friction/damping, object bounds outside the
`scene.yaml` workspace, unknown camera or material names, and jitter magnitudes
above the documented ceilings (`MAX_CAMERA_TRANSLATION_JITTER_M = 0.10`,
`MAX_CAMERA_ROTATION_JITTER_RAD = 0.35`,
`MAX_LIGHT_DIRECTION_JITTER_RAD = 0.50`). Colour-temperature ranges must lie in
the `kelvin_to_rgb` domain `[1000, 40000] K`. Profile names are restricted to
`[A-Za-z0-9_-]+`, so a request cannot escape the profile directory.

Object height is not sampled: it is derived as
`table_top + cube.size / 2` from `scene.yaml`. `enabled: false` (or the `none`
profile) returns a nominal sample whose poses and colours equal the scene
defaults and whose jitter/physics changes are identity.

### Determinism rules

`sample(config, episode_index)` is a **pure function**. It never reads or
writes global NumPy or Python RNG state. The base seed is `config.seed`:

```python
SeedSequence([config.seed, episode_index]).spawn(6)
# fixed order: object_pose, texture, lighting, camera, physics, distractors
```

Each group draws only from its own child stream. Disabling or adding a group
therefore never changes another group's values, and a sample does not depend on
call order or on how many samples were drawn earlier. A disabled group consumes
nothing from its stream. Tests cover same-input identity, different-seed
difference, order independence, global-RNG isolation, and stream independence
(disabling lighting leaves the object poses unchanged).

The reset service uses **`episode_index=0`**; the request `seed` overrides the
profile's default seed. A per-node counter is deliberately not used, so the same
seed and profile produce byte-identical `applied_config_json` regardless of
earlier resets. Later tasks can pass an explicit episode index to the sampler.

### Profiles

| profile | enabled | intent |
| --- | --- | --- |
| `none` | no | nominal template; all groups identity |
| `light` | yes | small pose/colour/lighting/camera/physics variation |
| `default` | yes | reset-service default; wide but bounded variation |
| `heavy` | yes | wide ranges plus a flat `material_pool` |

### Backend mapping and capability matrix

| group | fake | mujoco | isaac |
| --- | --- | --- | --- |
| object pose | **APPLIED** (stored observation) | **APPLIED** (free-joint `qpos`) | **UNVERIFIED** |
| texture / colour | **APPLIED** (flat RGBA); `material_pool` **PARTIAL** (flat approximation) | **APPLIED** (`geom_rgba` colour jitter); `material_pool` **PARTIAL** (no texture-image swap) | **UNVERIFIED** |
| lighting | **SKIPPED** (unlit) | **APPLIED** (`light_intensity`, `light_diffuse`/`light_specular`, `light_dir`/`light_dir0`) | **UNVERIFIED** |
| camera | **APPLIED** (local pose jitter) | **APPLIED** (`cam_pos`/`cam_quat` local jitter) | **UNVERIFIED** |
| physics | **SKIPPED** (kinematic) | **APPLIED** (`body_mass` + proportional `body_inertia`, `geom_friction`, `dof_damping`; `mj_setConst`) | **UNVERIFIED** |
| distractors | **SKIPPED** (no pool) | **SKIPPED** (no pooled parked bodies) | **UNVERIFIED** |

The applier gates each group on `BackendCapabilities` **before** merging the
backend result. A group is reported `APPLIED` only when the backend method
actually executed and returned `APPLIED`; a backend that returns nothing is
`SKIPPED`, and a capability gap is `PARTIAL`/`SKIPPED` with the capability
reason. `FakeBackend` records applied values in `applied_randomization` so tests
can verify them. `IsaacBackend` cannot be constructed; the applier raises a
clear refusal rather than pretending.

`material_pool` is intentionally **PARTIAL**: MuJoCo can change a geom's RGBA
at runtime but cannot swap texture images, so the applier applies a flat colour
from the fixed palette and says so. Distractors are **SKIPPED** because the
scene has no pooled parked bodies and Task 3 does not create bodies at runtime.

### Reset integration and quiesce

`sim2real.sim.reset_handler.reset_environment` is pure and does the following:

1. validate the seed, profile, and optional object-pose override **before**
   touching the backend, so a bad request cannot half-apply a scene;
2. load the profile and build the sample for `(seed, 0)`;
3. `backend.reset(seed, config)` restores nominal state and model parameters;
4. `apply_randomization(backend, sample)` applies the sample and returns the
   report;
5. an explicit override is validated against the workspace and applied as an
   object-pose-only step (to the first configured cube, `cube_red`);
6. on any application error, reset to nominal again and report the rollback.

The node calls this only inside the existing quiesce path, now holding the
backend lock across the whole window:

```
begin_reset()  -> wait for in-flight render, generation += 1
reset state    -> nominal
apply sample   -> model parameters + object poses
publish static TF (jittered cameras)
end_reset()
```

`begin_reset()` increments a scene generation. `RenderSnapshot` carries that
generation; the coordinator rejects a snapshot from an older generation and
counts `skipped_stale`. The node also clears `_latest_snapshot` during reset, so
the first camera frame after a reset comes from the new scene. This closes the
race where a camera callback read a pre-reset snapshot just before the reset and
submitted it afterwards.

Camera jitter changes static extrinsics, so the node republishes `/tf_static`
before releasing the quiesce. It uses a plain transient-local `TFMessage`
publisher rather than `tf2_ros.StaticTransformBroadcaster`, because the latter
deduplicates by child frame ID and silently ignores a changed transform. Late
joining subscribers receive the latest transient-local message.

The response keeps the existing wire fields (`success`, `message`,
`applied_seed`, `applied_config_json`). `applied_config_json` contains the
sample, the per-group report, the profile name, and the profile config hash, and
is validated by `metadata.validate_metadata_json` under the 64 KiB cap. Unknown
profiles and invalid overrides return `success=false` with a clear message and
no scene mutation.

### Known limitations (randomization)

- Reset-scoped, not continuous; there is no per-step randomization.
- `material_pool` is PARTIAL in MuJoCo and approximated in the fake backend.
- Distractors are SKIPPED; no runtime body creation.
- Isaac randomization is UNVERIFIED and refused by the applier.
- Camera jitter is applied in the camera's MuJoCo local frame, before the
  optical conversion; fake and MuJoCo use the same convention.
- The service always uses `episode_index=0`; explicit indices are available to
  the sampler for later tasks.
- The static-TF publisher is a plain transient-local publisher, not tf2's
  static broadcaster, for the reason above.

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

## Scripted expert and rollout generation (Task 4)

### Module layout

```
src/sim2real/expert/
  config.py          strict loading of configs/expert/*.yaml
  instruction.py     template parser -> TaskSpec(object_id, zone_id)
  ik.py              DampedLeastSquaresIK and IKResult
  planner.py         CartesianInterpolationPlanner + guarded MoveItPlanner
  state_machine.py   explicit states, transition table, guards, timeouts
  perturbations.py   deterministic intentional perturbations
  rollout.py         run_episode / run_episodes and EpisodeRecord
  staging.py         atomic NPZ + JSONL staging
  conversion.py      pure EpisodeRecord -> RolloutEpisode field dict
src/sim2real/sim/kinematics.py   the single Panda FK / TCP / numeric Jacobian
```

Nothing in this package imports `rclpy`, MuJoCo, or a renderer.

### One forward-kinematics source

The sim node publishes dynamic TF from `SimBackend.link_poses()` with
`tf_source=backend_fk`. Critically, `FakeBackend`'s previous FK was an explicitly
placeholder sine/cosine approximation. Task 4 replaces it with
`sim2real.sim.kinematics.forward_kinematics`, a pure-NumPy replication of the
pinned Menagerie MJCF body tree (`link1..link7`, `hand`). The same function
object supplies:

- `FakeBackend.link_poses()` (the TF path),
- `FakeBackend` end-effector and wrist-camera poses,
- the expert's TCP pose and finite-difference Jacobian.

A test asserts object identity, not just equal values. `MuJoCoBackend` keeps the
compiled model's own FK as ground truth. At five random configurations inside
the joint limits the measured maximum discrepancy is **3.45e-8** (position and
rotation element-wise); the test tolerance is `1e-6`. The residual comes from
the decimal quaternions in the MJCF versus exact trigonometric constants.

`tcp_pose(q)` is `hand_pose(q) @ tcp_offset` with `tcp_offset=[0,0,0.1]` from
`configs/sim/panda.yaml`. The Jacobian is central finite differences with
`step_rad=1e-6` (one-sided at a joint limit), documented in
`configs/expert/default.yaml`.

### Instruction parsing

`configs/expert/instructions.yaml` is the only source of accepted phrasings. A
template may contain `{colour}` and `{side}`; matching is case-insensitive,
collapses whitespace, and ignores trailing `.`, `!`, `?`. A match resolves to
`cube_<colour>` and `zone_<side>` and is validated against the loaded scene.
Unknown colours or zones, empty/whitespace text, and unmatched text raise
`UnparseableInstructionError`. There is intentionally **no default** colour or
zone.

### IK and planning

`IKSolver.solve(target_pose, q_init, panda)` returns
`IKResult(q, success, position_error, orientation_error, iterations)`.
`DampedLeastSquaresIK` clamps to joint limits every iteration, bounds each step
by `max_step_rad`, and stops at `max_iterations` or the position/orientation
tolerances. An unreachable target returns `success=False`; it never raises for
reachability and never loops unbounded.

`CartesianInterpolationPlanner` solves each Cartesian waypoint with IK and
time-scales each joint-space segment with a quintic (minimum-jerk) profile
`10u^3 - 15u^4 + 6u^5`. Segment duration is
`max(1.875*|dq|/v_max, sqrt(5.7735*|dq|/a_max), min_duration)` using the Panda
velocity/acceleration limits from `configs/sim/panda.yaml`. This yields zero
endpoint velocity and acceleration by construction.

`MoveItPlanner` is a **guarded, unexercised adapter**. MoveIt 2 is not installed
on this machine (`ros2 pkg list` and `/opt/ros/jazzy/share` have no MoveIt
entries). Construction raises `PlannerUnavailable`; even if MoveIt were
importable, `plan_to_pose` refuses with `PlannerUnavailable` until the adapter is
implemented and verified. No claim is made that MoveIt integration works.

### State machine

```
IDLE -> APPROACH_PREGRASP -> DESCEND -> GRASP -> LIFT -> TRANSIT
     -> PLACE_DESCEND -> RELEASE -> RETREAT -> DONE
any non-terminal -> FAILED  (guard or planner/IK failure)
any non-terminal -> TIMEOUT (per-state timeout)
```

`TRANSITIONS` is an explicit `(state, event) -> (next_state, guard_name)` table.
Illegal events raise `StateMachineError`. Guards are `pose_reached`,
`grasp_confirmed`, `gripper_open`, and `success`; a failed guard transitions to
`FAILED` with a reason. `tick(dt)` enforces per-state timeouts and transitions to
`TIMEOUT`. Every transition is logged as
`{from, to, event, reason, time_s}` and stored in the episode metadata. The
machine is pure deterministic code with no global state.

The motion phases are: pregrasp above the cube, descend to the grasp pose,
close and hold, lift, transit above the zone, descend to the place pose, open and
hold, retreat. A short settle loop holds the final joint target until the TCP
reaches the phase target (MuJoCo position actuators need a few control ticks);
the fake backend settles on the first hold.

### Success definition

An episode succeeds only when the target cube is within the zone radius in xy
(+`zone_radius_tolerance_m`), at rest at table height (`table_top + cube_size/2`
within tolerance), the gripper is open, and the cube position is stable for
`stable_steps` (range <= `stability_tolerance_m`). All thresholds live in
`configs/expert/default.yaml`.

### Intentional perturbations

`configs/expert/perturbations.yaml` defines:

| name | effect | expected quality |
| --- | --- | --- |
| `jitter` | deterministic high-frequency sinusoid on joint targets | degraded |
| `hesitation` | deterministic repeated targets (pauses) after motion phases | degraded |
| `detour` | extra offset waypoint during transit | degraded |
| `overspeed` | shorter quintic durations (`time_scale`) | degraded |
| `bad_grasp` | grasp target offset beyond the grasp radius + early open during lift | failure |
| `low_sweep` | low transit clearance toward the table | degraded |

`expected_quality` is diagnostic only; the recorded outcome is measured. Selection
is a pure function of `(seed, episode_index, perturbed_fraction)` using
`SeedSequence`. `perturbed_fraction` is a node/CLI parameter; `include_perturbed:
false` forces 0.

### FakeBackend grasp heuristic (test double, not physics)

When the gripper command is below `0.01` and the TCP is within `0.05 m` of a
cube, that cube is attached at its current relative transform and follows the
TCP. Opening the gripper above `0.02` releases it at table height. A TCP below
the table top by more than `0.005 m` is reported as a table contact. These
thresholds are fixed and documented; this model exists so the full pipeline is
CPU-testable, and it must not be read as contact dynamics.

### Rollout runner

`run_episode(backend, task, seed, ...) -> EpisodeRecord`:

1. derives a per-episode randomization seed from `(run seed, episode index)` with
   `SeedSequence([seed, episode_index, 0x52414E44])` (it does **not** rely on the
   reset service's `episode_index=0`);
2. resets through `sim2real.sim.reset_handler.reset_environment` with the chosen
   profile, so Task 3's capability gating still governs;
3. records the post-reset observation, plans and executes each phase, and
   records every control step;
4. catches backend/runner exceptions and records `ABORTED`; planner/IK failure
   records `FAILURE`; per-state and `max_steps` overflow record `TIMEOUT`.

Unsuccessful, timed-out, and aborted episodes are always recorded, never
dropped. `run_episodes` continues with the next seed. The runner creates no
`RenderCoordinator`; nothing renders, so no reset/render overlap is possible
in-process. The ROS node is responsible for not running rendering concurrently
with its (optional) MuJoCo batch.

`EpisodeRecord` carries the task ids, seed, backend, per-step joint
positions/velocities, EE pose, action (7 joint targets + gripper), per-step
`unintended_collision` derived from `backend.collision_report()`,
state-transition log, perturbation name, outcome, failure reason, and the full
randomization report. MuJoCo contact details involving the target cube and a
finger are treated as the intended grasp; any other reported contact is
`unintended_collision=true`. Task 5 will refine contact metrics.

### Atomic staging and conversion

`StagingBuffer` writes `episode_<id>.npz` (arrays) and `episode_<id>.json`
(metadata) to temporary files in the run directory, fsyncs, then `os.replace`s
them into place, and appends `{episode_id, seed, outcome, path, sha256,
written_at}` to `manifest.jsonl`. A crash before the rename leaves no final file.
Duplicate episode ids are rejected unless `overwrite=True`; overwrite keeps one
manifest row. Readers verify sha256, log and skip corrupt/incomplete entries, and
never abort a batch. Images are off by default. See `docs/DATASETS.md` for the
layout, size estimates, and limitations.

`episode_to_rollout_fields` is a pure conversion to the `RolloutEpisode` field
set. It rejects empty episodes, non-monotonic step stamps, invalid outcomes, and
metadata whose `num_steps` disagrees with the step list; `metadata_json` is
validated against the 64 KiB/NaN-rejecting metadata contract. Images are never
embedded.

### ROS adapter

`expert_collector_node` runs the chosen backend **in-process**. Rationale: the
expert executes at the control rate, and a ROS round-trip per step would add
scheduler latency and make trajectories depend on the executor rather than the
seed. The node is not a simulator and does **not** publish `/joint_states`;
`sim_node` remains the only `/joint_states` publisher when it runs.

It serves `/sim2real/generate_rollouts`, validates `num_rollouts` and
`sampling_mode`, runs the batch on a worker thread while the executor keeps
spinning, stages each episode, and publishes each completed `RolloutEpisode` on
`/sim2real/staging_episode` (reliable, transient-local, depth 10). A second
concurrent call is rejected rather than queued. `result_metadata_json` contains
`run_id`, seeds, requested/generated/failed counts, per-outcome counts, perturbed
count, profile, fraction, and the staging path. `FAILURE_TARGETED` returns
`success=false` with "FAILURE_TARGETED not implemented until Task 9".

## Deferred: joint-command subscription

No joint-command topic is subscribed in Task 2. The node holds its current
target and steps the backend. Command intake, safety filtering, and e-stop are
Task 11; publishing raw VLA output directly to a controller remains forbidden.

## Measured runtime (2026-10-31 Task 2b/3, 8 vCPU / 9.7 GiB WSL2)

MuJoCo **3.17.0**, NumPy 2.5.4 in `.venv` and system NumPy 1.26.4 in
`.venv-ros`.

### Before/after publishing decoupling

| Backend / images | `/joint_states` before | `/joint_states` after | `/tf` after | overhead image after | peak RSS after |
| --- | --- | --- | --- | --- | --- |
| fake / on | 20.1 Hz (Task 2) | 20.00 Hz | 19.99 Hz | 10.0 Hz | 113,100 KiB |
| mujoco / off | 20.0 Hz (this session) | 19.99 Hz | 20.00 Hz | n/a | 340,604 KiB |
| mujoco / on | **3.28 Hz** (this session; 1.87 Hz in Task 2) | **20.00 Hz** | **20.01 Hz** | **3.86 Hz** | **646,152 KiB** |

Before-fix `/tf` for mujoco/on was 1.89 Hz in this session. The pre-fix image rate
was 1.66 Hz in Task 2; the exact pre-fix image average was not captured after
the WSL limits changed, so it is not restated as 3.86's baseline. The important
result is that control/TF now run at the requested 20 Hz while rendering stays
slower under llvmpipe.

### Task 3 runtime (2026-10-31)

| Backend | seed 7 twice | seed 8 | unknown profile | report | `/joint_states` across reset |
| --- | --- | --- | --- | --- | --- |
| fake | identical JSON | differs | `success=false` | object_pose/texture/camera APPLIED; lighting/physics/distractors SKIPPED | **19.90 Hz** |
| mujoco headless | identical JSON | differs | `success=false` | object_pose/texture/lighting/camera/physics APPLIED; distractors SKIPPED | **19.70 Hz** |

Camera jitter changed the republished `/tf_static` overhead translation from
`(0.4645, -0.0132, 1.0951)` to `(0.4374, 0.0143, 1.1041)` on both backends.
The visual check wrote overhead PNGs for nominal and seeds 1-4 to
`log/task3_preview/` (git-ignored); mean RGB and cube positions varied per
seed. This is a visual sanity check, not a perception metric.

### Renderer benchmark (`scripts/bench_render.py`, 3 frames/camera)

| Configuration | Renderer string | overhead ms | wrist ms | peak RSS |
| --- | --- | --- | --- | --- |
| default `MUJOCO_GL=egl` | `llvmpipe (LLVM 20.1.2, 256 bits)` | 178.0 | 137.3 | 866,736 KiB |
| `GALLIUM_DRIVER=d3d12` | `D3D12 (AMD Radeon(TM) Graphics)` | 71.2 | 66.2 | 473,684 KiB |
| `GALLIUM_DRIVER=d3d12` + `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` | `D3D12 (NVIDIA GeForce RTX 4050 Laptop GPU)` | 83.0 | 80.4 | 442,376 KiB |

The Mesa variable is documented at
<https://docs.mesa3d.org/drivers/d3d12.html> as a case-insensitive substring
match. Both D3D12 paths name a hardware adapter; the NVIDIA path names the
RTX 4050 explicitly. Cross-renderer comparison against the llvmpipe frame of the
same seed: RGB mean absolute difference **0.08** (AMD) / **0.12** (NVIDIA) out of
255, depth MAD **1.16 mm**, with 415 of 65,536 validity-mask pixels differing
(mostly sky/edge pixels).

These are **OPT-IN** settings only: no default changed, CI never uses them, and
software rendering remains the documented baseline.


### Task 4 runtime (2026-11-01, 8 vCPU / 9.7 GiB WSL2)

| measurement | result |
| --- | --- |
| Core suite | **232 passed, 0 failed, 5 skipped, 0 errors** |
| ROS suite (`.venv-ros`) | **242 passed, 0 failed, 0 skipped** |
| `colcon test-result --verbose` | **7 tests, 0 errors, 0 failures** |
| FK vs MuJoCo, 5 random configs | max discrepancy **3.45e-8** (tolerance 1e-6) |
| Fake backend, raw `step()` | **2,578 control steps/s** |
| MuJoCo `backend.step()` | **1,344 control steps/s = 33,611 physics steps/s** (25 substeps/control step) |
| MuJoCo raw `mj_step` | **45,927 physics steps/s** |
| MuJoCo expert, 10 clean seeded episodes | **7/10 success**, 598 control steps/s, **14,961 physics steps/s** |
| Fake CLI, 20 episodes, 30% perturbed | **18 success / 2 failure** (both intended `bad_grasp`), 1,538 steps in 1.73 s (889 control steps/s end-to-end), 0.733 MiB |
| MuJoCo CLI, 20 episodes, 30% perturbed | **9 success / 11 failure**; clean 7/13, `bad_grasp` 0/2 (intended), `hesitation` 1/1, `jitter` 1/1, `detour` 0/1, `low_sweep` 0/1; 2.66 s, 0.727 MiB |
| `/tf_static` late joiner after a jittered reset | one TFMessage with **all 3** transforms (world→panda_link0, world→overhead, hand→wrist); depth 1, transient-local |
| ROS collector runtime | 3/3 episodes generated (1 perturbed); `/sim2real/staging_episode` echo received a complete episode |

The MuJoCo success rate is **measured, not tuned to a threshold**. The main
failure mode is an off-centre grasp under randomized cube pose: the cube lands
0.06-0.12 m from the zone centre, or the fingers cannot fully reopen around a
wedged cube. Yaw-aligning the gripper was tried and made the batch worse, so it
was reverted. No test asserts a success rate.

### Known limitations (Task 4)

- **MoveIt 2 is not installed and was not exercised.** `MoveItPlanner` raises
  `PlannerUnavailable`; no MoveIt behaviour is claimed.
- **The MuJoCo success rate is 7/10 on the clean seeded smoke and 9/20 with 30%
  perturbation.** Grasp centring and gripper reopening are the limiting factors.
  There is no retry or grasp-quality feedback.
- **`FakeBackend`'s grasp is a kinematic test double, not physics.** It attaches
  a cube within 5 cm of the TCP and ignores contact forces.
- **Perturbations are open-loop.** They vary commands; they are not a learned
  failure distribution (`FAILURE_TARGETED` is Task 9).
- **Images are not captured by the runner.** The staging API estimates disk and
  render time, but per-episode image capture is not wired in.
- **`unintended_collision` is coarse.** It maps the backend's contact report;
  MuJoCo cube↔finger contacts are treated as intended. Task 5 will add real
  collision metrics.
- **The collector runs one batch at a time** and rejects a concurrent call.
- **The collector is not a `/joint_states` publisher** and must not be confused
  with `sim_node`. Only one node may own `/joint_states`.
- **Task 3 regression fixed:** `mj_setConst` resets `MjData` to `qpos0`; the
  physics group now restores `qpos`/`qvel`. Before the fix, applying the `default`
  profile silently homed the arm and discarded sampled object poses in MuJoCo.

## Environment separation

| Environment | Created by | Contains | Must not contain |
| --- | --- | --- | --- |
| `.venv` | `uv sync` | core library + dev tools | ROS, torch, LeRobot |
| `.venv-ros` | `scripts/setup_envs.sh` | system Python 3.12 + ROS Jazzy via `--system-site-packages`, editable core package, pytest/pyyaml/ruff, MuJoCo 3.17.0 + rendering deps installed `--no-deps` | numpy/torch installs that shadow system packages |
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

## CI and fetched assets

The core job runs `uv sync --group sim`, then `./scripts/fetch_assets.sh`, then
`pytest` with `SIM2REAL_SKIP_RENDER=1`. The ROS job does not install MuJoCo and
does not fetch assets.

Options if the Panda meshes are absent:

1. **Current CI:** fetch them (about 33 MB per run), so MuJoCo physics tests run.
2. **No fetch:** the conftest skips every `sim`/`render` test with an explicit
   reason; fake-backend tests still pass. This is verified by temporarily hiding
   the mesh directory.
3. **Prebuilt cache:** cache the mesh directory between runs. Not implemented.

No additional asset-fetch step was added in Task 2b. The existing Task 2 step was
kept; `scripts/fetch_assets.sh` now verifies the checked-out SHA and the
Apache-2.0 LICENSE before copying.

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
| 17 | 2026-10-31 | Cameras are 256x256 at 10 Hz; control is 20 Hz with 2 ms physics | Software OpenGL and bounded WSL resources; Task 2b measured state 20 Hz, images 2.67 Hz |
| 18 | 2026-10-31 | Sim clock starts at zero and `/clock` is not published; camera triplets share one stamp | Deterministic recording; downstream uses ApproximateTimeSynchronizer (queue 10, slop 0.05 s) |
| 19 | 2026-10-31 | Menagerie meshes are git-ignored and restored by `scripts/fetch_assets.sh` | The pinned folder is 36.6 MB; only small XML, license, and attribution are committed |
| 20 | 2026-10-31 | Joint-command subscription deferred to Task 11 | Safety filtering and e-stop must exist before external commands reach the backend |
| 21 | 2026-10-31 | Camera topics use reliable keep-last 5, not best-effort sensor-data QoS | RViz Camera displays request reliable; a reliable publisher still serves best-effort consumers |
| 22 | 2026-10-31 | State/TF publication is decoupled from rendering with snapshots and a render worker | MuJoCo rendering on the control executor starved `/joint_states` to 1.87–3.28 Hz |
| 23 | 2026-10-31 | All MuJoCo rendering runs on one dedicated thread | EGL contexts are thread-bound; crossing executor threads caused `EGL_BAD_ACCESS` |
| 24 | 2026-10-31 | At most one render in flight; busy ticks are skipped and counted | No unbounded queue, bounded latency, observable counters at shutdown |
| 25 | 2026-10-31 | Reset quiesces rendering before mutating the scene | Task 3 will change model parameters; overlap with rendering is unsafe |
| 26 | 2026-10-31 | D3D12 renderer paths are opt-in benchmark settings only | They name hardware adapters but are not the default and are never used in CI |
| 27 | 2026-10-31 | Camera callbacks submit renders asynchronously; the worker publishes | CI showed a blocking camera callback could still starve control on a smaller runner |
| 28 | 2026-10-31 | Randomization samples use `SeedSequence([seed, episode_index])` with one child stream per group | Pure, order-independent, and disabling one group cannot perturb another |
| 29 | 2026-10-31 | The reset service uses `episode_index=0` and the request seed overrides the profile seed | Same seed/profile must be byte-identical regardless of earlier resets |
| 30 | 2026-10-31 | Randomization is applied only inside the existing reset quiesce | Model parameters are shared with the render worker; overlap is unsafe |
| 31 | 2026-10-31 | Snapshot generations reject pre-reset snapshots | A stale snapshot must never be rendered with the new model |
| 32 | 2026-10-31 | `/tf_static` uses a plain transient-local publisher, not `StaticTransformBroadcaster` | tf2's static broadcaster deduplicates by child frame ID and ignores a changed camera extrinsic |
| 33 | 2026-10-31 | `material_pool` is reported PARTIAL in MuJoCo; distractors are SKIPPED | Runtime texture-image swaps and runtime body creation are not supported by the scene |
| 34 | 2026-10-31 | The ROS rate test uses a 60% CI floor and keeps the 80% local floor | Shared CI runners are noisier; 60% still catches the 4.98-6.5 Hz starvation regressions |
| 35 | 2026-11-01 | One pure-NumPy Panda FK is shared by `FakeBackend` TF and the expert; MuJoCo remains ground truth | The old fake FK was a placeholder; duplicating FK would let TF and the planner disagree. Measured discrepancy vs MuJoCo is 3.45e-8 |
| 36 | 2026-11-01 | Default expert planner is Cartesian waypoint IK plus bounded joint quintics; MoveIt is a guarded, unexercised adapter | MoveIt 2 is not installed; a runnable, testable planner is required now, and no MoveIt claim may be made |
| 37 | 2026-11-01 | The expert state machine uses an explicit transition table with guards and per-state timeouts | Illegal transitions raise; guard failures become `FAILED` with a reason; timeouts become `TIMEOUT` |
| 38 | 2026-11-01 | Perturbation choice is a pure function of `(seed, episode_index, fraction)`; `expected_quality` is diagnostic only | Reproducible demonstrations; the measured outcome is recorded, not the label |
| 39 | 2026-11-01 | Staging writes atomic NPZ + JSON sidecar + sha256 manifest; images are off by default | Crash-safe iteration and verification; image capture needs size/time approval and becomes video in Task 7 |
| 40 | 2026-11-01 | The collector runs the backend in-process, does not publish `/joint_states`, and uses a worker thread per batch | Deterministic control-rate execution without ROS latency; `sim_node` stays the only `/joint_states` owner; the executor keeps spinning |
| 41 | 2026-11-01 | `/sim2real/staging_episode` is reliable, transient-local, depth 10 | A filter/recorder starting just after a batch still receives recent completed episodes; no images are embedded |
| 42 | 2026-11-01 | MuJoCo physics randomization saves/restores `qpos`/`qvel` around `mj_setConst` | `mj_setConst` resets `MjData` to `qpos0`; without the fix the physics group homed the arm and discarded sampled object poses |

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
