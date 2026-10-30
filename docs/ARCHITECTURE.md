# Architecture

Status: **Task 1 scaffold only.** No simulator, dataset, training, mapping,
inference, or safety implementation exists yet. This document records the
intended structure and the contracts agreed in Task 1; it does not claim those
components work.

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
| dev/test | CPU-only algorithms, contracts, CI | fake backend (planned Task 2) | scaffolded |
| high-fidelity | MuJoCo first, Isaac Sim optional | MuJoCo / Isaac Sim | MuJoCo pending Task 2; **Isaac Sim UNAVAILABLE on this machine** (published requirements) |
| physical | real Franka Panda | ROS 2 + ros2_control | out of scope; no hardware |

The project has no physical robot. Simulator results are not a safety case.

## Hardware reality (2026-10-30)

Official Isaac Sim 6.2.0 requirements:
<https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html>

| Resource | Isaac Sim 6.2.0 minimum | This machine | Result |
| --- | --- | --- | --- |
| GPU | GeForce RTX 4080 | RTX 4050 Laptop | below |
| VRAM | 16 GB | 6 GB | below |
| RAM | 32 GB | 5.8 GiB visible in WSL | below |
| CPU cores | 4 | 2 vCPU | below |
| Storage | 50 GB SSD | 930 GB free | meets |
| Driver | Linux 595.58.03 | 551.76 | below tested version |

Consequences:

- **Isaac Sim high-fidelity tier: UNAVAILABLE on this machine**, based on
  published requirements rather than a local test. No install is authorized;
  the guarded adapter and documentation remain.
- **SmolVLA fine-tuning feasibility: UNKNOWN** until Task 8 measures it. Cloud
  or Colab-style GPU runs are an alternative, with checkpoints and provenance
  metadata brought back into the repository.
- **MuJoCo rendering plus ROS on 5.8 GiB RAM is a risk.** Task 2 must use small
  camera resolutions and measure process memory.

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
| `sim2real_sim` | `ament_python` | Simulator backend adapter, `/joint_states` owner, reset service | skeleton only |
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
| `/joint_states` | `sensor_msgs/msg/JointState` | sim node **or** ros2_control (exactly one) | recorder, VLA adapter, safety watchdog | TBD (task 2) | TBD (task 2) | Exactly one publisher at any time. Never run the sim node and ros2_control together; duplicate publishers would corrupt state estimation. |
| `/tf` | `tf2_msgs/msg/TFMessage` | sim node | perception, VLA adapter, safety watchdog | TBD (task 2) | TBD (task 2) | Dynamic transforms must share one time base with `/joint_states`. |
| `/tf_static` | `tf2_msgs/msg/TFMessage` | sim node / robot_state_publisher | perception, VLA adapter | TBD (task 2) | TBD (task 2) | Static camera extrinsics; a stale value silently corrupts mapping. |
| `/camera/overhead/color/image_raw` | `sensor_msgs/msg/Image` | sim node | recorder, perception | TBD (task 2) | TBD (task 2) | Encoding `rgb8`. Images stay on disk, not inside `RolloutEpisode`. |
| `/camera/overhead/depth/image_raw` | `sensor_msgs/msg/Image` | sim node | recorder, perception | TBD (task 2) | TBD (task 2) | Encoding `32FC1` in metres. Never reinterpret as millimetres. |
| `/camera/overhead/camera_info` | `sensor_msgs/msg/CameraInfo` | sim node | perception | TBD (task 2) | TBD (task 2) | Must match the depth image resolution and intrinsics. |
| `/camera/wrist/color/image_raw` | `sensor_msgs/msg/Image` | sim node | recorder, perception | TBD (task 2) | TBD (task 2) | Encoding `rgb8`. |
| `/camera/wrist/depth/image_raw` | `sensor_msgs/msg/Image` | sim node | recorder, perception | TBD (task 2) | TBD (task 2) | Encoding `32FC1` in metres. |
| `/camera/wrist/camera_info` | `sensor_msgs/msg/CameraInfo` | sim node | perception | TBD (task 2) | TBD (task 2) | Must match the wrist depth image. |
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
| `/sim2real/reset_env` | `sim2real_msgs/srv/ResetEnv` | evaluation harness / operator | sim node | TBD (task 2) | TBD (task 2) | Reset must return the applied seed; a failed reset must not be treated as success. |
| `/sim2real/generate_rollouts` | `sim2real_msgs/srv/GenerateRollouts` | dataset tooling | expert/sim adapter | TBD (task 4) | TBD (task 4) | Bounded batch sizes; failures are reported, not silently retried forever. |

## Frames and conventions

| frame | parent | convention | notes |
| --- | --- | --- | --- |
| `world` | none (root) | z up | Single simulation world frame. |
| `panda_link0` | `world` | robot base | Mount transform TBD (task 2). |
| `panda_hand` | TF chain from `panda_link0` | EE frame | Used by `TrajectoryStep.ee_frame_id`. |
| `camera_overhead_optical_frame` | TBD (task 2/7) | optical: x right, y down, z forward | Depth/color must share it. |
| `camera_wrist_optical_frame` | TBD (task 2/7) | optical: x right, y down, z forward | Wrist camera moves with the arm. |

Additional rules:

- ROS 2 frame IDs are relative, without a leading `/`.
- `/joint_states` has exactly **one** publisher: the sim node **or**
  ros2_control, never both.
- Optical frames use REP-103/REP-105 conventions: x right, y down, z forward.
  This differs from the ROS body convention (x forward, y left, z up).
- `/vla/raw_action` joint order is fixed by `contracts.CONTROLLED_JOINT_NAMES`.

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
| 10 | 2026-10-30 | License is provisionally Apache-2.0 | Common ROS 2 default; user choice still pending |
| 11 | 2026-10-30 | Repository root is discovered dynamically (`git rev-parse --show-toplevel`); no fixed path is assumed | Docs and scripts must work from any checkout location; the verified checkout is `/home/kunal/Python/sim2real` |
| 12 | 2026-10-30 | Core and ROS test baselines use separate status lines | The core suite can skip ROS tests; mixing counts hid the core result (the old line recorded 113, a ROS count) |
| 13 | 2026-10-30 | Isaac Sim 6.2.0 tier marked UNAVAILABLE on this machine | Published minimums (16 GB VRAM, 32 GB RAM, 4 cores, RTX 4080) exceed measured hardware; not tested |
| 14 | 2026-10-30 | SmolVLA fine-tuning feasibility deferred to Task 8 | 6 GB VRAM may be insufficient; must be measured, with cloud GPU as a fallback |
| 15 | 2026-10-30 | Remove `<buildtool_depend>ament_python</buildtool_depend>`; rely on `<export><build_type>ament_python</build_type>` | The official Jazzy template does not declare that buildtool dependency, and rosdep has no rule for it |

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
6. **`docs/MASTER_SPEC.md` is the real specification** copied by the user
   (717 lines, LF-only). Its committed-copy SHA-256 is recorded in `CLAUDE.md`;
   comparison with the original Windows file is pending.
7. **License is provisional** (Apache-2.0 in `pyproject.toml` and all
   `package.xml` files). No `LICENSE` file exists; user choice is pending.
8. **ROS build verified locally and in CI**: `colcon build --symlink-install`
   and `colcon test` passed on 2026-10-30 with ROS Jazzy. CI run `41053040940`
   reports both jobs `success` via the GitHub API; the close-out commit still
   needs user confirmation in the Actions tab.
9. **`.venv-ros` includes an editable install of the core package** so the same
   suite can run with ROS available. It installs no dependencies (`--no-deps`),
   preserving the system NumPy.
10. **`ament_python` buildtool dependency removed.** The official
    `ros2 pkg create --build-type ament_python` template declares only
    `<export><build_type>ament_python</build_type>`. Removing our extra
    `<buildtool_depend>` let `rosdep install --from-paths ros2_ws/src
    --ignore-src --simulate` succeed **without** `--skip-keys`; `colcon build`
    and `colcon test` still pass.
