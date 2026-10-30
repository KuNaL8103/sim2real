
# MASTER PROMPT: End-to-End Sim2Real Robotics Data Pipeline and VLA Fine-Tuning Framework

## ROLE

You are a **Principal Embodied AI Engineer, Robotics Architect, ROS 2 Jazzy specialist, and expert Claude Code prompt engineer**.

I want to build the following project from scratch using **Claude Code in my terminal**, one stage at a time. I will execute each stage in a separate Claude Code session, verify the results, and push the changes to GitHub after every successfully completed stage.

**Your job is NOT to write the project's implementation code.** Your job is to produce a rigorous architecture plan and a complete set of 12 sequential, self-contained, ready-to-paste Claude Code prompts.

Every prompt must tell Claude Code precisely what to inspect, build, test, verify, document, commit, and push. Each prompt must be usable in a fresh Claude Code session without relying on unspoken conversational context.

Do not skip tasks, summarize prompts, or use phrases such as “follow the same pattern for the remaining tasks.”

---

## QUESTIONS TO SURFACE FIRST — DO NOT WAIT FOR ANSWERS

At the very top of your response, briefly list these hardware questions:

1. What GPU model and VRAM capacity are available?
2. Is a physical Franka Panda robot available, or will development initially use simulation and mock hardware?
3. Are NVIDIA drivers and a compatible CUDA environment already installed?

Then proceed with these sensible defaults without waiting for answers:

- One NVIDIA GPU with 24 GB VRAM.
- Franka Panda arm, tabletop pick-and-place environment.
- No physical robot initially; use simulation and `ros2_control` mock hardware.
- Ubuntu 24.04, ROS 2 Jazzy, and Python 3.12.
- Repository URL: `<YOUR_GITHUB_REPO_URL>`.

Do not invent the repository URL, hardware availability, installed dependencies, test results, or successful simulator execution.

## PROJECT

**Title:** End-to-End Sim2Real Robotics Data Pipeline and VLA Fine-Tuning Framework using Isaac Sim / MuJoCo and ROS 2 Jazzy

**Repository:** `<YOUR_GITHUB_REPO_URL>`

### Project objective

Build a modular system that generates synthetic robot demonstrations, filters them for quality, combines them with open-source robot datasets, fine-tunes a vision-language-action (VLA) policy, evaluates it in closed-loop simulation, generates more targeted data from failures, and exposes the policy through a reusable ROS 2 control stack with safety enforcement.

The finished project should be technically credible, reproducible, modular, testable without expensive hardware, and explainable in a robotics or embodied-AI interview. Prefer a small, working, well-tested implementation over a large framework that only appears complete.

## TECHNICAL STACK AND ARCHITECTURE

Use this stack unless official documentation exposes a genuine compatibility problem. If there is a conflict, flag it clearly, cite the relevant official documentation, and propose the smallest viable adjustment before committing to a design.

### 1. Operating system and development tools

- Ubuntu 24.04.
- ROS 2 Jazzy.
- Python 3.12.
- `uv` for managing non-ROS Python development and machine-learning environments.
- `colcon` for building the ROS 2 workspace.
- `pytest` for unit tests.
- `ruff` for linting and formatting checks.
- GitHub Actions for CPU-only continuous integration.
- Docker for reproducible development or test environments.
- Hydra/YAML configuration for simulator, randomization, data, training, evaluation, and safety parameters.
- Weights & Biases or MLflow for optional experiment tracking.

Do not mix system ROS Python dependencies into an incompatible `uv` environment. Document how the system Python used by ROS 2 interacts with the separately managed machine-learning environment.

### 2. Simulator architecture

The system must have a simulator-agnostic `SimBackend` interface.

- **MuJoCo 3.x:** Fast, headless, default development and smoke-test backend. Use supported offscreen rendering where available.
- **NVIDIA Isaac Sim:** High-fidelity synthetic-data backend, using its compatible ROS 2 bridge and Replicator capabilities where supported.
- **Franka Panda:** Seven-degree-of-freedom tabletop manipulation environment.
- **Robot control:** `ros2_control` and MoveIt 2, with a scripted expert or IK-based fallback if MoveIt is unavailable.
- **Rendering:** Synchronized overhead and wrist RGB-D streams.
- **Published state:** `/joint_states`, `/tf`, `/tf_static`, camera RGB-D topics, and camera calibration.
- **Domain randomization:** Object poses, textures, lighting, camera extrinsics, mass, friction, damping, and relevant distractors.

Before selecting an Isaac Sim release, verify the latest version that supports the required Ubuntu and ROS 2 configuration using current official NVIDIA documentation. Provide the exact version and source URLs. Do not assume compatibility merely because a version number looks plausible.

Isaac Sim must remain the high-fidelity target for the completed synthetic-data system. MuJoCo should provide the fast default development path and permit useful progress when Isaac Sim or a GPU is unavailable.

Check existing installations first. **Ask me before installing Isaac Sim, CUDA, large simulator dependencies, or multi-gigabyte datasets.** Do not install them silently or use `sudo` without authorization.

### 3. Initial repository and ROS 2 package structure

Use a monorepo with an initial layout similar to:

    .
    ├── CLAUDE.md
    ├── README.md
    ├── pyproject.toml
    ├── Dockerfile
    ├── .gitignore
    ├── .github/workflows/ci.yml
    ├── docs/
    │   ├── ARCHITECTURE.md
    │   ├── DATASETS.md
    │   ├── SAFETY.md
    │   └── DEPLOYMENT.md
    ├── configs/
    │   ├── sim/
    │   ├── randomization/
    │   ├── expert/
    │   ├── filtering/
    │   ├── mapping/
    │   ├── dataset/
    │   ├── training/
    │   ├── evaluation/
    │   └── safety/
    ├── scripts/
    ├── tests/
    ├── data/                    # generated/downloaded data; do not commit it
    └── ros2_ws/
        ├── src/
        │   ├── sim2real_msgs/
        │   ├── sim2real_sim/
        │   ├── sim2real_perception/
        │   ├── sim2real_vla/
        │   └── sim2real_control/
        └── ...

Refine this layout if required, but explain any architectural change in `docs/ARCHITECTURE.md`.

Use `ament_cmake` for the ROS 2 interface package and `ament_python` or `ament_cmake` as appropriate for the other packages. All ROS 2 packages must have suitable `package.xml` metadata, setup/build configuration, launch files where applicable, and meaningful tests.

Keep large datasets, simulator assets, model weights, credentials, local caches, logs, and generated rollout files out of Git.

### 4. Custom ROS 2 interfaces

Create these interfaces in `sim2real_msgs` during Task 1 so later tasks share stable contracts:

- `TrajectoryStep.msg`
- `RolloutEpisode.msg`
- `FilterMetrics.msg`
- `SafetyStatus.msg`
- `GenerateRollouts.srv`
- `ResetEnv.srv`

Define explicit, documented fields that cover timestamps, joint state, end-effector pose, action, language instruction, episode identity, random seed, task outcome, filter metrics, safety status, and useful metadata.

For example:

- `TrajectoryStep.msg`: timestamp, step index, task instruction, joint state, end-effector pose, and action.
- `RolloutEpisode.msg`: episode ID, backend, seed, task instruction, success/failure information, trajectory steps, and serialized metadata.
- `FilterMetrics.msg`: dimensionless squared jerk, SPARC, path length, joint-limit or velocity-limit margins, collision count, success, acceptance score, decision, and rejection reasons.
- `SafetyStatus.msg`: safe/unsafe state, emergency-stop status, failure reason, minimum obstacle distance, and relevant limit ratios.
- `ResetEnv.srv`: seed, randomization enablement/profile, optional object-pose override, and response describing the applied configuration.
- `GenerateRollouts.srv`: requested rollout count, random seed, task instruction, sampling mode, optional failure-distribution reference, and result metadata.

These are minimum requirements, not a reason to overcomplicate messages. Select sensible ROS-native field types, avoid duplicating unnecessary information, and validate serialized metadata.

### 5. ROS 2 topic and frame contracts

Use consistent names across packages, launch files, configuration, tests, and documentation.

**Robot state and transforms**

- `/joint_states`
- `/tf`
- `/tf_static`

**Overhead camera**

- `/camera/overhead/color/image_raw`
- `/camera/overhead/depth/image_raw`
- `/camera/overhead/camera_info`

**Wrist camera**

- `/camera/wrist/color/image_raw`
- `/camera/wrist/depth/image_raw`
- `/camera/wrist/camera_info`

Use `message_filters` or an equivalent supported synchronization mechanism where synchronized camera observations are required. Document the chosen time synchronization policy and frame naming convention.

Use stable frame names such as `world`, `panda_link0`, `panda_hand`, `camera_overhead_optical_frame`, and `camera_wrist_optical_frame`, respecting the actual Franka URDF and optical-frame conventions.

**Language and rollout data**

- `/task_instruction`
- `/sim2real/staging_episode`
- `/sim2real/filtered_episode`
- `/sim2real/filter_metrics`

**Workspace mapping**

- `/workspace/pointcloud` using `sensor_msgs/PointCloud2`
- `/workspace/tsdf_mesh`, using a documented visualization message type
- Optional `/workspace/occupancy_map` if OctoMap is implemented

**VLA and control**

- `/vla/raw_action`, using a documented and consistent ROS action-chunk representation.
- `/joint_trajectory_controller/joint_trajectory`, using `trajectory_msgs/JointTrajectory`.
- `/safety/status`.
- `/estop` as a service, preferably using `std_srvs/Trigger`, with latching semantics and a documented recovery path.

Document the topic ownership, message types, producers, consumers, rates, and safety implications.

### 6. Domain randomization

Implement a configuration-driven, seeded randomization engine.

Include:

- Object pose and orientation.
- Textures and materials.
- Lighting intensity, direction, and color temperature.
- Camera extrinsic jitter.
- Physics parameters such as object mass, sliding/torsional friction where supported, and joint damping.
- Optional distractor placement.

Use a config with explicit bounds and types, for example:

    randomization:
      enabled: true
      seed: 42
      lighting:
        intensity_range: [...]
        color_temperature_range: [...]
      camera:
        translation_jitter_m: [...]
        rotation_jitter_rad: [...]
      physics:
        object_mass_kg: [...]
        friction: [...]
        joint_damping: [...]
      object_pose:
        translation_bounds_m: [...]
        yaw_range_rad: [...]

The values must be valid for the simulator and robot. Document backend-specific limitations rather than pretending that every randomization parameter has identical effects in MuJoCo and Isaac Sim.

### 7. Expert demonstrations and trajectory filtering

Build a scripted language-conditioned pick-and-place expert with an explicit, testable state machine.

Use MoveIt 2 where available; provide a bounded IK/controller abstraction for development environments without the full planning stack. Generate both clean and intentionally perturbed candidate trajectories so the filter has meaningful positive and negative cases.

Compute and document:

- Dimensionless squared jerk, using a well-defined trajectory normalization such as \(J_d = \frac{T^5}{L^2}\int_0^T \|\dddot{x}(t)\|^2\,dt\), with suitable handling of zero-length or near-zero trajectories.
- Spectral Arc Length (SPARC), calculated as the negative arc length of the appropriately normalized movement-speed spectrum over a documented frequency range.
- Joint- and velocity-limit margins using the configured robot limits.
- Path length.
- Collision count.
- Task success and failure reasons.

Do not implement placeholder metrics that merely return constants. Test the metrics using synthetic trajectories with known qualitative behavior.

Use configuration-driven acceptance criteria, a deterministic score, and structured rejection reasons. Route accepted demonstrations to a curated dataset buffer and rejected demonstrations to diagnostic logs. Protect the data pipeline from malformed or incomplete episodes.

### 8. 3D workspace mapping

Use Open3D Scalable TSDF Volume integration and `tf2_ros` to fuse synchronized RGB-D frames from the overhead and wrist cameras.

Start with a configurable default TSDF voxel size, for example `0.01 m`, and truncation distance, for example `0.04 m`. Validate the settings and keep memory use bounded.

Publish `/workspace/pointcloud` and a documented TSDF mesh/visualization output. Implement a fast obstacle-distance or occupancy-query helper for downstream safety checks. OctoMap is optional, but if implemented, document how the fused representation is converted into the occupancy representation and how unknown space is treated.

Use tiny recorded or procedurally generated RGB-D fixtures for tests. The mapping unit tests must not require a live simulator, camera, GUI, or GPU.

### 9. Dataset format and mixing

Use the latest compatible version of Hugging Face LeRobot and its dataset APIs. Prefer the current stable dataset format supported by the pinned dependency; **do not hardcode LeRobot v2 if the compatible supported format has moved on**.

Provide conversion adapters for:

- Curated synthetic episodes.
- An appropriately licensed subset of Open X-Embodiment or BridgeData, converted from its source representation such as RLDS where necessary.

Do not download large datasets without asking me first.

Align observation and action spaces explicitly. Compute and store normalization statistics including `mean`, `std`, `q01`, and `q99` for relevant joint and end-effector action dimensions. Document coordinate conventions, units, missing values, episode boundaries, language instructions, and the selected embodiment mapping.

Implement a configurable mixing sampler. Test it using tiny local fixtures instead of requiring real dataset downloads.

### 10. VLA fine-tuning

Use SmolVLA as the default model family where supported by the pinned LeRobot stack, with an appropriately sized model such as SmolVLA-450M if the actual available model supports that configuration.

Keep OpenVLA-7B as an optional configuration, not a mandatory dependency for local testing.

Use Hugging Face PEFT LoRA with configurable parameters, including an initial configuration such as:

    model:
      family: smolvla
      checkpoint: <verified_model_identifier>
    lora:
      enabled: true
      r: 32
      alpha: 64
      dropout: 0.05
    training:
      smoke_test: true
      max_steps: ...
      batch_size: ...
      seed: ...

Verify the selected model's actual module names and adapter compatibility instead of assuming the same LoRA target modules work across model families.

Include action chunk encoding/decoding and normalization/de-normalization, checkpointing, reproducible configurations, optional W&B/MLflow tracking, and offline action-validation metrics.

Run a tiny smoke test only after checking available resources and dependencies. Ask before installing large machine-learning dependencies or downloading model weights. GPU-free unit tests must use small synthetic fixtures and mocked model interfaces.

### 11. Closed-loop evaluation and the data flywheel

Evaluate a policy in closed-loop simulation with reproducible task seeds and configuration.

Record success rate, task completion, trajectory metrics, collision information, termination reason, and failure metadata.

Implement failure-mode analysis over relevant variables such as initial object pose, friction, lighting, and randomization seed. A GMM or KDE may be used when enough failure samples exist. Require a documented minimum sample count and a bounded fallback sampling strategy for sparse failure data.

When failures are detected, trigger targeted rollout generation through `GenerateRollouts.srv` so the next training cycle focuses on underperforming regions.

Test that:

- Failure records are valid and reproducible.
- The sampler stays within configured simulation bounds.
- Sparse or invalid data does not cause crashes.
- The orchestrator does not start endless or uncontrolled retraining loops.
- Every generated training cycle can be traced to its configuration, source episodes, model checkpoint, and evaluation run.

### 12. ROS 2 inference and safety

Implement a reusable VLA inference node and a distinct safety watchdog/control layer.

The inference node subscribes to synchronized camera observations, `/joint_states`, and `/task_instruction`, and publishes raw predicted action chunks to `/vla/raw_action`.

The safety node must validate actions before sending them to the joint trajectory controller. Implement:

- Workspace bounds.
- Joint position and velocity constraints.
- Acceleration and jerk limits where applicable.
- Obstacle-distance checks against the workspace map.
- EMA action smoothing, with a configurable coefficient.
- A watchdog for stale observations or action output.
- A latched emergency stop service at `/estop`.
- Clear safe-state behavior when the map, transforms, or required observations are missing or invalid.

Test the safety layer with malformed, stale, out-of-bounds, excessive-velocity, collision-proximity, and emergency-stop cases.

Use `ros2_control` mock hardware for HIL-style integration tests. Never claim a mocked test establishes physical safety. Physical deployment must be a separate, explicitly documented step requiring user confirmation and appropriate safety precautions.

---

## NON-NEGOTIABLE EXECUTION RULES

### Rule 1 — No fabricated progress

Every future Claude Code prompt must instruct Claude Code to inspect the actual repository and verify all previous-stage claims before starting.

Require inspection of:

- `CLAUDE.md` and `docs/ARCHITECTURE.md`.
- `git status`, recent `git log`, and the configured Git remote.
- Relevant source and configuration files.
- The current test output.

Do not assert that a prior feature exists, a topic works, a commit was pushed, or a test passed unless the actual repository or command output proves it.

Because the final cumulative test count is unknown before implementation, each prompt must instruct Claude Code to retrieve the latest recorded count from `CLAUDE.md`, run the full suite, and report the actual results. **Never invent or hardcode a predicted cumulative test count.**

Use the previous tasks' explicit deliverables as the expected handoff state, but reconcile any mismatch with the actual repository before proceeding. Never overwrite unrelated user changes.

### Rule 2 — Every stage must be independently testable

All 12 tasks must include tests that run without a physical robot, GPU, live camera, simulator GUI, or external dataset download wherever practical.

Use mocks, temporary directories, in-memory fixtures, small synthetic episodes, fake backends, and sample sensor messages where appropriate.

Each task must specify a minimum target for newly added tests, while separately reporting the actual full-suite results. The final test report must distinguish passed, failed, skipped, and errored tests.

A test that was skipped must not be reported as passed. Do not claim a ROS topic, RViz visualization, simulator integration, training run, or physical deployment worked without carrying out the relevant verification.

### Rule 3 — Inspect dependencies before installing anything

Before every installation or expensive setup operation:

1. Inspect the operating system, ROS distribution, Python version, package availability, existing environments, and relevant installed dependencies.
2. Check available GPU/VRAM where relevant.
3. Explain what is missing and how much extra installation or download may be required.
4. Ask for my approval before installing large dependencies, Isaac Sim, CUDA, or multi-GB datasets.
5. Continue with hardware-free implementation and tests while an optional dependency is unavailable.

Do not silently install dependencies, download large weights, or use `sudo`.

### Rule 4 — One stage per session

Each prompt must represent roughly 1–3 hours of focused work and must have a clear minimum viable deliverable.

Do not expand a task into an unbounded framework. If an optional feature threatens the core deliverable, complete the smallest working version, add an explicit limitation to the documentation, and record the deferred work.

### Rule 5 — Every task prompt must contain a concrete stateful handoff

For Task 2 through Task 12, the very first line must be exactly:

`Read CLAUDE.md and docs/ARCHITECTURE.md before we start — both are current.`

Follow it with a specific expected current-state recap:

- Previously implemented package names and components.
- Relevant topics, services, and data flows already established.
- What the previous stages were supposed to verify.
- What has not yet been implemented.
- The requirement to verify the real Git history, test output, and remote before trusting those claims.

Never fabricate test counts or previous commits. Tell Claude Code to read the latest verified test count from `CLAUDE.md` and re-run the suite before making further changes.

Task 1 is the sole exception: the documentation does not yet exist, so its prompt must start from a new or uninitialized repository and create those files.

### Rule 6 — Exact file paths, contracts, and verification commands

Every prompt must specify:

- Files to create or modify.
- ROS 2 package and node names.
- Topic and service names and message types.
- Relevant configuration keys.
- Required mathematical metrics where applicable.
- CLI commands to build, source, test, and inspect the implementation.
- The exact minimum tests to add and the expected behavior those tests must prove.
- Any manual verification steps.

Do not demand irrelevant commands for a task. For example, a mapping-only unit-test task should not require Isaac Sim to launch. A camera integration task may need a manual visualization check, but its unit tests must still be headless.

### Rule 7 — Manual verification checkpoints

When a task needs a manual GUI, RViz, teleoperation, simulator, or hardware check:

- Provide the exact commands and explain what I should observe.
- Clearly distinguish commands Claude Code can run from actions I must perform.
- If confirmation is required before committing, stop and wait for my confirmation.
- Never say that a manual step passed before I confirm it.
- If the required hardware or application is unavailable, complete the hardware-free tests, mark the integration check as pending, and state what remains unverified.

### Rule 8 — Git workflow and documentation handoff

Use the repository's existing branch and remote safely. The intended publishing target is `origin main`, and the configured remote should match `<YOUR_GITHUB_REPO_URL>`.

At each successful stage:

1. Run the relevant package build, lint checks, and full test suite.
2. Update `CLAUDE.md`, including the actual verified current status and actual test count.
3. Update `docs/ARCHITECTURE.md` if the architecture, package responsibilities, interfaces, or data flow changes.
4. Run `git status` and review the diff; do not include datasets, weights, credentials, local artifacts, or unrelated user changes.
5. Stage the appropriate files, commit with the exact Conventional Commit message specified for that task, and push to `origin main`.
6. Verify the resulting commit and working-tree state. Report the actual push outcome.

Never force-push, discard user modifications, overwrite unexpected remote work, or claim a push succeeded if it failed. If the remote is not configured or GitHub refuses the push, explain the blocker and give the exact recovery commands without pretending the task has been published.

Do not perform the commit until the task's required verification gates have passed.

---

## REQUIRED 12-TASK SEQUENCE

Use exactly these 12 tasks. You may refine the internal deliverables to improve dependencies, but do not remove, merge, or add tasks.

### Task 1 — Repository Scaffold, Documentation, CI, ROS Workspace and Interfaces

Create the repository structure, ROS 2 workspace, package skeletons, `CLAUDE.md`, `docs/ARCHITECTURE.md`, configuration layout, `uv` development setup, `ruff`/`pytest` configuration, Dockerfile, `.gitignore`, and CPU-only GitHub Actions workflows.

Create `sim2real_msgs` and the six initial interfaces: `TrajectoryStep.msg`, `RolloutEpisode.msg`, `FilterMetrics.msg`, `SafetyStatus.msg`, `GenerateRollouts.srv`, and `ResetEnv.srv`.

This is the starting task. The required documentation does not yet exist, and no simulator, dataset, training, mapping, or inference implementation should be assumed to exist.

**Minimum new-test target:** at least 6 meaningful hardware-free tests or contract checks. Report actual results.

**Commit message:** `feat: scaffold sim2real workspace and interfaces`

### Task 2 — Simulator Backends, Franka Scene, RGB-D and ROS 2 Integration

Implement `SimBackend` and a runnable headless MuJoCo backend with a Franka Panda tabletop scene. Add a guarded Isaac Sim adapter and verify the supported Isaac Sim/ROS 2 version against official documentation.

Publish `/joint_states`, `/tf`, `/tf_static`, and the agreed overhead/wrist RGB-D and camera-info topics. Implement launch/configuration files and provide RViz verification instructions. The Isaac installation and high-fidelity run must be optional until approved.

**Minimum new-test target:** at least 8 backend, scene-contract, and mock-publishing tests.

**Commit message:** `feat: add simulator backends and Franka scene`

### Task 3 — Seeded Domain Randomization Engine

Implement the `domain_randomizer` module inside the appropriate package. Integrate seeded object-pose, material, lighting, camera-extrinsics, and supported physics randomization with `ResetEnv.srv`.

Add YAML configs, validation, reproducibility checks, backend capability handling, and tests. Do not claim parameters are randomized on a backend unless the implementation actually does it.

**Minimum new-test target:** at least 8 tests covering determinism, bounds, invalid configuration, and backend-independent behavior.

**Commit message:** `feat: add seeded domain randomization`

### Task 4 — Scripted Expert and IK Rollout Generator

Implement `expert_collector_node` and the rollout runner. Create a language-conditioned pick-and-place state machine with a MoveIt 2 integration where available and a bounded IK/controller abstraction for testable environments.

Generate clean and perturbed demonstrations into a staging buffer, record seeds and task outcomes, and handle simulator resets and unsuccessful rollouts.

**Minimum new-test target:** at least 8 tests of the state machine, trajectory schema, failure handling, and fake-backend execution.

**Commit message:** `feat: add scripted expert rollout collector`

### Task 5 — Active Trajectory Filtering and Curated Dataset Writer

Implement `trajectory_filter_node`, including dimensionless squared jerk, SPARC, path length, configured joint/velocity-limit margins, collision counts, task success, configurable acceptance criteria, and structured rejection reports.

Write accepted episodes into a versioned, documented LeRobot-compatible dataset layout with rich metadata. Keep staging, curated, and rejected outputs separate. Test metric edge cases and data integrity using synthetic trajectories.

**Minimum new-test target:** at least 12 tests, including known smooth/jerky trajectories, invalid episodes, threshold behavior, and writer/rejection-log correctness.

**Commit message:** `feat: filter rollouts and write curated datasets`

### Task 6 — Multi-View RGB-D Mapping and Obstacle Queries

Implement `workspace_mapper_node` using Open3D Scalable TSDF Volume and `tf2_ros`. Fuse overhead/wrist camera data when synchronized, publish `/workspace/pointcloud` and a documented TSDF mesh visualization, and implement a fast obstacle-distance or occupancy-query helper.

Use small recorded or procedural fixtures and mock transforms in hardware-free tests. OctoMap may be added if the representation conversion and query semantics are documented.

**Minimum new-test target:** at least 8 mapping, transform, input-validation, and obstacle-query tests.

**Commit message:** `feat: add RGB-D workspace mapping`

### Task 7 — LeRobot Dataset Builder and Open X-Embodiment Bridge

Implement `dataset_builder` to convert curated synthetic episodes and a licensed Open X-Embodiment/BridgeData subset into the selected compatible LeRobot format.

Align observations, actions, units, coordinate conventions, language instructions, and episode boundaries. Compute `mean`, `std`, `q01`, and `q99` normalization statistics. Implement a configurable dataset-mixing sampler and validate source/target schemas.

Use small fixtures for CI; ask before downloading large datasets.

**Minimum new-test target:** at least 10 tests of conversion, normalization, schema validation, sampling ratios, and malformed samples.

**Commit message:** `feat: add LeRobot and Open X-Embodiment dataset mixer`

### Task 8 — VLA LoRA Fine-Tuning Engine

Implement `vla_trainer` using the compatible pinned LeRobot and Hugging Face PEFT stack. SmolVLA is the default model family where supported; OpenVLA-7B remains an optional configuration.

Implement configurable LoRA parameters, action chunk handling, normalization/de-normalization, smoke-test training, checkpoint export, offline L1 action validation, and experiment tracking. Verify model compatibility and target modules instead of assuming them.

Use mocked model interfaces and tiny synthetic batches for GPU-free tests. Ask before installing large dependencies or downloading model weights.

**Minimum new-test target:** at least 8 tests of configuration, normalization round trips, action chunk shapes, checkpoint metadata, and mocked training behavior.

**Commit message:** `feat: add LoRA VLA fine-tuning pipeline`

### Task 9 — Closed-Loop Evaluation and Failure-Mode Sampling

Implement `closed_loop_eval_node` to evaluate checkpoints in closed-loop simulation and record success, failure, collision, and trajectory metrics.

Implement bounded failure-mode analysis using a GMM or KDE over relevant initial states and domain-randomization variables. Trigger targeted rollouts through `GenerateRollouts.srv`. Include sparse-data handling, deterministic seeds, run metadata, and limits that prevent uncontrolled retraining loops.

**Minimum new-test target:** at least 10 tests covering evaluation records, failure distributions, sparse samples, targeted sampling bounds, and orchestration errors.

**Commit message:** `feat: add closed-loop evaluation and failure sampling`

### Task 10 — ROS 2 VLA Inference Node

Implement `vla_inference_node` to consume synchronized RGB observations, `/joint_states`, and `/task_instruction`, load a compatible checkpoint through an abstraction, and publish action chunks to `/vla/raw_action`.

Add configurable inference settings, timestamp/sequence validation, normalization/de-normalization, bounded action dimensions, and clean failure behavior when the model or inputs are unavailable.

Use fake models and sensor messages for CPU-only tests. Do not load a large model merely to run the normal unit suite.

**Minimum new-test target:** at least 8 tests of input validation, preprocessing, mocked predictions, output schemas, and stale-input handling.

**Commit message:** `feat: add ROS 2 VLA inference node`

### Task 11 — Safety Watchdog, HIL Mock Hardware and Control

Implement `safety_watchdog_node` to validate raw action chunks against workspace bounds, joint/velocity/acceleration limits, obstacle-distance checks, stale data, and model/input failures.

Add configurable EMA smoothing, watchdog behavior, a latched `/estop` service, safe-state handling, and forwarding to `/joint_trajectory_controller/joint_trajectory` only after validation.

Integrate `ros2_control` mock hardware where supported and test the control chain without a physical robot. Clearly separate mock-hardware evidence from real-hardware safety validation.

**Minimum new-test target:** at least 12 tests of limits, unsafe actions, emergency stop, stale messages, unknown obstacle state, and valid action forwarding.

**Commit message:** `feat: add HIL safety watchdog and control`

### Task 12 — End-to-End Orchestration, Deployment Guide and Release

Implement `pipeline_orchestrator.py` or an equivalent explicit orchestration component to connect data generation, randomization, expert collection, filtering, dataset mixing, training, evaluation, and targeted data regeneration.

Add integrated launch files, configuration examples, end-to-end tests, architecture and data-flow diagrams, README instructions, failure-recovery procedures, and a deployment guide for mock hardware and optional physical Franka bring-up.

Run the practical validation levels that are available:

- CPU-only unit tests.
- ROS 2 build and integration tests.
- Headless MuJoCo tests.
- Isaac Sim integration tests if installed and approved.
- `ros2_control` mock-hardware HIL tests.
- Physical-hardware checks only if the robot is actually available and I explicitly approve the procedure.

A missing simulator or robot must be reported as an unverified integration level, not disguised as a pass. Finish by reviewing documentation, Git status, dependency requirements, reproducibility, and release readiness. Create a release tag only after the actual validation results support the release claims.

**Minimum new-test target:** at least 6 end-to-end orchestration, configuration, failure-recovery, or integration-contract tests, in addition to the full existing suite.

**Commit message:** `feat: integrate pipeline and document deployment`

---

### Windows Host, WSL2 and VS Code Development Environment

My computer runs Windows, and I intend to develop this Linux-targeted robotics project using Ubuntu 24.04 through WSL2 and VS Code.

The target development environment is Ubuntu 24.04, ROS 2 Jazzy, and Python 3.12.

Before implementing Task 1, perform an environment preflight.

1. Determine whether Claude Code is running in native Windows, WSL2, or a native Linux environment.
2. If running on Windows without WSL2, explain how I can install or verify WSL2 with Ubuntu 24.04 and connect VS Code to that environment.
3. If WSL2 is already configured, inspect the actual Ubuntu distribution and version before recommending additional setup.
4. Verify whether ROS 2 Jazzy, Python 3.12, Git, and the necessary development tools are installed.
5. Check GPU availability and the environment's suitability for the planned simulators without assuming that GPU access or simulator compatibility is available.
6. Provide the exact commands for the detected environment. Clearly identify which commands must run in Windows PowerShell and which must run in the Ubuntu/WSL terminal.
7. Do not install WSL, Ubuntu, ROS 2, CUDA, Isaac Sim, large dependencies, or datasets without my explicit approval.
8. Do not use sudo without my authorization.
9. If WSL2 is unavailable, explain the setup steps and stop before making system-level changes. Continue with repository planning or other safe work where possible.
10. Keep the Git repository inside the Linux filesystem, for example `~/projects/sim2real`, rather than under `/mnt/c`, unless a specific requirement justifies otherwise.
11. Explain how to reopen the repository in VS Code using the WSL extension and verify that the integrated terminal is actually running in Ubuntu.
12. Document the verified development environment and any outstanding setup requirements in `CLAUDE.md`.

Do not claim the environment is ready until the relevant checks have actually been performed.

## REQUIRED OUTPUT FORMAT

Your response must be organized into these sections.

### Part A — Brief Architecture Reference and 12-Task Plan

At the very top, list the three hardware/environment questions and immediately state that the plan proceeds with the specified defaults.

Then give:

1. A concise architecture summary.
2. The core package responsibilities.
3. The data flow from simulation through filtering, dataset conversion, training, evaluation, failure-driven generation, inference, and safety.
4. The agreed ROS 2 topics, services, and key message contracts.
5. One table listing all 12 tasks, with columns:
   - Task number.
   - Title.
   - Goal.
   - Deliverables.
   - Definition of done.

The architecture must clearly distinguish development/test backends from high-fidelity simulation and physical-hardware deployment.

### Part B — Copy-Pasteable Claude Code Prompts

Generate the prompts in the order listed in the task breakdown.

**For the first response, provide Part 1: the architecture reference, the full 12-task table, and Tasks 1–6 written out in complete detail.** Do not shorten the prompts to save space. After Task 6, provide the risk list from Part C, then end by asking me to reply exactly:

`Continue with Tasks 7 to 12`

In the continuation, provide Tasks 7–12 in full. Do not repeat the already-completed prompts unnecessarily.

Every task prompt must be in its own fenced code block. Put its task number and title immediately above the code block.

#### Required contents of every Claude Code prompt

Every prompt must contain all of the following:

1. **Stateful opening.** Task 1 is the only exception to the required opening line. Task 2 onward must begin with the exact line:

   `Read CLAUDE.md and docs/ARCHITECTURE.md before we start — both are current.`
2. **Current-state recap.** Accurately specify the packages/components that the preceding tasks were expected to implement, their relevant verified topic/service contracts, what has not yet been implemented, and how the previous changes should be recorded in Git. Tell Claude Code to verify the repository, actual Git history, and recorded test output instead of trusting unverified claims.
3. **A clear work statement.** Use the phrase `Here's what I want to work on today:` and describe the exact scope of the current stage.
4. **Concrete implementation instructions.** Give filenames, package and node names, topic names, message types, configuration keys, expected behavior, and relevant mathematical or data-format contracts.
5. **Dependency checks.** Require Claude Code to inspect existing installations before changing the environment. Require my approval before installing Isaac Sim, CUDA, large dependencies, or multi-GB datasets.
6. **Hardware-free tests.** Require meaningful tests with mocks and synthetic fixtures. Specify the minimum new-test target from the task definition, as well as the relevant full-suite command.
7. **Verification commands.** Include the relevant commands for ROS package builds, sourcing `install/setup.bash`, `pytest`, `ruff`, and `colcon test` or result summaries. Add `ros2 topic list`, `ros2 topic echo`, launch checks, and RViz instructions where appropriate. Do not run irrelevant or unavailable commands merely to fill a checklist.
8. **Honest test reporting.** Require actual passed, failed, skipped, and errored counts from the test output. The previously verified count must be read from `CLAUDE.md`; never hardcode or invent the actual cumulative total. Do not claim optional integration tests passed when they were not run.
9. **Manual checkpoint where applicable.** If a manual check is essential, provide exact commands, explain what I should observe, and wait for my confirmation before committing. When the relevant hardware/software is unavailable, mark the check as pending and distinguish that from unit-test results.
10. **Documentation handoff.** Update the `Current status` section of `CLAUDE.md` with the actual verified state and test results. Update `docs/ARCHITECTURE.md` whenever interfaces, package responsibilities, architecture, or data flow changes. Supply the precise status note to add after the relevant checks have passed.
11. **Git handoff.** Before committing, run the full test suite, review `git status` and the diff, and make sure only intended files are included. Stage the correct files, use the exact Conventional Commit message from the task definition, push to `origin main`, and verify the actual result. Never force-push or discard unrelated changes.
12. **A clear stopping point.** State precisely what is complete, what remains unverified, and what the next task should begin with.

Each prompt must be self-contained and suitable for direct pasting into Claude Code. Do not refer to a prompt above or below it as the source of essential instructions.

### Part C — Risks and Recovery Procedure

Finish with a concise, practical risk list covering:

- GPU/VRAM and disk requirements.
- Isaac Sim/ROS 2 compatibility.
- Dependency conflicts between system ROS Python and `uv`.
- Large datasets and model-weight downloads.
- LeRobot/Open X-Embodiment schema and embodiment alignment.
- Sim-to-real differences in camera calibration, lighting, physics, timing, and control.
- Real-hardware safety, emergency-stop behavior, and unknown map regions.
- What Claude Code should do if tests fail or a stage is interrupted midway.

For recovery, require Claude Code to preserve uncommitted user work, inspect the current repository state, identify the exact failing tests or command, make the smallest focused fix, rerun the relevant tests and full suite, update the documentation, and commit only after verification. If the fix requires an expensive dependency or physical robot, explain the blocker and leave a clear, testable continuation point.

---

## FINAL QUALITY STANDARD

The deliverable is successful only if:

- All 12 tasks form a coherent dependency order.
- Each task has a concrete definition of done.
- Each Claude Code prompt is complete, independently usable, and scoped to one working session.
- Task 1 creates the documentation and interfaces needed by subsequent tasks.
- The design makes practical progress possible with MuJoCo and CPU-only tests before large dependencies are approved.
- Isaac Sim remains the high-fidelity synthetic-data target, subject to verified compatibility.
- The current compatible LeRobot format and model interfaces are checked rather than guessed.
- The prompts distinguish mocked behavior, headless integration tests, GUI checks, HIL tests, and physical deployment.
- Every stage has honest verification, a Git handoff, and a factual `CLAUDE.md` update.
- No test counts, model results, topic behavior, commits, pushes, or deployment results are fabricated.
- The system remains modular, explainable, appropriately scoped, and credible to a robotics/ML hiring manager.

Start now with **Part 1: Architecture Reference + the Complete 12-Task Table + Tasks 1–6 in Full + Risks**. Then stop and ask me to reply `Continue with Tasks 7 to 12`.
