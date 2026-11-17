# VLA inference node (Task 10)

`vla_inference_node` (`sim2real_vla`) turns synchronized camera images, joint
state, and a task instruction into **raw, unsafe** action chunks. It is
policy-agnostic: the real checkpoint is reached only through the verified
policy-server protocol, and the node never imports torch.

**This node makes no claim about policy quality.** It is the transport and
timing pipeline. Task 11 owns safety filtering, clipping, and forwarding to a
controller.

## Node graph

```
/camera/overhead/color/image_raw ─┐
/camera/wrist/color/image_raw ────┼─ ApproximateTimeSynchronizer (slop = max_camera_skew_s)
                                  ▼
/joint_states ───────────────► vla_inference_node ──► /vla/raw_action (JointTrajectory, RAW/UNSAFE)
/task_instruction ───────────►        │              └─► [Task 11 safety node] ─► controller
                                      └──────────────► /vla/status (String, JSON, 1 Hz)
```

- Depth topics are **not used**. Training used RGB only; the model's expected
  cameras are `overhead` and `wrist`.
- `/joint_states` must contain `panda_joint1..panda_joint7` and
  `panda_finger_joint1`. The 8-D state is the seven arm positions (rad) followed
  by the single `panda_finger_joint1` value (m, `[0, 0.04]`). It is **not** the
  sum of the two finger positions; it is the Menagerie `finger_joint1` slide
  coordinate, exactly the backend's `RobotState.gripper_position`.
- Images are `rgb8`, HxWx3, at the publisher's resolution (the simulator uses
  **256x256**). They are converted with plain NumPy; there is no `cv_bridge`.
- The server maps `overhead -> observation.images.camera1` and
  `wrist -> observation.images.camera2` using the checkpoint's
  `camera_rename_map`. The node sends the short keys `overhead` and `wrist`.

## Contract and QoS

| topic | type | direction | rate | QoS |
| --- | --- | --- | --- | --- |
| `/camera/overhead/color/image_raw` | `sensor_msgs/msg/Image` | subscribe | publisher rate | reliable keep-last 5 |
| `/camera/wrist/color/image_raw` | `sensor_msgs/msg/Image` | subscribe | publisher rate | reliable keep-last 5 |
| `/joint_states` | `sensor_msgs/msg/JointState` | subscribe | 20 Hz | reliable keep-last 10 |
| `/task_instruction` | `std_msgs/msg/String` | subscribe | on change | reliable keep-last 1 |
| `/vla/raw_action` | `trajectory_msgs/msg/JointTrajectory` | publish | on completed inference, <= `max_inference_rate_hz` | reliable keep-last 10 |
| `/vla/status` | `std_msgs/msg/String` | publish | 1 Hz | reliable keep-last 10 |

`/vla/status` is an additive contract change recorded in
`src/sim2real/contracts.py` and the `docs/ARCHITECTURE.md` tables.

## Parameters

All parameters are validated; unknown keys, wrong types, and out-of-range
values fail at startup. Defaults live in `configs/inference/default.yaml` and
are shipped with the package.

| parameter | default | unit | meaning |
| --- | --- | --- | --- |
| `policy_backend` | `fake` | — | `fake` (hold-current-state test policy) or `server` |
| `checkpoint_path` | `""` | path | required for `server` |
| `server_python` | `""` | path | empty: `$SIM2REAL_POLICY_PYTHON`, else `<repo>/.venv-ml/bin/python`; resolved and checked |
| `server_script` | `""` | path | empty: `<repo>/scripts/policy_server.py` |
| `startup_timeout_s` | `120` | s | cold start (torch import + model load); measured remote load 34.71 s, user-reported T4 provenance |
| `call_timeout_s` | `30` | s | one `act` call |
| `policy_period_s` | `0.1` | s | dataset/policy period (10 Hz) |
| `n_action_steps` | `10` | steps | horizon published (1.0 s); training chunk is 50 |
| `max_state_age_s` | `0.5` | s | reject older `/joint_states` (local receipt age) |
| `max_camera_skew_s` | `0.05` | s | synchronizer slop and header-stamp skew bound |
| `max_observation_age_s` | `1.0` | s | reject older camera receipts and mailbox entries |
| `max_inference_rate_hz` | `10` | Hz | submission throttle |
| `sample_seed` | `-1` | — | `-1` = unseeded (logged); non-negative = deterministic inference seed |
| `latency_compensation` | `true` | — | drop already-past chunk points; see below |
| `max_restarts` | `3` | count | restart budget after failures |
| `restart_backoff_s` | `0.5` | s | exponential back-off base, capped at 60 s |
| `max_instruction_chars` | `512` | chars | empty or oversized instructions are rejected |
| `max_chunk_length` | `100` | rows | server chunk bound |
| `status_rate_hz` | `1` | Hz | `/vla/status` timer |

`sample_seed` is an inference seed, not a rollout seed. Task 10 draws no new
rollout seeds.

## State machine

`/vla/status` reports one of:

- `STARTING` — worker starting; no readiness claim yet.
- `MODEL_UNAVAILABLE` — the backend could not start (bad interpreter, missing
  checkpoint, metadata mismatch, startup timeout) or a call failed and a restart
  is being attempted. Nothing is published on `/vla/raw_action`.
- `WAITING_FOR_INPUT` — backend ready, but an instruction, joint state, or
  camera pair is missing or stale. Nothing is published.
- `READY` — backend ready and all inputs are fresh; waiting for the next
  observation.
- `RUNNING` — a request is in flight. The ROS executor is not blocked.
- `FAILED` — the restart budget is exhausted. Terminal: nothing is published,
  not even zeros or the last chunk.

The node never claims readiness before the server's `ready` packet. The worker
calls `CheckpointPolicy.start()` and waits for the handshake before reporting
`READY`.

## Asynchronous inference

The measured real call is about **1.33-1.39 s** (median) on the local RTX 4050
at 210 MHz — **throttled, not a performance figure**. `policy_period_s` is
0.1 s. A live node therefore cannot call the policy from a ROS callback:

- one worker thread owns all server I/O;
- the mailbox is a single slot: a newer observation replaces a waiting one;
- there is no unbounded queue, so obsolete observations cannot accumulate;
- request ids increase monotonically;
- each result carries the observation stamp, request id, monotonic send/receive
  times, and measured latency.

`/vla/status` exposes submitted/replaced/completed/failed/timed-out counts,
restart count, last latency, and the last error.

### Simulation time versus live time

Closed-loop **simulation** evaluation is synchronous: `policy.act` blocks the
runner and simulation time does not advance while the GPU runs. A live ROS node
does not get that free time; wall time keeps moving during inference. The
latency-aware path exists because of this difference.

## Action chunk representation and time convention

`/vla/raw_action` is `trajectory_msgs/msg/JointTrajectory`:

- `joint_names = [panda_joint1..panda_joint7, panda_finger_joint1]`;
- point `k` positions are the **absolute** joint targets (rad) and the absolute
  finger target (m, `[0, 0.04]`, dataset definition);
- `header.stamp` = the **observation stamp**, not the publish time;
- `header.frame_id` = empty (joint trajectories are frame-agnostic);
- `time_from_start(k) = k * policy_period_s` with `k = 0` the first action;
- at most `n_action_steps` points are published (default 10).

Why `k*dt`: the evaluation runner's default `linear` execution applies action
`k` at observation time + `k*dt` and linearly interpolates consecutive actions.
A `JointTrajectory` controller conventionally places point `k` at `(k+1)*dt`,
one policy period later. **Task 11 owns which convention the final controller
path uses**; the raw chunk keeps the observation stamp and per-point offsets, so
it can be retimed without losing information. The earlier `(k+1)*dt` table
entry was superseded by decision log entry 66.

### Normalization and de-normalization

The policy server runs LeRobot's official preprocessor and postprocessor and
returns **de-normalized absolute targets**. The node never normalizes or
de-normalizes again. A test proves the real client's values (dtype conversion
to ROS float64 aside) and row order are passed through unchanged.

### Latency compensation

When `latency_compensation` is true, the node drops points whose target time has
already passed:

- `now` is measured without assuming the camera clock equals the wall clock:
  `equivalent_now = observation_stamp + (result_receive_monotonic -
  observation_receive_monotonic)`;
- a point exactly at `now` is kept;
- `header.stamp` is **not** rebased, so a consumer sees the true age;
- if nothing useful remains, the chunk is not published, a stale counter
  increments, and a throttled warning is logged;
- set it to false to publish the full raw chunk and let Task 11 retime it.

With the measured ~1.3 s call and the default 1.0 s horizon, a real-checkpoint
run is expected to drop every selected point. That is an honest latency result,
not a safety or quality claim.

## Failure behaviour

- **Model/server unavailable at start** (bad path, wrong interpreter, metadata
  mismatch, startup timeout): the node stays alive, publishes nothing on
  `/vla/raw_action`, reports `MODEL_UNAVAILABLE` with the reason, retries within
  `max_restarts`, then stays terminal `FAILED`.
- **Inputs missing/stale/invalid** (no instruction, no joint state, camera skew
  too large, observation too old, NaN, wrong encoding/shape/step/length, empty or
  oversized instruction, sequence regression): the observation is rejected, a
  per-reason counter increments, nothing is published.
- **Server crash, EOF, or hang**: the per-call timeout or end-of-stream is
  detected, the subprocess is killed, and the worker restarts under the bounded
  policy. All server I/O stays in the worker thread.
- **Invalid chunk** (NaN/inf, wrong shape, empty, too long): rejected and
  counted. An out-of-range finger target is **passed through as raw** and flagged
  in status; the safety layer owns clipping.
- **Shutdown**: SIGINT/`rclpy.shutdown()` stops the worker, aborts/terminates the
  server, and leaves no child process. A test checks the PID is gone.
- The node never publishes zeros or the last chunk as a fallback.

## What this node does NOT do

- No safety checks, workspace bounds, velocity/acceleration/jerk limits, EMA
  smoothing, or obstacle checks.
- No gripper clipping.
- No controller connection and no publication to
  `/joint_trajectory_controller/joint_trajectory`.
- No claim that "the VLA controls the robot successfully".

The **Task 11 safety watchdog** is the only consumer allowed to forward
`/vla/raw_action` to a controller. It must choose the final time convention and
add the missing jerk/effort limits to `panda.yaml` first.

## Documented exception for starting Task 10

The pre-registered overfit decision rule (`docs/TRAINING.md` section 17,
`configs/evaluation/overfit_rule.yaml`) has **not** been met:

- Remote overfit run (Colab T4, **user-reported provenance**, hand-patched
  launcher/requirements, **no audited `as_used/` copy**): 1,000 steps;
  first-step arm L1 **0.0333 rad** (bar 0.010, FAIL); gripper first-step L1
  **0.00688 m** (bar 0.005, FAIL); first-10-beats-hold **1.0** (PASS); gripper
  clip delta **0.00170 m** (PASS).
- Closed loop on recorded training seeds 10000/10004, 3 repeats each:
  **0/6 success, 6/6 TIMEOUT**.
- Earlier results: the 9b policy "1/30" was a success-definition false positive
  (effective **0/30**); the real 1,000-step full-data checkpoint was **0/4** on
  training seeds.
- The instruction effect on the output is below sampling noise (9e).
- The local GPU is capped at **210 MHz**; local timings are **"throttled, not a
  performance figure"**. Measured model load **34.71 s**; median policy call
  about **1.33-1.39 s** on that throttled GPU.

Proceeding is acceptable because this node is policy-agnostic: it talks to the
verified policy-server protocol, is tested with fake models, and makes **no
claim about policy quality**. The deliverable is the pipeline, not a good
policy. Consequences: (1) a real-checkpoint run of this node is expected to fail
the task; (2) nothing may be described as "the VLA controls the robot
successfully"; (3) the pre-registered rule is unchanged and still unmet;
(4) retraining remains an open item.

## Launching

Fake backend (default, CPU only, publishes a hold-current-state chunk):

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
ros2 launch sim2real_vla vla_inference.launch.py
```

Real checkpoint (explicit; no controller is started):

```bash
ros2 launch sim2real_vla vla_inference.launch.py \
  policy_backend:=server \
  checkpoint_path:=outputs/remote/overfit_remote/checkpoint-00001000
```

The launch file never starts or connects a controller. `/vla/raw_action` must
not be wired to the joint trajectory controller until Task 11 exists.
