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
| `startup_timeout_s` | `180` | s | cold start (torch import + model load); measured local time-to-ready 107.38 s (Task 10b), remote load 34.71 s (user-reported T4 provenance) |
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

## Live two-camera validation (Task 10b)

The pre-fix live check under MuJoCo+llvmpipe measured overhead color at
**3.25 Hz** and wrist color, wrist depth and wrist camera_info at **0** messages
in 20 s; sim_node shutdown reported `published=123 skipped_busy=600`. The cause
was scheduling, not the simulator: `RenderCoordinator` has one slot and the
async `_camera_tick` always submitted `overhead` first, so `wrist` was always
`skipped_busy` when a render outlasted the camera tick. Before commit `51fc817`
(2026-10-31) the callback rendered both cameras synchronously, so both were
served; the async change introduced the starvation. The Task 2b docs' "images
2.67 Hz" is not evidence that the wrist ROS topic worked: the rate test and
the Task 2b test subscribed to overhead only.

After the pair-scheduler fix, a 20 s live probe measured:

| topic | rate |
| --- | --- |
| overhead color | 1.85 Hz |
| overhead depth | 1.85 Hz |
| overhead camera_info | 1.85 Hz |
| wrist color | 1.80 Hz |
| wrist depth | 1.80 Hz |
| wrist camera_info | 1.80 Hz |

Shutdown stats: `published=134` with `published_by_camera={overhead: 67,
wrist: 67}`, `skipped_busy=387`, `failed=0`. A 100-pair stamp probe measured
**0.0 s** overhead/wrist stamp skew for every pair (`stamp_intersection=100`);
the two cameras are rendered from the same snapshot, sequentially in arrival
time (median arrival gap about 0.25 s). The node's default `max_camera_skew_s`
is 0.05 s and the synchronizer slop is 0.05 s, so same-stamp pairs are accepted.

With `policy_backend:=fake`, no shim and the real two-camera stream:
**100 action messages in 60 s**, 59 status samples, node state `READY`, no
failures or restarts, and `image_pairs_received`/`observations_submitted`
increased steadily. Every action had **9 points** with `time_from_start`
**0.1-0.9 s** when `latency_compensation=true`; with
`latency_compensation=false` the same node published **10 points** at
**0.0-0.9 s**. The 0.0 s point is not a formatting difference: the trim rule
keeps `time_from_start >= elapsed`, and any positive elapsed time (even the
fake policy's ~0.0003 s call) is greater than 0.0 s, so the first point is
dropped. The raw chunk still carries the observation stamp and all per-point
offsets; Task 11 chooses the final convention.

## Live latency budget (Task 10b; throttled, not a performance figure)

The checkpoint `outputs/remote/overfit_remote/checkpoint-00001000` exists and
passed `scripts/verify_remote_checkpoint.py --handshake skip` with
`UNVERIFIED: bundle provenance not checked` and `overrides: []`. There is no
audited `outputs/remote/overfit_remote/as_used/` copy; the remote run remains
**user-reported provenance** with hand-patched launcher/requirements.

A direct cold-start measurement (no ROS, `.venv-ml` server) took **77.33 s**
to `ready`; the valid sim-backed run below took **107.38 s** to `ready`. All
local timings are on the **210 MHz / SW-power-capped** GPU and are labelled
**"throttled, not a performance figure"**.

The approved 16-minute block was cut short after run 2; runs 3 and 4 are **NOT
RUN** under that hard stop. The valid run used sim_node + real server,
`latency_compensation=false`, `n_action_steps=10`, 90 s after ready:

| metric | run 2 (valid) |
| --- | --- |
| time to ready | **107.38 s** |
| chunks received / published | **43 / 42** |
| chunks dropped stale | **0** |
| call latency samples | **42** |
| median / p95 / max call latency | **1.88085 s / 3.61568 s / 5.90400 s** |
| submitted / replaced | **149 / 105** |
| median action interval | 1.88091 s |
| published points still in the future | **0.0** (median/min/max; all selected points were already past at publish) |
| point count | 10 per chunk, 0.0-0.9 s |

Run 1 (server alone with a synthetic 10 Hz camera publisher) reached ready at
**74.41 s** but its camera stream stalled after ready (`OBSERVATION_TOO_OLD`);
it produced only 5 chunks and is **not** a valid 90 s window. Its measured
latency was median 3.69 s, max 6.21 s over 5 calls. That harness stall is a
test-publisher limitation, not a policy error. Runs 3 and 4 were not started.

The earlier Task 10 report's "21 chunks vs 53 latency samples" came from
sampling the 1 Hz `last_latency_s` field, not from per-call samples; the new
status carries a bounded per-call `latency` summary, and run 2 shows 42 samples
for 43 received chunks. The measured median is **1.88 s** here, versus the
earlier **1.33-1.39 s** on the same throttled machine; the difference is not
attributed beyond what was measured (sim rendering/CPU contention and a longer
cold start were present in this run). The measured consequence is plain: a
10-point chunk covers only **0.9 s**, while calls took **1.88-5.90 s**, so live
chunks are almost entirely past by the time they arrive unless they are
retimed/extended. Task 11 must choose the stale-chunk and retiming policy; this
node does not change its convention or defaults.

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

## Manual checkpoint (PENDING user confirmation)

These are for the user; the automated session does not mark them passed. Use
separate terminals, each with:

```bash
cd ~/Python/sim2real
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
```

**1) Simulator (terminal A).** Headless MuJoCo, images enabled:

```bash
export MUJOCO_GL=egl
ros2 launch sim2real_sim sim.launch.py backend:=mujoco headless:=true
```

Expect `/joint_states`, overhead and wrist color/depth/camera_info. Under
MuJoCo+llvmpipe the measured post-fix rates were **about 1.8-1.9 Hz per
camera** (each color, depth and camera_info), with both cameras in a pair
sharing one stamp. If wrist stays at zero, check that the pair scheduler fix is
built and look for `published_by_camera` in the shutdown stats.

**2) Inference node + rate (terminal B).** Fake backend; no model:

```bash
ros2 launch sim2real_vla vla_inference.launch.py policy_backend:=fake
ros2 topic hz /vla/raw_action
```

Expect a chunk at up to 10 Hz once an instruction exists; the fake policy holds
the current state.

**3) Status (terminal C).**

```bash
ros2 topic echo /vla/status --once --full-length
```

Expect `state` to move `STARTING -> WAITING_FOR_INPUT/READY`, `worker.state`
`READY`, `depth_used=false`, and counters. Publish an instruction once:

```bash
ros2 topic pub --once --qos-reliability reliable --qos-durability volatile \
  -w 1 /task_instruction std_msgs/msg/String \
  "{data: \"pick up the red cube and place it in the left zone\"}"
```

Expect `chunks_published` to increase and `/vla/raw_action` to show 9-10 points.

**4) Server-kill recovery (real backend, terminal B).** Start with
`policy_backend:=server` and a valid `checkpoint_path`, wait for
`worker.state=READY`, then:

```bash
pkill -f policy_server.py
ros2 topic echo /vla/status --full-length
```

Expect `state=MODEL_UNAVAILABLE` with the crash reason, **no**
`/vla/raw_action` publication while unavailable, then either recovery to
`READY` within the bounded restarts or terminal `FAILED`. Check
`ps aux | grep policy_server` afterwards: no child should remain.
