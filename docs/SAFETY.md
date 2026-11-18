# Safety

## Status (Task 11 and 11b)

The software safety watchdog and a **mock-hardware** control stack are
implemented. The watchdog is the only component allowed to forward
`/vla/raw_action` to a controller. It validates, smooths, retimes, and publishes
to `/joint_trajectory_controller/joint_trajectory`; raw VLA output is never
connected to a controller directly.

**Mock hardware is not physical safety.** The control stack uses
`mock_components/GenericSystem`, which mirrors position commands to position
states. It does not model dynamics, latency, torque, velocity enforcement, or
collisions. Passing these tests says nothing about a real robot, a real person,
or a real workspace. This repository makes no functional-safety, ISO 10218 /
ISO/TS 15066, or deployment-readiness claim. A software `/estop` is **not** a
safety-rated emergency stop; a real robot needs a hardware e-stop in the
safety-rated circuit.

## Data flow

```
vla_inference_node
  -> /vla/raw_action        JointTrajectory, RAW and UNSAFE
  -> safety_watchdog_node   validate -> EMA -> retime -> final check
  -> /joint_trajectory_controller/joint_trajectory
  -> ros2_control (mock in this repository)
  -> /safety/status         SafetyStatus (10 Hz)
  -> /safety/diagnostics    JSON counters (1 Hz)
```

`/safety/status` is a state signal for supervisors and recorders.
`/safety/diagnostics` is diagnostic only. Neither is a certified safety bus.

## Layer-by-layer design

### 1. Input validation (before any motion)

- The chunk must be a `JointTrajectory` with exactly
  `panda_joint1..7` followed by `panda_finger_joint1`, in that order.
- `positions` must be finite, have 8 values per point, contain at least one
  point, and contain at most `max_chunk_points` (50).
- `time_from_start` must be finite, non-negative, and strictly increasing.
- The header stamp must not be more than `future_stamp_tolerance_s` in the
  future.
- The finger target is clipped into `[0.0, 0.04]` m. The number of clipped raw
  points is counted and reported (`finger_clipped`, `finger_clipped_points`);
  clipping is not a rejection by itself.

### 2. Stale-chunk policy (default `reject`)

Points at or before the current age are dropped. This always drops the raw
`k=0` point once measurable time has passed, matching the Task 10b observation
that a real chunk's first point is usually already gone.

- `reject`: reject the whole chunk with `STALE_CHUNK` when every point is past
  or the age exceeds `max_chunk_age_s`; otherwise forward the surviving points at
  their **original** offsets. This can only delay motion, never speed it up.
- `retime_if_continuous`: drop past points and replay the surviving points from
  now as `(j+1)*command_period_s`. The continuity gate and every limit check
  still apply to the final retimed trajectory.

**Measured consequence (Task 10b, user-reported):** with a 0.9 s chunk and
1.88 s median / 5.90 s max calls on the throttled local GPU, the default
`reject` policy rejects essentially every real chunk: all points are past the
0.5 s `max_chunk_age_s`. The live chain will not move with the real checkpoint
on this machine. A 50-point chunk spans 4.9 s (arithmetic estimate, not
measured). This is a deliberate safe default, not a policy-quality claim.

### 3. EMA smoothing

`y_k = alpha * x_k + (1-alpha) * y_{k-1}`, with `y_{-1}` = the measured 8-D
state. `alpha` defaults to 0.5 and must be in `(0, 1]`; `alpha = 1` is an exact
passthrough. The **final** smoothed trajectory is what is validated, not the
raw input.

### 4. Continuity gate

The smoothed first command point must be within
`max_first_point_distance_rad` (0.5 rad) of the measured arm state. This catches
a plausible-looking chunk that starts somewhere else.

### 5. Workspace check

The end-effector (TCP) position is computed with the single-source
`sim2real.sim.kinematics.tcp_pose` for every command point. It must lie inside
the configured safety-frame bounds. The bounds are derived from
`configs/sim/scene.yaml`:

- table top: `z = 0.40 + 0.04/2 = 0.42 m`;
- table extents: `x in [0.45 - 0.35, 0.45 + 0.35] = [0.10, 0.80]`,
  `y in [-0.35, 0.35]`;
- scene workspace, which is the table/zone region of interest:
  `x in [0.25, 0.65]`, `y in [-0.30, 0.30]`, `z in [0.42, 0.75]`.

The safety config uses those exact scene bounds. Only the TCP point is checked;
there is no whole-arm or self-collision check.

### 6. Joint limits

Every command is checked with finite differences over the **final** command
times, with the measured state prepended at `t = 0` because the controller
interpolates from the current state to the first point. Ratios are reported
against margin-shrunk limits (`1.0` means exactly at the configured limit).

| quantity | source | applied bound |
| --- | --- | --- |
| position | `configs/sim/panda.yaml` `joint_limits` | each joint's range shrunk by `safety_margin_fraction` toward its centre |
| velocity | `velocity_limits` | `0.8 *` FER limit |
| acceleration | `acceleration_limits` | `0.8 *` FER limit |
| jerk | `jerk_limits` | `0.8 *` FER limit |
| effort / torque rate | `effort_limits`, `torque_rate_limits` | recorded for completeness; **not enforced** because this layer has no dynamics model |

`safety_margin_fraction: 0.8` leaves a 20% reserve for controller tracking
error, model mismatch, and unmodeled dynamics. It is an engineering margin, not
a certification or a claim that 20% is sufficient on hardware.

The finger has no FER velocity/acceleration/jerk limit in this project; it is
clipped to `[0, 0.04]` m and otherwise follows the command. A real Franka
gripper uses a separate action server, so the real path must split the 8th
value off instead of sending it to the JointTrajectoryController (documented,
not implemented).

### 7. Obstacle distance

The node rebuilds an `ObstacleQuery` from `/workspace/pointcloud` in a worker
thread. The mapper publishes frame `world`, which is also the configured safety
frame, so no tf2 lookup is needed. A cloud in any other frame is rejected with
`TRANSFORM_UNAVAILABLE`; the smallest correct option here is to reject rather
than add a transform path that has not been exercised.

Unknown is not safe:

- missing map -> `MAP_UNAVAILABLE`;
- `MapStatus` `NO_DATA` -> `MAP_NO_DATA`;
- `STALE` or an age over `max_map_age_s` -> `MAP_STALE`;
- `LOW_COVERAGE` or coverage below the configured proxy -> `MAP_LOW_COVERAGE`;
- an end effector outside the query grid -> `OUTSIDE_MAP` (a clamp is not proof
  of clearance).

`map_required: false` is an explicit commissioning/test mode: an unavailable
map is then reported as unknown (`NaN`, `obstacle_map_valid=false`) rather than
blocking. It is not a safe production setting.

**Limitations.** Only the end-effector point plus a configured safety radius is
checked. There is no swept volume, whole-arm, or self-collision checking.
The table and the manipulated object appear in the map. A naive threshold can
reject a legitimate grasp, so the default threshold is deliberately small
(1 cm) and the effect is measured rather than assumed.

**Task 11 H21 result (synthetic map, kinematic fake backend).** The Task 11
check used a **synthetic** point map (600 table points + 450 cube-surface
points) and the `FakeBackend` expert, not the real mapper cloud. Seeds
20900-20909, 40 chunks: **10 rejected**, all `OBSTACLE_TOO_CLOSE`. Every
rejection was chunk window 1 (start step 20) and the same TCP: the target red
cube. The 2 cm query grid reported **0.0 m** at command point 0 because the TCP
at the cube centre shared an occupied cell. The exact surface distance at that
point was **0.01861 m**; the closest exact point in the chunk was **0.00451 m**
at command point 6 during lift (finger target 0.00063 m). The nearest point was
always the target cube, never the table: nearest non-table `0.01861 m`, nearest
excluding the target cube `0.03525 m` (point 0). The table was not the cause.

**Real mapper cloud (2026-11-18, MuJoCo headless + workspace mapper, 60 s).**
667 points, frame `world`; z min/median/max **0.41994 / 0.419996 / 0.74742 m**;
**81.7%** of points within 5 mm and **84.9%** within 20 mm of the table plane
z=0.42; x extent **0.265-0.645 m**, y extent **-0.265-0.275 m**. Cube points
were present near the red (33 within 3 cm, min 0.02069 m) and blue (10, min
0.02149 m); the green cube had none within 3 cm (min 0.03242 m). Mapper
per-camera counters at n=280 were **overhead 140 / wrist 139**. The mapper
self-filter leaves some robot points (z up to 0.747), so the real cloud is not
just the table and cubes.

For the real cloud and the same expert grasp TCP
`[0.44977, -0.11995, 0.44139]`: exact nearest **0.01974 m** (red cube);
table-masked exact **0.01974 m**; target-cube-excluded exact **0.02248 m**;
both **0.02248 m**. The 2 cm grid reported **0.0 m**, 1 cm grid **0.010 m**
(borderline), 5 mm grid **0.01581 m**. The strict grid is a quantisation
problem, not a table problem.

**Measured strict vs masked (Task 11b, fake backend, synthetic map):**

| seed set | strict accepted/rejected | masked accepted/rejected |
| --- | --- | --- |
| 20900-20909 | **30 / 10** (all `OBSTACLE_TOO_CLOSE`) | **40 / 0** |
| 20910-20919 (held-out) | **30 / 10** (all `OBSTACLE_TOO_CLOSE`) | **40 / 0** |

**Options evaluated.**

- (a) Table-plane masking alone: with the existing grid it still rejects
  10/40; with exact distance it also rejects 10/40 because the target cube is
  the nearest point. Not sufficient alone.
- (b) Task-conditioned target-object allowance: excluding only the commanded
  target object inside its approach volume removes all 10 rejections. The
  originally suggested "finger above a value" gate would **not** fix the lift
  point (finger 0.00063 m at command point 6), so the implemented allowance
  uses the target object's identity and proximity, not finger opening.
- (c) Smaller `min_obstacle_distance_m`: the minimum clearance the expert needs
  is **4.51 mm**; a 4 mm threshold would pass. That is a blind global
  relaxation (an unexpected object at 4.4 mm would pass) and was not chosen.
- (d) **Chosen: `obstacle_mode: masked`** — exact surface-point distance (not
  the 2 cm EDT cell), table points inside `table_mask_band_m` ignored, and the
  known target object ignored only inside `target_allowance_radius_m` of its
  centre. The threshold remains **1 cm**. Default stays `strict`.

**Injected obstacle test:** a 2 cm surface block at 2, 5, and 9 mm from the TCP
is rejected in **both** strict and masked modes. At 12 mm it is outside the
1 cm threshold and passes, as specified. The target-object exception is
deliberate: an unexpected object **at the target object's configured volume**
would be allowed; an object elsewhere near the TCP is not.

**Live re-check (2026-11-18, no GPU).** The mock control stack + safety node +
`vla_inference_node policy_backend:=fake` + the saved real cloud (published as
a test-only fake map; `sim_node` did **not** run with the broadcaster) was run
in both modes. Strict: **108** forwarded commands before the injected block,
**0** after; last reason `OBSTACLE_TOO_CLOSE`, min distance **0.0**. Masked:
**112** before, **0** after; min distance **0.0040 m**. The strict count is
higher than masked in this run only because of scheduling; both forwarded real
commands before injection and stopped after. The saved cloud is not a
substitute for live mapping; it is the measured cloud replayed for the
single-owner-safe check.

The map is a surface TSDF; it does not carve free space, and a moved object
leaves a ghost until reset.

**Recorded decision (2026-11-18, user).** `configs/safety/default.yaml` keeps
`obstacle_mode: strict` as the shipped default. A separate
`configs/safety/sim_mock.yaml` sets `obstacle_mode: masked` and changes
**nothing else**; it is used only by `pipeline_sim.launch.py`,
`pipeline_mock_hw.launch.py`, and the CPU end-to-end tests. Its masked-mode
evidence is the **kinematic FakeBackend with a synthetic map above**, not the
real mapper and not hardware. Real-hardware use requires the strict default
unless the operator changes it explicitly.

**Target-position consequence (documented, not changed).**
`/task_instruction` is parsed against `configs/sim/scene.yaml`, so the allowance
is anchored to the **nominal configured** cube position and half extent, not to
a measured pose. With a randomized cube offset, a perception error, or a
distractor inside that configured volume/radius, the obstacle gate ignores
those points by design. Everything outside the allowance keeps the 1 cm
threshold.

**Rate-test note.** Task 11 A1 lowered the sim-node rate-test floor from
**16 Hz** (0.8 x 20) to **8 Hz** (0.4 x 20) for load robustness after the test
measured **20.00 Hz in 10/10 isolated runs**. The 8 Hz floor still catches the
documented 1.87-3.28 Hz and 4.98-6.5 Hz starvation regressions.

The map is a surface TSDF; it does not carve free space, and a moved object
leaves a ghost until reset.

### 8. Final command time convention

The controller documentation (Jazzy, fetched 2026-11-17,
<https://control.ros.org/jazzy/doc/ros2_controllers/joint_trajectory_controller/doc/trajectory.html>)
states that a zero `header.stamp` means "start now" and that
`time_from_start` is relative to that start. The safety node therefore emits:

- `header.stamp = 0`;
- point `j` at `(j+1) * command_period_s` (0.1 s default);
- absolute positions only (no velocities/accelerations/effort).

This differs from the raw chunk's `k*dt` from the observation stamp and from
the evaluation's linear execution at `observation + k*dt`: after inference
latency the original schedule is partly in the past, so the retained targets are
replayed from "now". The raw `/vla/raw_action` convention is unchanged.

The controller documentation also says a new trajectory **replaces** the old
one (the current implementation forgets the old trajectory), and that
positions-only interpolation is linear with discontinuous velocity at waypoints.
A header stamp in the past and elapsed `time_from_start` points are **not**
specified on the official Jazzy page; those behaviours are **UNVERIFIED**.

### 9. Watchdog and hold

The watchdog runs at 20 Hz. It enters `UNSAFE` when:

- the latest `/joint_states` is older than `max_joint_state_age_s` (0.25 s), or
- the map is required and missing/stale (or older than `max_map_age_s`), or
- the configured watchdog/outage bound is exceeded.

On entering `UNSAFE` or `ESTOP`, it publishes **exactly one** hold command:
the current measured 8-D positions, with one point at `hold_duration_s`. It
never publishes zeros, the last chunk, or an unvalidated fallback. If the
measured state itself is invalid (non-finite, wrong size, outside hardware joint
limits, or finger outside `[0, 0.04]`), it publishes nothing and reports that.
It recovers automatically from `UNSAFE` when fresh inputs return; it does not
resume the old chunk.

A rejected chunk does **not** latch and does not by itself publish a hold. A
previously accepted trajectory may finish, but no new chunk is forwarded until
one passes every check.

### 10. E-stop and recovery

`/estop` (`std_srvs/Trigger`) latches `ESTOP`, publishes one hold, and returns
success. Calling it again is idempotent.

`/estop_reset` (`std_srvs/Trigger`) is the only way out:

1. the latch must be set;
2. joint states and the required map must be fresh and valid;
3. they must have stayed fresh for `estop_reset_hold_s` (2 s default) **since
   the e-stop**;
4. no other unsafe condition may hold.

Otherwise the service returns `success=false` with a clear message. A
successful reset clears the latch and enters `STARTING`/`OK`; it does **not**
command motion. The next motion requires a new validated chunk.

**A software `/estop` is not a safety-rated emergency stop.** It depends on the
ROS process, the OS scheduler, DDS delivery, and the controller. A real robot
requires a hardware e-stop in the safety-rated circuit and a documented
safe-state procedure.

## Limits verification

The model is the Franka Emika Robot (FER)/Panda represented by the pinned
Menagerie `panda.xml`, **not FR3**. Values below are from the official Franka
page, section "Limits for Franka Emika Robot (FER)", fetched **2026-11-17**:
<https://frankarobotics.github.io/docs/robot_specifications.html>

| Quantity | Joint 1..7 | In `configs/sim/panda.yaml` |
| --- | --- | --- |
| `q_min` (rad) | -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973 | **yes** |
| `q_max` (rad) | 2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973 | **yes** |
| `dq_max` (rad/s) | 2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100 | **yes** |
| `ddq_max` (rad/s^2) | 15, 7.5, 10, 12.5, 15, 20, 20 | **yes** |
| `dddq_max` (rad/s^3) | 7500, 3750, 5000, 6250, 7500, 10000, 10000 | **yes** (added Task 11) |
| `tau_max` (Nm) | 87, 87, 87, 87, 12, 12, 12 | **yes** (added Task 11) |
| `dtau_max` (Nm/s) | 1000 | **yes** (added Task 11) |

Position, velocity, and acceleration were already verified in Task 6b. Jerk,
effort, and torque rate were added in Task 11 with the source and date in
`configs/sim/panda.yaml`. The safety layer enforces position/velocity/
acceleration/jerk; effort and torque rate are not computed because there is no
dynamics model in this layer.

## Safe-state behaviour summary

| condition | state | forwarding | hold |
| --- | --- | --- | --- |
| startup, no joint state yet | `STARTING` | blocked | no |
| all inputs fresh, map valid | `OK` | validated chunks only | no |
| stale joint state / stale or missing required map | `UNSAFE` | blocked | one, if measured state valid |
| `/estop` | `ESTOP` (latched) | blocked | one, if measured state valid |
| chunk rejected (limits/workspace/continuity/stale) | state unchanged | that chunk blocked | no |

`SafetyStatus` mapping: `STARTING -> STATE_DEGRADED`, `OK -> STATE_NOMINAL`,
`UNSAFE -> STATE_SAFE_STOP`, `ESTOP -> STATE_ESTOP`. `is_safe` is true only in
`OK` with no latch.

## What is NOT covered

- whole-arm, swept-volume, and self-collision checking;
- singularity detection or avoidance;
- payload, inertia, or external-force estimation;
- torque/effort enforcement (no dynamics model in the safety layer);
- controller-level torque, velocity, or jerk enforcement (the mock controller
  mirrors positions);
- hardware e-stop, safety-rated relays, or a safety PLC;
- real-robot communication loss, watchdog wiring, or fieldbus timing;
- calibration error, encoder faults, or frame/extrinsic errors;
- the table/manipulated-object ambiguity near a grasp (measured above);
- mutex/real-time guarantees under arbitrary OS load.

## Evidence and non-claims

- Core tests are hardware-free and cover the limit thresholds, workspace/FK,
  obstacle/map states, EMA, malformed chunks, finger clipping, continuity,
  stale policies, retiming, the gate/e-stop state machine, config validation,
  contracts/QoS, and the expert false-positive check.
- ROS tests cover the rclpy adapter, status fields, services, single-owner
  launch guard, torch-free import, and a guarded mock-hardware HIL test.
- The HIL test uses `mock_components/GenericSystem`. It is **not** a physical
  safety test and does not establish that a real Franka would stop or follow
  safely.
- No physical robot is connected. No functional-safety certification is
  claimed. A real deployment requires a separate hazard analysis, hardware
  e-stop, validated controller limits, and a documented bring-up.

## Manual checkpoint (for the user; PENDING)

These commands are **for the user**. Do not mark this checkpoint passed without
your confirmation. Use three terminals after building the workspace
(`source /opt/ros/jazzy/setup.bash && cd ros2_ws && colcon build`).

**Terminal A — mock control stack (yours):**

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
ros2 launch sim2real_control control_mock.launch.py
```

Observe: `ros2_control_node` loads `mock_components/GenericSystem`;
`joint_state_broadcaster` and `joint_trajectory_controller` activate; the
safety node starts. Do **not** start `sim_node`; the launch refuses the
combination.

**Terminal B — fake map and chunks (yours):**

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
.venv-ros/bin/python scripts/publish_fake_safety_inputs.py --duration-s 120
```

Observe: a synthetic table map is published at 2 Hz and a raw chunk every 1 s.
This script never publishes `/joint_states`.

**Terminal C — inspection (yours):**

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
ros2 topic hz /joint_trajectory_controller/joint_trajectory
ros2 topic echo /safety/status
ros2 topic echo /safety/diagnostics --once
```

Expected: controller-topic messages arrive about once per second; status is
`is_safe=true`, `state=0` (NOMINAL), `obstacle_map_valid=true`, finite
observation/action ages, and the ratios are at or below 1.0. The mock joint
follows the last target within 0.01 rad.

**E-stop and reset (Terminal C, yours):**

```bash
ros2 service call /estop std_srvs/srv/Trigger {}
# immediately:
ros2 service call /estop_reset std_srvs/srv/Trigger {}
# expected: success=false, "inputs fresh for ... < 2.0 s"
# after at least 2 s of fresh inputs:
ros2 service call /estop_reset std_srvs/srv/Trigger {}
# expected: success=true; no motion is commanded by the reset
```

Expected after `/estop`: `estop_latched=true`, `state=3` (ESTOP), one hold
command from the measured state, and no new multi-point chunks until reset.

**UNSAFE check (yours):**

Stop Terminal B with Ctrl-C. Within `max_joint_state_age_s`/`max_map_age_s`,
`/safety/status` becomes `state=2` (SAFE_STOP), `is_safe=false`, and the safety
node publishes one hold. Restart Terminal B; the status returns to NOMINAL
without resuming the old chunk.

**RViz (optional, yours):**

```bash
rviz2
```

Add a RobotModel display (the mock URDF is published by
`robot_state_publisher`) and a PointCloud2 display on `/workspace/pointcloud`
(frame `world`). The mock model should follow the commands. RViz mapping
confirmation remains PENDING until you report it.

A software `/estop` is not a safety-rated emergency stop. This checkpoint is
mock-hardware evidence only.
