# Status

This is the tracked status record. The local status notes file is untracked and ignored;
this file is authoritative for clones and CI.

- Last verified test run: passed=667 failed=0 skipped=32 errors=0 | date=2026-11-18 | base_commit=27bf958
- Last verified ROS test run: passed=699 failed=0 skipped=5 errors=0 | date=2026-11-18 | base_commit=27bf958
- Task 1 status: complete

## Release validation status: integrated pipeline, validation matrix, deployment docs (2026-11-18)

The original design brief is kept locally on disk and is not part of the tracked
repository; git history still contains an earlier copy.

**Implemented.** A pure-Python orchestrator (`src/sim2real/orchestrator/`) with
the explicit stages `GENERATE -> FILTER -> BUILD -> TRAIN -> EVALUATE ->
ANALYZE_FAILURES -> TARGETED_GENERATE`; strict `configs/pipeline/{default,fake_e2e}.yaml`;
`scripts/pipeline_orchestrator.py`; deterministic stage ids; resume and
quarantine; atomic cycle manifests and `--lineage`; seed-ledger enforcement;
hard bounds (cycles, episodes/cycle, wall time, writes, GPU time); approval
gates; the existing flywheel ledger reused. Integrated launches
`pipeline_sim.launch.py` and `pipeline_mock_hw.launch.py`, a labelled test-only
`fake_observation_publisher`, and `configs/safety/sim_mock.yaml`. New docs:
`docs/VALIDATION.md`, `docs/MANUAL_CHECKS.md`, `docs/RECOVERY.md`, a rewritten
`README.md` and `docs/DEPLOYMENT.md` (mock bring-up + documented-only physical
procedure), and the architecture diagrams/orchestrator section.

**Decision recorded (user, 2026-11-18).** `configs/safety/default.yaml` stays
`obstacle_mode: strict`. `configs/safety/sim_mock.yaml` sets `masked` and
changes nothing else; it is used only by the simulation/mock launch files and
CPU end-to-end tests and is validated only on the kinematic fake backend with a
synthetic map. Real hardware requires the strict default unless the user changes
it.

**Counts in this session (scikit-learn absent):**

| suite | result |
| --- | --- |
| core ROS-free `env -u PYTHONPATH -u AMENT_PREFIX_PATH uv run pytest -ra` (after the swap) | **667 passed / 0 failed / 32 skipped / 0 errors** |
| `.venv-ros` pytest (built ROS + sourced, after the swap) | **699 passed / 0 failed / 5 skipped / 0 errors** |
| `colcon test` / `colcon test-result --verbose` | **27 tests / 0 errors / 0 failures / 5 skipped** |
| `scripts/run_ml_tests.sh` (existing `.venv-ml`) | **4 passed / 1 deselected** |
| `scripts/ci_parity.sh` | **passed**; core half **664 passed / 35 skipped**; ROS half **27 / 0 / 0 / 13 skipped** (scipy/open3d shadowed) |
| `scripts/ci_parity.sh --clean-clone` (commit `9cce529`) | **663 passed / 36 skipped**, rc 0 |
| `.venv-ml` rebuild and swap | **PASSED / DONE** (see below) |

New tests: 10 core orchestrator tests plus 5 ROS integration tests; the safety watchdog
guarded HIL test is reused. Skips are explicit (no `rclpy`, no `.venv-ml`, no
decoded PNGs, optional sklearn cross-check).

**Clean-clone finding (fixed in `9cce529`).** The first clean-clone run errored
because `BUILD` in `core_validation` mode carried a display string that the
interpreter check treated as `.venv-ml`; when a real `.venv-ml` existed, the
core validation silently ran there. The environment is now resolved explicitly
(`core_validation -> .venv`, `lerobot -> .venv-ml`) with a regression assertion,
and the clean clone passed **663 passed / 36 skipped**.

**`.venv-ml` rebuild and swap (approved, run).** Built `.venv-ml.new` with the
pins in `scripts/setup_envs.sh` and `ML_TORCH_BACKEND=cu126`; `scipy`, `mujoco`,
and `open3d` are **absent**; resolved torch **2.11.0+cu126**, torchvision
**0.26.0+cu126**, lerobot **0.6.1**, peft **0.21.3**, accelerate **1.16.0**
(previously 0.21.2 / 1.15.0), transformers **5.5.4**, num2words **0.5.14**;
disk **6.7 GB**. Against `.venv-ml.new`: `ML_PYTHON=.venv-ml.new/bin/python
scripts/run_ml_tests.sh` **4 passed / 1 deselected**; both `--help` commands
rc 0; one policy-server handshake on
`outputs/remote/overfit_remote/checkpoint-00001000` **succeeded on CPU in
50.32 s** (chunk 50, fps 10, overhead/wrist; parent imported no
torch/scipy/mujoco). **Swapped on 2026-11-18** with exactly two renames
(`.venv-ml -> .venv-ml.old`, `.venv-ml.new -> .venv-ml`); post-swap
`.venv-ml/bin/python` imports torch/lerobot/peft, `uv pip list` shows
scipy/mujoco/open3d absent, `scripts/run_ml_tests.sh` is **4 passed / 1
deselected**, and one CPU handshake succeeded in **52.62 s**. The ROS-free core
suite is **667 passed / 32 skipped** after the swap versus **666 / 33** before, and
`.venv-ros` is **699 / 5** versus **698 / 6**; the +1/-1 in each is
`test_ml_env_independence.py` un-skipping now that `.venv-ml` matches
`setup_envs.sh`. **`.venv-ml.old` is retained until the user says
delete.** `.venv-ml` and `.venv-ml.old` match the git-ignore pattern `.venv*/`.

**Live checks.** Headless MuJoCo + mapper (60 s): `/joint_states` **18.04 Hz**,
overhead **1.863 Hz**, wrist **1.845 Hz**, `/workspace/pointcloud` **1.999 Hz**,
**666 points** frame `world`, mapper counters overhead **80** / wrist **79** at
n=160, no errors. Real fake-backend cycle: **13.74 s**, all seven stages
completed, full lineage. Mock-hardware launch: controller-topic **publisher
count 1**, **5.37 Hz** forwarded, diagnostics **164 accepted / 9-point chunk**,
`/estop` latched and stopped dispatch; three transient `WATCHDOG_MAP_STALE`
events during startup then a valid map (`age 0.356 s`, coverage 0.031). The
CLI reset boundary is not reliable (discovery exceeded the 2 s hold); the
automated in-process test covers refusal and success.

**Documented gap.** `sim_node` has no command subscription, so the live ROS
chain cannot close a loop through MuJoCo; closed-loop evaluation uses the
in-process runner. Masked-mode target allowance comes from
`/task_instruction` parsed against the **nominal** `configs/sim/scene.yaml` cube,
so a randomized offset, a perception error, or a distractor inside that
configured volume is ignored. No bridge was built.

**CI.** Both CI jobs for the push before the orchestrator stage at `27bf958` are
**user-confirmed green** (2026-11-18, Actions tab; supporting evidence only).
CI for the pipeline orchestrator push still needs the user's Actions-tab confirmation.

**Open/pending.** The pre-registered overfit rule is unchanged and **UNMET**
(arm 0.0333 vs 0.010, gripper 0.00688 vs 0.005, closed loop 0/6); the policy is
under-trained. L5 real-checkpoint smoke **NOT RUN**; L6 Isaac **UNAVAILABLE**;
L7 physical **NOT RUN**; latency runs 3/4 **NOT RUN**; remote provenance
**user-reported**; manual checkpoints (training, full-data run, inference, safety, obstacle mode, RViz, flywheel,
`.venv-ml` swap, CI) **PENDING** in `docs/MANUAL_CHECKS.md`; MoveIt/OctoMap
unverified. No policy-quality or physical-safety claim is made. A release tag is
proposed but **not created**; it needs the user's explicit yes.


The latency measurements used the code commit `1d8d882` (scheduler, status
instrumentation, colcon pytest visibility). Both baseline conventions, side by
side:

| convention | core ROS-free |
| --- | --- |
| scikit-learn **absent** (current `.venv`, measured 2026-11-17) | **608 passed / 20 skipped** |
| scikit-learn **present** (9f convention) | not re-run; installing sklearn was not approved. The optional cross-check previously added one pass and removed one skip |

`scripts/ci_parity.sh` reported **606 passed / 22 skipped** (it sets
`SIM2REAL_SKIP_RENDER=1`) and `colcon test-result` now counts the
`sim2real_vla` pytest results: **13 tests / 0 failures / 0 errors / 0
skipped** (6 `sim2real_msgs` + 1 CTest + 6 pytest). `.venv-ros` was **627/6**
and `run_ml_tests.sh` **4 passed / 1 deselected**. `.venv-ml` was not rebuilt;
see the deviations below.

## Obstacle-mode measurement status (2026-11-18)

**Why the 10 safety chunks were rejected.** The safety watchdog H21 check used a
**synthetic** map (600 table points + 450 cube-surface points) and the
kinematic `FakeBackend` expert. All 10 rejections were chunk window 1 (start
step 20) at the same TCP: the **target red cube**, not the table. The 2 cm grid
EDT reported **0.0 m** at command point 0 because the TCP at the cube centre
shared an occupied cell. Exact distances: **0.01861 m** to the cube at point 0,
and **0.00451 m** at command point 6 during lift (finger target **0.00063 m**).
Nearest non-table **0.01861 m**; nearest excluding the target cube **0.03525 m**.
The strict grid was a quantisation artifact, not a table-plane rejection.

**Real mapper cloud (MuJoCo headless + mapper, 60 s).** 667 points, frame
`world`; z min/median/max **0.41994 / 0.419996 / 0.74742 m**; **81.7%** within
5 mm and **84.9%** within 20 mm of the table plane z=0.42; x **0.265-0.645 m**,
y **-0.265-0.275 m**. Cube points: red **33 within 3 cm** (min 0.02069 m), blue
**10** (min 0.02149 m), green **0 within 3 cm** (min 0.03242 m). Mapper
per-camera counters at n=280: **overhead 140 / wrist 139**. The self-filter
leaves some robot points (z up to 0.747), so the cloud is not only table/cubes.
For the expert grasp TCP, exact nearest **0.01974 m** (red cube), table-masked
**0.01974 m**, target-cube-excluded **0.02248 m**, both **0.02248 m**. Grid:
2 cm **0.0**, 1 cm **0.010**, 5 mm **0.01581 m**.

**Options evaluated.** (a) table mask alone: still 10/40 with the grid and
10/40 with exact distance (target cube is nearest); not sufficient. (b)
task-conditioned target allowance: 0/40; the suggested finger-open gate would
not fix the lift point (finger 0.00063), so the implemented allowance uses the
target object's identity/proximity. (c) a 4 mm global threshold would pass the
measured 4.51 mm minimum but is a blind relaxation; not chosen. (d) **chosen:
`obstacle_mode: masked`** = exact surface-point distance + table band
(`0.42 ± 0.01 m`; table top = 0.40 + 0.04/2) + target-object allowance inside
`target_allowance_radius_m=0.05`. The threshold stays **1 cm**. `strict` is
still the default; **your decision is needed** before making masked the default.

**Measured false-positive rates (fake backend, synthetic map):**

| seed set | strict | masked |
| --- | --- | --- |
| 20900-20909 | **30 accepted / 10 rejected** (all `OBSTACLE_TOO_CLOSE`) | **40 / 0** |
| 20910-20919 (held-out) | **30 accepted / 10 rejected** (all `OBSTACLE_TOO_CLOSE`) | **40 / 0** |

**Injection test.** A 2 cm surface block at 2, 5, and 9 mm from the TCP is
rejected in **both** modes. At 12 mm it is outside the 1 cm threshold and
passes, as specified. An unexpected object at the target object's configured
volume would be allowed by design; elsewhere near the TCP it is not.

**Live re-check (no GPU).** Mock control stack + safety node +
`vla_inference_node policy_backend:=fake` + the saved real cloud as a
**test-only fake map** (`sim_node` did **not** run with the broadcaster).
Strict: **108** forwarded commands before the injected block, **0** after,
last reason `OBSTACLE_TOO_CLOSE`, min distance **0.0**. Masked: **112** before,
**0** after, min distance **0.0040 m**. Both modes forwarded real commands and
stopped after the block. The saved cloud is not a substitute for live mapping.

**Counts (2026-11-18, sklearn absent):** core **656 passed / 0 failed / 28
skipped / 0 errors**; `.venv-ros` **683 / 0 / 6 / 0**; colcon **22 tests / 0
errors / 0 failures / 0 skipped**; `run_ml_tests.sh` **4 passed / 1
deselected**; `ci_parity` **654 passed / 30 skipped** (ROS half **22 / 0 / 0 /
8 skipped**); clean clone at `3779336` **653 passed / 31 skipped**.

**Rate-test note:** the sim-node rate floor was lowered from **16 Hz** to
**8 Hz** in safety carry-over A1 for load robustness after **20.00 Hz in 10/10 isolated
runs**. It still catches the documented 1.87-3.28 Hz and 4.98-6.5 Hz
starvation regressions.

Both CI jobs for the push before the obstacle-mode measurement at `3aad245` are **user-confirmed green**
(2026-11-18, Actions tab; supporting evidence only). The obstacle-mode measurement push still
needs the user's Actions-tab confirmation.

## Safety watchdog and mock control status (2026-11-17)

Implemented the ROS-free safety core (`src/sim2real/safety/`), the
`sim2real_control` `safety_watchdog_node`, the latched `/estop` and
`/estop_reset` services, `/safety/status` and additive `/safety/diagnostics`,
strict `configs/safety/default.yaml`, the verified FER jerk/effort/torque-rate
limits in `configs/sim/panda.yaml`, and a **mock-hardware** ros2_control stack
(plain mock URDF, `joint_state_broadcaster`, `joint_trajectory_controller`,
`mock_components/GenericSystem`). `sim2real_control` is no longer a skeleton.

The safety node is the only publisher of
`/joint_trajectory_controller/joint_trajectory`. It validates joint
names/order, finite values, point count, increasing times, future stamp, finger
clipping, stale policy, EMA, continuity, TCP workspace, obstacle distance, and
position/velocity/acceleration/jerk on the **final** smoothed/retimed
trajectory. It publishes one hold from the measured state on `UNSAFE`/`ESTOP`
and never zeros or the last chunk. A rejected chunk does not latch; `ESTOP`
does. A software e-stop is **not** safety-rated, and mock hardware is **not**
physical safety.

Both CI jobs for the push before the safety stage at `4a92cbe` are **user-confirmed green**
(2026-11-17, Actions tab; supporting evidence only). The safety push
(`2519aad`) still needs the user's Actions-tab confirmation.

**Verified counts in this session (sklearn absent in the core `.venv`):**

| suite | result |
| --- | --- |
| core ROS-free `env -u PYTHONPATH -u AMENT_PREFIX_PATH uv run pytest -ra` | **649 passed / 0 failed / 28 skipped / 0 errors** |
| `.venv-ros` pytest (built ROS + sourced) | **676 passed / 0 failed / 6 skipped / 0 errors** |
| `colcon build` / `colcon test` / `colcon test-result --verbose` | **5 packages built; 22 tests / 0 errors / 0 failures / 0 skipped** |
| `scripts/run_ml_tests.sh` | **4 passed / 1 deselected** |
| `scripts/ci_parity.sh` | **647 passed / 30 skipped**, ROS half `22 tests / 0 failures / 8 skipped` (scipy/open3d shadowed) |
| `scripts/ci_parity.sh --clean-clone` at `6fbdc9d` | **646 passed / 31 skipped** |

**Carry-over A1.** The two timing-sensitive tests were made deterministic. The
0.4 s hello/call-timeout test now waits for the explicit READY state before
submitting and uses a client timeout with margin; it passed **10/10** alone. The
sim-node rate test now uses a 5 s window and a 40% floor (8 Hz), which still
catches the documented 1.87-3.28 Hz and 4.98-6.5 Hz starvation regressions; it
passed **10/10** alone. Both are test-only changes.

**Carry-over A2.** `startup_timeout_s` default raised from 120 s to **180 s** in
both `configs/inference/default.yaml` and `InferenceConfig`, with a test that
they agree and that the validator still bounds the value. The latency budget measurement reported
time-to-ready **107.38 s**; this is the only default change.

**Carry-over A3 (live two-camera mapping, 2026-11-17).** With MuJoCo headless
plus `workspace_mapper_node`: `/workspace/pointcloud` measured **2.001 Hz**
(window min 0.488 s, max 0.504 s), `frame_id=world`, **666 points**,
`point_step=16`. The mapper's own new per-camera counters at n=160 were
`{'overhead': 80, 'wrist': 79}`: **both cameras are integrated**. Before the wrist scheduling fix the wrist was starved (overhead-only), so earlier workspace mapping claims
that implied two-camera operation were overhead-only evidence; the corrected
wording is in `docs/ARCHITECTURE.md`. RViz visual confirmation remains
**PENDING** for the user.

**Time convention and stale decision.** The safety command uses
`header.stamp=0` and point `j` at `(j+1)*0.1 s` from now. The Jazzy controller
docs (fetched 2026-11-17) say zero means start now. The raw
`/vla/raw_action` convention is unchanged. Default `stale_chunk_policy=reject`:
all-past or age over `max_chunk_age_s=0.5 s` rejects; surviving points keep
their original offsets. `retime_if_continuous` is opt-in. With the user-reported 1.88 s median / 5.90 s max calls, the default rejects essentially
every real chunk here; the live chain will not move with the real checkpoint.
A 50-point chunk spans 4.9 s (arithmetic estimate, not measured).

**Expert false-positive check (H21, CPU only).** Seeds **20900-20909** (range
20900-20919 reserved in the seed ledger): 10 clean FakeBackend expert episodes,
**40 chunks** through the validator with a synthetic table + cube map:
**10 rejected**, all **`OBSTACLE_TOO_CLOSE`**, exactly one per episode at the
grasp/place approach. Limits were not loosened. This is the measured cost of a
TCP-only 1 cm obstacle gate near the manipulated object; options are documented
in `docs/SAFETY.md`.

**Mock HIL evidence (NOT physical safety).** The mock stack + safety node +
fake map/chunk publisher was run live: a valid chunk was forwarded (3 points,
zero header stamp), the mock joint followed the final target within 0.01 rad,
`/estop` latched and stopped further forwarding, immediate reset was refused,
reset after the 2 s hold succeeded, and a 0.6 s-old chunk was rejected. The
same path is a guarded automated test in `sim2real_control`; it skips with an
explicit reason if `controller_manager` is unavailable.

**Live headless graph check (fake inference backend, no GPU, no `sim_node`).**
The mock stack, `vla_inference_node policy_backend:=fake`, and
`scripts/publish_fake_safety_inputs.py --cameras` were run together on
`ROS_DOMAIN_ID=81`. `ros2 node list` showed `controller_manager`,
`joint_state_broadcaster`, `joint_trajectory_controller`,
`safety_watchdog_node`, `vla_inference_node`, `robot_state_publisher`, and the
mock hardware node. `ros2 topic info
/joint_trajectory_controller/joint_trajectory -v` reported **Publisher count
1, node `safety_watchdog_node`**. One instruction produced a real
`/vla/raw_action` chunk; the safety node forwarded a command with
`header.stamp=0`; `/safety/status` then reported `is_safe=true`, `state=0`,
finite `min_obstacle_distance_m=0.1`, position ratio `0.3388`, and zero
velocity/acceleration/jerk ratios for the hold chunk. A direct client call to
`/estop` latched, an immediate `/estop_reset` was refused ("inputs have not
been fresh for 2.0 s"), and a reset after the hold succeeded. The CLI
`ros2 service call` can spend longer than 2 s on service discovery, so the
manual checkpoint should use an already-discovered client when testing the
refusal boundary.

**Single-owner rule.** `sim.launch.py` refuses `start_ros2_control:=true` and
`control_mock.launch.py` refuses `start_sim:=true`; the pure guard is tested.
`/joint_states` still has exactly one owner. Manual process starts remain a
configuration error.

**Open/pending for the safety stage:** manual checkpoint (PENDING user confirmation; no
pass claimed), CI confirmation by the user in the Actions tab, RViz visual
mapping confirmation, remote provenance (user-reported, no audited `as_used/`),
latency runs 3/4 NOT RUN, `.venv-ml` rebuild (needs approval), MoveIt, Isaac
Sim, and any physical bring-up (release validation, explicit confirmation required).
No policy-quality claim and no physical-safety claim is made.

## VLA inference node status (2026-11-17)

Implemented `vla_inference_node` (`sim2real_vla`) plus the ROS-free core in
`src/sim2real/inference/`: observation assembly/validation, a single-slot
latest-observation-wins async worker, chunk conversion, latency trimming,
strict config, fake policy, and the real `CheckpointPolicy` client path. The
node never imports torch; the model stays behind `scripts/policy_server.py`.
`/vla/status` is an additive `std_msgs/msg/String` contract. The launch default
is `policy_backend:=fake`; nothing in this node connects to a controller.

Both CI jobs for the push before the inference stage at `fae2724` are **user-confirmed green**
(2026-11-17, Actions tab; supporting evidence only). The VLA inference node push
(`1ed799b`) still needs the user's Actions-tab confirmation.

**Measured results in this session (all local unless stated):**

| suite | result |
| --- | --- |
| core ROS-free `env -u PYTHONPATH -u AMENT_PREFIX_PATH uv run pytest -ra` | **601 passed / 0 failed / 19 skipped / 0 errors** |
| `.venv-ros` pytest (sourced ROS + built interfaces) | **619 passed / 0 failed / 6 skipped / 0 errors** |
| `colcon test --packages-select sim2real_vla` | **Ran 6 tests, OK** |
| `colcon test-result --all` | **7 tests / 0 failures / 0 errors / 0 skipped** (the 6 `sim2real_vla` unittest cases are not emitted as xunit; the 7 are `sim2real_msgs`) |
| `scripts/run_ml_tests.sh` | **4 passed / 1 deselected** |
| `scripts/ci_parity.sh` | **599 passed / 21 skipped** |
| `scripts/ci_parity.sh --clean-clone` at `5555ec3` | **598 passed / 22 skipped** (extra `.venv-ml`-absent skips) |

Counts moved from the 9i baseline (534/13 core, 547/5 ROS) by the new tests.
The dataset-frame parity test is **skipped honestly**: the checked-out
`data/lerobot/train_v1` copy has metadata only, no decoded PNG frames. A skipped
test is not a pass.

**Live fake-backend graph check.** `sim_node` (MuJoCo headless, llvmpipe) plus
`vla_inference_node policy_backend:=fake` and a one-off instruction produced
`/vla/raw_action` with `frame_id=""`, joint names
`panda_joint1..7,panda_finger_joint1`, 9 points (latency trimming dropped the
first), and `time_from_start` 0.1-0.9 s relative to the observation stamp.
`/vla/status` reported `READY`, 93 published chunks, zero failures/restarts,
and the fake call latency ~0.0001 s. **Pre-existing issue found:** under
MuJoCo+llvmpipe, `sim_node`'s single-slot render coordinator always starts with
overhead, so `/camera/wrist/color/image_raw` received **0** messages (overhead
~4 Hz; render stats `published=1212 skipped_busy=6600`). The live check fed the
same overhead frame to the wrist topic with an identical stamp to exercise the
node's real sync path. This is a sim_node scheduling defect, not an inference
claim; it is an open item.

**Optional real-checkpoint smoke (throttled, not a performance figure).**
`checkpoint-00001000` exists and passed
`scripts/verify_remote_checkpoint.py --handshake skip` with
`UNVERIFIED: bundle provenance not checked` and `overrides: []` in
`RUN_REPORT.json`; **no `outputs/remote/overfit_remote/as_used/` copy exists**.
The real node ran for 120 s with `latency_compensation:=false` (so raw chunks
would publish despite the call exceeding the 1 s horizon): **21 chunks**, 53
status latency samples, **median 2.526 s**, **max 10.250 s**, 319 submitted /
294 replaced (latest-wins mailbox), stale 0 (compensation deliberately off),
gripper-out-of-range flagged and passed through raw, 0 failures/restarts in the
window. The model was already ready before the probe's first READY sample, so
time-to-ready was not separately instrumented; the 9i measured load on this
machine was **34.71 s**. This says nothing about policy quality. A shutdown
race (abort during `act`) was found here and fixed: the worker no longer counts
a shutdown abort as a restart and the client no longer dereferences a cleared
process handle.

**Unverified here:** remote-run provenance (user-reported, hand-patched
launcher/requirements, no audited `as_used/`); ros2_controllers timing
semantics for the `(k+1)*dt` controller convention (not fetched); MoveIt;
Isaac Sim.

**Open items:** the policy remains under-trained (pre-registered rule unmet);
`.venv-ml` still holds unapproved `scipy 1.18.1` and `mujoco 4.0.1` and needs a
clean rebuild before final validation; `sim_node` wrist-camera starvation needs its own
fix; the manual checkpoint below is **PENDING** user confirmation.

## Wrist camera fix and latency budget status (2026-11-17)

**Cause (file/line evidence).** `sim_node._camera_tick` (`ros2_ws/src/sim2real_sim/sim2real_sim/sim_node.py`, lines 318-333) looped over `contracts.CAMERAS` and called `RenderCoordinator.submit`. `RenderCoordinator` has one slot (`src/sim2real/sim/rendering.py:73-90`); the first submit sets `_busy`, so the second camera was always rejected as `skipped_busy` when a render outlasted the 0.1 s camera tick. The order was fixed (`overhead` first). Before `51fc817` (2026-10-31, `fix: submit camera renders asynchronously`) `_camera_tick` used blocking `render()` for both cameras, so both were served. The rendering-decoupling notes' "images 2.67 Hz" and its rate test covered overhead only; they are not evidence that the wrist topic worked.

**Fix.** `CameraPairScheduler` in `src/sim2real/sim/rendering.py` keeps one pending snapshot, submits at most one camera per slot, retries the same camera while the coordinator is busy, and advances only after every camera in the pair has been submitted. The first camera alternates round-robin between pairs. Both cameras therefore share the snapshot stamp. `RenderCoordinator` now also exposes per-camera `published` and `skipped_busy` counters. State/TF/joint-state publication is unchanged and remains independent of rendering; the reset quiesce clears the pending pair.

**Measured live result (throttled, not a performance figure).** Pre-fix: overhead color 3.25 Hz, wrist color/depth/camera_info **0** in 20 s, stats `published=123 skipped_busy=600`. Post-fix: overhead color/depth/info **1.85 Hz**, wrist color/depth/info **1.80 Hz**, stats `published=134`, `published_by_camera={overhead: 67, wrist: 67}`, `skipped_busy=387`, `failed=0`. A 100-pair stamp probe: **100/100 same-stamp pairs, 0.0 s skew** (sequential arrival gap median ~0.25 s). The rate test passed **3/3 alone at 20.00 Hz** (`published_by_camera={overhead: 4, wrist: 4}`, `skipped_busy=28`).

**Inference on the real two-camera stream (fake policy, no shim).** 100 actions in 60 s; status `READY`; no failures/restarts; `image_pairs_received` and `observations_submitted` increased. `latency_compensation=true` published **9 points at 0.1-0.9 s**; `latency_compensation=false` published **10 points at 0.0-0.9 s**. Explanation from code: `trim_for_latency` keeps `time_from_start >= elapsed`; any positive elapsed time (even the fake call's ~0.0003 s) drops the 0.0 s point. The raw chunk still carries the observation stamp and offsets.

**Latency budget.** Cold start measured directly: **77.33 s**. The approved 16-minute GPU block was hard-stopped after run 2; runs 3 and 4 are **NOT RUN**. Valid run 2 (sim_node, real server, `latency_compensation=false`, `n_action_steps=10`, 90 s after ready): time to ready **107.38 s**; chunks received/published **43/42**; stale **0**; **42** per-call latency samples; median **1.88085 s**, p95 **3.61568 s**, max **5.90400 s**; submitted/replaced **149/105**; median action interval 1.88091 s; published future fraction **0.0** (all selected points already past at publish). Run 1 (server alone, synthetic 10 Hz stream) reached ready at **74.41 s** but the synthetic stream stalled after ready (cameras stale); it produced 5 chunks (median 3.69 s, max 6.21 s) and is **not** a valid 90 s window. The earlier "21 chunks vs 53 latency samples" is reconciled as 1 Hz sampling of `last_latency_s`; the new status tracks per-call samples (42 for 43 chunks in run 2). The median 1.88 s is above the earlier 1.33-1.39 s; only the measured conditions (sim rendering/CPU contention and a longer cold start) are stated, no causal claim. Consequence: a 10-point chunk covers 0.9 s while calls took 1.88-5.90 s, so the safety watchdog must choose stale handling and retiming; no default or convention changed.

**Colcon visibility.** Root cause: `colcon`'s pytest extension only matches when the package declares a pytest test dependency; otherwise it runs `python -m unittest`, which emits no xunit. Added `tests_require`/`extras_require={"test": ["pytest"]}` and a package-local `pytest.ini`; `colcon test-result` now reads `build/sim2real_vla/pytest.xml` and reports **13 tests / 0 failures** (6 msgs + 1 CTest + 6 vla).

**Test counts (measured, sklearn absent in the core `.venv`):** core **608 passed / 0 failed / 20 skipped / 0 errors**; `.venv-ros` **627 / 0 / 6 / 0**; colcon **13 / 0 / 0 / 0**; `run_ml_tests.sh` **4 passed / 1 deselected**; `ci_parity` **606 / 22**; `ci_parity --clean-clone` **605 passed / 23 skipped** (one extra `.venv-ml`-absent skip). New tests cover the pair scheduler, per-camera fairness, reset clearing, per-call latency/future-fraction summaries, time-to-ready, and the node status fields.

**Unverified:** remote provenance (user-reported; no `as_used/`); ros2_controllers `JointTrajectoryController` timing semantics (not fetched); MoveIt; Isaac Sim. **Open:** policy remains under-trained; `.venv-ml` still has unapproved `scipy 1.18.1`/`mujoco 4.0.1` and needs a clean rebuild before final validation; manual checkpoint PENDING.

## Documented exception for starting the VLA inference node

The pre-registered overfit decision rule (`docs/TRAINING.md` section 17,
`configs/evaluation/overfit_rule.yaml`) has **NOT** been met. Facts verified
from the docs and files: remote overfit run (Colab T4, **user-reported
provenance**, hand-patched launcher/requirements, no audited `as_used/` copy
unless `outputs/remote/overfit_remote/as_used/` exists, and it does not):
1,000 steps, first-step arm L1 **0.0333 rad** (bar 0.010: FAIL), gripper
first-step L1 **0.00688 m** (bar 0.005: FAIL), first-10-beats-hold fraction
**1.0** (PASS), gripper clip delta **0.00170 m** (PASS). Closed loop on recorded
training seeds 10000/10004, 3 repeats each: **0/6 success, 6/6 TIMEOUT**.
Earlier results: the real-checkpoint policy "1/30" was a success-definition false positive
(effective **0/30**); the real 1,000-step full-data checkpoint was **0/4** on
training seeds. The instruction effect on the output is below sampling noise
(GPU gate). The local GPU is capped at **210 MHz**; local timings are **"throttled, not
a performance figure"**. Measured model load **34.71 s**; median policy call
about **1.33-1.39 s** on that throttled GPU.

Why proceeding is acceptable: the inference node is policy-agnostic. It talks
to the verified policy-server protocol, is tested with fake models, and makes
**no claim about policy quality**. The deliverable is the pipeline, not a good
policy. Consequences that must be stated: (1) a real-checkpoint run of this node
is expected to fail the task; (2) nothing may be described as "the VLA controls
the robot successfully"; (3) the pre-registered rule is unchanged and still
unmet; (4) retraining remains an open item.

## Remote overfit result status (2026-11-17)

- **POLICY_ERROR cause:** environment/harness timeout, not policy output. All
  six closed-loop records (`eval_closed_loop_seed20000`, `..._seed20004`) have
  `steps = []` and `PolicyError: policy server timed out after 30.0 s`. The
  server sends `hello`, loads the base model, then `ready`; the 30 s client
  timeout covered that load. Measured here: `hello` 0.33 s, **`ready` 34.71 s**;
  the first `act` then returned finite `[50, 8]` in 13.02 s. Fixed with a
  separate `startup_timeout_s` (`--policy-startup-timeout`, default 180 s) and a
  regression test.
- **Pre-registered closed loop on training seeds:** checkpoint-00001000, linear,
  `n_action_steps=10`, seeds 10000/10004 x 3 repeats (distinct recorded sample
  seeds; r0 = episode seed, r1 = 9001xx, r2 = 9002xx): **0/6 success, 6/6
  TIMEOUT, 0/6 POLICY_ERROR**. Every episode left the 0.1 rad tube at policy
  step 1-5. Timings are throttled, not a performance figure. The 20000/20004
  runs are evaluation seeds (generalisation), not this check.
- **Criteria 1-3:** first-step arm L1 **0.0333 rad** (bar 0.010, FAIL),
  first-10 beats hold **1.00** (PASS), gripper **0.00688 m** (bar 0.005, FAIL),
  clip delta 0.00170 m (PASS). Overall **FAIL**.
- **Bias vs spread (first step, same 7 frames x 5 seeds):** remote 200
  0.0634/0.0493 (single L1 0.0748); 500 0.0348/0.0369 (0.0472); 1,000
  0.0223/0.0280 (0.0329). Spread is comparable to the error, but **bias alone
  is 2.2x the 0.010 bar**, so a perfect sampler could not pass. Error is still
  falling; the 100-step block loss flattened (0.232 -> 0.234 over the last 200).
- **Deviations:** the d7a0c90 remote run patched the clock gate, edited
  `requirements.txt` and the MANIFEST hash, used a `uv` Python, and used
  `skip-worktree`; therefore `git_dirty=false` does **not** prove the code
  matched `d7a0c90`. Full table in `docs/REMOTE_TRAINING.md` "Remote overfit result".
- **Local environment deviation:** `.venv-ml` contains `scipy 1.18.1` and
  `mujoco 4.0.1` (+ `mujoco-core`, `mujoco-studio`, `mujoco-studio-wasm`).
  These were **unapproved local installs** made during the remote-overfit
  evaluation; the two-environment design says `.venv-ml` must not contain them.
  Left in place (no uninstall without approval). The ML CLIs were exercised
  with `scipy`/`mujoco`/`open3d` blocked and `PYTHONPATH=src`, which proves
  import independence for those paths only; a clean rebuild from
  `scripts/setup_envs.sh --with-ml` is required before final validation
  (pipeline orchestrator) and needs separate approval. `scripts/setup_envs.sh` was not
  modified.
- **Gate 0 note:** an untracked copy of the overfit config was found at the repo
  root, byte-identical to `configs/training/smolvla_lora_overfit_remote.yaml`
  (sha256 `ab7a7730...f2d27ba`). It was moved, not deleted, to
  `/tmp/smolvla_lora_overfit_remote.yaml.stray`; the tree was then clean with no
  skip-worktree/assume-unchanged flags. Not added to Git; `.gitignore`
  unchanged.
- **Tooling:** `--allow-low-clock` (default hard stop; records
  `overrides: ["low_clock"]` in `RUN_REPORT.json`, memory floor never
  overridable), as-used bundle hashes in `preflight.json`, an
  `<archive>.bundle_info.json` exporter sidecar, local-version pins stripped
  from `requirements.txt`, and verifier `--bundle-info` / `--accept-overrides`.
- **Follow-up (prepared, not run):**
  `configs/training/smolvla_lora_overfit_remote_2000.yaml` changes only
  `max_steps` 1,000 -> 2,000 at the same lr/batch. Estimated 0.023 rad
  single-sample L1 at 2,000 steps (probably still above 0.010) and ~40.3 min on
  the measured 1.21 s/step T4 rate; both are estimates.
- **Inference work was started under the documented exception below.** Criteria fail,
  the loop is 0/6 on training seeds, and the run carried protocol overrides.
  The pre-registered rule and bars were not edited.

## Authoritative bundle after the remote overfit analysis

`outputs/remote_bundle/sim2real_remote_3ddf112_20261117T084815Z.tar.gz`:
**95,934,401 bytes**, sha256
`3103d5a631efc0ef07164b9725baa7315510e52eceebb50f1217413311dcaa44`, manifest
commit `3ddf1123234a98e7a882092e143d6fc1c4121de9`, 14 manifest files, payload
`requirements.txt` sha256 `e544dad5...92361a` (no `+cpu`/`+cu126` pins; the torch
wheel index is a documented line). The sidecar is
`sim2real_remote_3ddf112_20261117T084815Z.tar.gz.bundle_info.json`. It passed
`--check-only` from `repo.bundle` (14 hashes, Python 3.12, GPU printed, no
install/download/training). The superseded d7a0c90 archive is kept for the
provenance record and must not be used for new runs.

## GPU gate and branch R status (2026-11-15)

- Both CI jobs for `d7a0c90` are **user-confirmed green** (2026-11-15,
  Actions tab); the author also observed run `43717234280` with both jobs
  success. Supporting evidence only.
- Repo visibility, unauthenticated:
  `GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=true git ls-remote
  https://github.com/KuNaL8103/sim2real.git HEAD` returned
  `d7a0c90c4077a8f9bf4ffbbe79aa841c1b0f51a5 HEAD`, equal to local HEAD.
- 60 s synthetic matmul gate: SM clock **210/210/210 MHz** (min/median/max),
  power **4.68/7.17/7.25 W**, temperature **54/56/57 C**, throttle **`0x24`**
  (SW Power Cap + SW Thermal Slowdown) on 30/30 samples. **Branch R**: no
  training, no new verifier accept-path test, no teacher-forced or closed-loop
  evaluation.
- The authoritative d7a0c90 bundle passed `--check-only` in a clean temp
  clone: 13 manifest hashes, clean manifest commit, Python 3.12, `lerobot`
  Requires-Python `>=3.12`, GPU/driver printed, no install/download/training.
- `scripts/ci_parity.sh --clean-clone` was added and passed; the flag is
  covered by a test.
- Inference work was subsequently started under the documented exception (the rule is
  still unmet and retraining is open).

## Cleanup-and-hardening status (2026-11-15)

- The user confirmed **both CI jobs green** for the 9f push at `b620c91`
  (supporting evidence only; confirmed in the Actions tab).
- Both CI jobs for the 9f push at `b620c91` are **user-confirmed green**
  (2026-11-15, Actions tab); this is supporting evidence, not a substitute for
  the local runs below.
- The GitHub repository resolves unauthenticated:
  `GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=true git ls-remote
  https://github.com/KuNaL8103/sim2real.git HEAD` returned
  `b620c91f16cef462a7b74ab3e78ae01d8c6b43d8 HEAD` (public).
- Pre-registered decision rule committed first at `17d877b` in
  `docs/TRAINING.md` section 17 and `configs/evaluation/overfit_rule.yaml`.
- Added `scripts/eval_teacher_forced.py` plus CPU tests; `scripts/remote_train.sh`
  now has `--check-only`, `--from-bundle`, interpreter/Requires-Python checks,
  driver/CUDA printing, network preflight, and preflight recording. The exporter
  now adds `repo.bundle`, the payload copy of `remote_train.sh`, Python metadata,
  and the repo.bundle hash.
- The 9f archive extracts to exactly `data/` and `outputs/`; its payload is
  `outputs/remote_bundle_payload/`. The old archive was rejected by the new
  `--check-only` because its manifest lacks `python.major_minor`; the bundle hardening
  rebuild fixes that.
- The verifier on the real `checkpoint-00001000` correctly refused it: the old
  checkpoint metadata has no `adapter_hash` field (created before the field
  existed). Config hash, stats hash and commit matched; no verifier defect was
  found. `--no-run-report` was added as an explicit UNVERIFIED path.
- Wider first-chunk analysis on dataset indices 0/1/2 (frames 0-5/10/20, 5
  seeds) confirms the frame-0 pattern: mean error **0.02683-0.02785 rad** vs
  motion **0.00080-0.00134 rad** (mean ratio **22-37x**); sample-seed std is
  comparable to or larger than the mean error.
- Action parameterization statistics are documented as a hypothesis only: arm
  delta std is **0.13-0.38x** the absolute-action std, but nothing was changed.

## Remote bundle status (2026-11-13)

Part A (first-chunk error analysis) is complete. Part B (bundle, remote script,
local verifier, tests) is implemented but no remote run was executed. Part C
(overfit protocol), Part D (single-instruction count) and Part E (docs/status)
are documented. No training, upload, install, download, sudo/apt, Docker, or
new rollout generation ran in this session.

Part A answer: observation/preprocessor parity is exact (state 0.0, both camera
tensors 0.0, language tokens 0.0 after LeRobot preprocessing) and the runner
executes the postprocessed chunk's first action with max abs difference 0.0.
The gap is the policy's first-chunk action error relative to motion: at frame 0
the recorded motion is 0.00016-0.00228 rad per arm dimension while the
five-sample mean error is 0.004-0.079 rad (9-60x motion); the largest error is
about 80% of the 0.1 rad tube before the first two control steps. "Small"
teacher-forced L1 is small only in absolute radians, not relative to the
per-step motion. Details and full per-dimension tables are in
`docs/EVALUATION.md`.

## Remote bundle

**Superseded (bundle hardening, kept for the provenance record; do not use for new
runs):** `outputs/remote_bundle/sim2real_remote_d7a0c90_20261115T131152Z.tar.gz`,
**95,909,403 bytes (91.47 MiB)**, sha256
`658df5514c7345c97343fc0dc6681f6d3d6adc5367a12a6f4660f4d0770afe4e`, MANIFEST
commit `d7a0c90c4077a8f9bf4ffbbe79aa841c1b0f51a5`. It is the archive the first
remote run used (with the local edits listed in the remote overfit analysis deviations table).
`c3802a5` is also superseded.

The current authoritative archive is the remote overfit analysis bundle listed above
(`sim2real_remote_3ddf112_20261117T084815Z.tar.gz`). The remote checks out the
**MANIFEST commit**, not the repository HEAD, so later docs-only commits are
safe. An archive contains the full 40-episode `train_v1` dataset, the training
configs, pinned requirements, `repo.bundle` for offline cloning, the payload
copy of `remote_train.sh`, Python/`lerobot` Requires-Python metadata, and a
MANIFEST with every file hash. No model weights or HF cache are included.

## Pending checks

- [x] Run the overfit protocol on a remote GPU; `RUN_REPORT.json` and the
  checkpoint folder were returned (2026-11-16, user-reported T4) with protocol
  overrides; see the remote overfit analysis.
- [x] Verify the returned checkpoint locally with
  `scripts/verify_remote_checkpoint.py` (CPU handshake): `checkpoint verified`,
  all 10 metadata/adapter hashes match `RUN_REPORT.json`.
- [x] Run `scripts/eval_teacher_forced.py` with
  `configs/evaluation/overfit_rule.yaml` (overall FAIL: arm 0.0333, gripper
  0.00688 m) and the closed loop on the training seeds (0/6, all TIMEOUT).
- [x] Implement `vla_inference_node`, its ROS-free inference core, launch/config,
  contract/QoS rows, and CPU/ROS tests (inference node).
- [x] Run the core, `.venv-ros`, colcon, ML, CI-parity, clean-clone, and
  fake-backend live graph checks; record the real-checkpoint smoke above.
- [x] Fix the sim_node wrist starvation fairly, prove both cameras and same-stamp
  pairs live, add deterministic scheduler/reset tests, and measure the 20 Hz
  control rate 3/3 (latency measurement).
- [x] Add per-call latency/future-fraction and `time_to_ready_s` status fields;
  measure one valid real-checkpoint run (run 2) and record runs 3/4 as NOT RUN
  under the approved 16-minute hard stop.
- [x] Make colcon count the `sim2real_vla` tests (13 tests / 0 failures).
- [ ] **VLA inference node manual checkpoint (PENDING user confirmation):** inspect both
  cameras, `/vla/status`, `/vla/raw_action`, and server-kill recovery with the
  commands in `docs/VLA_INFERENCE.md`. Do not mark this passed without the user.
- [ ] **Safety watchdog manual checkpoint (PENDING user confirmation):** run the mock
  stack and fake chunk publisher, observe `/safety/status`, trigger `/estop`,
  try `/estop_reset` immediately (expect refusal) and after the hold (expect
  success), stop the map publisher and observe UNSAFE, and (if RViz is
  available) the mock model following commands. Commands are in
  `docs/SAFETY.md` and `docs/STATUS.md`; do not mark this passed without the
  user.
- [ ] **Decision needed:** keep `obstacle_mode: strict` as the default (shipped
  now) or make `masked` the default. Strict rejects 10/40 safety chunks;
  masked rejects 0/40 on both seed sets, but it deliberately ignores the
  commanded target object inside its approach radius. Do not change the default
  without the user's explicit yes.
- [ ] Confirm this push's GitHub Actions tab (supporting evidence only; the user
  confirms the Actions tab). The obstacle-mode measurement CI is not yet confirmed.
- [ ] RViz visual confirmation of `/workspace/pointcloud` with both cameras
  (PENDING; the headless 2.001 Hz / 666-point / both-camera check passed).
- [ ] Run the prepared 2,000-step follow-up
  (`configs/training/smolvla_lora_overfit_remote_2000.yaml`) on a GPU that
  passes the clock/memory gate; not run and not uploaded in the remote overfit analysis.
- [ ] Rebuild `.venv-ml` from `scripts/setup_envs.sh --with-ml` before final
  validation (removes the unapproved scipy/mujoco installs); needs approval.
- [x] Inference interpretation: **started under the documented exception**. Criteria
  1-3 fail, the closed loop is 0/6 on training seeds, and the run carried
  protocol overrides; the rule remains unmet.

## GPU finding

The local RTX 4050 Laptop is degraded. The GPU gate measured
**210/210/210 MHz** SM (min/median/max), **4.68/7.17/7.25 W**, **54/56/57 C**,
throttle **`0x24`** on 30/30 samples. The GPU health gate and remote-bundle checks measured the same 210 MHz
signature. Treat the local GPU as degraded: short inference diagnostics only;
every timing is labelled **"throttled, not a performance figure"**. No local
training.
