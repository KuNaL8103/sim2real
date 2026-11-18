# Validation matrix

Every row states what the level proves, what it does **not** prove, the status,
and the evidence (command, date, commit). A skipped test is not a pass. Levels
that were not run are marked NOT RUN; unavailable hardware is marked UNAVAILABLE.

Date of this record: 2026-11-18. Base commit before the Task 12 commit:
`27bf958`.

## Summary

| Level | What it covers | Status |
| --- | --- | --- |
| L1 | CPU unit/contract tests, lint, CI parity, clean clone | see rows below |
| L2 | ROS 2 build, colcon tests, `.venv-ros` integration | see rows below |
| L3 | Headless MuJoCo rates + both cameras; orchestrator fake cycle | see rows below |
| L4 | Mock-hardware HIL (controller_manager + safety + fake inference) | see rows below |
| L5 | Real-checkpoint inference smoke | **NOT RUN** |
| L6 | Isaac Sim high-fidelity tier | **UNAVAILABLE** |
| L7 | Physical robot | **NOT RUN** |

## L1 — CPU unit, contract, lint, parity

| Evidence | Result | Proves | Does NOT prove |
| --- | --- | --- | --- |
| `env -u PYTHONPATH -u AMENT_PREFIX_PATH uv run pytest -ra` (2026-11-18, tree at the Task 12 commit) | **666 passed / 0 failed / 33 skipped / 0 errors** (scikit-learn absent) | core algorithms, contracts, safety core, orchestrator state machine, resume/quarantine/gates, 10 new orchestrator tests | nothing about GPU, ros2_control, or a real robot |
| `uv run ruff check .` / `ruff format --check .` | all checks passed; 294 files already formatted | style/import policy | runtime behaviour |
| `./scripts/ci_parity.sh` (2026-11-18) | **passed**; core half **664 passed / 35 skipped** (`SIM2REAL_SKIP_RENDER=1` adds two render skips); ROS half colcon **27 tests / 0 errors / 0 failures / 13 skipped** | the CI jobs' commands reproduce locally, including the ROS half with scipy/open3d shadowed | CI itself |
| `./scripts/ci_parity.sh --clean-clone` | recorded after the task commit; see the status record | a fresh clone of the committed tree passes | uncommitted work |
| `scripts/run_ml_tests.sh` (default `.venv-ml`) | **4 passed / 1 deselected** | ML-marked tests in the existing ML env | the clean rebuild (below) |
| `.venv-ml.new` rebuild checks (2026-11-18) | **PASSED** — see below | a clean ML environment without scipy/mujoco | the swap (PENDING) |

### `.venv-ml.new` rebuild (Task 12)

Built with the pins in `scripts/setup_envs.sh` (no pin changes), CUDA index
`https://download.pytorch.org/whl/cu126`:

| Check | Result |
| --- | --- |
| `uv pip list` for scipy/mujoco/open3d | **all absent** |
| resolved versions | torch **2.11.0+cu126**, torchvision **0.26.0+cu126**, lerobot **0.6.1**, peft **0.21.3**, accelerate **1.16.0**, transformers **5.5.4**, num2words **0.5.14** |
| disk | **6.7 GB** |
| `ML_PYTHON=.venv-ml.new/bin/python scripts/run_ml_tests.sh` | **4 passed / 1 deselected** |
| `scripts/eval_teacher_forced.py --help` | rc 0 |
| `scripts/verify_remote_checkpoint.py --help` | rc 0 |
| policy-server handshake, `outputs/remote/overfit_remote/checkpoint-00001000`, one run, **CPU** (`CUDA_VISIBLE_DEVICES=""`) | **ok in 50.32 s**; checkpoint hash `f47b5e47…`, chunk 50, fps 10, cameras overhead/wrist; parent process imported no torch/scipy/mujoco |

The existing `.venv-ml` was not touched and no swap was performed. The swap is
**PENDING** the operator's confirmation; keep the old environment as
`.venv-ml.old` when it happens.

## L2 — ROS 2 build and integration

| Evidence | Result | Proves | Does NOT prove |
| --- | --- | --- | --- |
| `colcon build --symlink-install --parallel-workers 2` | 5 packages built | the adapters, interfaces, and new launches import/build | runtime correctness |
| `colcon test` + `colcon test-result --verbose` | recorded in the status record | package-level tests | physical behaviour |
| `.venv-ros/bin/python -m pytest -ra` (sourced ROS) | recorded in the status record | full ROS-marked integration suite, including the new pipeline graph tests | policy quality or safety |
| `test_pipeline_mock_integration.py` (5 tests) | **5 passed** | fake observations → `vla_inference_node` → `safety_watchdog_node` → controller topic; safety is the only publisher; NaN chunk and missing map produce nothing; strict profile forwards a clean chunk and rejects a 4 mm obstacle; `sim_mock` differs only in `obstacle_mode` | a trained policy; physical safety |
| HIL guarded test (`test_safety_watchdog.py::TestHilIntegration`) | reused from Task 11; skips with an explicit reason if `controller_manager` is unavailable | the mock stack follows commands, e-stop stops dispatch, reset needs fresh inputs | physical safety |

## L3 — Headless MuJoCo and the orchestrator

`ros2 launch sim2real_perception mapper.launch.py backend:=mujoco use_rviz:=false`
(2026-11-18, ROS domain 90, ~60 s):

| Measurement | Result |
| --- | --- |
| `/joint_states` | **18.04 Hz** (window 323) |
| `/camera/overhead/color/image_raw` | **1.863 Hz** |
| `/camera/wrist/color/image_raw` | **1.845 Hz** |
| `/workspace/pointcloud` | **1.999 Hz** |
| point cloud | **666 points**, `frame_id=world` |
| mapper per-camera counters at n=160 | **overhead 80 / wrist 79** |
| launch errors | none |

These are measured under llvmpipe; they are not performance figures and no
threshold is asserted.

Orchestrator on the real code path:

```bash
uv run python scripts/pipeline_orchestrator.py --config configs/pipeline/fake_e2e.yaml \
    --root /tmp/opencode/orch_e2e --execute --cycles 1
```

Result: `cycle_0001: COMPLETED wall=13.74s`, all seven stages completed, seeds
20950-20959, 10 source episodes, a curated version, a labelled fake checkpoint,
`eval_cycle_0001`, a KDE failure distribution, and a flywheel decision
`REQUEST_TARGETED_GENERATION`.

## L4 — Mock-hardware HIL

`ros2 launch sim2real_control pipeline_mock_hw.launch.py` (2026-11-18, ROS
domain 93):

| Check | Result |
| --- | --- |
| `ros2 topic info /joint_trajectory_controller/joint_trajectory` | **Publisher count: 1** |
| forwarded commands | **5.37 Hz**; diagnostics `accepted_chunks=164`, `last_chunk.command_points=9` |
| `/estop` | `success=True`, `e-stop latched; hold command published` |
| after `/estop` | `is_safe=false`, `estop_latched=true`, `state=3`; no further multi-point dispatch |
| map | valid at the end (`age_s=0.356`, `coverage_ratio=0.031`, `reasons=[]`); three transient `WATCHDOG_MAP_STALE` events during startup under load |
| immediate `/estop_reset` via CLI | **not a reliable boundary test**: `ros2 service call` spent longer than the 2 s hold on discovery, so the reset succeeded. The refusal boundary is covered by the automated in-process test (`test_estop_latches_blocks_and_reset_requires_hold`, which uses already-discovered clients) |

`mock_components/GenericSystem` does not model dynamics, latency, torque,
velocity enforcement, or collisions. This is **not** physical safety.

## L5 — Real-checkpoint inference smoke

**NOT RUN.** The user did not approve a new real-checkpoint run. It could not
drive the arm even if run: `sim_node` has no command subscription, and with the
measured 1.88 s median / 5.90 s max calls on this throttled GPU against a 0.9 s
chunk, the default `stale_chunk_policy=reject` refuses essentially every real
chunk. The Task 10b optional smoke (21 chunks/120 s, median 2.526 s, max
10.250 s, no GPU performance claim) is the only real-checkpoint inference
evidence and is user-reported from that session.

## L6 — Isaac Sim

**UNAVAILABLE.** Published Isaac Sim 6.2.0 minimums are 16 GB VRAM and 32 GB
RAM; this machine has 6 GB VRAM and 9.7 GiB WSL RAM. Requirements-based, not
tested. See `docs/ISAAC_SIM.md`. No install is authorized.

## L7 — Physical robot

**NOT RUN.** No physical robot is connected. No physical-safety claim is made.
`docs/DEPLOYMENT.md` contains a documented-only bring-up checklist.

## Open items

- The pre-registered overfit rule is **UNMET** (arm first-step L1 0.0333 rad vs
  0.010; gripper 0.00688 m vs 0.005; closed loop 0/6 on training seeds). The
  policy is under-trained. Retraining needs a healthy GPU, more data/steps, and
  an explicit approval.
- No ROS closed loop through MuJoCo: `sim_node` has no command subscription.
  Closed-loop evaluation uses the in-process runner. A bridge is an open design
  decision, not implemented.
- Latency runs 3 and 4 are **NOT RUN** under the approved hard stop.
- Remote-run provenance is **user-reported**; no audited `as_used/` copy exists.
- Manual checkpoints for Task 8 smoke, Task 8b, Task 10, Task 11, Task 11b,
  RViz mapping, the flywheel ledger, the `.venv-ml` swap, and CI are **PENDING**
  in `docs/MANUAL_CHECKS.md`.
- MoveIt 2 and OctoMap are not verified.
- Masked obstacle mode is validated only on the kinematic fake backend with a
  synthetic map. The default remains `strict`.
- The `.venv-ml` swap is pending; the old environment still contains the
  unapproved scipy/mujoco.

## Release decision (proposal; not executed)

**A release could honestly claim:**

- the pipeline stages are implemented and tested on the CPU (fake backend), in
  headless MuJoCo, and on mock hardware;
- the orchestrator has deterministic stage ids, resume, quarantine, hard
  bounds, approval gates, and full lineage manifests;
- the safety watchdog is the only publisher of the controller topic and refuses
  raw VLA output; e-stop is latched and reset requires fresh inputs;
- the dataset, filtering, evaluation, failure-analysis, and targeted-generation
  code paths run end to end with fakes and with measured MuJoCo data.

**A release must NOT claim:**

- a working VLA policy or task success;
- physical safety, functional safety, or standards compliance;
- Isaac Sim support on this machine;
- a validated ROS → MuJoCo closed loop, MoveIt, OctoMap, or real Bridge data.

Proposed tag: `v0.12.0-task12-validation`. Annotated tag text would summarise
the above and point to this file. **No tag has been created**; it requires the
operator's explicit yes and a user-confirmed green CI run.
