# Status

This is the tracked status record. Local `CLAUDE.md` is untracked and ignored;
this file is authoritative for clones and CI.

- Last verified test run: passed=601 failed=0 skipped=19 errors=0 | date=2026-11-17 | base_commit=5555ec3
- Last verified ROS test run: passed=619 failed=0 skipped=6 errors=0 | date=2026-11-17 | base_commit=5555ec3
- Task 1 status: complete

The 2026-11-17 run used the Task 9i tree at `3ddf112`. Both baseline
conventions, side by side:

| convention | core ROS-free |
| --- | --- |
| scikit-learn **absent** (current `.venv`) | **534 passed / 13 skipped** |
| scikit-learn **present** (9f convention) | **535 passed / 12 skipped** (inferred: the optional cross-check adds one pass and removes one skip; not re-run because installing sklearn was not approved) |

`scripts/ci_parity.sh` reported **532 passed / 15 skipped** (it sets
`SIM2REAL_SKIP_RENDER=1`) and `colcon test-result` **7 tests / 0 failures /
0 errors / 0 skipped**; `.venv-ros` was **547/5** and `run_ml_tests.sh` **4
passed / 1 deselected**. `scripts/ci_parity.sh --clean-clone` passed at
`3ddf112` with **531 passed / 16 skipped** (one extra skip: no `.venv-ml`, so
the real-pin test skips). `.venv-ml` was not rebuilt; see the deviations below.

## Task 10 status: ROS 2 VLA inference node (2026-11-17)

Implemented `vla_inference_node` (`sim2real_vla`) plus the ROS-free core in
`src/sim2real/inference/`: observation assembly/validation, a single-slot
latest-observation-wins async worker, chunk conversion, latency trimming,
strict config, fake policy, and the real `CheckpointPolicy` client path. The
node never imports torch; the model stays behind `scripts/policy_server.py`.
`/vla/status` is an additive `std_msgs/msg/String` contract. The launch default
is `policy_backend:=fake`; nothing in this node connects to a controller.

Both CI jobs for the pre-Task-10 push at `fae2724` are **user-confirmed green**
(2026-11-17, Actions tab; supporting evidence only). The Task 10 push
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
node's real sync path. This is a sim_node scheduling defect, not a Task 10
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
clean rebuild before Task 12; `sim_node` wrist-camera starvation needs its own
fix; the manual checkpoint below is **PENDING** user confirmation.

## Documented exception for starting Task 10

The pre-registered overfit decision rule (`docs/TRAINING.md` section 17,
`configs/evaluation/overfit_rule.yaml`) has **NOT** been met. Facts verified
from the docs and files: remote overfit run (Colab T4, **user-reported
provenance**, hand-patched launcher/requirements, no audited `as_used/` copy
unless `outputs/remote/overfit_remote/as_used/` exists, and it does not):
1,000 steps, first-step arm L1 **0.0333 rad** (bar 0.010: FAIL), gripper
first-step L1 **0.00688 m** (bar 0.005: FAIL), first-10-beats-hold fraction
**1.0** (PASS), gripper clip delta **0.00170 m** (PASS). Closed loop on recorded
training seeds 10000/10004, 3 repeats each: **0/6 success, 6/6 TIMEOUT**.
Earlier results: the 9b policy "1/30" was a success-definition false positive
(effective **0/30**); the real 1,000-step full-data checkpoint was **0/4** on
training seeds. The instruction effect on the output is below sampling noise
(9e). The local GPU is capped at **210 MHz**; local timings are **"throttled, not
a performance figure"**. Measured model load **34.71 s**; median policy call
about **1.33-1.39 s** on that throttled GPU.

Why proceeding is acceptable: the inference node is policy-agnostic. It talks
to the verified policy-server protocol, is tested with fake models, and makes
**no claim about policy quality**. The deliverable is the pipeline, not a good
policy. Consequences that must be stated: (1) a real-checkpoint run of this node
is expected to fail the task; (2) nothing may be described as "the VLA controls
the robot successfully"; (3) the pre-registered rule is unchanged and still
unmet; (4) retraining remains an open item.

## Task 9i status: first remote overfit result interpreted (2026-11-17)

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
  matched `d7a0c90`. Full table in `docs/REMOTE_TRAINING.md` "Task 9i".
- **Local environment deviation:** `.venv-ml` contains `scipy 1.18.1` and
  `mujoco 4.0.1` (+ `mujoco-core`, `mujoco-studio`, `mujoco-studio-wasm`).
  These were **unapproved local installs** made during the remote-overfit
  evaluation; the two-environment design says `.venv-ml` must not contain them.
  Left in place (no uninstall without approval). The ML CLIs were exercised
  with `scipy`/`mujoco`/`open3d` blocked and `PYTHONPATH=src`, which proves
  import independence for those paths only; a clean rebuild from
  `scripts/setup_envs.sh --with-ml` is required before final validation
  (Task 12) and needs separate approval. `scripts/setup_envs.sh` was not
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
- **Task 10 remains blocked.** Criteria fail, the loop is 0/6 on training
  seeds, and the run carried protocol overrides. The pre-registered rule and
  bars were not edited.

## Authoritative bundle after Task 9i

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

## Task 9h GPU gate and branch R status (2026-11-15)

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
- Task 10 remains blocked pending a healthy-GPU overfit result.

## Task 9g cleanup-and-hardening status (2026-11-15)

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
  `--check-only` because its manifest lacks `python.major_minor`; the Task 9g
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

## Task 9f-remote status (2026-11-13)

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

**Superseded (Task 9g, kept for the provenance record; do not use for new
runs):** `outputs/remote_bundle/sim2real_remote_d7a0c90_20261115T131152Z.tar.gz`,
**95,909,403 bytes (91.47 MiB)**, sha256
`658df5514c7345c97343fc0dc6681f6d3d6adc5367a12a6f4660f4d0770afe4e`, MANIFEST
commit `d7a0c90c4077a8f9bf4ffbbe79aa841c1b0f51a5`. It is the archive the first
remote run used (with the local edits listed in the Task 9i deviations table).
`c3802a5` is also superseded.

The current authoritative archive is the Task 9i bundle listed above
(`sim2real_remote_3ddf112_20261117T084815Z.tar.gz`). The remote checks out the
**MANIFEST commit**, not the repository HEAD, so later docs-only commits are
safe. An archive contains the full 40-episode `train_v1` dataset, the training
configs, pinned requirements, `repo.bundle` for offline cloning, the payload
copy of `remote_train.sh`, Python/`lerobot` Requires-Python metadata, and a
MANIFEST with every file hash. No model weights or HF cache are included.

## Pending checks

- [x] Run the overfit protocol on a remote GPU; `RUN_REPORT.json` and the
  checkpoint folder were returned (2026-11-16, user-reported T4) with protocol
  overrides; see Task 9i.
- [x] Verify the returned checkpoint locally with
  `scripts/verify_remote_checkpoint.py` (CPU handshake): `checkpoint verified`,
  all 10 metadata/adapter hashes match `RUN_REPORT.json`.
- [x] Run `scripts/eval_teacher_forced.py` with
  `configs/evaluation/overfit_rule.yaml` (overall FAIL: arm 0.0333, gripper
  0.00688 m) and the closed loop on the training seeds (0/6, all TIMEOUT).
- [x] Implement `vla_inference_node`, its ROS-free inference core, launch/config,
  contract/QoS rows, and CPU/ROS tests (Task 10).
- [x] Run the core, `.venv-ros`, colcon, ML, CI-parity, clean-clone, and
  fake-backend live graph checks; record the real-checkpoint smoke above.
- [ ] **Manual checkpoint (PENDING user confirmation):** inspect the fake
  backend, `/vla/status`, and server-kill recovery with the commands in the
  Task 10 handoff. Do not mark this passed without the user.
- [ ] Confirm this push's GitHub Actions tab (supporting evidence only; the user
  confirms the Actions tab).
- [ ] Run the prepared 2,000-step follow-up
  (`configs/training/smolvla_lora_overfit_remote_2000.yaml`) on a GPU that
  passes the clock/memory gate; not run and not uploaded in Task 9i.
- [ ] Rebuild `.venv-ml` from `scripts/setup_envs.sh --with-ml` before final
  validation (removes the unapproved scipy/mujoco installs); needs approval.
- [x] Task 10 interpretation: **blocked**. Criteria 1-3 fail, the closed loop is
  0/6 on training seeds, and the run carried protocol overrides.

## GPU finding

The local RTX 4050 Laptop is degraded. The Task 9h 60 s matmul gate measured
**210/210/210 MHz** SM (min/median/max), **4.68/7.17/7.25 W**, **54/56/57 C**,
throttle **`0x24`** on 30/30 samples. Task 9e/9f measured the same 210 MHz
signature. Treat the local GPU as degraded: short inference diagnostics only;
every timing is labelled **"throttled, not a performance figure"**. No local
training.
