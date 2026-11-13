# Closed-loop evaluation (Task 9)

Status: **implemented**. Task 9 evaluates policies in closed loop in simulation,
records rich per-episode results, analyses failures over randomization features,
requests failure-targeted rollouts, and enforces flywheel guards with a
persistent ledger. **Task 9 never starts training.**

Everything here is simulation-only. Simulator success is not a safety case, and
no physical robot is connected.

## Timing contract

| Quantity | Value | Why |
| --- | --- | --- |
| dataset / policy period | 0.1 s (10 Hz) | Training data was written at decimation 2 of the 20 Hz loop. |
| backend control rate | 20 Hz | Each policy action covers **2 control steps**. |
| `eval.execution` | **`linear`** (default) | Linear joint-space interpolation between consecutive 10 Hz actions; `hold` is the legacy zero-order hold. |
| `eval.n_action_steps` | **10** (1.0 s) | Not the training value 50 (5.0 s, longer than an episode). |
| success definition | Task 4 expert `_success_check` | Same thresholds, so paired comparisons are meaningful. |

A chunk is re-planned every `n_action_steps` policy steps (20 control steps at
the defaults). In `linear` mode, control step `2i` receives action `a_i` and
step `2i+1` receives the midpoint `(a_i + a_{i+1})/2`; the last action in a
chunk is held constant for both steps. In `hold` mode both steps receive `a_i`.
The earlier 9b/9c numbers were measured in **hold mode** and are legacy.

Oracle on the 32 train-split seeds (recorded 10 Hz actions, n_action_steps 10):

| execution | success | Wilson 95% |
| --- | --- | --- |
| `hold` (legacy) | 24/32 = 0.750 | [0.579, 0.867] |
| `linear` (default) | **31/32 = 0.969** | [0.843, 0.994] |
| 20 Hz no-decimation ceiling | **32/32 = 1.000** | [0.893, 1.000] |

`linear` is the default because it recovers 7 of the 8 hold failures. The ROS 2
Jazzy `joint_trajectory_controller` documentation states that the `spline`
interpolator uses **linear** interpolation when only positions are specified
(<https://control.ros.org/jazzy/doc/ros2_controllers/joint_trajectory_controller/doc/trajectory.html>),
which matches the interpolation style. However, the `/vla/raw_action` contract
uses `time_from_start=(k+1)*dt`, i.e. point `k` is at `(k+1)*0.1 s`, while the
evaluation applies `a_k` at control step `2k` (`k*0.1 s`). A live trajectory
therefore lags the evaluation convention by one policy period unless the
publisher uses `k*dt` or the consumer accounts for the offset. This is a Task 10
contract decision, not a measurement claim.

`n_action_steps` sweep (fake backend, `noisy_expert`, 30 seeds from 20000, same
noise seed, expert 30/30 on every setting):

| `n_action_steps` | policy success | Wilson 95% | policy-only failures |
| --- | --- | --- | --- |
| 5 | 13/30 | [0.274, 0.608] | 17 |
| 10 (default) | 10/30 | [0.192, 0.512] | 20 |
| 25 | 8/30 | [0.142, 0.445] | 22 |

The default 10 keeps the 1.0 s replan interval documented in the contract. The
proxy policy does slightly better at 5, but this is one noisy test policy on the
kinematic fake backend, not a policy-quality conclusion. No real-model sweeps
were run.

## Policy interface

```python
Policy.reset(task_instruction: str, seed: int) -> None
Policy.act(observation: PolicyObservation) -> ActionChunk
ActionChunk.actions: (K, 8) absolute joint targets (7 arm joints + gripper)
ActionChunk.step_period_s: float
Policy.needs_images: bool
Policy.info: dict  # name, checkpoint hash, ...
```

Implementations:

- `ScriptedExpertPolicy` — runs the Task 4 expert on its **own** backend at reset
  and replays the recorded executed actions. It is the reference upper bound.
- `NoisyExpertPolicy` — expert actions plus deterministic Gaussian noise and
  step dropout; the scale can be coupled to the sampled initial object x.
- `RandomPolicy`, `ZeroPolicy` — test/diagnostic policies.
- `CheckpointPolicy` — guarded client of `scripts/policy_server.py`.

Policies never step the backend. A policy exception, NaN, or wrong shape becomes
`POLICY_ERROR`; the batch continues.

## Terminations

`SUCCESS`, `TIMEOUT`, `COLLISION`, `LIMIT_VIOLATION`, `POLICY_ERROR`,
`BACKEND_ERROR`, `PHYSICS_INSTABILITY`.

- `collision_terminate` defaults to `false` (measure, do not stop). Remember
  `collision_count` is **not** a safety signal.
- `terminate_on_limit` defaults to `true`.
- The physics guard mirrors the Task 4 guard (non-finite state, huge qacc,
  object outside the workspace margin) without modifying expert code.

## Records

`EvalEpisodeRecord` (`schema_version: 1`) is JSON-serialisable and rejects
non-finite values, wrong shapes, unknown termination reasons, and a `success`
flag inconsistent with the reason.

Fields: episode/run id, seed, episode index, randomization profile and the full
flattened sample, initial object pose, friction/mass/lighting/camera values,
task instruction, `success`, `task_completion`, `collision_count`, termination
reason, Task 5 metrics (dimensionless squared jerk, SPARC, path length, limit
margins; `degenerate` marker instead of raising), failure metadata (terminal
state and last policy info), policy id and checkpoint hash, config hash, git
commit and dirty flag, `n_action_steps`, renderer, and every control step.

`task_completion = 1 - clip(final_xy_distance / start_xy_distance, 0, 1)` to the
zone centre; a zero start distance is handled explicitly.

`EvalRun` stores the seed list derived with `SeedSequence([eval_seed, index,
"EVAL"])` and aggregates: success rate with a Wilson 95% interval, counts per
termination reason, and mean completion.

Replay: a record plus a deterministic policy reproduces the same outcome and
trajectory metrics. This is tested by running the same seeds twice and comparing
the full record dictionaries.

## Paired reference

The MuJoCo expert fails a large fraction of clean episodes, so a policy trained
on filtered demonstrations is not expected to beat it. `run_paired_eval`
evaluates the **same seeds** with `ScriptedExpertPolicy` and the candidate and
reports:

- expert success (count and Wilson interval),
- policy success,
- policy success on seeds where the expert succeeded,
- **policy-only failures** (expert succeeded, policy failed).

Measured MuJoCo baseline (physics only, **no rendering**, 30 seeds
20000-20029, expert 24/30 = 0.80, Wilson [0.627, 0.905]):

| candidate | success | Wilson 95% | success on expert-success seeds | policy-only failures |
| --- | --- | --- | --- | --- |
| `ZeroPolicy` | 0/30 | [0.000, 0.138] | 0/24 | 24 |
| `NoisyExpertPolicy` (sigma 0.01, dropout 0.05) | 2/30 | [0.018, 0.213] | 2/24 = 0.083 [0.023, 0.258] | 22 |

The fake-backend development run (60 seeds 20000-20059, `noisy_expert`) measured
policy **18/60** (0.30, Wilson [0.199, 0.425]), expert **60/60**
([0.940, 1.000]), policy-on-expert-success **18/60**, and **42** policy-only
failures; mean completion 0.305. The fake expert is much easier than MuJoCo, so
these are not MuJoCo numbers.

## Failure analysis

### Feature space

`object_x, object_y, object_yaw, friction_sliding, object_mass, light_intensity,
color_temperature, camera_jitter_norm`, normalised to [0,1] using the profile
bounds (`sim2real/randomization/features.py`).

`to_feature_vector(sample)` and `from_feature_vector(features, base_seed,
episode_index)` round-trip. `from_feature_vector` rebuilds a **complete** sample:
groups outside the feature space are drawn normally and deterministically.
Out-of-bounds inputs are clipped and reported; if the target cube lands within
`min_separation_m` of another cube, the other cube is re-placed deterministically
and the repair is reported.

### Failure definition

- `policy_only` (default): seeds where the **expert succeeded and the policy
  failed**, from a paired evaluation.
- `any_failure`: every failed episode.

`policy_only` is the default because `any_failure` would also target regions
where the scripted expert fails; targeted generation would produce
demonstrations the trajectory filter rejects (wasted compute). `policy_only`
without paired data is refused with a clear message.

### Fitting chain and minimum counts

| valid failures | model | notes |
| --- | --- | --- |
| >= `min_failures_gmm` (30) | Gaussian mixture | BIC over 1..`max_components`, `reg_covar` floor, fixed `random_state`; scikit-learn when importable, else a NumPy full-covariance EM with the same selection rule |
| >= `min_failures_kde` (10) | Gaussian KDE | Silverman bandwidth with a configured floor |
| 1-9 | bounded neighbour perturbation | Gaussian noise at `neighbor_sigma_fraction` of the feature range around each failure |
| 0 | `NO_TARGETING` | no generation; the flywheel stops |

NaN, wrong dimensions, and empty inputs are rejected and recorded. Duplicate
failures that collapse the covariance degrade to the next fallback and record
why.

### Sampler

`TargetedSampler` draws `targeted_fraction` (0.7) from the model and the rest
uniformly over [0,1]. Draws are rejection-sampled into bounds with a max-tries
limit; failures are clipped and counted. The same seed gives the same samples.

### Artifact

`data/failure_distributions/<id>.json` stores the type, parameters, failure
count, feature names and bounds, seed, source eval run id, failure definition,
config hash, fallback reason, and a **content hash**. The collector refuses a
missing, malformed, or hash-mismatched file.

## Targeted generation

`GenerateRollouts.srv` with `sampling_mode: 1` (`SAMPLING_FAILURE_TARGETED`)
loads the artifact, samples feature vectors, maps each to a full valid
randomization sample, runs the expert with that sample applied directly, and
stages episodes atomically. Every seed is derived inside **30000-30999**, the
distribution id is stored in each episode's metadata, and
`result_metadata_json` lists ids, seeds, counts, sampler report, and the real
expert yield. `num_rollouts` above the cap is rejected.

The MuJoCo expert still fails roughly 40% of clean episodes, so roughly 60% of
targeted episodes are expected to be usable and fewer pass the filter. Report
the measured yield; do not promise it.

Measured development run (fake backend, 10 rollouts, seed 30000, GMM fitted on
42 policy-only failures): **10/10** expert success, 7 targeted / 3 uniform
draws, 0 clip fallbacks, 8 rejection tries, every seed in 30000-30999. The fake
expert is easier than MuJoCo, so this 100% yield is **not** a MuJoCo estimate.

## Flywheel guards and ledger

`FlywheelController` appends one JSON object per cycle to
`data/flywheel/ledger.jsonl`. Fields: `cycle_id`, `parent_cycle_id`,
`config_hash`, `source_dataset_versions`, `episode_ids`, `episode_hash`,
`checkpoint_hash`, `eval_run_id`, `failure_distribution_id`,
`rollouts_requested`, `rollouts_generated`, `decision`, `reason`,
`success_rate`, `expert_success_rate`, `n_failures`, `policy_only_failures`,
`created_at`.

Guards:

| guard | default | effect |
| --- | --- | --- |
| `max_cycles` | 10 | hard cycle cap |
| `max_total_rollouts` | 200 | hard cumulative cap |
| `max_rollouts_per_request` | 32 | per-request cap, also enforced by the collector |
| `min_failures_to_trigger` | 3 | do not target noise |
| `min_success_improvement` | 0.05 | improvement needed to reset the streak |
| `max_consecutive_no_improvement` | 3 | stop a plateau |
| `cooldown_s` | 0.0 | minimum wall time between cycles |
| `STOP` file | `data/flywheel/STOP` | kill switch |
| idempotency | — | the same `eval_run_id` is refused twice |

Decisions: `REQUEST_TARGETED_GENERATION`, `REQUEST_RETRAIN` (reserved; Task 9
never trains), `STOP`. Corrupt or partial ledger lines are reported and skipped.

## Policy server

- Runs `scripts/policy_server.py` in `.venv-ml` with `PYTHONPATH=src`.
- Protocol: 4-byte big-endian length prefix + UTF-8 JSON header; raw array blobs
  follow in the same packet. No pickle. File descriptor 1 is redirected to
  stderr and a duplicated descriptor carries only protocol frames, so library
  progress output cannot corrupt the stream.
- Handshake: protocol version, checkpoint metadata SHA-256, chunk size, fps,
  expected cameras, state/action dimensions. A mismatch is refused before any
  model is loaded. After the model is loaded the server sends an explicit
  `ready` frame; the client does not send observations before it arrives.
- A live server is reused across episodes (a `reset` frame calls the policy's
  reset); it is not reloaded per episode.
- Deterministic sampling: `--sample-seed` (or a `sample_seed` field in the
  `reset` frame) seeds the flow-matching noise as `torch.manual_seed(seed +
  call_index)`. The client sends the episode seed by default. Same seed + same
  input reproduces the chunk exactly; a different seed differs (measured max
  abs 0.263).
- Observation: 8-D state, the named uint8 images, and the task string. Reply:
  action chunk shape/dtype plus the policy step period.
- Timeouts, malformed frames, and server death are detected and reported as
  `POLICY_ERROR`.
- The real LeRobot preprocessor/postprocessor run inside the server so nothing
  is normalised twice. Fake-model mode is tested without torch.

## Seeds

Evaluation uses **20000-20999 only** (never training data). Targeted generation
uses **30000-30999** (training data). Episode seeds are derived with
`SeedSequence` and stored in the run manifest; the base evaluation seed is also
recorded.

## Renderer

The training dataset was rendered with **llvmpipe**. Evaluation records
`renderer` in every episode and in the run manifest: `fake-analytic` for the
fake backend, `llvmpipe` for MuJoCo by default, and `d3d12...-OPT_IN` if the
opt-in D3D12 environment is present. Opt-in settings are never used silently.

## Task 9b real-checkpoint verification (2026-11-11)

Test subject: `outputs/task8/train_v1_lora/checkpoint-00001000` (LoRA, 1,000
steps, constant LR 1e-4, `git_commit=bf74c84`, `git_dirty=false`, chunk 50,
`n_action_steps=50`, fps 10, metadata SHA-256
`b87f5094d414d97590166ba8691989dd930b21b4c166ddf5de4a4f28234422e3`).

### Policy server on the real checkpoint

Handshake: protocol 1, checkpoint hash `b87f5094…22e3`, `chunk_size=50`,
`fps=10.0`, state dim 8, action dim 8, cameras `overhead`/`wrist` (renamed to
`observation.images.camera1` / `camera2`). First process load took **39.82 s**
(HF cache warm). A later `reset` reuses the live server.

50 calls with real-shaped observations (8-D state, two 256x256 uint8 images,
task string), warm-up excluded:

| quantity | value |
| --- | --- |
| warm-up call | 4.229 s |
| median | **1.331 s** |
| mean | 1.457 s |
| p95 | **2.237 s** |
| max | 3.247 s |
| min | 1.255 s |
| peak VRAM | **1,077 MiB** (before: 0 MiB) |
| peak server RSS | **2,455,672 KiB**; `VmHWM` 3,841,184 KiB (3.66 GiB) |

Returned chunk: **(50, 8)**, wire dtype float32, client dtype float64 after
conversion, all values finite. Per-dimension min/max over 50 chunks:

| dim | min | max | Panda limit |
| --- | --- | --- | --- |
| 0 | -0.208128 | 0.078293 | [-2.8973, 2.8973] |
| 1 | -0.483027 | -0.034547 | [-1.7628, 1.7628] |
| 2 | -0.351209 | 0.114082 | [-2.8973, 2.8973] |
| 3 | -2.255185 | -1.471933 | [-3.0718, -0.0698] |
| 4 | -0.144063 | 0.024904 | [-2.8973, 2.8973] |
| 5 | 1.383635 | 1.854236 | [-0.0175, 3.7525] |
| 6 | -1.260738 | -0.665998 | [-2.8973, 2.8973] |
| 7 (gripper) | **-0.010464** | **0.063410** | [0.0, 0.04] |

The seven arm dimensions are inside the Panda limits. The eighth (gripper) is
outside `[0, 0.04]`; the simulator clips it before stepping. This is a
calibration observation on one synthetic observation, not a safety or quality
claim.

Failure paths on the real server: a wrong checkpoint hash is **refused**; a
server killed mid-call is **detected** (`policy server is not running`); a
1 ms client timeout is **handled** (`policy server timed out`).

### Cost

Live llvmpipe rendering, `n_action_steps=10`, 256x256 cameras:

| quantity | measurement |
| --- | --- |
| overhead render | **0.193 s/frame** mean (0.195 median, 0.203 max) |
| wrist render | **0.154 s/frame** mean (0.152 median, 0.159 max) |
| policy latency in eval | median **1.295 s/call** |
| 2-episode combined | 179.1 s (includes one 39.8 s model load) |
| straight 30-episode estimate | **44.8 min** |
| actual 30-episode wall | **1,592.6 s = 26.5 min** |

The 2-episode estimate was above 10 minutes, so the 30-episode run was started
only after explicit approval. It ran alone (no colcon, no other GPU work).

### 30-episode paired result

Seeds 20000-20029 (the same seeds as the earlier no-model MuJoCo baseline),
`n_action_steps=10`, renderer **`llvmpipe`**:

| metric | value |
| --- | --- |
| expert success | **24/30 = 0.800**, Wilson [0.6269, 0.9050] |
| policy success | **1/30 = 0.0333**, Wilson [0.00591, 0.16671] |
| policy success on expert-success seeds | **1/24 = 0.0417**, Wilson [0.00739, 0.20242] |
| policy-only failures | **23** |
| policy termination reasons | `SUCCESS` 1, `TIMEOUT` 29 |
| mean policy task completion | **0.01686** |
| expert termination reasons | `SUCCESS` 24, `TIMEOUT` 6 |

The expert result **exactly matches** the earlier no-model baseline (24/30),
which is the expected consistency check. The policy's single success was
episode index 10 (seed 132083934), an expert-success seed. Policy calls during
the run: 871, median 1.323 s, p95 1.480 s, max 12.409 s.

ZeroPolicy context on the same 30 seeds (no model, no rendering): **0/30**,
Wilson [0.0, 0.1135], all `TIMEOUT`, mean completion 0.00173 — matching the
earlier baseline.

**No policy-quality claim is made beyond these counts.** The policy was trained
for 1,000 of a 2,000-step bound on 40 episodes with accepted-data skew; 30
seeds and one task instruction cannot support a generalisation claim.

### Failure analysis on the real result

`failure_definition=policy_only` on the 23 policy-only failures selected the
**KDE** path (below the 30-failure GMM minimum, at or above the 10-failure KDE
minimum); no fallback reason was recorded. Artifact id
`64f51eb905e0e941`, `n_failures=23`. No targeted rollouts were generated.

### scikit-learn cross-check

`scikit-learn 1.9.1` was installed into core `.venv` only (dry-run: 4 packages,
~9.1 MiB download; 33 MiB installed for sklearn). It is not in `pyproject.toml`
or `uv.lock`, so CI does not require it. One optional test fits the NumPy EM and
sklearn `GaussianMixture` on the same seeded 3-cluster fixture (120 points,
8 features):

- both BIC argmins selected **3 components**;
- matched-component mean maximum absolute difference **8.3e-17**
  (tolerance 0.05);
- BIC relative difference **0.76%** (tolerance 2%).

The test skips with a reason when sklearn is absent.

### Test-claim audit

Task 9 tests were searched for success-rate/threshold assertions. Two groups
were relaxed (no policy-quality threshold is asserted anywhere):

- `tests/test_evaluation_policies.py` asserted `expert num_success >= 1`,
  `policy num_success == 0`, and zero policy-on-expert success. These were
  replaced with definitional identities: `num_eligible == expert num_success`,
  `policy_only == expert num_success - policy_on_expert`, and
  `0 <= policy_only <= num_pairs`.
- `tests/test_evaluation_records.py` had arbitrary Wilson thresholds
  (`wilson_95[0] > 0.4`, `wilson_95[1] < 0.6`). These were replaced by
  interval-membership checks on the synthetic aggregate fixtures. The exact
  `success_rate` values there are arithmetic checks of the aggregate function,
  not policy claims.

The ROS suite delta (+52 vs core +51) is exactly one extra test:
`ros2_ws/src/sim2real_sim/test/test_closed_loop_eval.py::test_run_eval_publishes_a_summary`,
which is ROS-marked and therefore skipped in the core-only run.

## Task 9c policy diagnosis (2026-11-11)

Goal: explain why the real checkpoint scored 1/30 in closed loop. No training,
no new rollout seeds. Existing train episodes were reused with their recorded
seeds.

### A. Observation parity (3 train-split episodes, no GPU)

Replayed the staged 20 Hz actions through the same reset/step/render path used by
the evaluator (llvmpipe, same cameras, 256x256) and compared every dataset frame.

| episode (dataset index) | frames | initial state max abs | max state abs per dim (7 joints + gripper) |
| --- | --- | --- | --- |
| 0 (`ep0000`) | 47 | 5.24e-8 | 3.7e-9 … 1.18e-7 |
| 1 (`ep0004`) | 45 | 5.24e-8 | 7.2e-9 … 1.14e-7 |
| 2 (`ep0008`) | 49 | 5.24e-8 | 7.4e-9 … 1.17e-7 |

State layout is the same 8-D order (`panda_joint1..7` + measured finger opening),
gripper units are metres, and the maximum difference over all dimensions is
**1.18e-7** (float32 storage precision).

Images are **byte-identical** for every compared frame: overhead **46/46,
44/44, 48/48** and wrist **46/46, 44/44, 48/48**, max pixel difference **0**.
Swapping RGB→BGR raises the mean absolute difference to ~27-35, so channel order
is correct. No orientation flip was needed.

Decimation phase: dataset frame `t` matches control step **2t** (max abs
**1.18e-7**), while control step `2t-1` differs by up to **0.0829 rad** (median
0.0164). There is **no off-by-one**.

Camera/rename mapping: the checkpoint metadata maps
`observation.images.overhead -> observation.images.camera1` and
`...wrist -> ...camera2`; the server handshake advertises `overhead`/`wrist` and
maps them to those model keys.

### B. Server-input parity (GPU, 3 frames)

For train episode 0, frames t = 0, 10, 20, the dataset observation and the
runner-built observation have identical state (max abs 5.24e-8, 4.85e-8,
9.59e-9) and identical images (max pixel 0). The returned chunks differ by at
most 0.260/0.431/0.170, but two calls with the **same** dataset input differ by
0.269/0.223/0.305: the difference is the model's flow-matching sampling noise,
not an input mismatch. A test that needs deterministic chunks must fix the noise
seed.

Teacher-forced one-shot error against the recorded dataset actions (physical
radians, executed steps only):

| frame | first-step L1 mean | first-10-step L1 mean | per-dim first step |
| --- | --- | --- | --- |
| 0 | 0.0567 | 0.1189 | 0.0315, 0.1175, 0.0929, 0.0199, 0.0039, 0.0647, 0.1211, 0.0023 |
| 10 | 0.0118 | 0.0304 | 0.0059, 0.0074, 0.0004, 0.0069, 0.0074, 0.0140, 0.0354, 0.0172 |
| 20 | 0.0567 | 0.0581 | 0.0318, 0.0345, 0.0809, 0.0880, 0.0231, 0.0194, 0.1685, 0.0072 |

This is model action error, not an observation-pipeline error.

### C. Oracle-hold test (no GPU, no model)

An oracle policy returned the recorded 10 Hz dataset actions; the closed-loop
runner held each for 2 control steps with `n_action_steps=10`.

- Recorded 20 Hz expert outcome: **32/32 success** (all accepted train episodes;
  30 clean + 2 detour-perturbed).
- Oracle hold: **24/32 = 0.750**, Wilson [0.5789, 0.8675]; **8 TIMEOUTs**
  (6 clean, 2 detour-perturbed).
- Clean subset: **24/30 = 0.800**; one of those (dataset index 39, seed 10071)
  started with the cube within the 0.06 m success tolerance (`completion=0.0`),
  so effective manipulation success is **23/30**.
- Same truncation, independent replay: **hold 20/32**, **linear interpolation
  between consecutive 10 Hz actions 30/32**, **next-action 22/32**. The runner's
  24/32 is higher than the independent hold replay because it repeats the last
  action after the recorded actions run out.

So the 10 Hz decimation + zero-order hold **loses the odd 20 Hz action** and
fails 6 clean episodes the 20 Hz expert completed. Linear interpolation recovers
most of them in this comparison. This is an evaluation-side information loss,
not an observation-parity bug.

### D. Train-seed closed loop (real checkpoint, GPU, llvmpipe)

Four train-split episodes (seeds 10000, 10004, 10008, 10009; indices 0, 1, 2, 3),
recorded profile, live llvmpipe:

| n_action_steps | success | termination | completion | joint max abs vs dataset | joint mean abs |
| --- | --- | --- | --- | --- | --- |
| 10 | **0/4** | 4 TIMEOUT | 0 / 0.0594 / 0 / 0 | 0.437-0.538 | 0.068-0.093 |
| 50 | **0/4** | 4 TIMEOUT | ~0 | 0.350-0.426 | 0.049-0.067 |

Policy call median was 1.35-1.45 s. The policy fails on **training** seeds at both
the 1 s and the training 5 s chunk horizon, so the 1/30 evaluation result is not
a generalisation gap: it is a policy that does not reproduce its training
actions.

### E. Frames and gripper traces

Under `log/task9c/` (git-ignored):

- `train_seed10000_n10/` — 30 policy steps x overhead+wrist, `gripper_trace.csv`,
  `policy_calls.csv`
- `eval_seed600278683/` — failed 9b episode (index 0), same files
- `eval_seed132083934/` — the 9b "success" (index 10), one frame set
- `gripper_summary.csv` — selected calls across the three episodes

Observed behaviour from the produced frames and numbers: in the train-seed and
failed-eval episodes the arm moves from home to the lower-left, parks over the
left (yellow) zone, and **stalls**; the cubes remain at their initial positions.
The wrist view shows the gripper over the empty zone, not around a cube. The
gripper opens and closes (train seed 10000 measured range 0.0001-0.0399 m) but
never grasps, and frames 15 and 29 of the train episode are visually almost
identical. It does not approach the target cube.

The 9b "success" is a **false positive**: the cube starts 0.0565 m from the zone
centre (success tolerance 0.06 m), the arm is still at home, and the episode
ends after 4 control steps with `task_completion=0.0`. The policy only opened
the gripper enough to satisfy `gripper_open`; it did not pick or place anything.
The policy's effective manipulation success on 20000-20029 is therefore **0/30**,
not 1/30.

### F. Gripper range check

- Dataset action dimension 8 is an **absolute gripper target in metres**
  (0 = closed, 0.04 = open). Dataset state dimension 8 is the **measured finger
  opening in metres**.
- Dataset stats: action min 0.0, max 0.04, mean 0.01752, std 0.01985, q01 ≈ 0,
  q99 ≈ 0.04; state max 0.04 (min -2.2e-5 is numerical).
- Both backends clip the target to `[0, 0.04]`; MuJoCo maps it to actuator
  0..255. The observed policy output range [-0.010464, 0.063410] is outside by
  0.0105 below and 0.0234 above (≈0.53σ / 1.18σ), plausible as regression error
  at 1,000 steps but not physically valid; the simulator clips it.

### G. Inference latency vs simulation time

Closed-loop simulation evaluation is **synchronous**: `policy.act` blocks the
runner, and simulation time does not advance while the GPU inference runs. A
1.3 s policy call therefore costs wall time only, not simulated time. A live ROS
node will **not** pause the robot while it waits. Task 10 must therefore run
inference asynchronously, buffer action chunks with observation timestamps,
reject stale or late chunks, keep the control loop non-blocking, define a
safe fallback when no fresh action is available, and monitor end-to-end latency.

### Decision table (recorded, not acted on)

| observation | row that applies | action |
| --- | --- | --- |
| A/B state/image parity | **no mismatch** | Nothing to fix in the observation path. |
| C oracle hold 24/32 vs 20 Hz 32/32 | **parity/contract mismatch** | Evaluation-side information loss from the 10 Hz zero-order hold. Proposed: interpolate between consecutive 10 Hz actions (30/32 in the variant replay) or retain odd 20 Hz commands; requires approval and a 30-seed rerun. Not changed today. |
| D real policy 0/4 train at n=10 and 0/4 at n=50 | **under-training** (not generalisation) | Propose: complete the 2,000-step bound, add an LR schedule, more episodes/diversity, and verify gripper calibration. Do not retrain today. |
| 9b 1/30 "success" | **success-definition false positive** | Initial cube within 0.06 m of the zone. Effective policy success is 0/30. Proposed: require displacement or start outside the success radius in evaluation-only success accounting. Expert code unchanged. |
| everything else | next hypotheses | gripper range/clipping, chunk open-loop at n=50, and model action accuracy. Image scale/normalization is not implicated: inputs are exact and the server uses LeRobot pre/post with dataset stats. |

## Task 9d trust and overfit diagnosis (2026-11-13)

### A. Success-definition audit

The success tolerance is read from the success function:
`zone.radius * success_zone_scale + zone_radius_tolerance_m` =
`0.055 * 1.0 + 0.005` = **0.060 m**.

- **train_v1 (40 episodes):** **1** starts inside 0.060 m — dataset index 39,
  episode `cli_mujoco_10000_1793611074_ep0071`, seed 10071, initial distance
  **0.05108 m**, recorded outcome success. The other 39 start outside.
- **30 eval seeds (derived from 20000):** **1** starts inside — index 10, derived
  seed **132083934**, initial distance **0.056518 m**; it is also an expert
  success and was the 9b policy "success". Expert success excluding starts:
  **23/29 = 0.793**, Wilson **[0.6161, 0.9015]** (recorded all-seed expert 24/30).
- **Sampler minimum:** the default profile allows `cube_red` at
  `x=0.38, y=-0.20`, i.e. **0.020 m** from the left zone centre, below the
  0.060 m tolerance. The sampler **can** spawn inside the success region.

ZeroPolicy reconciliation on the exact 9b seed (132083934, index 10):

| step | xy error | allowed+tole | height error | gripper open | stable | result |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.056518 | 0.055 + 0.005 | 0.0 | true (0.040) | 1/5 | not stable yet |
| 1 | 0.056518 | same | 0.000153 | **false (0.02515)** | 2/5 | gripper closed |
| 4 (hold-open) | 0.056518 | same | 0.000108 | **true (0.03672)** | 5/5 | **SUCCESS** |

ZeroPolicy commands the measured opening (0.04); the MuJoCo fingers sag to
0.02515 after the first step, below the 0.035 release threshold. A policy that
holds the arm but commands the gripper target to 0.04 reaches 0.03672 and
succeeds at step 4 with `task_completion=0.0`. The **gripper component**, not
the cube-in-zone test, is the difference. Neither run manipulates the cube; the
9b "success" is a success-definition false positive and the effective policy
manipulation success is **0/30**.

`EvalEpisodeRecord` now carries `starts_in_success_region` (bool or null;
missing = unknown for legacy records), computed from the initial observation
with the success function's own tolerance. Staged targeted episodes carry the
same key in metadata. Aggregates report both all-episode and
`*_excluding_start_in_region` success/Wilson fields. Options and consequences:

- **Exclude starts from evaluation sampling:** changes the seed meaning (a seed
  may now be skipped), and previous 9b/9c counts are not directly comparable.
- **Require displacement:** needs a displacement threshold and would change the
  success function (expert code; not changed today).
- **Exclude starts from training data:** the existing `train_v1` has one such
  episode (index 39); removing it changes the dataset hash, split and filter
  statistics. The next dataset should exclude such spawns.

### B. Execution contract

**Reconciliation of runner 24/32 vs independent replay.** Side by side on the
same 32 train seeds: runner `hold` **24/32**, independent replay without
repeating the last action **22/32**, independent replay with repeat-last
**24/32** (identical episode set to the runner). The per-episode table shows the
exact difference at indices 21 and 24: the runner continues holding the last
action after the recorded actions run out, while the independent replay stopped
at the final recorded action. Phase, action-to-control-step assignment,
re-planning and randomization were otherwise identical. **Cause: truncation at
episode end plus repeat-last, not a phase or seed difference.** (The 9c script
reported 20 with a slightly different loop; the side-by-side numbers above are
the reproducible comparison.)

**Linear mode.** `eval.execution` is now `hold | linear`; `linear` is the
default. Control step `2i` uses `a_i`, step `2i+1` uses `(a_i + a_{i+1})/2`,
and the final action in a chunk is held constant. Tests cover the phase,
interpolation exactness and chunk-end behaviour.

Oracle on the 32 train seeds:

| mode | success | Wilson 95% | failures (final cube-zone distance, m) |
| --- | --- | --- | --- |
| hold (legacy) | 24/32 = 0.750 | [0.579, 0.867] | 8: 0.0610, 0.0632, 0.0650, 0.0698, 0.0768, 0.0794, 0.0910, 0.1070 |
| linear (default) | **31/32 = 0.969** | [0.843, 0.994] | 1: index 24, 0.1541 |
| 20 Hz no-decimation | **32/32 = 1.000** | [0.893, 1.000] | none |

The ROS 2 Jazzy documentation says positions-only `spline` trajectories use
**linear** interpolation, matching the interpolation style. But the live
`/vla/raw_action` contract uses `time_from_start=(k+1)*dt`, so a live
trajectory is one policy period behind the evaluation convention (`a_k` at
`k*dt`); Task 10 must decide whether to publish `k*dt` or account for the
offset. Hold-mode 9b/9c numbers are legacy.

### C. Reproducible, language-aware calls

- **Instruction parity:** the runner string and the dataset task string are
  exactly `'pick up the red cube and place it in the left zone'` (repr identical
  for three train episodes and `meta/tasks.parquet`). The server uses the same
  LeRobot preprocessor as training with the same stats, so tokenization is
  identical by construction; no independent token dump was produced.
- **Instruction sensitivity:** six instructions (3 cubes x 2 zones), 5 sampled
  chunks each, same state/images. Mean pairwise cross-instruction chunk
  difference **0.01903**, max **0.03737**; within-instruction repeat noise mean
  **0.01712**, max **0.01824**. The effect is only **~1.1x** the mean noise
  (max ~2.0x): language conditioning is **weak/uncertain**, not a strong
  instruction response. This is a diagnostic, not a quality claim.
- **Deterministic sampling:** the server accepts `--sample-seed` / a
  `sample_seed` reset field and seeds `torch.manual_seed(seed + call_index)`.
  Same seed + same input gives **identical** chunks (max abs 0.0); a different
  seed differs by **0.263**. The client sends the episode seed by default.
- **Teacher-forced error (5-sample mean, physical units):**

| frame | first-step L1 | first-10 L1 | hold-last first-step | hold-last first-10 |
| --- | --- | --- | --- | --- |
| 0 | 0.02425 | 0.07772 | 0.00000 | 0.07739 |
| 10 | 0.01451 | 0.02275 | 0.02373 | 0.03699 |
| 20 | 0.01958 | 0.03854 | 0.02639 | 0.11623 |

  The 9c frame-0/frame-20 single-sample values were 0.056724 and 0.056681; they
  were **not identical**, just coincidentally close. The averaged values are
  clearly distinct.
- The model beats hold-last on first-step and first-10 at frames 10/20, but not
  at frame 0 (where the recorded action equals the state) - consistent with the
  offline L1 pattern.

### D. Overfit sanity test

Chosen episodes: dataset indices 0 and 1 (`ep0000` seed 10000, `ep0004` seed
10004), both outside the success region, same instruction. Training: same stack,
batch **4**, 250 steps, lr **1e-4**, chunk 50, LoRA settings unchanged,
`episode_indices: [0, 1]`.

- Wall **15:06**, step median **2.992 s**, first 11.3 s; peak VRAM **1,733 MiB**,
  peak RSS **3,908 MiB**; GPU **210 MHz**, 59-60 C (SW Power Cap and SW Thermal
  Slowdown active; 9b ran at ~1,185 MHz). Checkpoint
  `outputs/task9d/overfit_v1/checkpoint-00000250`, `git_commit=e2a25e3`,
  `git_dirty=false`, metadata SHA-256 `9be5ce5a...`.
- Loss blocks (50 steps): **2.1746 -> 0.9834 -> 0.8210 -> 0.6818 -> 0.6396**;
  final **0.49675**, min 0.22072. This is **not** a converged overfit.
- Closed loop on the same two seeds, linear execution, n_action_steps 10, 3
  deterministic sampling repeats each: **0/6 SUCCESS**, all TIMEOUT, completion
  ~0, joint max **0.338-0.518 rad**, mean **0.065-0.092 rad**.
- Teacher-forced on those training frames: first-step L1 **0.037-0.078**,
  first-10 **0.058-0.107**, not better than the base checkpoint.

**Interpretation:** the overfit run did not reach a low training loss, so the
test is **inconclusive as a pipeline-bug detector**. It does not show that the
pipeline can overfit and then fail in closed loop; it shows that 250 steps at
1e-4 under the 210 MHz/15-minute constraint were insufficient to fit even two
episodes. A/B parity is exact, so the leading hypothesis remains insufficient
optimization, not an observation/normalization bug. Task 10 waits for a true
overfit (or a GPU-clock fix) before further training.

### Decision table (Task 9d)

| observation | row | consequence |
| --- | --- | --- |
| 1/40 train and 1/30 eval start inside tolerance | **trivial-start episodes found** | Flagged; plan exclusion for the next dataset; 9b policy success is effectively **0/30**. |
| Linear oracle 31/32 vs hold 24/32, 20 Hz 32/32 | **linear beats hold** | Adopt `linear` for evaluation and Task 10; hold results are legacy. |
| Overfit 0/6 but training loss 0.64, step-capped at 210 MHz | **overfit fails / inconclusive** | Pipeline bug hunt before further training; Task 10 waits. Parity exact and loss not converged make insufficient optimization the first hypothesis. |
| Instruction cross-difference 0.019 vs noise 0.017 | **instruction sensitivity weak/uncertain** | Language conditioning is not clearly learned; the next plan starts with one instruction and checks instruction-conditioned behaviour. |

Next pipeline hypotheses with evidence: optimization budget/LR (loss still
falling, 250 steps only); state/action alignment (A/B exact - against);
normalization/action units (server uses dataset stats and parity is exact -
against); gripper range (clipped, but the arm never reaches the cube -
secondary); camera mapping/unused camera3 (byte-identical, expected 2 cameras -
against); frozen vision/expert-only (intended; may need more capacity/steps);
instruction handling (weak sensitivity - observed but not the train-seed failure
cause).

## CLI

```bash
uv run python scripts/run_eval.py --backend fake --policy noisy_expert \
    --num 60 --seed 20000 --profile default --out data/eval/dev_eval
uv run python scripts/fit_failure_distribution.py \
    --eval-run data/eval/dev_eval --out data/failure_distributions
uv run python scripts/collect_rollouts.py --backend fake --num 10 --seed 30000 \
    --failure-distribution data/failure_distributions/<id>.json \
    --out data/staging/targeted_dev
uv run python scripts/flywheel_status.py
```

`run_eval.py` pairs the scripted expert by default (`--no-paired-expert` to
disable) because `policy_only` failure analysis needs paired data in
`pairs.json`.

## Known limitations

- Only simulation; no physical robot and no safety claim.
- **Task 9d trust findings:** the success region radius is **0.060 m**; 1/40
  train episodes and 1/30 eval seeds start inside it, so the 9b 1/30 policy
  success is a false positive (effective **0/30**). `starts_in_success_region`
  is now recorded and aggregates report with/without such starts. Linear
  execution succeeds **31/32** oracle episodes vs hold **24/32**; linear is the
  default and hold-mode 9b/9c numbers are legacy.
- **Task 9d overfit run is inconclusive, not a pipeline-bug proof.** 250 steps
  on 2 episodes reached training loss **0.49675** (block mean 0.640) and closed
  loop failed **0/6** on the same seeds. The GPU was pinned at **210 MHz** with
  SW Power Cap/Thermal Slowdown active (15:06 wall), so the run did not
  converge. A true overfit is still required before Task 10 or further
  training.
- **Instruction sensitivity is weak.** Six instructions x 5 samples gave a mean
  cross-instruction chunk difference **0.01903** vs within-instruction noise
  **0.01712** (max 0.03737 vs 0.01824). Language conditioning is not clearly
  learned or used.
- The fake backend is kinematic; success rates there are not MuJoCo numbers.
- **Task 9c diagnosis:** observation parity is exact (state ≤1.18e-7; images
  byte-identical; frame t = control step 2t). The 10 Hz zero-order hold loses
  the odd 20 Hz action: the dataset-action oracle succeeds **24/32** (6 clean
  failures) while the 20 Hz expert succeeds 32/32; linear interpolation reaches
  30/32 in the variant replay. The real policy also fails **0/4 train-seed
  episodes at both n_action_steps 10 and 50**, so under-training dominates; this
  is not a generalisation gap.
- **The 9b 1/30 "success" is a success-definition false positive.** The cube
  starts 0.0565 m from the zone centre (tolerance 0.06 m), the arm stays near
  home, and the episode ends after 4 steps with `task_completion=0.0`. Effective
  policy manipulation success on 20000-20029 is **0/30**.
- **The real policy's gripper output is outside the physical range.** Across 50
  chunks the eighth dimension ranged [-0.010464, 0.063410] against
  dataset/backend [0, 0.04]; the simulator clips it. The dataset action is an
  absolute finger target in metres and state is measured opening.
- **The real server is slow to load and relatively slow per call.** First
  process load measured **39.8 s**; 50-call median **1.331 s**, p95 **2.237 s**.
  A 30-episode paired run took **26.5 min** with llvmpipe rendering. A live
  server is reused across resets.
- **Closed-loop sim pauses the world during inference.** The evaluator is
  synchronous, so 1.3 s calls cost wall time, not sim time. A live ROS node will
  not pause; Task 10 must use asynchronous inference, timestamped chunk
  buffering, staleness rejection and a safe fallback.
- **scikit-learn is a core `.venv`-only, optional cross-check dependency.** It is
  not in `pyproject.toml`/`uv.lock`; CI skips the cross-check with a reason.
  Installed size is ~33 MiB for sklearn plus `joblib`/`threadpoolctl`/`cloudpickle`.
- **Rendered episode frames for Task 9b were not saved.** Task 9c saved frames
  for only 2 eval and 1 train episode under `log/task9c/`.
- `REQUEST_RETRAIN` is a reserved decision; Task 12 will orchestrate any
  retraining.
- The checkpoint used for the pipeline test is a LoRA run that stopped at 1,000
  of the 2,000-step bound with a constant learning rate (no scheduler), so it is
  not a fully converged policy.
- One renderer per dataset: use llvmpipe for the real model unless the D3D12
  choice is explicitly approved and recorded.
