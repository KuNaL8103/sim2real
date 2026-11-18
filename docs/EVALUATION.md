# Closed-loop evaluation

Status: **implemented**. The evaluation stage evaluates policies in closed loop in simulation,
records rich per-episode results, analyses failures over randomization features,
requests failure-targeted rollouts, and enforces flywheel guards with a
persistent ledger. **The evaluation stage never starts training.**

Everything here is simulation-only. Simulator success is not a safety case, and
no physical robot is connected.

## Timing contract

| Quantity | Value | Why |
| --- | --- | --- |
| dataset / policy period | 0.1 s (10 Hz) | Training data was written at decimation 2 of the 20 Hz loop. |
| backend control rate | 20 Hz | Each policy action covers **2 control steps**. |
| `eval.execution` | **`linear`** (default) | Linear joint-space interpolation between consecutive 10 Hz actions; `hold` is the legacy zero-order hold. |
| `eval.n_action_steps` | **10** (1.0 s) | Not the training value 50 (5.0 s, longer than an episode). |
| success definition | the scripted expert's `_success_check` | Same thresholds, so paired comparisons are meaningful. |

A chunk is re-planned every `n_action_steps` policy steps (20 control steps at
the defaults). In `linear` mode, control step `2i` receives action `a_i` and
step `2i+1` receives the midpoint `(a_i + a_{i+1})/2`; the last action in a
chunk is held constant for both steps. In `hold` mode both steps receive `a_i`.
The earlier real-checkpoint and parity-diagnosis numbers were measured in **hold mode** and are legacy.

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
publisher uses `k*dt` or the consumer accounts for the offset. This is an inference contract decision, not a measurement claim.

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

- `ScriptedExpertPolicy` — runs the scripted expert on its **own** backend at reset
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
- The physics guard mirrors the scripted-expert guard (non-finite state, huge qacc,
  object outside the workspace margin) without modifying expert code.

## Records

`EvalEpisodeRecord` (`schema_version: 1`) is JSON-serialisable and rejects
non-finite values, wrong shapes, unknown termination reasons, and a `success`
flag inconsistent with the reason.

Fields: episode/run id, seed, episode index, randomization profile and the full
flattened sample, initial object pose, friction/mass/lighting/camera values,
task instruction, `success`, `task_completion`, `collision_count`, termination
reason, the filtering metrics (dimensionless squared jerk, SPARC, path length, limit
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

Decisions: `REQUEST_TARGETED_GENERATION`, `REQUEST_RETRAIN` (reserved; the evaluation stage
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

### Seed ledger

| range | owner | status |
| --- | --- | --- |
| 1000-1099 | filter tuning | used |
| 2000-2059 | reserved | used |
| 3000-3039 | reserved | used |
| 4000-4129 | reserved | used |
| 6000-6059 | filter hold-out | used |
| 7000-7002 | replay prototype | used |
| 8000-8059 | reserved | used |
| 10000-10199 | training/evaluation diagnosis | used |
| 20000-20999 | closed-loop evaluation | used |
| **20900-20919** | **Safety expert false-positive check** | **20900-20909 used; 20910-20919 reserved** |
| **20950-20969** | **Pipeline fake-backend end-to-end tests** | **reserved for the fake backend only; a MuJoCo run must not use it** |
| **21000-21099** | **pipeline generation default (`configs/pipeline/default.yaml`)** | **reserved; not yet run** |
| 30000-30999 | targeted generation | used |

The safety range is carved out of the evaluation range for this one diagnostic
only. It is not an evaluation seed set and must not be reused for a new choice
without updating this ledger.

The 20950-20969 range is the orchestrator's CPU end-to-end carve-out. The
orchestrator refuses those seeds unless `generation.backend: fake`; it also
refuses any drawn seed inside a used range. `21000-21099` is reserved for the
shipped default pipeline profile but has **not** been run.

## Renderer

The training dataset was rendered with **llvmpipe**. Evaluation records
`renderer` in every episode and in the run manifest: `fake-analytic` for the
fake backend, `llvmpipe` for MuJoCo by default, and `d3d12...-OPT_IN` if the
opt-in D3D12 environment is present. Opt-in settings are never used silently.

## Real-checkpoint verification (2026-11-11)

Test subject: the full-data checkpoint under `outputs/` (LoRA, 1,000
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

Evaluation tests were searched for success-rate/threshold assertions. Two groups
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

## Observation-parity diagnosis (2026-11-11)

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

Under the diagnostics log directory (git-ignored):

- `train_seed10000_n10/` — 30 policy steps x overhead+wrist, `gripper_trace.csv`,
  `policy_calls.csv`
- `eval_seed600278683/` — failed real-checkpoint episode (index 0), same files
- `eval_seed132083934/` — the real-checkpoint "success" (index 10), one frame set
- `gripper_summary.csv` — selected calls across the three episodes

Observed behaviour from the produced frames and numbers: in the train-seed and
failed-eval episodes the arm moves from home to the lower-left, parks over the
left (yellow) zone, and **stalls**; the cubes remain at their initial positions.
The wrist view shows the gripper over the empty zone, not around a cube. The
gripper opens and closes (train seed 10000 measured range 0.0001-0.0399 m) but
never grasps, and frames 15 and 29 of the train episode are visually almost
identical. It does not approach the target cube.

The real-checkpoint "success" is a **false positive**: the cube starts 0.0565 m from the zone
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
node will **not** pause the robot while it waits. The VLA inference node must therefore run
inference asynchronously, buffer action chunks with observation timestamps,
reject stale or late chunks, keep the control loop non-blocking, define a
safe fallback when no fresh action is available, and monitor end-to-end latency.

The VLA inference node implemented that asynchronous path (latest-observation-wins mailbox,
observation stamps, stale-chunk counting, `/vla/status`); see
`docs/VLA_INFERENCE.md`. The pre-registered overfit rule is still unmet: inference work proceeded under the documented exception recorded in `docs/TRAINING.md` section
17 and `docs/VLA_INFERENCE.md`. A real-checkpoint run of the node is expected to
fail the task, and no policy-quality claim follows from it.

### Decision table (recorded, not acted on)

| observation | row that applies | action |
| --- | --- | --- |
| A/B state/image parity | **no mismatch** | Nothing to fix in the observation path. |
| C oracle hold 24/32 vs 20 Hz 32/32 | **parity/contract mismatch** | Evaluation-side information loss from the 10 Hz zero-order hold. Proposed: interpolate between consecutive 10 Hz actions (30/32 in the variant replay) or retain odd 20 Hz commands; requires approval and a 30-seed rerun. Not changed today. |
| D real policy 0/4 train at n=10 and 0/4 at n=50 | **under-training** (not generalisation) | Propose: complete the 2,000-step bound, add an LR schedule, more episodes/diversity, and verify gripper calibration. Do not retrain today. |
| real-checkpoint 1/30 "success" | **success-definition false positive** | Initial cube within 0.06 m of the zone. Effective policy success is 0/30. Proposed: require displacement or start outside the success radius in evaluation-only success accounting. Expert code unchanged. |
| everything else | next hypotheses | gripper range/clipping, chunk open-loop at n=50, and model action accuracy. Image scale/normalization is not implicated: inputs are exact and the server uses LeRobot pre/post with dataset stats. |

## Trust and overfit diagnosis (2026-11-13)

### A. Success-definition audit

The success tolerance is read from the success function:
`zone.radius * success_zone_scale + zone_radius_tolerance_m` =
`0.055 * 1.0 + 0.005` = **0.060 m**.

- **train_v1 (40 episodes):** **1** starts inside 0.060 m — dataset index 39,
  episode `cli_mujoco_10000_1793611074_ep0071`, seed 10071, initial distance
  **0.05108 m**, recorded outcome success. The other 39 start outside.
- **30 eval seeds (derived from 20000):** **1** starts inside — index 10, derived
  seed **132083934**, initial distance **0.056518 m**; it is also an expert
  success and was the real-checkpoint policy "success". Expert success excluding starts:
  **23/29 = 0.793**, Wilson **[0.6161, 0.9015]** (recorded all-seed expert 24/30).
- **Sampler minimum:** the default profile allows `cube_red` at
  `x=0.38, y=-0.20`, i.e. **0.020 m** from the left zone centre, below the
  0.060 m tolerance. The sampler **can** spawn inside the success region.

ZeroPolicy reconciliation on the exact real-checkpoint seed (132083934, index 10):

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
The real-checkpoint "success" is a success-definition false positive and the effective policy
manipulation success is **0/30**.

`EvalEpisodeRecord` now carries `starts_in_success_region` (bool or null;
missing = unknown for legacy records), computed from the initial observation
with the success function's own tolerance. Staged targeted episodes carry the
same key in metadata. Aggregates report both all-episode and
`*_excluding_start_in_region` success/Wilson fields. Options and consequences:

- **Exclude starts from evaluation sampling:** changes the seed meaning (a seed
  may now be skipped), and previous real-checkpoint and parity counts are not directly comparable.
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
episode end plus repeat-last, not a phase or seed difference.** (The parity-diagnosis script
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
`k*dt`); the VLA inference node must decide whether to publish `k*dt` or account for the
offset. Hold-mode real-checkpoint and parity numbers are legacy.

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

  The parity-diagnosis frame-0/frame-20 single-sample values were 0.056724 and 0.056681; they
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
  Slowdown active; the earlier run used ~1,185 MHz). Checkpoint
  the overfit checkpoint under `outputs/`, `git_commit=e2a25e3`,
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
optimization, not an observation/normalization bug. The VLA inference node waits for a true
overfit (or a GPU-clock fix) before further training.

### Decision table

| observation | row | consequence |
| --- | --- | --- |
| 1/40 train and 1/30 eval start inside tolerance | **trivial-start episodes found** | Flagged; plan exclusion for the next dataset; the real-checkpoint policy success is effectively **0/30**. |
| Linear oracle 31/32 vs hold 24/32, 20 Hz 32/32 | **linear beats hold** | Adopt `linear` for evaluation and the VLA inference node; hold results are legacy. |
| Overfit 0/6 but training loss 0.64, step-capped at 210 MHz | **overfit fails / inconclusive** | Pipeline bug hunt before further training; the VLA inference node waits. Parity exact and loss not converged make insufficient optimization the first hypothesis. |
| Instruction cross-difference 0.019 vs noise 0.017 | **instruction sensitivity weak/uncertain** | Language conditioning is not clearly learned; the next plan starts with one instruction and checks instruction-conditioned behaviour. |

Next pipeline hypotheses with evidence: optimization budget/LR (loss still
falling, 250 steps only); state/action alignment (A/B exact - against);
normalization/action units (server uses dataset stats and parity is exact -
against); gripper range (clipped, but the arm never reaches the cube -
secondary); camera mapping/unused camera3 (byte-identical, expected 2 cameras -
against); frozen vision/expert-only (intended; may need more capacity/steps);
instruction handling (weak sensitivity - observed but not the train-seed failure
cause).

## GPU gate, drift and instruction diagnosis (2026-11-13)

### GPU health gate: FAILED - training parts not run

60 s synthetic 2048x2048 matmul load in `.venv-ml`, sampling every 2 s (32
samples, 2280 matmuls): SM clock **210 / 210 / 210 MHz** (min/median/max),
power **4.36 / 7.245 / 7.35 W**, temperature **52 / 55 / 56 C**, throttle reason
`0x24` on **32/32** samples (SW Power Cap + SW Thermal Slowdown). This is the
same 210 MHz cap as the trust and overfit diagnosis. The gate requires median >=1000 MHz and no power
cap, so **B (real overfit) and C (fix variant) were not run**. No system or
driver setting was changed. The GPU needs a charger/power-plan/vendor-software
check first.

### A. Closed-loop drift (existing 1,000-step checkpoint)

Linear execution, n_action_steps 10, deterministic sampling. Teacher-forced
mode overwrote the arm qpos with the recorded dataset state at every re-plan
point; free mode did not.

| seed | mode | max joint diff (rad) | mean joint diff | first 0.1 rad exit | fraction >0.1 rad |
| --- | --- | --- | --- | --- | --- |
| 10000 | teacher | 0.617 | 0.128 | step 4, frame 2 | 0.864 |
| 10000 | free | 0.499 | 0.099 | step 4, frame 2 | 0.954 |
| 10004 | teacher | 0.551 | 0.080 | step 4, frame 2 | 0.836 |
| 10004 | free | 0.374 | 0.077 | step 4, frame 2 | 0.860 |
| 10008 | teacher | 0.575 | 0.075 | step 5, frame 2 | 0.778 |
| 10008 | free | 0.456 | 0.088 | step 5, frame 2 | 0.874 |

Policy call median was 1.37 s. **The policy leaves the 0.1 rad tube in the
first chunk (frame 2) in every run, with or without state resets.** Teacher
forcing the arm state at re-plan points does not improve the trajectory; the
error is already in the first chunk, so this is not primarily accumulated
drift. It is a per-chunk action error (under-training or a model/pipeline
issue), which B would separate but B is blocked by the GPU.

### D. Instruction test with 10 samples

Deterministic sampling, 6 instructions x 10 seeds, 3 train observations.
Values are mean absolute chunk differences over the first 8 action dimensions.

Fine-tuned checkpoint:

| frame | cross-instruction mean | within-instruction noise | ratio |
| --- | --- | --- | --- |
| 0 | 0.01710 | 0.02906 | 0.589 |
| 10 | 0.01768 | 0.02081 | 0.849 |
| 20 | 0.01708 | 0.02066 | 0.827 |

Base `lerobot/smolvla_base` (same observations, 10 seeds):

| frame | cross-instruction mean | within-instruction noise | ratio |
| --- | --- | --- | --- |
| 0 | 0.00499 | 0.00862 | 0.579 |
| 10 | 0.03250 | 0.04850 | 0.670 |
| 20 | 0.03458 | 0.06503 | 0.532 |

**Instruction effects are below the sampling noise in all six cases, for both
the base and fine-tuned models.** The language pathway is not measurably
affecting the action chunk with this setup. The thin overfit result (ratio 1.1) did
not survive 10 samples. The training plan should start with one instruction and
include an explicit instruction-conditioned test before multi-instruction
training.

### B/C status and conclusion

B (real overfit) and C (fix variant) are **NOT RUN** because the GPU health
gate failed. The overfit run remains unconverged and inconclusive. Decision:
**GPU unhealthy -> hardware fix first** (charger, Windows power plan, vendor
software). The VLA inference node stays blocked until the GPU clock is healthy and a true
overfit either succeeds (pipeline sound) or fails at low training loss (pipeline
bug hunt). No policy-quality claim.

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
- **The GPU health gate failed.** 60 s synthetic load held SM at **210 MHz**
  with SW Power Cap + SW Thermal Slowdown active (4.36-7.35 W, 52-56 C), so the
  real overfit and fix-variant runs were **not run**. The GPU needs a charger /
  Windows power-plan / vendor-software check first.
- **Closed-loop drift starts in the first chunk.** With the existing
  checkpoint, all three train seeds leave the 0.1 rad tube at **frame 2**;
  teacher-forced state resets at re-plan points do not help (max drift 0.55-0.62
  rad teacher, 0.37-0.50 free). This is per-chunk action error, not accumulated
  drift, and B is needed to separate under-training from a pipeline issue.
- **Instruction sensitivity is below noise at 10 samples.** Fine-tuned
  cross/within ratios 0.59-0.85; base model 0.53-0.67. The language pathway is
  not measurably affecting the action chunk; start with one instruction and
  test instruction conditioning explicitly.
- **Trust and overfit findings:** the success region radius is **0.060 m**; 1/40
  train episodes and 1/30 eval seeds start inside it, so the real-checkpoint 1/30 policy
  success is a false positive (effective **0/30**). `starts_in_success_region`
  is now recorded and aggregates report with/without such starts. Linear
  execution succeeds **31/32** oracle episodes vs hold **24/32**; linear is the
  default and hold-mode real-checkpoint and parity numbers are legacy.
- **The overfit run is inconclusive, not a pipeline-bug proof.** 250 steps
  on 2 episodes reached training loss **0.49675** (block mean 0.640) and closed
  loop failed **0/6** on the same seeds. The GPU was pinned at **210 MHz** with
  SW Power Cap/Thermal Slowdown active (15:06 wall), so the run did not
  converge. A true overfit is still required before the VLA inference node or further
  training.
- **Instruction sensitivity is weak.** Six instructions x 5 samples gave a mean
  cross-instruction chunk difference **0.01903** vs within-instruction noise
  **0.01712** (max 0.03737 vs 0.01824). Language conditioning is not clearly
  learned or used.
- The fake backend is kinematic; success rates there are not MuJoCo numbers.
- **The observation-parity diagnosis:** observation parity is exact (state ≤1.18e-7; images
  byte-identical; frame t = control step 2t). The 10 Hz zero-order hold loses
  the odd 20 Hz action: the dataset-action oracle succeeds **24/32** (6 clean
  failures) while the 20 Hz expert succeeds 32/32; linear interpolation reaches
  30/32 in the variant replay. The real policy also fails **0/4 train-seed
  episodes at both n_action_steps 10 and 50**, so under-training dominates; this
  is not a generalisation gap.
- **The real-checkpoint 1/30 "success" is a success-definition false positive.** The cube
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
  not pause; the VLA inference node must use asynchronous inference, timestamped chunk
  buffering, staleness rejection and a safe fallback.
- **scikit-learn is a core `.venv`-only, optional cross-check dependency.** It is
  not in `pyproject.toml`/`uv.lock`; CI skips the cross-check with a reason.
  Installed size is ~33 MiB for sklearn plus `joblib`/`threadpoolctl`/`cloudpickle`.
- **Rendered episode frames for the real-checkpoint evaluation were not saved.** The observation-parity diagnosis saved frames
  for only 2 eval and 1 train episode under the diagnostics log directory.
- `REQUEST_RETRAIN` is a reserved decision; the pipeline orchestrator will manage any
  retraining.
- The checkpoint used for the pipeline test is a LoRA run that stopped at 1,000
  of the 2,000-step bound with a constant learning rate (no scheduler), so it is
  not a fully converged policy.
- One renderer per dataset: use llvmpipe for the real model unless the D3D12
  choice is explicitly approved and recorded.

## First-chunk error analysis (2026-11-13)

Purpose: explain how a small teacher-forced L1 can coexist with leaving the
0.1 rad tube at frame 2. This is inference-only on the **throttled local GPU**
(210 MHz, SW Power Cap/Thermal Slowdown). The runs took 71 s and 75 s wall
including model load; that is **throttled, not a performance figure**.

Setup: the full-data checkpoint under `outputs/`, dataset
index 0 (seed 10000, `ep0000`), frames 0..5, action chunk 50, postprocessed
actions in physical units. The prediction is the mean first-step action over
five deterministic sample seeds **101-105** (not rollout seeds). "Motion" is the
recorded action change from frame `f` to `f+1`; "ratio" is mean absolute
first-step error divided by that motion.

### Recorded motion, error and ratio per dimension

Absolute first-step error (rad or m for d7), mean over 5 sample seeds:

| frame | d0 | d1 | d2 | d3 | d4 | d5 | d6 | d7 | mean | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.01262 | 0.07946 | 0.00426 | 0.04636 | 0.00819 | 0.02714 | 0.03650 | 0.00027 | 0.02685 | 0.07946 |
| 1 | 0.00999 | 0.05157 | 0.00340 | 0.01235 | 0.00385 | 0.01092 | 0.04315 | 0.00311 | 0.01729 | 0.05157 |
| 2 | 0.01702 | 0.03747 | 0.00022 | 0.01563 | 0.00498 | 0.01660 | 0.04714 | 0.00043 | 0.01744 | 0.04714 |
| 3 | 0.01824 | 0.00011 | 0.00362 | 0.00721 | 0.00677 | 0.01184 | 0.05155 | 0.00102 | 0.01254 | 0.05155 |
| 4 | 0.02330 | 0.04310 | 0.00986 | 0.00693 | 0.01050 | 0.01354 | 0.05022 | 0.00040 | 0.01973 | 0.05022 |
| 5 | 0.01490 | 0.04724 | 0.00774 | 0.04226 | 0.01395 | 0.05480 | 0.03588 | 0.00400 | 0.02760 | 0.05480 |

Recorded absolute motion `|a(f+1) - a(f)|` per dimension:

| frame | d0 | d1 | d2 | d3 | d4 | d5 | d6 | d7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.00039 | 0.00228 | 0.00047 | 0.00183 | 0.00016 | 0.00045 | 0.00084 | 0.00000 |
| 1 | 0.00736 | 0.04282 | 0.00878 | 0.03431 | 0.00293 | 0.00843 | 0.01580 | 0.00000 |
| 2 | 0.01723 | 0.10016 | 0.02053 | 0.08026 | 0.00685 | 0.01973 | 0.03696 | 0.00000 |
| 3 | 0.01974 | 0.11475 | 0.02352 | 0.09196 | 0.00785 | 0.02261 | 0.04235 | 0.00000 |
| 4 | 0.01274 | 0.07409 | 0.01519 | 0.05938 | 0.00507 | 0.01460 | 0.02735 | 0.00000 |
| 5 | 0.00270 | 0.01571 | 0.00322 | 0.01259 | 0.00107 | 0.00309 | 0.00580 | 0.00000 |

Ratio of error to motion (n/a where motion is exactly zero):

| frame | d0 | d1 | d2 | d3 | d4 | d5 | d6 | d7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 32.15 | 34.81 | 9.09 | 25.34 | 52.45 | 60.35 | 43.32 | n/a |
| 1 | 1.36 | 1.20 | 0.39 | 0.36 | 1.31 | 1.29 | 2.73 | n/a |
| 2 | 0.99 | 0.37 | 0.01 | 0.19 | 0.73 | 0.84 | 1.28 | n/a |
| 3 | 0.92 | 0.00 | 0.15 | 0.08 | 0.86 | 0.52 | 1.22 | n/a |
| 4 | 1.83 | 0.58 | 0.65 | 0.12 | 2.07 | 0.93 | 1.84 | n/a |
| 5 | 5.52 | 3.01 | 2.40 | 3.36 | 12.98 | 17.71 | 6.19 | n/a |

Per-seed first-step standard deviation is large on the arm: d1 0.046-0.064,
d3 0.087-0.103, d5 0.039-0.074 rad, so a single deterministic sample is noisy.
First-10 mean L1 over the five seeds was **0.07928, 0.07852, 0.07504, 0.06293,
0.04681, 0.03419 rad** at frames 0..5.

### Preprocessor parity at frame 0

The runner observation was captured inside `run_eval_episode` after the same
MuJoCo reset. Raw state max abs difference vs the dataset sample was
**5.24e-8**; camera pixel max difference was **0** for both overhead and wrist.
After LeRobot's own preprocessor, with the same rename map and dataset stats:

| key | shape | max abs difference |
| --- | --- | --- |
| `observation.state` | (1, 8) | **0.0** |
| `observation.images.camera1` | (1, 3, 256, 256) | **0.0** |
| `observation.images.camera2` | (1, 3, 256, 256) | **0.0** |
| `observation.language.tokens` | (1, 48) | **0.0** |
| `observation.language.attention_mask` | (1, 48) | **0.0** |
| `task` | list | equal |

The dataset batch had extra label/padding keys (`action`, `action_is_pad`,
`index`, `task_index`, `episode_index`, `timestamp`); the runner batch does not
send them at inference. No observation, preprocessing, tokenization, or
normalization difference was found.

### Executed command vs postprocessed chunk

A recording wrapper captured the exact chunk returned by `CheckpointPolicy`
(which already applies LeRobot's postprocessor) while the real runner executed
it. At control step 0 (linear execution, fraction 0):

- invoked action vs `chunk[0]`: **max abs difference 0.0**;
- control step 1 vs `chunk[0] + 0.5 * (chunk[1] - chunk[0])`: **max abs
  difference 0.0**;
- values stayed in physical units (chunk first min -1.54196, max 1.51481; the
  gripper dimension was 0.04176 m).

So the postprocessor is applied exactly once and the runner executes the
postprocessed chunk without re-normalising it.

### Answer

None of input parity, LeRobot preprocessing, double postprocessing, or unit
conversion explains the gap. The policy's first-step action error is **not
small relative to the motion**: at frame 0 the recorded arm motion is only
0.00016-0.00228 rad per dimension, while the error is 0.004-0.079 rad, i.e.
**9-60x** the per-step motion. The largest first-step error (0.079 rad) is
already ~80% of the 0.1 rad tube radius; one policy action is held for two
20 Hz control steps, so by frame 2 the tube is crossed even when the recorded
initial motion is almost zero. At moving frames the ratio is smaller (often
1-3 on joints) but the absolute error remains comparable to one frame of
motion. The 0.023-0.078 rad teacher-forced L1 is "small" only in absolute
radians; it is a large fraction of the 0.1 rad acceptance and larger than the
first-frame motion. Flow-matching sampling noise (per-seed std up to 0.10 rad)
and under-training, not an input/pipeline mismatch, are the measured causes.

## Wider first-chunk analysis (2026-11-13, throttled GPU)

Scope: train-split dataset indices **0, 1 and 2**, frames **0-5, 10, 20**, five
deterministic sample seeds **101-105**. The checkpoint is the same 1,000-step
full-data run (the full-data checkpoint under `outputs/`). The run took
**255 s** wall on the local GPU; that is **throttled, not a performance
figure**. It is a diagnostic only.

Per-frame summary (L1 in physical units; "motion" is `|a(f+1) - a(f)|`; the
ratio is the mean over the seven arm dimensions):

| episode | frame | mean error | max error | mean motion | max motion | mean ratio (7 arm dims) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0.02685 | 0.07946 | 0.00080 | 0.00228 | 36.79 |
| 0 | 1 | 0.01729 | 0.05157 | 0.01505 | 0.04282 | 1.24 |
| 0 | 2 | 0.01744 | 0.04714 | 0.03522 | 0.10016 | 0.63 |
| 0 | 3 | 0.01254 | 0.05155 | 0.04035 | 0.11475 | 0.54 |
| 0 | 4 | 0.01973 | 0.05022 | 0.02605 | 0.07409 | 1.14 |
| 0 | 5 | 0.02760 | 0.05480 | 0.00552 | 0.01571 | 7.31 |
| 0 | 10 | 0.01432 | 0.02418 | 0.01862 | 0.07101 | 7.24 |
| 0 | 20 | 0.02327 | 0.06459 | 0.00774 | 0.03125 | 56.42 |
| 1 | 0 | 0.02785 | 0.07486 | 0.00134 | 0.00271 | 22.07 |
| 1 | 1 | 0.02119 | 0.05523 | 0.02427 | 0.04901 | 0.87 |
| 1 | 2 | 0.01562 | 0.03452 | 0.05291 | 0.10685 | 0.42 |
| 1 | 3 | 0.00898 | 0.02789 | 0.05291 | 0.10685 | 0.36 |
| 1 | 4 | 0.01841 | 0.05119 | 0.02427 | 0.04901 | 0.78 |
| 1 | 5 | 0.01547 | 0.05220 | 0.00134 | 0.00271 | 10.53 |
| 1 | 10 | 0.01361 | 0.03893 | 0.00740 | 0.02719 | 4.32 |
| 1 | 20 | 0.01406 | 0.03123 | 0.00120 | 0.00473 | 70.08 |
| 2 | 0 | 0.02683 | 0.07534 | 0.00098 | 0.00208 | 28.77 |
| 2 | 1 | 0.02168 | 0.05558 | 0.01889 | 0.04028 | 1.18 |
| 2 | 2 | 0.01730 | 0.04145 | 0.04666 | 0.09949 | 0.47 |
| 2 | 3 | 0.00990 | 0.02141 | 0.05888 | 0.12554 | 0.31 |
| 2 | 4 | 0.01551 | 0.04395 | 0.04666 | 0.09949 | 0.41 |
| 2 | 5 | 0.01795 | 0.06134 | 0.01889 | 0.04028 | 1.00 |
| 2 | 10 | 0.01425 | 0.04043 | 0.03116 | 0.11432 | 2.36 |
| 2 | 20 | 0.02748 | 0.11962 | 0.02098 | 0.08104 | 1.32 |

Frame-0 per-dimension error (rad, or m for d7):

| episode | d0 | d1 | d2 | d3 | d4 | d5 | d6 | d7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.01262 | 0.07946 | 0.00426 | 0.04636 | 0.00819 | 0.02714 | 0.03650 | 0.00027 |
| 1 | 0.01388 | 0.07486 | 0.00566 | 0.04568 | 0.00425 | 0.03575 | 0.04244 | 0.00026 |
| 2 | 0.01435 | 0.07534 | 0.00134 | 0.04632 | 0.00431 | 0.03105 | 0.04175 | 0.00018 |

Frame-0 per-dimension recorded motion:

| episode | d0 | d1 | d2 | d3 | d4 | d5 | d6 | d7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.00039 | 0.00228 | 0.00047 | 0.00183 | 0.00016 | 0.00045 | 0.00084 | 0.00000 |
| 1 | 0.00112 | 0.00271 | 0.00136 | 0.00210 | 0.00042 | 0.00058 | 0.00244 | 0.00000 |
| 2 | 0.00073 | 0.00208 | 0.00094 | 0.00166 | 0.00038 | 0.00039 | 0.00163 | 0.00000 |

Frame-0 per-dimension error/motion ratio (n/a means zero motion):

| episode | d0 | d1 | d2 | d3 | d4 | d5 | d6 | d7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 32.15 | 34.81 | 9.09 | 25.34 | 52.45 | 60.35 | 43.32 | n/a |
| 1 | 12.35 | 27.58 | 4.15 | 21.70 | 10.10 | 61.25 | 17.37 | n/a |
| 2 | 19.59 | 36.18 | 1.42 | 27.96 | 11.49 | 79.14 | 25.63 | n/a |

**Does the 9f pattern hold?** Yes. At frame 0 all three episodes have almost no
recorded motion (max 0.00208-0.00271 rad) while the first-step error is
**0.02683-0.02785 rad** (max 0.07486-0.07946 rad); the mean ratio is **22-37x**.
The per-dimension ratios are 1.4-79x. At moving frames the ratio usually falls
below ~1-2, but frame 5 and 20 can be high again when the recorded motion is
near zero (7-70x). The same first-chunk error is therefore present on every
episode, not just episode 0.

**Sample-seed spread vs the mean error.** The standard deviation across the five
seeds is comparable to or larger than the mean error: mean-over-dimension
std/mean-error is **1.33, 2.07, 2.15, 3.07, 1.89, 1.37** at frames 0-5 for
episode 0; **1.31, 1.79, 2.30, 4.02, 1.59, 1.52** for episode 1; and
**1.36, 1.80, 2.20, 3.68, 2.01, 1.39** for episode 2. At frame 0 the arm-dimension
std is 0.018-0.091 rad, while the mean error is 0.027 rad. A five-seed mean is
still noisy; the pre-registered rule uses it deliberately to reduce this noise,
but it cannot make a single deterministic sample representative.

## GPU gate and branch R (2026-11-15)

The 60 s synthetic matmul gate ran on the local RTX 4050 in `.venv-ml`
(2,280 matmuls, 60.11 s). It sampled `nvidia-smi` every 2 s (30 samples):

| Quantity | Min | Median | Max |
| --- | --- | --- | --- |
| SM clock (MHz) | 210 | **210** | 210 |
| Power (W) | 4.68 | 7.17 | 7.25 |
| Temperature (C) | 54 | 56 | 57 |

Throttle reason was **`0x24`** (SW Power Cap + SW Thermal Slowdown) on every
sample. This fails the health rule (median SM clock >= 1,000 MHz with no power
cap active), so the session took **branch R**: no local or remote training and
no teacher-forced/closed-loop evaluation of a new checkpoint. The pre-registered
rule was not evaluated and was not edited.

The authoritative bundle `sim2real_remote_d7a0c90_20261115T131152Z.tar.gz`
(95,909,403 bytes, sha256 `658df551...afe4e`) passed `--check-only` in a clean
temp clone at `d7a0c90`: 13 manifest hashes, clean commit, Python 3.12,
`lerobot Requires-Python >=3.12`, GPU/driver printed, no install/download/
training. The VLA inference node remains blocked pending a healthy-GPU overfit result.

## First remote overfit result (2026-11-17)

The returned d7a0c90 remote run produced a checkpoint that verifies locally
(all metadata/adapter hashes match `RUN_REPORT.json`; stats and config hashes
match the bundle), but the run had recorded protocol overrides; see
`docs/REMOTE_TRAINING.md` "Remote overfit result". The pre-registered rule was applied
unchanged.

### The six `POLICY_ERROR` episodes were a harness timeout, not a policy fault

All six closed-loop records (evaluation seeds 20000 and 20004, 3 repeats each)
have `termination_reason = POLICY_ERROR`, `steps = []` and the identical
`failure.reason`:

```
PolicyError: policy server timed out after 30.0 s
```

Cause, established in this session:

1. `CheckpointPolicy._handshake` waits `timeout_s` (default **30.0 s**) for the
   server's `ready` packet (`policy_interface.py:523`). The server sends `hello`
   immediately, then imports torch and loads the base model, then sends
   `ready` (`policy_server.py:258-285`).
2. Re-running the same checkpoint and code on the local GPU measured
   **hello at 0.33 s, ready at 34.71 s** (model load 34.38 s), so the 30 s wait
   expires. The first real `act` then returned a finite `[50, 8]` chunk in
   **13.02 s**.
3. Because a failed handshake leaves the client not-ready, the next episode
   restarts the server and repeats the load, which is why all three repeats per
   seed fail identically with zero steps.

This is **environment/harness**, not policy output: no NaN/inf, no shape error,
no joint-limit rejection. It is aggravated by the throttled local GPU, and the
repo already documented a 39.8 s first load in `docs/EVALUATION.md` (parity diagnosis), i.e.
the 30 s default was already shorter than a documented cold start.

Fix (evaluation-side, no contract change): `CheckpointPolicy` now takes a
separate `startup_timeout_s` used only for the ready wait; `run_eval.py` exposes
`--policy-startup-timeout` (default 180 s) and `--policy-timeout` (default 30 s
per call). Regression test:
`tests/test_evaluation_policy_server.py::test_cold_start_timeout_is_separate_from_call_timeout`.

### Pre-registered closed loop on the TRAINING seeds

Checkpoint `checkpoint-00001000`, linear execution, `n_action_steps=10`,
recorded profile, seeds 10000 and 10004, 3 repeats each. Repeats use different,
recorded policy sample seeds (r0 = episode seed; r1 = 9001xx; r2 = 9002xx).
Timings are **throttled, not a performance figure**.

| seed | repeat | sample seed | success | termination | completion | joint max | joint mean | first 0.1 rad exit | fraction > 0.1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10000 | 0 | 10000 | False | TIMEOUT | 0.0000 | 0.3354 | 0.0504 | step 2 | 0.617 |
| 10000 | 1 | 900100 | False | TIMEOUT | 0.0000 | 0.2771 | 0.0540 | step 2 | - |
| 10000 | 2 | 900200 | False | TIMEOUT | 0.0000 | 0.4630 | 0.0748 | step 2 | - |
| 10004 | 0 | 10004 | False | TIMEOUT | 0.0000 | 0.5419 | 0.1097 | step 1 | - |
| 10004 | 1 | 900101 | False | TIMEOUT | 0.0000 | 0.5387 | 0.0964 | step 5 | - |
| 10004 | 2 | 900201 | False | TIMEOUT | 0.0000 | 0.4974 | 0.0932 | step 4 | - |

**0/6 success; 6/6 TIMEOUT; 0/6 POLICY_ERROR.** No repeat came within the 0.1 rad
tube: every episode left it by policy step 1-5, i.e. in the first chunk or the
one after it. Elapsed 65-165 s per episode.

The earlier 20000/20004 runs are **not** the pre-registered secondary check:
those seeds are evaluation seeds, so a result there would measure generalisation
to new poses. They also produced no usable measurement (0 control steps).

### Bias versus sampling spread (first-step arm L1)

Same frames (0, 1, 2, 10, 20, 30, 40 on dataset episodes 0 and 1) and the same
five sample seeds (101-105) for every checkpoint. `bias` is the absolute error
of the 5-seed mean prediction; `spread` is the mean per-dimension standard
deviation across seeds; `single L1` is the pre-registered mean single-sample
number; `mean5 L1` is the error of the 5-seed mean.

| checkpoint | frames | bias | spread | single L1 | mean5 L1 | spread / single L1 |
| --- | --- | --- | --- | --- | --- | --- |
| remote 200 | 14 | 0.0634 | 0.0493 | 0.0748 | 0.0634 | 0.659 |
| remote 500 | 14 | 0.0348 | 0.0369 | 0.0472 | 0.0348 | 0.781 |
| remote 1,000 | 14 | 0.0223 | 0.0280 | 0.0329 | 0.0223 | 0.850 |
| overfit 250 (pre-adapter_hash) | 14 | 0.0550 | 0.0507 | 0.0684 | 0.0550 | 0.741 |
| train_v1 1,000 (pre-adapter_hash) | 14 | 0.0186 | 0.0324 | 0.0318 | 0.0186 | 1.019 |

(Bias per arm dimension for remote 1,000: 0.0079, 0.0445, 0.0057, 0.0252,
0.0060, 0.0273, 0.0398 rad. The largest systematic errors are joints 1, 6 and 4.)

Reading:

- The pre-registered **0.0333** (locally recomputed 0.0329 on the reduced frame
  set) is **not dominated by sampling spread**. Spread is comparable to the
  error (0.0280 vs 0.0329), but averaging five samples only removes the
  0.0105 difference: the **bias alone is 0.0223 rad**, still **2.2x** the
  0.010 bar. A perfectly denoised sampler could not pass criterion 1 with this
  checkpoint.
- Error is **still falling** between 200, 500 and 1,000: 0.0748 -> 0.0472 ->
  0.0329 (a factor ~0.70 per doubling of steps). The bias falls faster
  (0.0634 -> 0.0348 -> 0.0223) than the spread (0.0493 -> 0.0369 -> 0.0280).
- The pre-registered **bars are unchanged**. Sampling-side ideas (averaging K
  chunks, more denoising steps) are **hypotheses only**, to be tested later;
  they cost latency roughly linearly and cannot by themselves clear a
  0.0223 rad bias.
- `train_v1` 1,000 (32 episodes) has a slightly *lower* bias (0.0186) than the
  two-episode overfit checkpoint (0.0223), so "more data" and "more steps" are
  not clearly separated at this budget.

