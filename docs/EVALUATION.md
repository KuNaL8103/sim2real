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
| backend control rate | 20 Hz | One policy action is **held for 2 control steps**. |
| `eval.n_action_steps` | **10** (1.0 s) | Not the training value 50 (5.0 s, longer than an episode). |
| success definition | Task 4 expert `_success_check` | Same thresholds, so paired comparisons are meaningful. |

A chunk is re-planned every `n_action_steps` policy steps (20 control steps at
the defaults). `ActionChunk.step_period_s` is checked but the runner uses the
configured hold count; a mismatch is a configuration error, not a silent change.

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
  follow in the same packet. No pickle.
- Handshake: protocol version, checkpoint metadata SHA-256, chunk size, fps,
  expected cameras, state/action dimensions. A mismatch is refused before any
  model is loaded.
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
- The fake backend is kinematic; success rates there are not MuJoCo numbers.
- Real-checkpoint evaluation was **NOT RUN** in Task 9; policy quality is not
  claimed.
- `REQUEST_RETRAIN` is a reserved decision; Task 12 will orchestrate any
  retraining.
- The checkpoint used for the pipeline test is a LoRA run that stopped at 1,000
  of the 2,000-step bound with a constant learning rate (no scheduler), so it is
  not a fully converged policy.
- One renderer per dataset: use llvmpipe for the real model unless the D3D12
  choice is explicitly approved and recorded.
