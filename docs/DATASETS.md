# Datasets

## Status

**Task 4 staging, Task 5 `curated_v1` filtering, and Task 7 LeRobot export
exist.** The scripted expert writes raw rollouts to an atomic staging directory.
`trajectory_filter_node` and `scripts/filter_staging.py` validate, measure, and
accept/reject them, then write a versioned `curated_v1` dataset and a rejection
log. Task 7 adds the two-layer dataset builder, official LeRobot v3 writer,
replay-image references, normalization statistics, an episode split, a
deterministic mixture sampler, and a fixture-first Bridge RLDS adapter.
`data/` is ignored by git except `data/.gitkeep`.

## Task 4 staging layout

```
data/staging/<run_id>/
  episode_<episode_id>.npz   arrays: step_index, stamp_ns, joint_positions (N,7),
                             joint_velocities (N,7), ee_pose (N,4,4),
                             action (N,8: 7 joint targets + gripper),
                             gripper_position (N), unintended_collision (N,)
  episode_<episode_id>.json  sidecar: ids, seed, backend, instruction, task ids,
                             outcome, failure_reason, perturbation, expected
                             quality, transitions, randomization report,
                             success metrics, final object poses, metadata_json
  manifest.jsonl             {episode_id, seed, outcome, path, sha256, written_at}
```

Properties:

- **Atomic.** Arrays and sidecar are written to temporary files, fsynced, then
  `os.replace`d. A crash before the rename leaves no final file. The manifest is
  appended and fsynced after both renames.
- **Verifiable.** `manifest.jsonl` records the NPZ SHA-256. Readers recompute it
  and skip corrupt/incomplete entries with a logged error.
- **Duplicate-safe.** Re-writing an existing `episode_id` raises
  `DuplicateEpisodeError`; `overwrite=True` replaces the files and keeps exactly
  one manifest row.
- **No images by default.** `staging.capture_images: false`. The runner does not
  capture images. `scripts/render_replay.py` replays a staged episode offline
  and writes PNG frames under `<run>/images/<episode_id>/`; per-episode image
  capture during a rollout is **not wired in**, and images are not part of the
  `curated_v1` write yet (Task 7).
- **No video yet.** The replay prototype writes PNG frames. Images become video
  or dataset tensors in Task 7; no encoding is chosen here.

## Replay image references (Task 6b)

`scripts/render_replay.py --staging <run> [--episode-id <id>] [--decimation N]`
rebuilds the scene through the same reset path, steps the recorded actions, and
writes one PNG per selected step and camera. It:

- prints a frame/byte/minute estimate first (RGB uint8 only, depth OFF);
- refuses an estimate above **1 GiB or 10 minutes** without `--i-approved`;
- never overwrites an existing image file;
- adds an `images` list to the episode sidecar and the manifest row:
  `{step_index, camera, path, sha256, skipped}`;
- records final object poses in the sidecar (`final_object_poses`) for the
  fidelity check, and compares them with the recorded values.

The `images` shape is the mapping the `curated_v1` writer accepts; the writer is
**not** changed in Task 6b. Design, measured render costs, and open questions
are in `docs/IMAGE_REPLAY.md`.

## `curated_v1` layout (Task 5)

```
data/curated/<dataset_name>/v<MAJOR.MINOR.PATCH>/
  meta/info.json                 schema_version, dataset/version, fps, robot_type,
                                 feature table (dtype/shape/units/frame),
                                 normalisation placeholders, filter config hash
                                 and copy, source run ids, creation time, code
                                 version, images section
  meta/episodes.jsonl            episode_index, episode_id, seed, task, length,
                                 outcome, perturbed flag, FilterMetrics, decision
  meta/tasks.jsonl               task_index, instruction
  dataset_card.md                what it is, units, limitations, images warning
  data/chunk-000/episode_000000.parquet
                                 timestamp, frame_index, episode_index, index,
                                 task_index, observation.state (8),
                                 action (8), next.done
```

Parquet columns are **LeRobot-STYLE**: names and shapes follow LeRobot
conventions, but this layout is **not validated against the LeRobot loader**
until Task 7. `observation.state` and `action` are fixed-size lists of 8:
seven arm joints in radians plus one gripper finger joint in metres. `index` is
a dataset-global frame index; `next.done` is true only on an episode's last row.

Writer rules:

- writes into a `.partial-*` directory, then atomically renames it;
- refuses to overwrite an existing version;
- a failure mid-write removes the partial directory and leaves no final version;
- each write creates a **new minor version** containing the previous accepted
  episodes plus the new ones (a complete snapshot, never an in-place patch);
- episode indices stay contiguous and duplicate episode ids are refused;
- images are never copied. Staging has no images by default, so `info.json`
  records `images: {"mode": "none", ...}` and `dataset_card.md` says prominently
  that the dataset **cannot train a VLA** until RGB-D frames are rendered later.

Rejection log:

```
data/rejected/<run_id>/rejections.jsonl   one row per rejected/invalid episode:
                                          episode id, seed, decision, structured
                                          reasons {code, metric, value,
                                          threshold}, metrics, validation issues
data/rejected/<run_id>/episodes/<id>.json diagnostic copy of the episode reference
```

Staging, curated, and rejected directories are separate. An accepted episode
never appears in the rejection log and vice versa; a test enforces this.

## Filtering commands

```bash
uv run python scripts/filter_staging.py \
    --staging data/staging/hold_fake \
    --config configs/filtering/default.yaml \
    --curated data/curated --rejected data/rejected \
    --dataset-name dev_curated
uv run python scripts/summarize_dataset.py data/curated/dev_curated
cat data/curated/dev_curated/v*/meta/info.json
head -n 3 data/rejected/*/rejections.jsonl
```

Use `configs/filtering/mujoco.yaml` for MuJoCo data. Thresholds and their
provenance are documented in `docs/FILTER_METRICS.md`.

## Sizes measured on 2026-11-01

| run | episodes | steps | disk |
| --- | --- | --- | --- |
| fake, 20 episodes, 30% perturbed | 20 | 1,538 | **0.733 MiB** |
| MuJoCo, 20 episodes, 30% perturbed | 20 | 1,518 | **0.727 MiB** |

That is roughly 37-38 KiB per episode without images. A 10,000-episode staging
run is therefore about **0.36 GiB** of arrays/metadata. Image capture dominates
by orders of magnitude; use `estimate_storage()` before enabling it.

## Commands

```bash
# Collect (fake or mujoco; mujoco runs physics without rendering)
uv run python scripts/collect_rollouts.py --backend fake --num 20 --seed 0 \
    --perturbed-fraction 0.3 --out data/staging/dev_run
uv run python scripts/collect_rollouts.py --backend mujoco --num 20 --seed 0 \
    --perturbed-fraction 0.3 --out data/staging/dev_run_mujoco

# Counts by outcome/perturbation, disk size, manifest verification
uv run python scripts/summarize_staging.py data/staging/dev_run/<run_id>

# ROS collector (UNIFORM only)
ros2 run sim2real_sim expert_collector_node --ros-args -p backend:=fake
ros2 service call /sim2real/generate_rollouts \
  sim2real_msgs/srv/GenerateRollouts \
  "{num_rollouts: 3, seed: 1, task_instruction: 'pick up the red cube and place it in the left zone', sampling_mode: 0, include_perturbed: true}"
ros2 topic echo --no-arr --once /sim2real/staging_episode
```

## Limitations

- Staging holds **unfiltered** episodes, including intended failures and
  timeouts. Curated output is the filtered subset; staging is not training data.
- Outcomes are measured by the expert's success rule (zone xy, rest height,
  stable steps, gripper open). Trajectory metrics are Task 5 and are documented
  separately.
- Per-step `unintended_collision` is a coarse mapping of the backend contact
  report. A Task 4 bug makes it true whenever **any** contact exists, including
  the intended cube↔finger grasp; see `docs/FILTER_METRICS.md`. Expert code was
  not changed in Task 5.
- The MuJoCo clean success rate measured so far is 7/10 (seeded smoke) and 9/20
  with 30% perturbation. Do not assume a high-quality dataset from this expert
  without filtering and inspection.
- `metadata_json` is capped at 64 KiB; large per-episode extras do not fit.
- No provenance hash of code/config yet beyond `run_id`, seeds, and the
  randomization report.

## Task 7 — LeRobot v3 export and Open X adapter (2026-11-02)

### Primary-source verification

Fetched on 2026-11-02. These are the facts used by the implementation; nothing
below is inferred from a memory of an older LeRobot release.

| Item | Verified finding | Source |
| --- | --- | --- |
| LeRobot release | **0.6.1** (published 2026-08-03) | <https://pypi.org/pypi/lerobot/json> |
| Dataset format | **v3.0** (`CODEBASE_VERSION = "v3.0"`) | <https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/datasets/dataset_metadata.py> |
| Import paths | `from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata` | <https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/datasets/__init__.py> |
| Creation API | `LeRobotDataset.create(repo_id, fps, features, root=..., robot_type=..., use_videos=...)`; `add_frame(frame)`; `save_episode()`; `finalize()`; reload with `LeRobotDataset(repo_id=..., root=...)` | <https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/datasets/lerobot_dataset.py> |
| Feature dtypes | NumPy dtype strings (`float32`, `int64`, ...), `image`, `video`, `string`, `language`; fixed shapes become scalars/sequences/arrays; names cannot contain `/` | <https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/datasets/feature_utils.py> |
| Images vs video | `image` stores PNG frames under `images/`; `video` stores MP4 shards under `videos/`. This build uses PNG images (`use_videos=False`). | <https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/datasets/dataset_writer.py> |
| Codec requirements | `lerobot[dataset]` pulls `av` and `torchcodec`; video encoding uses PyAV/FFmpeg. Image mode was used here, so MP4 encoding was not exercised. | <https://pypi.org/pypi/lerobot/0.6.1/json> |
| Metadata | `meta/info.json` (schema/fps/features/path templates), `meta/stats.json`, `meta/tasks.parquet`, `meta/episodes/chunk-*/file-*.parquet`, data `data/chunk-*/file-*.parquet`, video `videos/<camera>/chunk-*/file-*.mp4` | <https://github.com/huggingface/lerobot/blob/v0.6.1/docs/source/lerobot-dataset-v3.mdx> |
| Statistics | Native keys `mean`, `std`, `min`, `max`, `count`, `q01`, `q10`, `q50`, `q90`, `q99`; quantiles are histogram-based per episode and aggregated | <https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/datasets/compute_stats.py> |
| Python/torch | Python `>=3.12`; torch `>=2.7,<2.12`; torchvision `>=0.22,<0.27`; NumPy `>=2.0,<2.3` | <https://pypi.org/pypi/lerobot/0.6.1/json> |
| Open X / Bridge slug | The official GCS bucket lists **`bridge`** under `bridge/0.1.0_$folder$`; it does **not** list `bridge_orig`. The registered dataset name in the official spreadsheet is `bridge`. | <https://storage.googleapis.com/storage/v1/b/gdm-robotics-open-x-embodiment/o?prefix=bridge>; <https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g/export?format=csv> |
| Bridge size/rate | **25,460 episodes**, **387.49 GB** in the Open X spreadsheet, 5 Hz, 4 RGB + 1 depth + 1 wrist camera, EEF-position actions. The raw BridgeData V2 release is 60,096 trajectories at 5 Hz and 640x480. | spreadsheet above; <https://rail-berkeley.github.io/bridgedata/> |
| Bridge RLDS layout | `steps.observation.image_0..image_3`, optional `depth_0`, `observation.state` (7,), `action` (7,), `is_first`, `is_last`, `language_instruction`. The dlimp builder documents state `[XYZ, roll-pitch-yaw, gripper]` and action `[XYZ delta, roll-pitch-yaw delta, gripper absolute]`; it appends a zero action at the end. | <https://github.com/kvablack/dlimp/blob/main/rlds_converters/bridge_dataset/bridge_dataset_dataset_builder.py> |
| Bridge action frame | The upstream controller's `action2transform_local` says the action axes are **the same as world** and rotation is about the current EEF position. `transform2state` stores xyz in that frame and rpy relative to the controller default rotation. A base-frame-vs-world transform is **not** independently verified here; do not guess one. | <https://github.com/rail-berkeley/bridge_data_robot/blob/main/widowx_envs/widowx_envs/utils/transformation_utils.py> |
| Bridge license | Data and other materials: **CC BY 4.0**; code in the Open X repo is Apache-2.0. Treat Bridge as research/auxiliary data and do not redistribute converted data in this repository. | <https://rail-berkeley.github.io/bridgedata/>; <https://github.com/google-deepmind/open_x_embodiment> |

`bridge_orig` is a common name in third-party LeRobot conversions and an older
working note. It is **not** the official Open X bucket name. The adapter's
default config uses `bridge`; the source class is still named
`RLDSEpisodeSource` because the format, not the slug, is what matters.

### Builder layers

```
src/sim2real/dataset_builder/
  schema.py          UnifiedFeatureSpec, UnifiedEpisode, ImageFrame
  embodiments.py     EmbodimentSpec / ActionSpec from configs/dataset/embodiments.yaml
  adapters/curated_v1.py   curated parquet + replay PNG refs + ee_delta derivation
  adapters/rlds_bridge.py  abstract EpisodeSource, tiny fixture source, guarded TFDS source
  stats.py           streaming mean/std, exact or reservoir quantiles
  split.py           deterministic per-source train/val split
  mixing.py          manifest-based, resumable deterministic sampler
  validation.py      schema, index, image-policy, and mixture checks
  lerobot_writer.py  official LeRobot creation API only (guarded import)
configs/dataset/{default,embodiments}.yaml
configs/dataset/mixtures/example.yaml
```

The core and adapters are NumPy-only. `lerobot_writer.py` is the only module
that imports LeRobot/torch; it is tested with the `ml` marker. `.venv-ml` is
created only by `scripts/setup_envs.sh --with-ml`, uses CPU-only torch, and has
no ROS or system site-packages.

### Embodiment table

| embodiment | robot | fps | state | actions | cameras | license |
| --- | --- | --- | --- | --- | --- | --- |
| `panda_sim` | Franka Panda simulation | **10 Hz** (`20 / 2`) | 8: `panda_joint1..7` rad + finger opening m | `joint_position` 8 absolute targets; derived `ee_delta` 7 | `overhead`, `wrist` | Apache-2.0 (repo) |
| `widowx_bridge` | WidowX 250 / BridgeData V2 | **5 Hz** | 7: xyz m + rpy rad + gripper controller units | `bridge_ee_delta` 7: xyz delta m + rpy delta rad + absolute gripper | `third_person`, `side_1`, `side_2`, `wrist` | CC-BY-4.0 |

Panda `observation.state` and `action` keep the gripper in **metres** because
`TrajectoryStep.gripper_position` is the measured finger opening
(`0.0` closed, `0.04` open in `configs/expert/default.yaml`). The derived
`ee_delta` action normalizes the gripper to `0..1` using that span, and the last
step is a zero delta with the last gripper value. Its translation and rotation
vector use the recorded `ee_pose` and world axes:
`[dx,dy,dz] = p[t+1]-p[t]`, `rotvec(R[t+1] R[t]^T)`.

Bridge's exact gripper normalization is not independently verified; the adapter
does not convert it. State/action frames are recorded in the docs above rather
than silently transformed.

### Decimation and rate decision

Default `decimation: 2`. State, action, and image frames are selected at the
**same** source control steps (`0, 2, 4, ...`), and the stored fps is derived as
`control_rate_hz / decimation`. Panda is therefore **10 Hz**. This is a config
key; the writer never hardcodes 10. Consequence for Task 8: horizon in seconds =
`chunk / fps`; a 50-step chunk at 10 Hz spans **5 s**, longer than a typical
90-step/4.5 s episode, so padding and masking are mandatory rather than optional.

### Missing values, boundaries, and alignment

- Curated episodes are taken in curated seed order; `max_episodes` is required
  and bounded.
- Empty task strings are rejected, never silently written.
- Curated episodes with no replay images are an error unless
  `allow_imageless: true`; then the dataset is flagged `images: none`.
- Replay references must match the selected steps and their SHA-256. A missing
  image, mismatched digest, mixed renderer, or replay joint/object error above
  the configured tolerance rejects the episode.
- RLDS boundaries use `is_first`/`is_last` (and optional `is_terminal`). A
  malformed first/last flag skips the episode with a reason.
- Empty/missing Bridge language is either skipped or filled with the configured
  default; both counts are reported.
- NaN/inf state or action drops the episode with a reason. Missing Bridge camera
  images are skipped by default, or zero-filled only when explicitly configured.
- Images are resized with a pure-NumPy nearest-neighbour path, then converted to
  uint8. No TensorFlow is needed by the fixture source or CI.

### Statistics and normalization

`stats.py` computes `mean`, `std`, `min`, `max`, `count`, `q01`, `q10`, `q50`,
`q90`, `q99` per feature and per dimension. Mean/std use a Chan/Welford merge;
quantiles are exact for small data or reservoir-sampled for large data. Constant
dimensions receive a small positive std floor (`1e-8`) so no downstream
normalizer divides by zero. NaN input is an error.

On the built two-episode demo dataset, our mean/std agree with LeRobot's native
stats to `1.2e-7` / `1.8e-6`; our exact q01/q99 differ from LeRobot's
episode-aggregated histogram quantiles by up to `0.064`. LeRobot's
`meta/stats.json` is authoritative for training; our stats are a cross-check and
are written by `scripts/compute_stats.py`.

### Split and mixture

`split.py` uses a stable hash of `(seed, source, original_episode_id)`, sorts
within each source, and takes a deterministic validation count. Train and
validation are disjoint at the episode level. The split is stored in
`meta/sim2real_split.json` inside the LeRobot dataset; the loader itself sees
all episodes, so Task 8 must pass the split indices/filters explicitly.

`mixing.py` requires an explicit manifest. It supports `frames` or `episodes`
units, weights, optional `max_fraction`, replacement, deterministic seeding, and
save/restore of RNG state, position, counts, and the seen set. Unknown ids,
non-positive weights, empty sources, and weights that do not sum to one without
`normalize: true` are errors. Mixed action semantics are allowed only when the
manifest says `allow_mixed_action_semantics: true`; the sampler emits
`(source_id, episode_index, frame_index)` references and never concatenates
action columns.

### Renderer and image policy

The builder accepts **one renderer per dataset** and records the renderer,
GL environment settings, resolution, decimation, fps, cameras, image mode,
replay script SHA-256, replay error maxima, and source/embodiment summary in
`meta/sim2real_build.json`. `scripts/render_replay.py` now records its renderer
label and replay fidelity (`max_joint_error_rad`, `max_object_error_m`) in the
staging sidecar and manifest. Depth remains off by default. Images are the
project PNG format; no second renderer or image format was invented.

Default build size is `max_episodes: 40`. The builder prints image-frame count,
estimated image GiB, and estimated replay minutes before writing. A replay run
above 1 GiB or 10 minutes still needs `--i-approved` at the replay script. This
session used a 2-episode demo only.

### Task 7 runtime evidence (2026-11-02)

- `.venv-ml`: Python 3.12, `lerobot==0.6.1`, `torch==2.11.0+cpu`,
  `torchvision==0.26.0+cpu`, `torchcodec==0.11.1+cpu`, `av==15.1.0`,
  `datasets==4.8.5`, 1.5 GiB installed.
- MuJoCo training generation: **200 generated / 131 successful / 123
  accepted** with `configs/filtering/mujoco.yaml` and the default 0.25
  perturbation fraction. Clean-success accepted: **114/117**.
- Accepted initial red-cube x: min **0.3601 m**, median **0.4351 m**, max
  **0.5371 m**; accepted data skews toward the easier lower-x poses.
- Replayed the first two accepted episodes at decimation 2: **184 PNG frames**,
  **0.034 GiB**, max joint error **0.0 rad**, max object error **0.0 m**.
- Built `data/lerobot/train_v1_imagedemo`: **2 episodes, 92 frames, 10 fps, 4.7 MiB**,
  image mode `image`, one task, LeRobot stats include q01/q99.
- Bridge fixture: **5 available / 3 converted**, one language fill, one NaN
  drop, one missing-wrist drop; built `data/lerobot/bridge_fixture` at 5 fps.
- Replay determinism was extended from the earlier 38 frames (decimation 5) to
  **182/182 frames of episode 7000 at decimation 1**, two replays, identical
  SHA-256, max joint/object error 0.0.

### Licenses and redistribution

| source | license | use here |
| --- | --- | --- |
| sim2real repository / Panda simulation | Apache-2.0 | generated synthetic data |
| LeRobot 0.6.1 | Apache-2.0 | writer dependency |
| BridgeData V2 / Open X `bridge` | CC BY 4.0 (materials); Apache-2.0 (Open X code) | auxiliary fixture only; no real Bridge data downloaded or committed |
| Open X-Embodiment aggregation | CC BY 4.0 (materials) | metadata/research only |

No dataset, image, weight, or converted Bridge artifact is committed. `data/**`
is git-ignored. Any real Bridge conversion must stay under `data/` and carry the
CC BY 4.0 attribution.

## Planned work

| Item | Task | Status |
| --- | --- | --- |
| Trajectory metrics (dimensionless squared jerk, SPARC, path length, limit margins, collisions) | Task 5 | **implemented** |
| Acceptance engine and `trajectory_filter_node` | Task 5 | **implemented** |
| `curated_v1` on-disk episode format | Task 5 | **implemented (LeRobot-style, unvalidated)** |
| LeRobot dataset export and loader validation | Task 7 | **implemented** (official v3 API; 2-episode real dataset) |
| Open-X / LeRobot dataset mixing | Task 7 | **implemented** (fixture-first Bridge adapter + mixture sampler) |
| Dataset statistics and provenance | Task 7 | **implemented** (stats, split manifest, build metadata) |
| Video encoding path | Task 7 | **not exercised**; image mode only |
| Real Bridge subset | Task 7 | **not downloaded**; fixture only |

## Storage policy

- No dataset, recording, or exported episode is committed to this repository.
- `RolloutEpisode` does not embed images. Raw RGB-D frames stay on disk and are
  referenced from `metadata_json` (bounded to 64 KiB).
- `data/**` is ignored by git; only `data/.gitkeep` is tracked.
- Dataset schemas are versioned (`curated_v1`, then a new schema name on a
  breaking change). Each filter run writes a new minor version.
- Ask before a run estimated above 1 GiB or 10 minutes of image capture.
