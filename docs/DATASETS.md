# Datasets

## Status

**Task 4 staging and Task 5 `curated_v1` filtering exist.** The scripted expert
writes raw rollouts to an atomic staging directory. `trajectory_filter_node` and
`scripts/filter_staging.py` validate, measure, and accept/reject them, then write
a versioned `curated_v1` dataset and a rejection log. LeRobot loader validation
and dataset mixing remain Task 7. `data/` is ignored by git except
`data/.gitkeep`.

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
                             success metrics, metadata_json
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
- **No images by default.** `staging.capture_images: false`. When enabled, the
  intended format is uint8 RGB and float16 depth at the configured decimation
  and resolution. Per-episode image capture is **not wired into the runner yet**;
  `estimate_storage()` prints an upper-bound disk/time estimate before a run.
- **No video yet.** Images become video in the Task 7 LeRobot conversion; no
  encoding is chosen here.

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

## Planned work

| Item | Task | Status |
| --- | --- | --- |
| Trajectory metrics (dimensionless squared jerk, SPARC, path length, limit margins, collisions) | Task 5 | **implemented** |
| Acceptance engine and `trajectory_filter_node` | Task 5 | **implemented** |
| `curated_v1` on-disk episode format | Task 5 | **implemented (LeRobot-style, unvalidated)** |
| LeRobot dataset export and loader validation | Task 7 | planned |
| Open-X / LeRobot dataset mixing | Task 6 | planned |
| Dataset statistics and provenance | Task 6 | planned |

## Storage policy

- No dataset, recording, or exported episode is committed to this repository.
- `RolloutEpisode` does not embed images. Raw RGB-D frames stay on disk and are
  referenced from `metadata_json` (bounded to 64 KiB).
- `data/**` is ignored by git; only `data/.gitkeep` is tracked.
- Dataset schemas are versioned (`curated_v1`, then a new schema name on a
  breaking change). Each filter run writes a new minor version.
- Ask before a run estimated above 1 GiB or 10 minutes of image capture.
