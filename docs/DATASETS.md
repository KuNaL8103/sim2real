# Datasets

## Status

**Task 4 staging exists; `curated_v1` does not.** The scripted expert writes raw
rollouts to an atomic staging directory. Filtering, the `curated_v1` format,
LeRobot export, and dataset mixing remain unimplemented. `data/` is ignored by
git except `data/.gitkeep`.

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
  timeouts. They are not training data until Task 5 accepts them.
- Outcomes are measured by the expert's success rule (zone xy, rest height,
  stable steps, gripper open). There are no trajectory-quality metrics yet.
- Per-step `unintended_collision` is a coarse mapping of the backend contact
  report; MuJoCo cube↔finger contacts are treated as intended.
- The MuJoCo clean success rate measured so far is 7/10 (seeded smoke) and 9/20
  with 30% perturbation. Do not assume a high-quality dataset from this expert
  without filtering and inspection.
- `metadata_json` is capped at 64 KiB; large per-episode extras do not fit.
- No provenance hash of code/config yet beyond `run_id`, seeds, and the
  randomization report.

## Planned work

| Item | Task | Status |
| --- | --- | --- |
| Trajectory metrics (dimensionless squared jerk, SPARC, path length, limit margins, collisions) | Task 5 | planned |
| Acceptance engine and `trajectory_filter_node` | Task 5 | planned |
| `curated_v1` on-disk episode format | Task 5 | planned, schema not defined |
| LeRobot dataset export | Task 6 | planned |
| Open-X / LeRobot dataset mixing | Task 6 | planned |
| Dataset statistics and provenance | Task 6 | planned |

## Storage policy

- No dataset, recording, or exported episode is committed to this repository.
- `RolloutEpisode` does not embed images. Raw RGB-D frames stay on disk and are
  referenced from `metadata_json` (bounded to 64 KiB).
- `data/**` is ignored by git; only `data/.gitkeep` is tracked.
- Dataset schemas will be versioned. `curated_v1` is a placeholder name for
  Task 5, not a committed schema.
- Ask before a run estimated above 1 GiB or 10 minutes of image capture.
