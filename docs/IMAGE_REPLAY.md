# Image replay rendering (Task 6b prototype)

## Why replay

Rollouts do not render images: MuJoCo under llvmpipe costs about **0.2 s per
camera frame**, which would dominate collection. The scripted expert is
deterministic, so a staged episode can be replayed later and rendered offline.
This document defines the contract, the algorithm, the measured cost, and the
open questions for Task 7.

## Determinism contract

A replay is faithful only if all of the following match:

1. the same backend and simulator version (MuJoCo 3.17.0) and the same renderer
   (llvmpipe or the opt-in D3D12/NVIDIA path);
2. the same reset path: `seed`, `episode_index`, randomization profile, and the
   derived randomization seed (`derive_episode_seed(seed, episode_index)`);
3. the same recorded action sequence (`action[:7]` joint targets and
   `action[7]` gripper target) applied at the recorded control rate;
4. no stochastic physics beyond what the seed controls (the scene has none).

The replay script compares, for every step, the replayed joint positions with
the recorded `joint_positions`, and at the end the replayed object poses with the
staged `final_object_poses`. It reports the maximum absolute joint error (rad)
and the maximum object translation error (m). Two replays of the same episode
must produce byte-identical PNGs on the same renderer.

Measured prototype (seeds 7000-7002, both cameras): **max joint error 0.0 rad**,
**max final object error 0.0 m**. The earlier “38/38 identical hashes” covered
the decimation-5 selection for one episode, not every frame. On 2026-11-02 the
check was extended to **182/182 frames of episode 7000 at decimation 1** (two
overhead/wrist replays, same llvmpipe renderer); all SHA-256 values matched and
both replays reported 0.0 rad / 0.0 m error. The earlier 38/38 result is
therefore retained only as the narrower prototype check.

## Algorithm

```
open the staging run and select episode records
validate seed, episode_index, recorded steps, and finite 8-vector actions
estimate frames = ceil(steps / decimation) * cameras
refuse if > 1 GiB or > 10 min unless --i-approved
reset_environment(seed=derive_episode_seed(seed, episode_index),
                  profile=record.randomization_report["profile"],
                  episode_index=record.episode_index)
for step_index in 0 .. len(steps)-1:
    if step_index > 0: backend.step(action[:7], action[7])
    if step_index % decimation == 0:
        for camera in (overhead, wrist):
            write PNG under <run>/images/<episode_id>/<step>_<camera>.png
            record path + sha256 (never overwrite an existing file)
compare final joint state and object poses
update the episode sidecar and manifest with image references, renderer,
and replay error fields
```

Step 0 is the post-reset observation; frame timestamps are
`step_index / 20 Hz`. Decimation 2 keeps every second control step.

The PNG writer is the same stdlib implementation used by
`scripts/debug_keyframes.py` (`sim2real.expert.images.write_png`); there is no
second image format. Depth is **off** by default (`float16` would add 2 bytes
per pixel if enabled later).

## Measured render cost (2026-11-02, `scripts/bench_render.py --frames 3`, run alone)

| Renderer | overhead ms/frame | wrist ms/frame | peak RSS |
| --- | --- | --- | --- |
| llvmpipe (default) | **196.5** | **138.7** | 868,248 KiB |
| D3D12 + NVIDIA RTX 4050 (opt-in) | **54.7** | **56.4** | 444,256 KiB |

The D3D12 path is opt-in only, never a default, and never used by CI.

## Cost model

Measured prototype (3 episodes, 548 frames at decimation 1, both cameras):

| Renderer | per episode | frames | RGB size | wall time |
| --- | --- | --- | --- | --- |
| llvmpipe | 35.8 / 31.9 / 30.7 s | 548 | 0.100 GiB | 98.4 s |
| D3D12/NVIDIA | 17.5 / 16.2 / 17.2 s | 548 | 0.100 GiB | 50.9 s |

Formulas (not guesses): a frame is `256 x 256 x 3 = 196,608` bytes (0.1875 MiB);
an episode is about 90 control steps, so decimation 1 is about 180 frames and
decimation 2 about 90 frames. Using the measured per-episode time:

| Scenario | llvmpipe dec 1 | llvmpipe dec 2 | NVIDIA dec 1 | NVIDIA dec 2 |
| --- | --- | --- | --- | --- |
| 1 episode | ~33 s | ~16 s | ~17 s | ~8.5 s |
| 100 episodes | ~55 min | ~27 min | ~28 min | ~14 min |
| 500 episodes | ~4.6 h | ~2.3 h | ~2.4 h | ~1.2 h |
| 100 episodes storage | ~3.3 GiB | ~1.7 GiB | ~3.3 GiB | ~1.7 GiB |
| 500 episodes storage | ~16.7 GiB | ~8.3 GiB | ~16.7 GiB | ~8.3 GiB |

The script's estimate uses a conservative **0.25 s per camera frame** (0.5 s per
step) and therefore prints a larger wall time than the measured llvmpipe run.
The hard refusal thresholds are 1 GiB and 10 minutes without `--i-approved`.

## Staging fields and references

A staged episode already contains `seed`, `episode_index`,
`randomization_seed`, `randomization_report` (including the profile and sampled
object poses), the per-step actions, `control_rate_hz`, and the task ids.
Task 6b adds `final_object_poses` to the sidecar for the fidelity check; old
records still replay but report `max_object_error_m = null`.

Images live under `<run>/images/<episode_id>/<step>_<camera>.png`. The episode
sidecar and the manifest row gain an `images` list:

```json
{"step_index": 0, "camera": "overhead",
 "path": "images/run_ep0000/0000_overhead.png",
 "sha256": "…", "skipped": false}
```

This is the `images` mapping shape accepted by the `curated_v1` writer; the
writer is **not** changed in Task 6b. Existing image files are never
overwritten.

## Alignment

- The control rate is 20 Hz; one recorded step is 50 ms.
- Frame `k` is control step `k`; at decimation 1 the video is 20 fps, at
  decimation 2 it is 10 fps.
- The initial frame is the post-reset observation, so a 91-step episode has 91
  frames per camera at decimation 1.
- No cross-camera synchronization is needed: both cameras render the same
  post-step state.

## Open questions for Task 7

1. Which image size and cameras should the LeRobot builder use (256x256 both,
   or overhead only)?
2. Frames as PNG references or encoded video? PNG at 100 episodes is already
   about 3.3 GiB; video would reduce storage but needs an encoder.
3. Is depth needed (float16 optional, currently off)?
4. ~~Does LeRobot v3 accept external image references, or must frames be copied
   into the dataset layout?~~ **Answered in Task 7:** the official writer copies
   image bytes into its own PNG layout; our builder reads the replay PNGs and
   passes decoded arrays to `LeRobotDataset.add_frame`.
5. What is the storage/time budget for the first training set, and is the
   opt-in NVIDIA renderer acceptable for offline replay (never CI)?
6. Should the replay store camera intrinsics/extrinsics per episode, or rely on
   the staged randomization report?

## Known limitations

- The replay prototype was run on **3 MuJoCo episodes only** (seeds 7000-7002);
  it is not a production batch.
- Images are PNG frames, not video, and are not integrated into `curated_v1`.
- Depth is off.
- The measured times are for this WSL2 machine; another renderer or machine will
  differ.
