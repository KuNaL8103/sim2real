# VLA fine-tuning (Task 8)

Status: **implemented, smoke-tested, and trained for 1,000 steps on `train_v1`.**
A real LoRA checkpoint exists at
`outputs/task8/train_v1_lora/checkpoint-00001000` with clean provenance
(`git_dirty=false`). Offline L1 on 8 validation episodes beats both trivial
baselines per dimension, but there is **no closed-loop or generalisation
result** yet. The earlier two-step smoke run remains the pipeline check; the
1,000-step run is the Task 8b test subject for Task 9.

Pinned stack: **LeRobot 0.6.1**, dataset format **v3.0**, **torch 2.11.0+cu126**,
**PEFT 0.21.2**, **accelerate 1.15.0**, **transformers 5.5.4**, Python 3.12.3,
driver 551.76 (CUDA 12.4 capability), CUDA runtime 12.6.

## 1. Primary sources and what was verified

The installed code is the authority for the pinned version; the model card and
`config.json` are the authority for the checkpoint.

| Source | Revision / URL | Used for |
| --- | --- | --- |
| LeRobot tag `v0.6.1` | commit `7e241bd630a3719a56157a497ce5d08f244784f1`; <https://github.com/huggingface/lerobot/releases/tag/v0.6.1> | training entrypoint, PEFT integration, processor construction |
| Installed LeRobot | `.venv-ml/lib/python3.12/site-packages/lerobot` (`0.6.1`) | exact code read for every claim below |
| `lerobot/smolvla_base` | revision `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`; <https://huggingface.co/lerobot/smolvla_base> | model config, parameter count, license, camera keys |
| Backbone `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467` | VLM weights/tokenizer loaded by `load_vlm_weights: true` |
| PyTorch wheel index | <https://download.pytorch.org/whl/cu126> | CUDA wheel tested here |

`smolvla_base` files (Hub API, `?blobs=true`): `model.safetensors`
**906,712,520 B (864.7 MiB)**, `config.json` 2,299 B,
`policy_preprocessor.json` 1,872 B + 640 B step file,
`policy_postprocessor.json` 660 B + 640 B step file. The backbone
`model.safetensors` is **2,029,990,624 B (1.89 GiB)**, plus tokenizer/config
files. Total download is about **2.9 GB**; the cache is the standard
`~/.cache/huggingface/hub` (2.9 GB after the smoke).

### Verified SmolVLA facts

| Property | Verified value | Evidence |
| --- | --- | --- |
| Parameters | **450,046,176** (~450M); 446,772,624 BF16 + 3,273,552 F32 | Hub safetensors metadata; matches the measured frozen base count |
| License | **Apache-2.0** | model card / Hub tags |
| `chunk_size` | **50** | `configuration_smolvla.py`, `config.json` |
| `n_action_steps` | **50** | same |
| `max_state_dim` / `max_action_dim` | **32 / 32** | same |
| `normalization_mapping` | VISUAL **IDENTITY**, STATE **MEAN_STD**, ACTION **MEAN_STD** | same |
| Image preprocessing | resize with padding to **512x512**, `[0,1] -> [-1,1]`; SigLIP wants `[-1,1]` | `prepare_images` |
| `freeze_vision_encoder` | **true** | config |
| `train_expert_only` | **true** | config |
| `train_state_proj` | **true** | config |
| Camera keys | `observation.images.camera1`, `camera2`, `camera3` | `config.json` |
| Cameras expected by the base | **3** | `config.json` |
| Base state/action shapes | **6 / 6** | `config.json` |
| Missing-camera handling | `prepare_images` uses present keys; absent keys are only zero-filled when `empty_cameras > 0` | `modeling_smolvla.py` |
| Tokenizer | `pad_language_to: max_length`, `tokenizer_max_length: 48` | config |

**Our dataset has two cameras, the base expects three.** The training config
renames `overhead -> camera1` and `wrist -> camera2`. With `empty_cameras: 0`
the third slot is simply absent; the model runs with two image groups. This is a
deviation from the base model's pretraining and is recorded as a limitation, not
hidden.

### LeRobot's own PEFT entrypoint (it does exist in 0.6.1)

`TrainPipelineConfig` has `peft: PeftConfig | None`. The CLI exposes
`--peft.method_type`, `--peft.target_modules`, `--peft.r`, `--peft.lora_alpha`,
`--peft.full_training_modules`, and `--peft.init_type`. `lerobot_train.py`
calls `policy.wrap_with_peft(peft_cli_overrides=dataclasses.asdict(cfg.peft))`.
SmolVLA supplies a default target regex in
`SmolVLAPolicy._get_default_peft_targets()`; there is no separate LoRA CLI
family, only the `--peft` group.

**PEFT quirk found the hard way:** PEFT 0.21.2 converts a **list** of
`target_modules` to a set and then matches exact names, losing regex semantics.
A list containing the LeRobot default regex fails with
`NoMatchingPeftModuleError`. The fix used here is to resolve the regex against
the loaded model first and hand PEFT the exact full module names.

### LeRobot checkpoint layout vs ours

LeRobot's own training writes
`outputs/train/.../checkpoints/<step>/pretrained_model/` containing the policy
weights (or `adapter_model.safetensors` for PEFT), `config.json`,
`policy_preprocessor.json` + step safetensors,
`policy_postprocessor.json` + step safetensors, and `training_state/`.

Our export is `<output>/checkpoint-XXXXXXXX/` with `adapter/` (PEFT adapter
files, `adapter_config.json`, `lora_targets.json`), `trainer_state.json`, and
`checkpoint_metadata.json`. The metadata is what `scripts/validate_checkpoint.py`
verifies before evaluating.

### Policy preprocessor / postprocessor

`make_smolvla_pre_post_processors` builds:

```
input:  rename_observations -> add_batch_dim -> newline -> tokenizer
        -> to_device -> normalize
output: unnormalize -> to_cpu
```

The real path here uses LeRobot's official `make_pre_post_processors` with the
same overrides as `lerobot_train.py`: the dataset camera rename map and the
dataset's `meta/stats.json` (authoritative). The saved base processor is loaded
and overridden, so normalization is never applied twice.

## 2. Decisions A-C

### Decision A - library API, not the CLI

**Decision:** wrap LeRobot's **library** components (`SmolVLAConfig`,
`make_policy`, `wrap_with_peft`, `make_pre_post_processors`, `LeRobotDataset`)
inside a small deterministic loop; do not spawn `lerobot_train.py`.

**Evidence / reason:** `lerobot_train.py` is a draccus CLI. The library objects
are stable in 0.6.1, but the CLI's split is `dataset.eval_split` (a per-task
fraction), not our stored episode-level `meta/sim2real_split.json`, and it does
not emit our provenance metadata or offline L1 report. Wrapping the library
keeps LeRobot's own preprocessing and PEFT while making the split, checkpoint
metadata, and validation ours. The CLI remains a valid option for long remote
runs (see `docs/REMOTE_TRAINING.md`).

### Decision B - keep chunk 50, settle it in seconds

**Decision:** `chunk_size = 50`, `n_action_steps = 50`, so the horizon is
**50 / 10 Hz = 5.0 s**. Short episodes are padded and masked with `is_pad`.

**Evidence:** `chunk_size` is a runtime sequence length, not a weight shape.
`embed_suffix` builds `att_masks += [1] * chunk_size`, applies a **runtime**
sinusoidal time embedding, and `suffix_out[:, -chunk_size:]`; action tokens use
attention `position_ids` computed from the mask. There is no learned positional
embedding indexed by chunk position, so a smaller chunk is weight-compatible.
`n_action_steps` is separate: `select_action` fills a queue and re-plans only
when it empties, so it is an inference/replanning horizon.

50 was kept because it matches the pretrained regime and the measured peak VRAM
(1.1 GiB) is far below the 6 GiB card. A typical episode is ~4.6 s, so the last
steps of a chunk are padded; `action_is_pad` is mandatory. A smaller chunk
(e.g. 16 = 1.6 s) is a supported config change, not a weight change, but was not
trained here.

### Decision C - one normalization path per tensor

**Decision:** the real SmolVLA path uses LeRobot's pre/post processors and the
dataset's `meta/stats.json`. `src/sim2real/training/normalization.py` is used
only by the fake adapter, offline validation, and tests. Never both.

**Evidence:** the ML test compares our implementation with
`NormalizerProcessorStep` on the same fixture (mean_std and quantile, forward
and inverse); the measured maximum difference is **8.0e-08**. Our formulas copy
LeRobot's exactly, including the `MEAN_STD` inverse using `std` rather than
`std + eps`.

## 3. LoRA target modules and trainable fraction

Resolved against the **loaded model**, not hardcoded:

| Group | Count | Modules |
| --- | --- | --- |
| Action expert | **32** | `model.vlm_with_expert.lm_expert.layers.{0..15}.self_attn.{q,v}_proj` |
| Projections | **5** | `model.state_proj`, `model.action_in_proj`, `model.action_out_proj`, `model.action_time_mlp_in`, `model.action_time_mlp_out` |
| **Total** | **37** | |

Measured after wrapping with `r=32`, `alpha=64`, `dropout=0.05`, `bias=none`:

| Quantity | Value |
| --- | --- |
| Total parameters | **451,531,488** |
| Frozen base parameters | **450,046,176** |
| Trainable LoRA parameters | **1,485,312** |
| Trainable fraction | **0.003289 (0.329%)** |

The vision encoder and VLM backbone stay frozen per the verified
`freeze_vision_encoder`/`train_expert_only` flags. The adapter's
`resolve_target_modules` fails if a configured pattern matches nothing; a
negative check runs in CI via `scripts/list_lora_targets.py --fake-tree`.

## 4. Action chunks, padding, and horizon

`src/sim2real/training/actions.py` is the NumPy reference used by the fake
adapter, offline validation, and tests. It mirrors the real path's semantics:

- state/action vectors are zero-padded to 32 dimensions;
- a chunk past the episode end repeats the last action and sets
  `action_is_pad=True` (including an episode shorter than the chunk);
- `decode_chunk` strips padding back to the native 8 dimensions exactly;
- the horizon is always `chunk_size / fps`, so a 5 Hz source and a 10 Hz source
  are not conflated even if their chunk lengths match.

The real path uses LeRobot's dataset batching and its `action_is_pad`; the
offline validator inverts that mask (`is_pad=True` means **invalid**) before
computing L1.

## 5. Training data (`train_v1`)

Built from the existing accepted episodes, seed order, up to 40:

| Item | Value |
| --- | --- |
| Generated / successful / accepted (full run) | **200 / 131 / 123** |
| Accepted clean / clean-success | **114 / 117** |
| Accepted initial red-cube x (full run) | min **0.3601 m**, median **0.4351 m**, max **0.5371 m** |
| Perturbed-but-accepted (full run) | **9**: detour 4, overspeed 3, jitter 2 |
| Training subset | first **40** accepted episodes |
| Subset perturbations | **4**: detour 3, jitter 1 |
| Subset initial cube x | min **0.3630 m**, median **0.4141 m**, max **0.5104 m** |
| Episodes / frames / fps | **40 / 1,849 / 10** |
| Split (episode-level, seed 123) | **32 train / 8 val** |
| Images | **3,698 PNGs**, 256x256, overhead + wrist, **103 MB** |
| Dataset size | **92 MB** (`data/lerobot/train_v1`) |
| Renderer / decimation | **llvmpipe** / **2** |
| Replay fidelity | max joint **0.0 rad**, max object **0.0 m** |
| Rendering cost | 38 new episodes, 3,514 frames, **650.3 s (10.84 min)** |

The local dataset card is `data/lerobot/train_v1/DATASET_CARD.md` (git-ignored).
The first two episodes already had llvmpipe images from Task 7; the remaining 38
were rendered with the same renderer, so the dataset is single-renderer. No
renderer/GPU training ran concurrently.

## 6. Smoke run (measured)

Command:

```bash
.venv-ml/bin/python scripts/train_vla.py \
  --config configs/training/smolvla_lora.yaml --smoke --max-steps 2
```

| Measurement | Value |
| --- | --- |
| GPU | RTX 4050 Laptop, 6141 MiB; driver 551.76; CUDA runtime 12.6 |
| Steps | **2** |
| Batch size / chunk / cameras | 1 / 50 / 2 |
| Losses | **2.277515411376953 -> 3.060152530670166** (finite; no learning claim) |
| Step time | **5.06 s** first step (warm-up), **0.66 s** second step; steady-state **0.43-0.66 s** across runs |
| Peak VRAM (`torch.cuda.max_memory_allocated`) | **1,117.21 MiB** |
| Peak host RSS (`resource` / `/usr/bin/time`) | **3,807 MiB / 3,906 MiB (~3.7 GiB)** |
| Trainable / total parameters | **1,485,312 / 451,531,488** |
| Checkpoint | `outputs/task8/smolvla_smoke/checkpoint-00000002` |
| Adapter size | **5,952,536 B** (`adapter_model.safetensors`) |

Reproducibility: two runs from the same seed produced **identical losses**
(`[2.277515411376953, 3.060152530670166]`) and the same peak VRAM
(1,117.21 MiB). Step timing varies with warm-up.

The checkpoint's `checkpoint_metadata.json` records `git_dirty: true` because it
was created before the Task 8 commit; that is accurate, not a failure.

## 7. Offline action validation (L1)

`scripts/validate_checkpoint.py` on the smoke checkpoint (8 held-out episodes):

| Quantity | Value |
| --- | --- |
| Windows / valid steps | **366 / 8,561** |
| Horizon | **5.0 s** |
| Per-dimension L1 (native units) | 0.0634, 0.0973, 0.1566, 0.1931, 0.0692, 0.2032, 0.1980, 0.0194 |
| Normalized per-dimension L1 | 0.694, 0.795, 0.993, 0.893, 1.140, 1.416, 0.890, 0.979 |
| Horizon L1 (first 5 positions) | 0.0580, 0.0667, 0.0762, 0.0849, 0.0928 |
| Baseline `hold_last_state` per-dim | 0.0756, 0.1068, 0.1537, 0.2369, 0.0588, 0.1574, 0.2093, 0.0223 |
| Baseline `mean_action` per-dim | 0.0711, 0.0568, 0.1429, 0.1792, 0.0542, 0.1405, 0.1966, 0.0194 |

After two steps the policy is comparable to, not better than, the trivial
baselines. **No generalisation or quality claim is made.** Padded positions are
excluded; different chunk horizons are never compared.

## 8. Configuration and CLI

`configs/training/smolvla_lora.yaml` is the validated default
(`schema_version: 1`). `configs/training/openvla7b_lora.yaml` is explicitly
**not validated** (Llama-derived license and VRAM unverified; the adapter
refuses to run).

```bash
# Core-only (no torch): the loop, checkpoint, and split path
uv run python scripts/train_vla.py --config configs/training/smolvla_lora.yaml \
  --fake --smoke
uv run python scripts/validate_checkpoint.py \
  --checkpoint outputs/task8/fake_smoke/checkpoint-00000005 --fake
uv run python scripts/list_lora_targets.py --fake-tree

# Real model (.venv-ml): inspect and train
.venv-ml/bin/python scripts/list_lora_targets.py \
  --config configs/training/smolvla_lora.yaml
.venv-ml/bin/python scripts/train_vla.py \
  --config configs/training/smolvla_lora.yaml --smoke --max-steps 2
.venv-ml/bin/python scripts/validate_checkpoint.py \
  --checkpoint outputs/task8/smolvla_smoke/checkpoint-00000002 \
  --config configs/training/smolvla_lora.yaml

# ML tests (markers `ml and not gpu`; `--gpu` only after approval)
scripts/run_ml_tests.sh
```

Config validation rejects unknown keys, `r <= 0`, `alpha <= 0`,
`dropout` outside `[0,1)`, empty target modules, an unknown family, `max_steps <
1`, `batch_size < 1`, a missing seed, `n_action_steps > chunk_size`, an
incomplete camera rename map, and an unknown tracking backend, each with a
key path.

## 9. Checkpoint provenance

Every checkpoint writes `checkpoint_metadata.json` with: base model id and
revision, library versions (torch/CUDA, PEFT, LeRobot, accelerate, transformers,
NumPy, Python), GPU name, git commit and dirty flag, config hash and a copy of
the config, dataset root/version/hash, the episode-level split, the mixture
manifest (if any), the stats-file hash, seed, step, LoRA config, camera rename
map, action representation, chunk size, `n_action_steps`, state/action/max dims,
fps, horizon seconds, and creation time.

`load_checkpoint` refuses a mismatched stats hash, action representation,
rename map, or config hash. `scripts/validate_checkpoint.py` prints the verified
fields before evaluating.

## 10. Environment and wheel result

| Item | Result |
| --- | --- |
| CUDA wheel index | **cu126**; `torch==2.11.0+cu126` (830,526,889 B) |
| cu124 | no CPython 3.12 torch satisfying LeRobot `>=2.7,<2.12` (index checked) |
| cu128 | `torch 2.11.0+cu128` exists (820,326,880 B) but was **not tested** |
| Driver | 551.76 (CUDA 12.4); cu126 runtime 12.6 runs via minor-version compatibility |
| Scratch test | 2048x2048 matmul on `cuda`, sum **141957.375**, 40.1 MiB peak |
| `.venv-ml` | now **6.7 GB**: torch 2.11.0+cu126, torchvision 0.26.0+cu126, PEFT 0.21.2, accelerate 1.15.0, transformers 5.5.4, LeRobot 0.6.1, num2words 0.5.14 |
| Extra package | `num2words` 0.5.14 is required by `transformers/models/smolvlm/processing_smolvlm.py` (the processor raises at its `num2words` guard without it); `scripts/setup_envs.sh --with-ml` installs `num2words>=0.5,<1` |
| HF cache | **2.9 GB** under `~/.cache/huggingface/hub` |

`scripts/setup_envs.sh` has the opt-in
`ML_TORCH_BACKEND=cu126 ./scripts/setup_envs.sh --with-ml` lines; CI never
creates or installs `.venv-ml`.

## 11. Task 8b: train_v1 LoRA run (1,000 steps)

This is the checkpoint intended as a Task 9 test subject. It was trained from
clean commit `bf74c84` so the metadata records `git_dirty=false`.

| Setting | Value |
| --- | --- |
| Config | `configs/training/smolvla_lora_train_v1.yaml` |
| Base / revision | `lerobot/smolvla_base` / `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6` |
| Dataset / split | `data/lerobot/train_v1`, 40 episodes, 32 train / 8 val |
| Batch / chunk | **16** / 50 (`n_action_steps=50`) |
| Optimizer / lr | AdamW / **1e-4** |
| Precision | **fp32** (bf16/fp16 were not verified) |
| Seed | **0** |
| `save_every` | **250** |
| Config `max_steps` bound | **2,000** (approved) |
| **Actual steps** | **1,000** (stopped by user instruction) |

### Batch-size measurement (full data, 20 steps each)

| Batch | Peak VRAM | Free VRAM | Median step | Peak RSS |
| --- | --- | --- | --- | --- |
| 4 | 1,733 MiB | 4,408 MiB | 0.374 s | 3,796 MiB |
| 8 | 2,547 MiB | 3,594 MiB | 0.492 s | 3,829 MiB |
| **16** | **4,134 MiB** | **2,007 MiB** | **1.176 s** | **6,096 MiB** |

Batch 16 was the largest tested and left more than 1 GiB free. The realistic
full-data 20-step measurement was median **1.176 s**, mean **1.741 s**; the
2,000-step bound was estimated at 44-58 min, so it was approved.

### Loss curve (mean per 50 steps)

| Steps | Mean loss |
| --- | --- |
| 1-50 | 1.833490 |
| 51-100 | 1.029106 |
| 101-150 | 0.795580 |
| 151-200 | 0.662678 |
| 201-250 | 0.602787 |
| 251-300 | 0.524186 |
| 301-350 | 0.489446 |
| 351-400 | 0.495581 |
| 401-450 | 0.428381 |
| 451-500 | 0.376630 |
| 501-550 | 0.374111 |
| 551-600 | 0.375046 |
| 601-650 | 0.325024 |
| 651-700 | 0.322790 |
| 701-750 | 0.294869 |
| 751-800 | 0.301526 |
| 801-850 | 0.300011 |
| 851-900 | 0.268235 |
| 901-950 | 0.301328 |
| 951-1000 | 0.247039 |

Loss **decreased** from **3.544756** (step 1) to **0.320808** (step 1000);
minimum **0.141453** at step 978; maximum 5.049912. No NaN/abort.

### Runtime and telemetry

| Quantity | Value |
| --- | --- |
| Wall time | **28:53.06** for 1,000 steps |
| Step time | median **1.505 s**, mean **1.474 s**, first **5.234 s** |
| Peak VRAM | **4,377 MiB** (sampled) |
| Peak host RSS | **6,245,720 KiB (5.96 GiB)** (`/usr/bin/time`); sampler max 5,285 MiB |
| GPU start | 54 °C, 2130/7000 MHz, 12.66 W |
| GPU end | 73 °C, 2130/7000 MHz, 16.20 W |
| During run | max **88 °C**; SM clock median **1,185 MHz** (max sample 2,670 MHz); power median 24.0 W, max 79.6 W |

The GPU reached 88 °C with a 55 W board cap and a median SM clock well below the
observed maximum. That is consistent with power/thermal limiting; the exact
throttle reason was not instrumented.

### Checkpoint

| Item | Value |
| --- | --- |
| Path | `outputs/task8/train_v1_lora/checkpoint-00001000` |
| Metadata | 8,491 B, SHA-256 `b87f5094d414d97590166ba8691989dd930b21b4c166ddf5de4a4f28234422e3` |
| Adapter | `adapter_model.safetensors`, **5,952,536 B** |
| Whole checkpoint | **5.70 MiB** |
| Provenance | `git_commit=bf74c84`, **`git_dirty=false`**, stats hash `a25f62ab...` |
| Other checkpoints | 250, 500, 750 |

### Offline L1 (same 8 val episodes, 366 windows, 8,561 valid steps)

| Dim | Trained | Smoke | Hold-last | Trained beats hold by | Mean-action | Trained beats mean by |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.020793 | 0.063401 | 0.075608 | 0.054815 | 0.071098 | 0.050305 |
| 1 | 0.027214 | 0.097294 | 0.106815 | 0.079601 | 0.056781 | 0.029567 |
| 2 | 0.031092 | 0.156631 | 0.153720 | 0.122628 | 0.142897 | 0.111805 |
| 3 | 0.078327 | 0.193076 | 0.236889 | 0.158562 | 0.179172 | 0.100845 |
| 4 | 0.012209 | 0.069209 | 0.058824 | 0.046615 | 0.054167 | 0.041958 |
| 5 | 0.057156 | 0.203211 | 0.157372 | 0.100216 | 0.140472 | 0.083316 |
| 6 | 0.041395 | 0.197957 | 0.209320 | 0.167925 | 0.196635 | 0.155240 |
| 7 | 0.007821 | 0.019423 | 0.022346 | 0.014525 | 0.019413 | 0.011592 |

Normalized per-dimension L1 (trained): **0.2278, 0.2225, 0.1971, 0.3624,
0.2011, 0.3982, 0.1861, 0.3941**.

Horizon-wise, the trained mean over the 47 valid positions is **0.032904**,
versus **0.159288** for hold-last and **0.104959** for mean-action. The policy
beats hold-last at **k=2..46 (45 positions)** but **loses at k=0 and k=1**; it
beats mean-action at every valid horizon.

These are 8-episode offline numbers. They do **not** establish closed-loop
success, robustness, or generalisation.

## 12. What was NOT run

- The remaining **1,000 steps** of the approved 2,000-step bound; the run was
  stopped at checkpoint 1,000 by user instruction.
- OpenVLA-7B (adapter refuses; license/VRAM unverified).
- bf16 or fp16 training (config allows, but only fp32 was measured).
- Gradient checkpointing (flag exists; the stack was not verified to support it).
- cu128 or any other wheel index; a second GPU/driver.
- A remote-GPU run (`docs/REMOTE_TRAINING.md` is documentation only).
- Video/MP4 datasets, real Bridge data, or any Hub upload.
- `wandb`/`mlflow` backends (not installed); only the `none` tracker is exercised.
- Policy quality, closed-loop success, or sim2real transfer.
- Isaac Sim, MoveIt, RViz/viewer checks.

## 13. Limitations

- **40 episodes** (or fewer in other runs); no generalisation claim.
- **10 Hz** and a 50-step chunk span **5.0 s**, longer than a ~4.6 s episode;
  `is_pad` masks are mandatory.
- Accepted data **skews toward easy cube poses** (full-run median x 0.4351 m).
- **9 accepted episodes are perturbed** (detour/overspeed/jitter); they pass the
  filter and are included unless `exclude_perturbed: true`.
- `collision_count` is **not** a safety signal (`max_collisions: 3` is a
  workaround), so it is not used by training.
- The base expects **3 cameras**; this dataset supplies 2, and the third slot is
  absent (`empty_cameras: 0`).
- `n_action_steps = 50` matches pretraining but is not a tuned replanning
  horizon; Task 9 should choose it for closed-loop execution.
- The smoke checkpoint was written from a **dirty** tree (`git_dirty: true`).
- Two steps prove the pipeline runs; they do not prove learning.
- **The 1,000-step run is not the full approved 2,000-step bound.** The
  committed config records 2,000; the checkpoint metadata records the actual
  step (1,000). Do not describe it as a completed 2,000-step run.
- **Offline L1 is from 8 validation episodes only (366 windows).** The trained
  checkpoint beats both baselines per dimension, but hold-last is better at
  chunk positions k=0 and k=1. No closed-loop or generalisation claim.
- **The GPU reached 88 °C** with a median training SM clock of 1,185 MHz versus
  a 2,670 MHz maximum sample; the exact throttle reason was not instrumented.

## 14. Task 9d measured training plan (document only, not run)

The Task 9d diagnosis (see `docs/EVALUATION.md`) changes the next training plan
in four ways: the 9b "success" is a start-in-success-region false positive; the
evaluation execution contract is now `linear`; the overfit run was step-limited
at 210 MHz and did not converge; and instruction sensitivity is weak.

**Scope.** Start with **one instruction**: `pick up the red cube and place it in
the left zone`, randomized object poses. Do not train all six instructions
until a single-instruction policy shows a measurable instruction-conditioned
response and closed-loop progress.

**Episodes and arithmetic.** The MuJoCo expert succeeded **24/30 = 0.80** on
the evaluation seeds; the accepted-data filter accepted **123/131** successful
Task 7 episodes, and clean acceptance was **114/117**. Excluding start-in-region
spawns removes about **1/30 = 3.3%**. A 40-episode target therefore needs roughly
`40 / (0.80 * 0.94 * 0.97) ≈ 55` clean attempts; budget **70** attempts to allow
for the expert's ~20% failure rate and the observed easy-pose skew. This is an
estimate, not a guarantee.

**Steps and wall time.** The measured Task 9d batch-4 step is **2.99 s** at
210 MHz; batch 16 was **1.5 s** in 9b at ~1,185 MHz. At the current clock,
2,000 steps at batch 4 is about **100 min**; if the GPU clock is restored,
2,000 steps at batch 16 is about **50 min**, and 5,000 steps about **125 min**.
Choose the step count after the GPU clock is healthy and the overfit test
converges; state the estimate before running.

**Optimization.** Keep batch 16 if VRAM allows (measured 4.1 GiB), lr 1e-4 with
a short warm-up and cosine decay (for example 50 warm-up steps then decay to
10% of the peak over the run), fp32, chunk 50. The constant-LR 1,000-step run
left loss at 0.32; a schedule is the smallest evidence-based change.

**Dataset rate.** Keep the 10 Hz dataset and evaluate with `execution: linear`
(the oracle improves from 24/32 hold to 31/32 linear). A 20 Hz re-export would
double the data and is not justified before a single-instruction policy works.

**Evaluation protocol.** Seeds **20000-20029**, paired scripted expert, linear
execution, `n_action_steps=10`, deterministic sampling, llvmpipe, and report
success both **with and without** start-in-success-region episodes. The paired
expert baseline on those seeds is **24/30** (23/29 excluding the trivial start);
do not compare against the hold-mode 9b policy numbers without labelling them
legacy.

**What is not claimed.** No expected success rate, no generalisation claim, and
no claim that the current 1,000-step checkpoint is representative.

### Task 9e addendum (2026-11-13)

The GPU health gate failed: 60 s synthetic load held SM at **210 MHz** with
**SW Power Cap and SW Thermal Slowdown** active (power 4.36-7.35 W, 52-56 C).
The Task 9e overfit and fix-variant runs were therefore **not run**; the 9d
overfit remains unconverged and inconclusive.

Closed-loop drift with the existing checkpoint: with linear execution and
state teacher forcing at re-plan points, all three train seeds still leave the
0.1 rad tube at **frame 2** (first chunk); teacher-forced max drift **0.55-0.62
rad**, free **0.37-0.50 rad**. The error is already in the first chunk, not
accumulated over many steps.

Instruction test with 10 samples: cross-instruction chunk differences are
**below** within-instruction noise for both the fine-tuned checkpoint
(ratios 0.59-0.85) and `lerobot/smolvla_base` (0.53-0.67). The language pathway
is not measurably affecting the action chunk; start with one instruction and
verify instruction conditioning explicitly before multi-instruction training.

Do not start the next real training run until the GPU clock/power issue is
fixed and a true overfit (mean last-20-step loss below 0.1) is achieved. Task
10 remains blocked pending that evidence.

## 15. Remote overfit protocol (Task 9f; document only, not run)

This is the exact protocol to execute on a remote GPU that passes the bundle's
clock/memory gate. It is the pipeline-health test before Task 10, not a policy
quality experiment. No part of it ran in the Task 9f session.

**Config.** `configs/training/smolvla_lora_overfit_remote.yaml`:

| Setting | Value |
| --- | --- |
| Episodes | dataset indices **0 and 1** (`ep0000` seed **10000**, `ep0004` seed **10004**) |
| Batch / steps | **4** / **1000** |
| LR / precision | **1e-4 constant** / **fp32** |
| Chunk / `n_action_steps` | 50 / 50 (training), evaluation uses 10 |
| LoRA | r **32**, alpha **64**, dropout **0.05**, unchanged targets |
| Checkpoints | every **100** steps: 100, 200, ..., end (1000) |
| Seed | 0 |

**Convergence, defined before the run.** Converged means the **mean training
loss of the last 20 steps is below 0.1**. The training loss is the
flow-matching objective, not an L1 distance in radians. The Task 9d 250-step run
ended at **0.49675** and was still falling; 0.1 is about **5x lower** and is a
clear "the model can fit two episodes" signal. It is pre-registered here and
must not be redefined after seeing the curve.

**Batch and step arithmetic (healthy-GPU measurements, not remote guesses).**
The measured healthy-GPU step times were: batch 1 **0.43-0.66 s**, batch 4
median **0.374 s** (20-step full-data measurement), batch 16 median **1.176 s**
(20 steps) / **1.505 s** (1,000-step run). Peak VRAM was 1,733 MiB at batch 4
and 4,377 MiB at batch 16. Choose batch 4 and 1,000 steps:

- compute estimate: `1,000 x 0.374 s = 374 s = 6.2 min`;
- batch 16 would be `1,000 x 1.505 s = 1,505 s = 25.1 min`;
- batch 4 keeps VRAM near 1.7 GiB and is the faster total run.

Remote step time is **unknown until measured**. `RUN_REPORT.json` records it;
do not substitute the local 210 MHz 9d median (2.992 s/step), which would make
1,000 steps ~50 min and is explicitly a throttled figure.

**Execution.** Follow `docs/REMOTE_TRAINING.md`:

1. build/upload the bundle from the pushed commit (no upload in this session);
2. on the remote host, extract it at the repository root, clone the public repo
   at the manifest commit, and run `scripts/remote_train.sh` with the bundle
   config;
3. return `outputs/remote/<run_name>/RUN_REPORT.json` and the checkpoint
   folder(s) under `outputs/remote/<run_name>/`;
4. verify locally with `scripts/verify_remote_checkpoint.py`.

**Local verification (refuses a mismatch).**

```bash
scripts/verify_remote_checkpoint.py \
  --checkpoint outputs/remote/overfit_remote/checkpoint-00001000 \
  --config configs/training/smolvla_lora_overfit_remote.yaml \
  --run-report outputs/remote/overfit_remote/RUN_REPORT.json
```

The default handshake is CPU (`CUDA_VISIBLE_DEVICES` empty), so the result is
not a throttled GPU timing. Use `--handshake cuda` only for a short diagnostic;
label it throttled.

**Local closed-loop evaluation after verification.** Use the two training
seeds, linear execution, `n_action_steps=10`, deterministic sampling, and
**3 repeats per seed**. The runner resets with seed 10000 and 10004 and episode
index 0 (the 9d convention; dataset indices 0 and 1 are `ep0000`/`ep0004`).

```bash
.venv/bin/python - <<'PY'
from dataclasses import replace
from pathlib import Path

from sim2real.evaluation import CheckpointPolicy, load_evaluation_config
from sim2real.evaluation.episode_runner import run_eval_episode
from sim2real.expert import load_expert_config
from sim2real.filtering.config import load_filter_config
from sim2real.sim import create_backend

root = Path.cwd()
checkpoint = root / "outputs/remote/overfit_remote/checkpoint-00001000"
config = load_evaluation_config("configs/evaluation/default.yaml")
config = replace(
    config,
    settings=replace(
        config.settings, n_action_steps=10, execution="linear", max_control_steps=600
    ),
)
backend = create_backend("mujoco", "configs/sim")
policy = CheckpointPolicy(checkpoint, repo_root=root, deterministic_sampling=True, timeout_s=240)
filter_config = load_filter_config("configs/filtering/mujoco.yaml")
expert_config = load_expert_config("configs/expert")
try:
    for seed in (10000, 10004):
        for repeat in range(3):
            record = run_eval_episode(
                backend,
                policy,
                seed=seed,
                episode_index=0,
                config=config,
                eval_run_id=f"overfit_remote_{seed}_r{repeat}",
                config_hash=config.config_hash(),
                git_commit="remote-verified",
                git_dirty=False,
                randomization_dir="configs/randomization",
                filter_config=filter_config,
                expert_config=expert_config,
            )
            print(seed, repeat, record.success, record.termination_reason, record.task_completion)
finally:
    policy.close()
    backend.close()
PY
```

Teacher-forced L1 uses the same five deterministic sample-seed mean as the Part
A analysis in `docs/EVALUATION.md`: frames 0..5 of dataset index 0, first-step
and first-10 L1 in physical units, no closed-loop stepping. Report it next to
the closed-loop outcome; it is not a policy-quality metric.

**Interpretation table.**

| Outcome | Interpretation | Next action |
| --- | --- | --- |
| Converged and closed loop progresses/succeeds | pipeline is sound; the local failure was optimization + the degraded GPU | proceed to Task 10 preparation and a larger run |
| Converged but closed loop still fails | pipeline bug hunt, in this order: state/action alignment; normalization vs units; gripper dimension; camera mapping / unused camera3; frozen vision encoder and `train_expert_only`; LoRA coverage; instruction handling | test the single hypothesis with the existing checkpoint, no retraining first |
| Not converged | the limiter is named by the curve (still falling vs flat); the single smallest change is, in order: more steps (1,000 -> 2,000 at the same batch/lr), then LoRA rank, then LR | change one thing only and rerun the same protocol |

Do not call a non-converged run a pipeline bug; do not call a converged run a
policy-quality result.

## 16. Single-instruction dataset count (Task 9f; document only)

`train_v1` contains **one** instruction only. `meta/tasks.parquet` has one row,
`task_index=0`, text `pick up the red cube and place it in the left zone`; all
**40/40** accepted episodes use it. The six-instruction matrix used in the 9d
sensitivity test is 3 cubes x 2 zones, so the other five target combinations
have **0** accepted episodes in `train_v1`.

| Instruction (target cube -> zone) | Accepted episodes | Starts inside 0.060 m tolerance |
| --- | --- | --- |
| red -> left | **40** | **1/40** (index 39, `ep0071`, seed 10071, 0.05108 m) |
| red -> right | 0 | not applicable (no episodes) |
| green -> left | 0 | not applicable |
| green -> right | 0 | not applicable |
| blue -> left | 0 | not applicable |
| blue -> right | 0 | not applicable |

`train_v1` therefore cannot test instruction conditioning across six
instructions; a six-instruction dataset must be generated separately.

**Generation attempts for a 40-episode single-instruction dataset.** The
measured Task 7 run accepted **123/200 = 0.615** generated episodes overall
(200 generated, 131 successful, 123 accepted). Excluding start-in-success-region
spawns removes about **1/30 = 3.3%**:

`40 / (0.615 * (1 - 0.033)) = 40 / 0.595 = 67.2`

so budget **70 attempts**. The alternative evaluation-seed arithmetic
`40 / (0.80 * 0.94 * 0.967) = 40 / 0.727 = 55` uses the 24/30 MuJoCo expert
success and 123/131 filter acceptance; **70** covers both estimates and the
observed easy-pose skew. For six instructions at the same yield, the naive
budget is `6 x 70 = 420` attempts; that is not recommended before one
instruction works.
