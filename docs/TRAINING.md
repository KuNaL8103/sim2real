# VLA fine-tuning (Task 8)

Status: **implemented and smoke-tested on this machine.** SmolVLA + PEFT LoRA
runs for two optimizer steps at batch 1 and chunk 50. This proves the pipeline
runs and the loss is finite; it says **nothing** about policy quality.

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

## 11. What was NOT run

- OpenVLA-7B (adapter refuses; license/VRAM unverified).
- bf16 or fp16 training (config allows, but only fp32 was measured).
- Gradient checkpointing (flag exists; the stack was not verified to support it).
- cu128 or any other wheel index; a second GPU/driver.
- A remote-GPU run (`docs/REMOTE_TRAINING.md` is documentation only).
- Video/MP4 datasets, real Bridge data, or any Hub upload.
- `wandb`/`mlflow` backends (not installed); only the `none` tracker is exercised.
- Policy quality, closed-loop success, or sim2real transfer.
- Isaac Sim, MoveIt, RViz/viewer checks.

## 12. Limitations

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
