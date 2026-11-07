# Remote-GPU training runbook (documentation only)

Status: **documentation only.** No account was created, nothing was uploaded,
and no provider's GPU, limits, or pricing was tested or is claimed here. The
local RTX 4050 smoke run fits in about **1.1 GiB VRAM**, so a remote GPU is not
required for the current two-step smoke; this runbook is for longer runs or a
machine where a larger batch/chunk is needed.

## When to use it

- A longer run (many steps/epochs), a larger batch, or a different precision
  does not fit locally.
- You need a GPU architecture or driver that this machine does not have.
- The pinned stack must stay identical; a remote run that changes LeRobot,
  torch, PEFT, or the dataset semantics is a different experiment.

## What to export

| Item | Detail |
| --- | --- |
| Code | The exact git commit; include the uncommitted diff only if it is part of the experiment. Record `git rev-parse HEAD` and `git status --porcelain`. |
| Dataset | `data/lerobot/train_v1/` as built here: **92 MB**, 40 episodes, 1,849 frames, 10 fps, 32/8 split. Include `meta/info.json`, `meta/stats.json`, `meta/sim2real_split.json`, `meta/sim2real_build.json`, and the `data/`/`images/` trees. |
| Config | `configs/training/smolvla_lora.yaml` (and the dataset/embodiments configs it depends on). |
| Environment | `pyproject.toml` and `uv.lock`; the pinned versions are LeRobot 0.6.1, torch 2.11.0 with a compatible CUDA backend, PEFT 0.21.2, accelerate 1.15.0, transformers 5.5.4. |
| Checksums | `sha256sum` of the dataset metadata files and the config. `scripts/validate_checkpoint.py` compares the stats hash. |

Do **not** upload model weights: download `lerobot/smolvla_base` at revision
`5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6` and the
`HuggingFaceTB/SmolVLM2-500M-Video-Instruct` backbone at revision
`7b375e1b73b11138ff12fe22c8f2822d8fe03467` on the remote machine. They are
~2.9 GB and belong in that machine's Hugging Face cache.

## What NOT to upload

- Credentials, tokens, SSH keys, `.env` files, or Hugging Face/W&B logins.
- Unrelated datasets, personal files, or the repository's git-ignored caches.
- `outputs/`, `wandb/`, `mlruns/`, `.venv*`, `data/staging/`, or any model
  weights/checkpoints from a different run.
- Any data whose license you have not checked. This dataset is synthetic
  (Apache-2.0 project data); Bridge remains CC BY 4.0 and is not included.

## Run

The exact install command depends on the provider's CUDA/driver and is **not
verified here**. On a machine whose driver supports CUDA 12.6 or newer:

```bash
git clone <your-repo-url> sim2real && cd sim2real
git checkout <exact-commit>
# choose a torch build compatible with that driver; do not silently change
# LeRobot/PEFT/transformers versions
ML_TORCH_BACKEND=cu126 ./scripts/setup_envs.sh --with-ml
# or reproduce the pins manually in a fresh venv

# copy the dataset tree to data/lerobot/train_v1 and verify the hashes
sha256sum data/lerobot/train_v1/meta/{info,stats,sim2real_split}.json

.venv-ml/bin/python scripts/train_vla.py \
  --config configs/training/smolvla_lora.yaml \
  --max-steps <N> --output-dir outputs/task8/remote
```

If the remote GPU/driver needs a different wheel index, record which one worked
and the actual `torch.version.cuda` and `torch.cuda.get_device_name(0)` in the
run notes. Do not claim a wheel works without a matmul/step test.

## What to bring back

| Artifact | Why |
| --- | --- |
| `outputs/.../checkpoint-<step>/adapter/` | PEFT adapter weights and `adapter_config.json` |
| `checkpoint_metadata.json` | Base revision, versions, GPU, git commit, config hash, dataset/stats hash, split, seed, step, LoRA config, rename map, chunk/n_action, dims, fps, horizon |
| `trainer_state.json` | Step/epoch/offset/order/RNG for an exact resume |
| `losses.jsonl` | Per-step loss and step time |
| The stdout summary | Final step, losses, peak VRAM/RSS, parameter counts |
| `lora_targets.json` | The resolved module list actually adapted |

Bring them back under `outputs/` (git-ignored). Never commit weights.

## Verify locally

```bash
.venv-ml/bin/python scripts/validate_checkpoint.py \
  --checkpoint outputs/task8/remote/checkpoint-<step> \
  --config configs/training/smolvla_lora.yaml
```

The validator refuses the checkpoint if the stats-file hash, action
representation, camera rename map, or config hash differs from the local
dataset/config. If the remote dataset was regenerated, the stats hash will
differ by design; rebuild the local dataset or record the new hash explicitly
rather than bypassing the check.

## Unverified items (marked, not claimed)

- Any provider's GPU model, VRAM, driver, CUDA version, pricing, free tier,
  session limit, or network policy.
- Whether a specific cloud image ships a PyTorch/CUDA combination that matches
  this driver-tested cu126 build.
- Remote filesystem performance and dataset upload time.
- Whether the remote environment can reach the Hugging Face Hub; if not,
  pre-download the two pinned repositories and mount the cache.
