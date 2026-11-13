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

## Task 9f bundle runbook (2026-11-13)

This section supersedes the older export list above. It is still **documentation
only**: no upload, account, provider test, or remote training happened in this
session. The local GPU is pinned at **210 MHz** (SW Power Cap + SW Thermal
Slowdown) and is not used for training.

### What to upload

One archive, built locally from a **clean** pushed commit:

```bash
python scripts/export_remote_bundle.py \
  --dataset data/lerobot/train_v1 \
  --config configs/training/smolvla_lora_overfit_remote.yaml \
  --config configs/training/smolvla_lora_train_v1.yaml \
  --output-dir outputs/remote_bundle
```

The exporter prints the payload size **before** writing, refuses a dirty tree,
scans for forbidden files, and writes:

| Content | Detail |
| --- | --- |
| `data/lerobot/train_v1/` | Full 40-episode dataset, 92 MB on disk (parquet already compressed; the archive is approximately the same size) |
| `outputs/remote_bundle_payload/configs/training/` | The overfit protocol config and the full `train_v1` config |
| `outputs/remote_bundle_payload/requirements.txt` | Exact versions read from `.venv-ml`, with `--extra-index-url https://download.pytorch.org/whl/cu126` |
| `outputs/remote_bundle_payload/MANIFEST.json` | sha256 + size of every file, git commit, dataset stats/dataset hashes, expected `checkpoint_metadata` fields, model revisions |

The archive name includes the commit prefix and UTC timestamp. The exporter
also prints its own sha256. `outputs/` is git-ignored.

### What NOT to upload

- Credentials, tokens, SSH keys, `.env` files, HF/W&B logins, or account data.
- Hugging Face caches or model weights. The remote machine downloads
  `lerobot/smolvla_base` at revision
  `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6` and the backbone
  `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` at
  `7b375e1b73b11138ff12fe22c8f2822d8fe03467` itself.
- `outputs/`, existing checkpoints, `logs/`, `.venv*`, unrelated datasets,
  `data/staging/`, `data/curated/`, or W&B/MLflow directories.
- The repository's `.git` directory; the remote clones the public repo.

### Exact remote commands

On the remote machine, after uploading the archive (for example with `scp` or
`rsync`; no upload was performed here):

```bash
git clone https://github.com/KuNaL8103/sim2real.git sim2real
cd sim2real
git checkout <EXACT_COMMIT_FROM_MANIFEST>
tar -xzf /path/to/sim2real_remote_<commit7>_<UTC>.tar.gz -C .
scripts/remote_train.sh \
  --bundle-dir outputs/remote_bundle_payload \
  --config outputs/remote_bundle_payload/configs/training/smolvla_lora_overfit_remote.yaml \
  --output-dir outputs/remote/overfit_remote
```

`tar` extracts the dataset under `data/` and the payload under `outputs/`; both
are git-ignored, so `git status --porcelain` stays empty. The script fails
loudly if the tree is dirty, the HEAD differs from the manifest commit, a
manifest hash mismatches, CUDA is unavailable, the GPU has less than **8 GiB**,
or the median SM clock under load is below **1,000 MHz**.

### Network actions, printed before they run

1. `pip install -r outputs/remote_bundle_payload/requirements.txt` from PyPI
   plus the pinned PyTorch cu126 index.
2. Hugging Face downloads of the two pinned repositories above, unless already
   cached.

No upload, Hub push, account/login, W&B, or MLflow traffic is used. The script
does not call `sudo` or `apt`.

### Expected output

The script prints, in order:

- `MANIFEST verified: N files`;
- `code commit <sha> is checked out clean`;
- the network actions above;
- a GPU JSON block with the matmul checksum, total memory, median/min/max SM
  clock, power, temperature, and throttle reasons;
- the training summary JSON and per-step `losses.jsonl`;
- `RUN_REPORT.json` with steps, wall time, final loss, last-20 mean loss, median
  step time, peak VRAM/RSS, GPU telemetry, and checkpoint hashes.

Checkpoints are written every 100 steps: `checkpoint-00000100`,
`checkpoint-00000200`, ..., and the end (`checkpoint-00001000` for this config).

### Bring the result back

```bash
rsync -av <remote-user>@<remote-host>:~/sim2real/outputs/remote/overfit_remote/ \
  ./outputs/remote/overfit_remote/
```

This downloads the checkpoint folders, `losses.jsonl`, `train.log`,
`gpu_monitor.csv`, and `RUN_REPORT.json`. Keep the run name
`overfit_remote` (or use a unique `outputs/remote/<run_name>/` for every run).

### Verify locally

```bash
scripts/verify_remote_checkpoint.py \
  --checkpoint outputs/remote/overfit_remote/checkpoint-00001000 \
  --config configs/training/smolvla_lora_overfit_remote.yaml \
  --run-report outputs/remote/overfit_remote/RUN_REPORT.json
```

It refuses `git_dirty=true`, a commit absent from local history, a config hash,
stats/dataset hash, rename map, chunk/fps/`n_action_steps` mismatch, an adapter
hash mismatch, or a policy-server handshake refusal. The default handshake is
CPU with `CUDA_VISIBLE_DEVICES` empty; it prints which handshake was used. A
`--handshake cuda` run is a short throttled diagnostic and is not a performance
figure.

### Remote GPU decision rule

**Accept the machine only if:** under load, the median SM clock is not capped
and memory is at least **8 GiB**.

Why 8 GiB: the measured peak at batch 16 was **4,377 MiB**, and batch 4 was
1,733 MiB. Eight GiB leaves roughly 3.6 GiB of headroom for the driver,
allocator fragmentation, the base model, and the checkpoint/evaluation
processes. The script enforces 8,192 MiB and a conservative 1,000 MHz median
floor and prints the throttle reasons; a 210 MHz capped device fails. This
floor rejects the known-degraded local GPU, not a claim that every GPU above it
is suitable.

**UNVERIFIED provider claims.** No provider documentation was fetched in this
session. Every claim about a specific provider's GPU model, VRAM, session
limit, quota, pricing, network policy, or CUDA image is **UNVERIFIED** and is
not made here. The runbook applies to any free notebook GPU or rented GPU VM
that the user operates and that passes the gate above.

### Overfit protocol

The exact protocol (episodes, batch/steps arithmetic, convergence rule,
checkpoints, local closed-loop evaluation and interpretation table) is in
`docs/TRAINING.md` section 15. Do not redefine convergence after seeing the
loss curve.
