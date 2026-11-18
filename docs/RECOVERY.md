# Failure recovery

Every procedure here preserves evidence first. Never delete a failed artifact,
a log, a checkpoint, or a ledger line to "make it pass". The orchestrator moves
corrupt stage outputs to `quarantine/` instead of deleting them.

## Orchestrator: interrupted or failed stage

Symptom: a cycle stops with `PIPELINE STOPPED: <STAGE> FAILED/BLOCKED`.
The stage status is in `<run_root>/<run_id>/stages/<STAGE>.json`; the cycle
manifest (even on failure) is in `<run_root>/<run_id>/manifests/cycle_NNNN.json`.

```bash
CONFIG=configs/pipeline/fake_e2e.yaml
ROOT=data/pipeline/fake_e2e
RUN=<run_id from the manifest or directory name>

cat "$ROOT/$RUN/stages/TRAIN.json"
cat "$ROOT/$RUN/manifests/cycle_0001.json" | head -80

# Re-run the same command. Completed stages whose inputs and output hashes are
# unchanged are SKIPPED; the failed stage and everything after it run again.
uv run python scripts/pipeline_orchestrator.py --config "$CONFIG" --root "$ROOT" \
    --run-id "$RUN" --execute --cycles 1
```

If the failure was an approval gate (`APPROVAL REQUIRED`), the message says the
exact flag or checkpoint that is missing. Do not add flags pre-emptively.

## Orchestrator: corrupt or incomplete stage output

Symptom: `OUTPUT QUARANTINED: <STAGE> recorded output is corrupt/incomplete`.

The recorded artifact and the stage status were moved to
`<run_root>/<run_id>/quarantine/<STAGE>-<timestamp>/`, and the stage status is
`QUARANTINED`. The run refuses to consume the artifact.

```bash
ls -la "$ROOT/$RUN/quarantine/"
cat "$ROOT/$RUN/quarantine/<STAGE>-"*/QUARANTINE_REASON.txt
```

Decide why it changed (manual edit, interrupted write, wrong config, disk full),
fix the cause, then re-run the orchestrator. It will not silently reuse the
quarantined artifact. If the artifact is valuable, copy the quarantine directory
somewhere safe before re-running.

## Orchestrator: seed-ledger refusal

Symptom: `SeedLedgerError: seed N overlaps ...`.

Choose an unused range. Do not edit `configs/pipeline/*.yaml` to a used range.
The ranges are listed in `docs/EVALUATION.md` and printed with:

```bash
uv run python scripts/pipeline_orchestrator.py --ledger
```

The 20950-20969 carve-out is fake-backend end-to-end only.

## Orchestrator: cycle gate refusal

Symptom: `cycle N refused: another cycle requires --approve-cycle` or
`the supplied checkpoint is not new`.

Another cycle needs **both** an explicit `--approve-cycle` and a new verified
checkpoint. Do not reuse the previous checkpoint. Supply a new one:

```bash
uv run python scripts/pipeline_orchestrator.py --config configs/pipeline/default.yaml \
    --execute --cycles 2 --approve-cycle \
    --supply-checkpoint outputs/task8/remote/checkpoint-00002000 \
    --allow-gpu --allow-download
```

## Remote/failed training

Preserve the whole run directory (`losses.jsonl`, stdout, `trainer_state.json`,
checkpoint, `RUN_REPORT.json`). Do not edit `RUN_REPORT.json` or a MANIFEST to
make verification pass.

```bash
uv run python scripts/verify_remote_checkpoint.py \
    --checkpoint <checkpoint-dir> \
    --config configs/training/smolvla_lora.yaml \
    --run-report <RUN_REPORT.json> \
    --handshake skip
```

If the verifier refuses, read the first failing line. The common causes are a
different dataset stats hash, a different config hash, a missing `adapter_hash`
in an old checkpoint, or recorded overrides (`low_clock`) without
`--accept-overrides`. Rebuild the local dataset or record the new hash
explicitly; do not bypass the check.

If the run itself failed, keep the partial checkpoint and logs, note the last
successful step, and resume only from a checkpoint whose metadata matches the
local config/dataset.

## Safety: e-stop latched

Symptom: `/safety/status` shows `estop_latched=true`, `state=ESTOP`.

1. Remove the cause. Do not clear the latch first.
2. Confirm fresh `/joint_states` and a valid `/workspace/pointcloud`.
3. Call `/estop_reset`; an immediate call is expected to fail with
   "inputs have not been fresh for 2.0 s". Wait for the hold and call again.
4. The reset commands **no motion**. A new validated chunk is required.

```bash
ros2 topic echo /safety/status --once
ros2 service call /estop_reset std_srvs/srv/Trigger {}
```

A software e-stop is not a safety-rated emergency stop. On hardware, use the
hardware e-stop.

## Safety: stale/missing map or joint state

Symptom: `state=SAFE_STOP`, `is_safe=false`, reason `MAP_STALE`,
`MAP_UNAVAILABLE`, or `JOINT_STATE_STALE`.

Restore the publisher (mapper/`sim_node`, or `ros2_control` broadcaster). The
watchdog recovers automatically when inputs are fresh; it does not resume the
old chunk. Do not set `map_required: false` to "fix" this in production; that
mode reports unknown instead of blocking.

## Policy server crash or timeout

Symptom: `/vla/status` shows `restarts`, `last_error`, or no new chunks.

- The node restarts the server up to `max_restarts` with a backoff.
- Check the server process and the checkpoint hash in the handshake.
- First load is slow (measured 34.71-107 s depending on conditions). The startup
  timeout is separate from the call timeout; do not lower it to "speed up" the
  first chunk.
- With the real checkpoint on this laptop, calls take 1.88 s median / 5.90 s max
  against a 0.9 s chunk (`throttled, not a performance figure`). The safety
  default `stale_chunk_policy=reject` will refuse those chunks. That is the safe
  behaviour, not a bug to bypass.

## Dirty tree before a commit

Do not discard unrelated changes. Either commit the intended files on a branch
or use a separate worktree:

```bash
git status --short
git diff --stat
git worktree add /tmp/sim2real/sim2real_clean HEAD
```

Never use `git checkout -- .`, `git reset --hard`, or force-push.

## CI failure

Reproduce locally before changing code:

```bash
./scripts/ci_parity.sh            # core + ROS halves, with CI's shadowing
./scripts/ci_parity.sh --clean-clone
env -u PYTHONPATH -u AMENT_PREFIX_PATH uv run ruff check .
env -u PYTHONPATH -u AMENT_PREFIX_PATH uv run pytest -ra
```

A skip is not a pass. If the failure is a flaky rate test, run it alone and
record the measured rate; do not lower a floor without evidence.

## Disk-full or oversized write

The orchestrator refuses a stage whose configured estimate exceeds
`max_write_bytes` unless `--allow-large-write` (or `--allow-download` for a
training download) is passed. Free space or change the configured bound
deliberately. Do not delete the append-only flywheel ledger; move old run
directories only after the manifests are backed up.
