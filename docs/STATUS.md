# Status

This is the tracked status record. Local `CLAUDE.md` is untracked and ignored;
this file is authoritative for clones and CI.

- Last verified test run: passed=512 failed=0 skipped=11 errors=0 | date=2026-11-13 | base_commit=556925d
- Last verified ROS test run: passed=524 failed=0 skipped=4 errors=0 | date=2026-11-13 | base_commit=556925d
- Task 1 status: complete

## Task 9f-remote status (2026-11-13)

Part A (first-chunk error analysis) is complete. Part B (bundle, remote script,
local verifier, tests) is implemented but no remote run was executed. Part C
(overfit protocol), Part D (single-instruction count) and Part E (docs/status)
are documented. No training, upload, install, download, sudo/apt, Docker, or
new rollout generation ran in this session.

Part A answer: observation/preprocessor parity is exact (state 0.0, both camera
tensors 0.0, language tokens 0.0 after LeRobot preprocessing) and the runner
executes the postprocessed chunk's first action with max abs difference 0.0.
The gap is the policy's first-chunk action error relative to motion: at frame 0
the recorded motion is 0.00016-0.00228 rad per arm dimension while the
five-sample mean error is 0.004-0.079 rad (9-60x motion); the largest error is
about 80% of the 0.1 rad tube before the first two control steps. "Small"
teacher-forced L1 is small only in absolute radians, not relative to the
per-step motion. Details and full per-dimension tables are in
`docs/EVALUATION.md`.

## Remote bundle

The bundle is built under `outputs/remote_bundle/` (git-ignored) and is about
**90.7 MiB** (first build: 95,133,662 bytes; payload 95,519,481 bytes, 11
files). It contains the full 40-episode `train_v1` dataset, the overfit remote
config, the full `train_v1` config, pinned requirements from `.venv-ml`
(torch 2.11.0+cu126, PEFT 0.21.2, transformers 5.5.4, accelerate 1.15.0,
LeRobot 0.6.1, num2words 0.5.14), and a MANIFEST with every file hash. The
manifest records the pushed commit; the remote run must clone exactly that
commit. No model weights or HF cache are included.

## Pending checks

- [ ] Run the overfit protocol on a remote GPU that passes the bundle's matmul
  and clock/memory gate; return `RUN_REPORT.json` and the checkpoint folder.
- [ ] Verify the returned checkpoint locally with
  `scripts/verify_remote_checkpoint.py` (CPU handshake by default).
- [ ] Run the local closed-loop and teacher-forced evaluation in
  `docs/TRAINING.md` section 15 after verification.
- [ ] Confirm this push's GitHub Actions tab (supporting evidence only; the user
  confirms the Actions tab).
- [ ] Task 10 remains blocked until the remote overfit result is interpreted.

## GPU finding

The local RTX 4050 Laptop is stuck at **210 MHz SM clock** (210 MHz graphics,
405 MHz memory) with **SW Power Cap** and **SW Thermal Slowdown** active at
idle, 3 W, 51 C. A 60 s load in Task 9e also held 210 MHz (4.36-7.35 W,
52-56 C, throttle `0x24` on 32/32 samples). The local GPU may be used for
short inference diagnostics only; any timing from it is labelled
**"throttled, not a performance figure"**. No local training.
