# Status

This is the tracked status record. Local `CLAUDE.md` is untracked and ignored;
this file is authoritative for clones and CI.

- Last verified test run: passed=522 failed=0 skipped=12 errors=0 | date=2026-11-15 | base_commit=319103c
- Last verified ROS test run: passed=535 failed=0 skipped=4 errors=0 | date=2026-11-15 | base_commit=319103c
- Task 1 status: complete

The 2026-11-15 run used the tree at `319103c` with scikit-learn **absent**
(removed by an earlier ci_parity `uv sync`), so the core count is **522 passed /
12 skipped**; with sklearn present the optional cross-check would add one pass
and remove one skip (the 9f baseline was 512/11). `.venv-ros` was **535/4**,
`colcon test-result` **7 tests / 0 failures**, and `run_ml_tests.sh` **4 passed
/ 1 deselected**. `scripts/ci_parity.sh` passed at the end of the session. The
difference from 9f is explained by the new tests: 6 teacher-forced, 1 verifier
and 4 remote-preflight tests (plus the status test already present).

## Task 9g cleanup-and-hardening status (2026-11-15)

- The user confirmed **both CI jobs green** for the 9f push at `b620c91`
  (supporting evidence only; confirmed in the Actions tab).
- Both CI jobs for the 9f push at `b620c91` are **user-confirmed green**
  (2026-11-15, Actions tab); this is supporting evidence, not a substitute for
  the local runs below.
- The GitHub repository resolves unauthenticated:
  `GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=true git ls-remote
  https://github.com/KuNaL8103/sim2real.git HEAD` returned
  `b620c91f16cef462a7b74ab3e78ae01d8c6b43d8 HEAD` (public).
- Pre-registered decision rule committed first at `17d877b` in
  `docs/TRAINING.md` section 17 and `configs/evaluation/overfit_rule.yaml`.
- Added `scripts/eval_teacher_forced.py` plus CPU tests; `scripts/remote_train.sh`
  now has `--check-only`, `--from-bundle`, interpreter/Requires-Python checks,
  driver/CUDA printing, network preflight, and preflight recording. The exporter
  now adds `repo.bundle`, the payload copy of `remote_train.sh`, Python metadata,
  and the repo.bundle hash.
- The 9f archive extracts to exactly `data/` and `outputs/`; its payload is
  `outputs/remote_bundle_payload/`. The old archive was rejected by the new
  `--check-only` because its manifest lacks `python.major_minor`; the Task 9g
  rebuild fixes that.
- The verifier on the real `checkpoint-00001000` correctly refused it: the old
  checkpoint metadata has no `adapter_hash` field (created before the field
  existed). Config hash, stats hash and commit matched; no verifier defect was
  found. `--no-run-report` was added as an explicit UNVERIFIED path.
- Wider first-chunk analysis on dataset indices 0/1/2 (frames 0-5/10/20, 5
  seeds) confirms the frame-0 pattern: mean error **0.02683-0.02785 rad** vs
  motion **0.00080-0.00134 rad** (mean ratio **22-37x**); sample-seed std is
  comparable to or larger than the mean error.
- Action parameterization statistics are documented as a hypothesis only: arm
  delta std is **0.13-0.38x** the absolute-action std, but nothing was changed.

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

The Task 9f archive is `outputs/remote_bundle/sim2real_remote_b620c91_20261113T143925Z.tar.gz`
(**95,133,660 bytes, sha256 `967dc109...41b8`**, built from `b620c91`). The
Task 9g bundle is rebuilt only after the final push, from the final pushed HEAD
with a clean tree. It contains the full 40-episode `train_v1` dataset, the two
training configs, pinned `.venv-ml` requirements, `repo.bundle` for offline
cloning, the payload copy of `remote_train.sh`, Python/`lerobot` Requires-Python
metadata, and a MANIFEST with every file hash. Its manifest `git_commit` is the
authoritative commit the remote run must clone; `c3802a5` is superseded and must
not be used. No model weights or HF cache are included. The final archive size
and sha256 are printed by the builder and reported in the session close-out.

## Pending checks

- [ ] Run the overfit protocol on a remote GPU that passes the bundle's matmul
  and clock/memory gate; return `RUN_REPORT.json` and the checkpoint folder.
- [ ] Verify the returned checkpoint locally with
  `scripts/verify_remote_checkpoint.py` (CPU handshake by default).
- [ ] Run `scripts/eval_teacher_forced.py` with
  `configs/evaluation/overfit_rule.yaml` after verification, then the local
  closed-loop evaluation in `docs/TRAINING.md` section 15.
- [ ] Confirm this push's GitHub Actions tab (supporting evidence only; the user
  confirms the Actions tab).
- [ ] Task 10 remains blocked until the remote overfit result is interpreted.

## GPU finding

The local RTX 4050 Laptop measured **210 MHz SM** under 9e/9f load with
**SW Power Cap** and **SW Thermal Slowdown** active (4.36-7.35 W, 52-56 C,
throttle `0x24` on 32/32 samples). On 2026-11-15 idle it read **2130 MHz**
(P3, 12 W, 54 C) but the same throttle flags were still active; that single idle
reading does not prove the clock is fixed. Treat the local GPU as degraded:
short inference diagnostics only; every timing is labelled
**"throttled, not a performance figure"**. No local training.
