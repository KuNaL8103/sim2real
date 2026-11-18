# Manual checks (for the operator)

These checks require a human. The agent never fills the last column. A row is
only "confirmed" when you write the date and what you saw.

Build/source once before the ROS checks:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --parallel-workers 2 --base-paths ros2_ws/src
source ros2_ws/install/setup.bash
```

| # | Check | Commands (separate terminals) | What to observe | Confirmed by user / date |
| --- | --- | --- | --- | --- |
| 1 | Task 8 smoke inspection | `less outputs/task8/smolvla_smoke/losses.jsonl`; `ls outputs/task8/smolvla_smoke/checkpoint-00000002/`; `cat outputs/task8/smolvla_smoke/checkpoint-00000002/checkpoint_metadata.json` | Two finite losses, adapter files present, metadata says `git_dirty: true` (accurate, not a failure) | |
| 2 | Task 8b checkpoint comparison | `less outputs/task8/train_v1_lora/losses.jsonl`; `cat outputs/task8/train_v1_lora/checkpoint-00001000/checkpoint_metadata.json`; `less outputs/task8/analysis/val_trained.log outputs/task8/analysis/val_smoke.log` | Loss falls from 3.54 to 0.32; metadata `git_dirty=false`, step 1000; trained L1 beats both baselines | |
| 3 | Flywheel ledger | `cat data/flywheel/ledger.jsonl`; `uv run python scripts/flywheel_status.py` | Append-only JSON lines with cycle ids, hashes, decision; `will_train: false` | |
| 4 | Task 10 inference node | Terminal A: `ros2 launch sim2real_vla vla_inference.launch.py`; Terminal B: `ros2 topic echo /vla/status`; `ros2 topic echo /vla/raw_action --once` | Status `READY`, counters/restarts, raw chunk with 8 joint names and absolute targets. Kill the policy server and restart it to see recovery | |
| 5 | Task 11 safety mock/HIL | Terminal A: `ros2 launch sim2real_control pipeline_mock_hw.launch.py`; Terminal B: `ros2 topic info /joint_trajectory_controller/joint_trajectory -v`; `ros2 topic echo /safety/status`; `ros2 service call /estop std_srvs/srv/Trigger {}`; `ros2 service call /estop_reset std_srvs/srv/Trigger {}` (immediate, then after 2 s) | Publisher count 1 = `safety_watchdog_node`; commands forwarded; e-stop latches; immediate reset refused; reset after hold succeeds; no motion from the reset | |
| 6 | RViz mapping | Terminal A: `ros2 launch sim2real_perception mapper.launch.py backend:=mujoco use_rviz:=true` (or start RViz separately with `sim2real_perception/rviz/mapper.rviz`) | Fixed frame `world`; colored table/cube cloud; TSDF mesh over it; both cameras contribute; robot links absent from the query cloud. MuJoCo + full camera displays can crash WSLg; use the lean config | |
| 7 | Integrated simulation launch | `ros2 launch sim2real_control pipeline_sim.launch.py backend:=fake`; `ros2 topic hz /vla/raw_action`; `ros2 topic echo /safety/status` | Four nodes start; fake inference publishes chunks; safety validates; **no controller** (documented: sim_node does not consume commands) | |
| 8 | CI confirmation | GitHub Actions tab for the pushed commit | Both jobs green. Supporting evidence only; the agent does not confirm CI | |
| 9 | `.venv-ml` swap | After you confirm the `.venv-ml.new` checks: `mv .venv-ml .venv-ml.old && mv .venv-ml.new .venv-ml`; `./scripts/run_ml_tests.sh`; `.venv-ml/bin/python -c "import scipy"` (must fail) | ML tests pass; scipy/mujoco absent; keep `.venv-ml.old` until you say delete | |
| 10 | Remote checkpoint provenance | Inspect `outputs/remote/overfit_remote/as_used/` and the Task 9i deviation table in `docs/REMOTE_TRAINING.md` | Whether an audited `as_used/` exists. Until then provenance is user-reported | |

## Not checks, but pending decisions

- Whether to make `obstacle_mode: masked` the default (currently `strict`).
- Whether to build the ROS -> MuJoCo command bridge (documented gap; see
  `docs/ARCHITECTURE.md` and `docs/DEPLOYMENT.md`).
- Whether to spend GPU/cloud time on more training data and steps.
- Whether to create a release tag (see `docs/VALIDATION.md`).
