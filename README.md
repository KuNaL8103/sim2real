# sim2real

End-to-end sim2real robotics data pipeline and VLA fine-tuning framework for a
Franka Panda, built around ROS 2 Jazzy, MuJoCo (Isaac Sim optional), LeRobot,
and SmolVLA + LoRA.

**Task 6 status: RGB-D workspace mapping is implemented.** Simulator backends,
seeded domain randomization, the scripted pick-and-place expert, trajectory
filtering, `curated_v1` output, and the workspace mapper exist. LeRobot export,
training, inference, and safety are not implemented.

## What exists today

- **Pure-Python core** (`src/sim2real`, no `rclpy`): contract constants, bounded
  metadata validation, status parsing, `SimBackend` + `FakeBackend` + MuJoCo
  backend + guarded Isaac stub, seeded domain randomization, the scripted
  pick-and-place expert (FK/IK/planner/state machine/perturbations/staging),
  trajectory filtering with `curated_v1` writing, and RGB-D mapping
  (`TSDFMap`, robot self-filter, transform providers, vectorized obstacle
  query).
- **ROS 2 Jazzy adapters**: `sim_node`, `expert_collector_node`,
  `trajectory_filter_node`, and `workspace_mapper_node`.
- `sim2real_msgs` interfaces, including the Task 5 measured `TrajectoryStep`
  fields.
- Documentation, versioned YAML configs, scripts, CI, and hardware-free tests.
- Verified on 2026-11-01: core **306 passed / 7 skipped**, ROS `.venv-ros`
  **318 passed**, `colcon test-result` **7 tests / 0 failures**. Details and
  provenance are in `CLAUDE.md` and `docs/ARCHITECTURE.md`.

## Quick start

Core / CI (the ROS-free run is the official baseline):

```bash
uv sync --group sim
env -u PYTHONPATH -u AMENT_PREFIX_PATH uv run ruff check .
env -u PYTHONPATH -u AMENT_PREFIX_PATH uv run ruff format --check .
env -u PYTHONPATH -u AMENT_PREFIX_PATH uv run pytest -ra
```

ROS 2 (only if Jazzy is installed):

```bash
source /opt/ros/jazzy/setup.bash
cd ros2_ws
colcon build --symlink-install --parallel-workers 2
colcon test --packages-select sim2real_perception sim2real_sim sim2real_msgs
colcon test-result --verbose
source install/setup.bash
cd ..
.venv-ros/bin/python -m pytest -ra
```

## Workspace mapping (Task 6)

```bash
# Fake backend: fastest, no simulator rendering
ros2 launch sim2real_perception mapper.launch.py backend:=fake use_rviz:=true

# MuJoCo: slower under llvmpipe, but the full scene
ros2 launch sim2real_perception mapper.launch.py backend:=mujoco use_rviz:=true
```

The mapper:

- fuses overhead and wrist RGB-D frames through tf2 poses into one
  workspace-bounded Open3D TSDF;
- discards depth outside `[depth_min_m, depth_trunc_m]` and points outside
  `configs/mapping/default.yaml` workspace bounds **before** integration;
- publishes `/workspace/pointcloud` (xyz+rgb, frame `world`) and
  `/workspace/tsdf_mesh` (`TRIANGLE_LIST` Marker, frame `world`);
- serves `/workspace/reset_map` (`std_srvs/Trigger`);
- resets automatically when a jittered camera changes `/tf_static` beyond the
  documented epsilon; a reset without camera jitter needs an explicit
  `/workspace/reset_map`;
- provides `sim2real.mapping.ObstacleQuery`, a vectorized 3-D Euclidean
  distance transform over the surface points for the Task 11 safety layer.

### Open3D and NumPy

`open3d-cpu==0.20.0` is pinned in the core dev group and installed into
`.venv-ros` with `--no-deps`, so system NumPy 1.26.4 stays visible; `.venv` uses
NumPy 2.5.4. The CPU wheel is 105 MB (the CUDA wheel is 400.8 MB) and imports
headlessly with no display.

Open3D 0.20.0's legacy `ScalableTSDFVolume` produces empty geometry from metric
float32 depth (upstream issue
[#7592](https://github.com/isl-org/Open3D/issues/7592), open PR #7594): its
activation step re-scales an already-metric depth image by 1000. The mapper
therefore uses a workspace-bounded `UniformTSDFVolume`, whose metric-depth path
is correct. This is a documented deviation, not a claim that the scalable
volume works. Open3D 0.19.0 is unaffected but needs a large ML dependency set
for its eager `open3d.ml` import.

### Mapping limitations

- The map is a **surface** map. There is no free-space carving, so an empty
  region is not proof of clearance; a moved object leaves a ghost until reset.
- Unknown space is expressed only by `MapStatus` (`NO_DATA`, `STALE`,
  `LOW_COVERAGE`, `TF_FAILURES`). An invalid status is **unsafe** for Task 11.
- The robot self-filter is a coarse sphere per link; a grasped object near the
  hand can be removed with the hand.
- On WSLg, MuJoCo + RViz needs the lean
  `sim2real_perception/rviz/mapper.rviz` (no Camera/Image displays) and an 8 s
  RViz delay; the mapper launch provides both. The full `sim2real.rviz` camera
  panels can crash WSLg under MuJoCo.
- OctoMap is deferred.

## Safety

This project has no physical robot and no implemented safety system. Simulator
success is not a safety case. The Task 6 map is a surface TSDF, not an
occupancy or collision map; see `docs/SAFETY.md`.

## Layout

```
src/sim2real/        pure-Python core (no rclpy)
ros2_ws/src/         ROS 2 interface + adapter packages
configs/             per-subsystem YAML (schema_version: 1)
docs/                architecture, datasets, safety, deployment, master spec
scripts/             environment setup, test entry points, diagnostics
tests/               hardware-free tests and analytic fixtures
data/                ignored; holds datasets/rollouts locally
```

## Documentation

- `CLAUDE.md` — verified environment, canonical commands, current status,
  known limitations, decision log summary.
- `docs/ARCHITECTURE.md` — design, contract table, QoS, mapping pipeline,
  extrinsic convention, Open3D decision, limitations.
- `docs/SAFETY.md` — what the map can and cannot guarantee.
- `docs/FILTER_METRICS.md` — filtering metrics and threshold provenance.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
