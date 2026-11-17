"""Task 11 H21: clean expert trajectories through the safety validator.

Seeds 20900-20909 are reserved for this check only.  The test reports the
false-positive rate and every rejection reason; it deliberately does not
assert a threshold, because loosening limits to make it pass without telling
the user would hide a real safety/trajectory mismatch.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from sim2real import contracts
from sim2real.expert import parse_instruction, run_episodes
from sim2real.expert.config import load_expert_config
from sim2real.mapping.config import WorkspaceBounds
from sim2real.mapping.obstacle_query import MapStatus, ObstacleQuery
from sim2real.safety import (
    RawChunk,
    ValidationContext,
    load_safety_config,
    validate_chunk,
)
from sim2real.sim import create_backend
from sim2real.sim.config import load_sim_configs

ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(20900, 20910))
STAMP = 1_000_000_000


def _map_points(configs) -> np.ndarray:
    scene = configs.scene
    table_top = float(scene.table_position[2] + scene.table_size[2] / 2.0)
    xs = np.linspace(scene.workspace_min[0], scene.workspace_max[0], 20)
    ys = np.linspace(scene.workspace_min[1], scene.workspace_max[1], 30)
    points = [[x, y, table_top + 0.001] for x in xs for y in ys]
    for cube in scene.cubes.values():
        half = cube.size / 2.0
        axis = np.linspace(-half, half, 5)
        for u in axis:
            for v in axis:
                x, y, z = cube.position
                points.extend(
                    [
                        [x + u, y + v, z + half],
                        [x + u, y + v, z - half],
                        [x + u, y + half, z + v],
                        [x + u, y - half, z + v],
                        [x + half, y + u, z + v],
                        [x - half, y + u, z + v],
                    ]
                )
    return np.asarray(points, dtype=np.float64)


def _chunks(record):
    steps = record.steps
    if len(steps) < 20:
        return
    for start in range(0, len(steps) - 20, 20):
        window = steps[start : start + 20 : 2]
        if len(window) != 10:
            continue
        measured = np.array(
            [*steps[start].joint_positions, steps[start].gripper_position], dtype=np.float64
        )
        positions = np.asarray([step.action for step in window], dtype=np.float64)
        yield (
            RawChunk(
                stamp_ns=STAMP,
                joint_names=tuple(contracts.CONTROLLED_JOINT_NAMES),
                positions=positions,
                time_from_start_s=tuple(0.1 * index for index in range(len(window))),
            ),
            measured,
        )


def test_clean_expert_chunks_through_validator():
    configs = load_sim_configs(ROOT / "configs" / "sim")
    task = parse_instruction(
        "pick up the red cube and place it in the left zone",
        configs.scene,
        ROOT / "configs" / "expert" / "instructions.yaml",
    )
    safety, panda = load_safety_config(
        ROOT / "configs" / "safety" / "default.yaml",
        panda_config_path=ROOT / "configs" / "sim" / "panda.yaml",
    )
    bounds = WorkspaceBounds(
        tuple(float(v) for v in safety.workspace_min_m),
        tuple(float(v) for v in safety.workspace_max_m),
    )
    query = ObstacleQuery(
        _map_points(configs),
        bounds,
        safety.map_query_resolution_m,
        safety.map_max_query_distance_m,
    )
    status = MapStatus(valid=True, age_s=0.0, num_integrations=1, coverage_ratio=1.0, reasons=())
    backend = create_backend("fake", str(ROOT / "configs" / "sim"))
    try:
        records = run_episodes(
            backend,
            task,
            SEEDS,
            instruction="pick up the red cube and place it in the left zone",
            run_id="task11_safety_fp",
            randomization_profile="none",
            config=load_expert_config(ROOT / "configs" / "expert"),
        )
    finally:
        backend.close()

    total = 0
    rejected = 0
    reasons: Counter[str] = Counter()
    per_episode: list[tuple[int, int, int]] = []
    for record in records:
        episode_total = 0
        episode_rejected = 0
        for chunk, measured in _chunks(record):
            decision = validate_chunk(
                chunk,
                measured,
                safety,
                panda,
                ValidationContext(
                    now_stamp_ns=STAMP,
                    now_monotonic_s=1.0,
                    received_monotonic_s=1.0,
                    query=query,
                    map_status=status,
                ),
            )
            total += 1
            episode_total += 1
            if not decision.accepted:
                rejected += 1
                episode_rejected += 1
                reasons[decision.reason or "UNKNOWN"] += 1
        per_episode.append((record.seed, episode_total, episode_rejected))

    print(
        "H21 expert false-positive:",
        f"episodes={len(records)} chunks={total} rejected={rejected}",
        f"reasons={dict(sorted(reasons.items()))}",
        f"per_episode={per_episode}",
        flush=True,
    )
    assert len(records) == len(SEEDS)
    assert total > 0
    assert rejected <= total
