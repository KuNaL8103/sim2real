"""Task 11b: strict vs masked obstacle gating on measured expert chunks."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from sim2real.expert import parse_instruction, run_episodes
from sim2real.expert.config import load_expert_config
from sim2real.mapping.config import WorkspaceBounds
from sim2real.mapping.obstacle_query import MapStatus, ObstacleQuery
from sim2real.safety import (
    MAP_NO_DATA,
    OBSTACLE_TOO_CLOSE,
    ValidationContext,
    check_obstacle_distance,
    load_safety_config,
    validate_chunk,
)
from sim2real.sim import create_backend
from sim2real.sim.config import load_sim_configs
from tests.test_safety_expert_false_positive import _chunks, _map_points

ROOT = Path(__file__).resolve().parents[1]
STAMP = 1_000_000_000
SEED_SETS = (tuple(range(20900, 20910)), tuple(range(20910, 20920)))


@pytest.fixture(scope="module")
def context():
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
        status=MapStatus(valid=True, age_s=0.0, num_integrations=1, coverage_ratio=1.0),
    )
    cube = configs.scene.cubes["cube_red"]
    backend = create_backend("fake", str(ROOT / "configs" / "sim"))
    try:
        records = {}
        for seeds in SEED_SETS:
            records[seeds] = run_episodes(
                backend,
                task,
                seeds,
                instruction="pick up the red cube and place it in the left zone",
                run_id=f"task11b_{seeds[0]}",
                randomization_profile="none",
                config=load_expert_config(ROOT / "configs" / "expert"),
            )
    finally:
        backend.close()
    return {
        "configs": configs,
        "safety": safety,
        "panda": panda,
        "query": query,
        "records": records,
        "target_center": np.asarray(cube.position, dtype=np.float64),
        "target_half": float(cube.size) / 2.0,
    }


def _evaluate(ctx, seeds, mode):
    config = replace(ctx["safety"], obstacle_mode=mode)
    accepted = rejected = 0
    reasons = Counter()
    for record in ctx["records"][seeds]:
        for chunk, measured in _chunks(record):
            decision = validate_chunk(
                chunk,
                measured,
                config,
                ctx["panda"],
                ValidationContext(
                    now_stamp_ns=STAMP,
                    now_monotonic_s=1.0,
                    received_monotonic_s=1.0,
                    query=ctx["query"],
                    map_status=ctx["query"].status,
                    target_center_m=ctx["target_center"],
                    target_half_extent_m=ctx["target_half"],
                ),
            )
            if decision.accepted:
                accepted += 1
            else:
                rejected += 1
                reasons[decision.reason or "UNKNOWN"] += 1
    return accepted, rejected, reasons


@pytest.mark.parametrize("seeds", SEED_SETS, ids=["20900-20909", "20910-20919"])
def test_masked_mode_removes_target_object_false_positives(context, seeds):
    strict_ok, strict_rej, strict_reasons = _evaluate(context, seeds, "strict")
    masked_ok, masked_rej, masked_reasons = _evaluate(context, seeds, "masked")
    print(
        f"seeds={seeds[0]}-{seeds[-1]} strict={strict_ok}/{strict_rej} "
        f"{dict(strict_reasons)} masked={masked_ok}/{masked_rej} "
        f"{dict(masked_reasons)}",
        flush=True,
    )
    assert strict_rej == 10, "strict regression baseline changed"
    assert strict_reasons == {OBSTACLE_TOO_CLOSE: 10}
    assert masked_rej == 0, masked_reasons
    assert masked_ok == strict_ok + strict_rej


def test_injected_obstacle_is_rejected_in_both_modes(context):
    safety = context["safety"]
    center = np.array([0.55, 0.0, 0.52])
    bounds = WorkspaceBounds((0.2, -0.4, 0.3), (0.8, 0.4, 0.9))
    status = MapStatus(valid=True, age_s=0.0, num_integrations=1, coverage_ratio=1.0)
    for distance in (0.002, 0.005, 0.009):
        block = _make_query(np.array([center + [0.0, 0.0, distance]]), bounds, status)
        for mode in ("strict", "masked"):
            result = check_obstacle_distance(
                center.reshape(1, 3),
                block,
                status,
                replace(safety, obstacle_mode=mode),
            )
            assert not result.ok, (mode, distance)
            assert result.reason == OBSTACLE_TOO_CLOSE, (mode, distance, result.reason)


def test_table_band_is_ignored_only_inside_the_band(context):
    safety = replace(context["safety"], obstacle_mode="masked")
    bounds = WorkspaceBounds((0.2, -0.4, 0.3), (0.8, 0.4, 0.9))
    status = MapStatus(valid=True, age_s=0.0, num_integrations=1, coverage_ratio=1.0)
    ee = np.array([[0.45, 0.0, 0.42]])
    inside = _make_query(np.array([[0.45, 0.0, 0.424], [0.65, 0.0, 0.52]]), bounds, status)
    assert check_obstacle_distance(ee, inside, status, safety).ok
    above = _make_query(np.array([[0.45, 0.0, 0.435]]), bounds, status)
    result = check_obstacle_distance(np.array([[0.45, 0.0, 0.435]]), above, status, safety)
    assert not result.ok
    assert result.reason == OBSTACLE_TOO_CLOSE


def test_masked_mode_without_target_context_still_masks_only_the_table(context):
    safety = replace(context["safety"], obstacle_mode="masked")
    bounds = WorkspaceBounds((0.2, -0.4, 0.3), (0.8, 0.4, 0.9))
    status = MapStatus(valid=True, age_s=0.0, num_integrations=1, coverage_ratio=1.0)
    ee = np.array([[0.45, -0.12, 0.46]])
    cube = _make_query(np.array([[0.45, -0.12, 0.46], [0.65, 0.0, 0.52]]), bounds, status)
    result = check_obstacle_distance(ee, cube, status, safety)
    assert not result.ok
    assert result.reason == OBSTACLE_TOO_CLOSE

    # With an explicit target-object context the same point is allowed inside
    # the approach volume; this is the deliberate, task-conditioned exception.
    allowed = check_obstacle_distance(
        ee,
        cube,
        status,
        safety,
        target_center_m=np.array([0.45, -0.12, 0.44]),
        target_half_extent_m=0.02,
    )
    assert allowed.ok


def test_new_config_keys_are_validated(tmp_path):
    raw = yaml.safe_load((ROOT / "configs" / "safety" / "default.yaml").read_text())
    path = tmp_path / "safety.yaml"
    raw["safety"]["obstacle_mode"] = "ignore"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="obstacle_mode"):
        load_safety_config(path, panda_config_path=ROOT / "configs" / "sim" / "panda.yaml")

    raw["safety"]["obstacle_mode"] = "masked"
    raw["safety"]["table_mask_band_m"] = -0.01
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="table_mask_band_m"):
        load_safety_config(path, panda_config_path=ROOT / "configs" / "sim" / "panda.yaml")

    raw["safety"]["table_mask_band_m"] = 0.01
    raw["safety"]["unknown_key"] = 1
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_safety_config(path, panda_config_path=ROOT / "configs" / "sim" / "panda.yaml")


def _make_query(points, bounds, status):
    return ObstacleQuery(points, bounds, 0.02, 0.5, status=status)


def test_masked_mode_with_an_empty_after_mask_map_is_not_clear(context):
    safety = replace(context["safety"], obstacle_mode="masked")
    bounds = WorkspaceBounds((0.2, -0.4, 0.3), (0.8, 0.4, 0.9))
    status = MapStatus(valid=True, age_s=0.0, num_integrations=1, coverage_ratio=1.0)
    table_only = _make_query(np.array([[0.45, 0.0, 0.424]]), bounds, status)
    result = check_obstacle_distance(np.array([[0.45, 0.0, 0.42]]), table_only, status, safety)
    assert not result.ok
    assert result.reason == MAP_NO_DATA
