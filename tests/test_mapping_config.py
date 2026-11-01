"""Mapping config validation and workspace derivation."""

from __future__ import annotations

import math

import pytest

from sim2real.mapping.config import (
    ConfigError,
    estimate_memory_mb,
    load_mapping_config,
)
from tests.fixtures.mapping import make_config, write_config


def test_default_config_loads_and_matches_scene_workspace():
    config = load_mapping_config()
    assert config.voxel_size_m == 0.01
    assert config.sdf_trunc_m == 0.04
    assert config.workspace_bounds.minimum == (0.25, -0.30, 0.42)
    assert config.workspace_bounds.maximum == (0.65, 0.30, 0.75)
    assert config.sdf_trunc_m >= 2 * config.voxel_size_m
    assert config.depth_min_m < config.depth_trunc_m
    assert config.estimated_memory_mb < config.max_memory_mb
    assert config.estimated_memory_mb < 300.0


def test_workspace_bounds_are_derived_when_absent(tmp_path):
    import yaml

    path = write_config(tmp_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    del raw["mapping"]["workspace_bounds"]
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_mapping_config(path)
    assert config.workspace_bounds.minimum == (0.25, -0.30, 0.42)


@pytest.mark.parametrize(
    ("overrides", "needle"),
    [
        ({"voxel_size_m": 0.0}, "mapping.voxel_size_m"),
        ({"voxel_size_m": -0.01}, "mapping.voxel_size_m"),
        ({"voxel_size_m": 0.05, "sdf_trunc_m": 0.09}, "mapping.sdf_trunc_m"),
        ({"depth_min_m": 2.0, "depth_trunc_m": 2.0}, "mapping.depth_min_m"),
        ({"depth_min_m": 2.0, "depth_trunc_m": 1.0}, "mapping.depth_min_m"),
        ({"voxel_size_m": float("nan")}, "mapping.voxel_size_m"),
        (
            {"voxel_size_m": 0.002, "sdf_trunc_m": 0.004, "max_memory_mb": 1.0},
            "mapping.max_memory_mb",
        ),
        ({"min_coverage_ratio": 1.5}, "mapping.min_coverage_ratio"),
        ({"unknown_key": 1}, "unknown keys"),
        ({"workspace_bounds": {"min": [0, 0, 0], "max": [0, 0, 0]}}, "workspace_bounds"),
    ],
)
def test_invalid_configs_report_key_path(tmp_path, overrides, needle):
    with pytest.raises(ConfigError) as excinfo:
        make_config(tmp_path, **overrides)
    assert needle in str(excinfo.value)


def test_memory_estimate_scales_with_voxel_size():
    from sim2real.mapping.config import WorkspaceBounds

    bounds = WorkspaceBounds((-0.2, -0.2, -0.1), (0.2, 0.2, 0.2))
    coarse = estimate_memory_mb(bounds, 0.02, 0.04)
    fine = estimate_memory_mb(bounds, 0.01, 0.04)
    assert fine > coarse > 0.0
    assert math.isfinite(fine)
