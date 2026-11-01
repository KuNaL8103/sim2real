"""Config helpers shared by mapping tests."""

from __future__ import annotations

from pathlib import Path

import yaml

from sim2real.mapping.config import MappingConfig, load_mapping_config

BASE_MAPPING: dict[str, object] = {
    "voxel_size_m": 0.01,
    "sdf_trunc_m": 0.04,
    "depth_trunc_m": 2.0,
    "depth_min_m": 0.05,
    "workspace_bounds": {"min": [-0.2, -0.2, -0.1], "max": [0.2, 0.2, 0.2]},
    "query_resolution_m": 0.02,
    "max_query_distance_m": 0.5,
    "publish_rate_hz": 2.0,
    "max_triangles": 1000,
    "max_integrations_before_reset": 100,
    "max_memory_mb": 256.0,
    "map_stale_after_s": 5.0,
    "min_coverage_ratio": 0.1,
    "sync_slop_s": 0.05,
    "reset_on_static_tf_change": True,
    "static_tf_translation_epsilon_m": 0.005,
    "static_tf_rotation_epsilon_rad": 0.01,
    "self_filter": {
        "link_frames": ["panda_link0", "panda_hand"],
        "radii_m": {"panda_link0": 0.08, "panda_hand": 0.06},
    },
}


def write_config(tmp_path: Path, **overrides: object) -> Path:
    body = dict(BASE_MAPPING)
    body.update(overrides)
    path = tmp_path / "mapping.yaml"
    path.write_text(yaml.safe_dump({"schema_version": 1, "mapping": body}), encoding="utf-8")
    return path


def make_config(tmp_path: Path, **overrides: object) -> MappingConfig:
    return load_mapping_config(write_config(tmp_path, **overrides))
