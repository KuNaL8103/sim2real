"""RGB-D workspace mapping core (Open3D TSDF, self-filter, obstacle query)."""

from __future__ import annotations

from .config import (
    ConfigError,
    MappingConfig,
    SelfFilterConfig,
    WorkspaceBounds,
    estimate_memory_mb,
    load_mapping_config,
    validate_mapping_config,
)
from .mapper import CAMERA_FRAMES, MapperCounters, WorkspaceMapper
from .obstacle_query import (
    LOW_COVERAGE,
    NO_DATA,
    STALE,
    TF_FAILURES,
    MapStatus,
    ObstacleQuery,
    build_query_from_points,
)
from .self_filter import RobotSelfFilter, SelfFilterResult
from .transforms import (
    StaticTransformProvider,
    TransformProvider,
    camera_static_transforms,
    rotation_angle_rad,
    static_transforms_changed,
    transform_delta,
)
from .tsdf_fusion import (
    OPEN3D_AVAILABLE,
    IntegrationResult,
    MeshData,
    Open3DUnavailableError,
    PointCloudData,
    TSDFMap,
    TSDFStats,
)

__all__ = [
    "OPEN3D_AVAILABLE",
    "CAMERA_FRAMES",
    "ConfigError",
    "IntegrationResult",
    "LOW_COVERAGE",
    "MapStatus",
    "MapperCounters",
    "MappingConfig",
    "MeshData",
    "NO_DATA",
    "ObstacleQuery",
    "Open3DUnavailableError",
    "PointCloudData",
    "RobotSelfFilter",
    "STALE",
    "SelfFilterConfig",
    "SelfFilterResult",
    "StaticTransformProvider",
    "TF_FAILURES",
    "TSDFMap",
    "TSDFStats",
    "TransformProvider",
    "WorkspaceBounds",
    "WorkspaceMapper",
    "build_query_from_points",
    "camera_static_transforms",
    "estimate_memory_mb",
    "load_mapping_config",
    "rotation_angle_rad",
    "static_transforms_changed",
    "transform_delta",
    "validate_mapping_config",
]
