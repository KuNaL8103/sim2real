"""Parse .msg/.srv files with a tiny ROS-free parser and enforce conventions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MSGS_DIR = ROOT / "ros2_ws" / "src" / "sim2real_msgs"

BUILTIN_TYPES = {
    "bool",
    "byte",
    "char",
    "float32",
    "float64",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "string",
    "wstring",
}
KNOWN_PACKAGES = {"builtin_interfaces", "geometry_msgs", "std_msgs", "sim2real_msgs"}

FIELD_TYPE_RE = re.compile(
    r"^(?:(?P<package>[a-z][a-z0-9_]*)/[A-Z][A-Za-z0-9_]*|"
    r"(?P<builtin>" + "|".join(sorted(BUILTIN_TYPES)) + r"))(?:\[\d*\])?$"
)

INTERFACES = sorted([*MSGS_DIR.glob("msg/*.msg"), *MSGS_DIR.glob("srv/*.srv")])

REQUIRED_FIELDS = {
    "msg/TrajectoryStep.msg": {
        "stamp",
        "step_index",
        "task_instruction",
        "joint_names",
        "joint_positions",
        "joint_velocities",
        "ee_pose",
        "ee_frame_id",
        "action",
        "action_space",
    },
    "msg/RolloutEpisode.msg": {
        "episode_id",
        "schema_version",
        "backend",
        "seed",
        "task_instruction",
        "start_stamp",
        "control_rate_hz",
        "is_perturbed",
        "outcome",
        "failure_reason",
        "steps",
        "metadata_json",
    },
    "msg/FilterMetrics.msg": {
        "episode_id",
        "dimensionless_squared_jerk",
        "sparc",
        "path_length_m",
        "min_joint_limit_margin",
        "min_velocity_limit_margin",
        "collision_count",
        "task_success",
        "acceptance_score",
        "decision",
        "rejection_reasons",
        "metadata_json",
    },
    "msg/SafetyStatus.msg": {
        "header",
        "is_safe",
        "estop_latched",
        "state",
        "failure_reason",
        "obstacle_map_valid",
        "min_obstacle_distance_m",
        "max_joint_limit_ratio",
        "max_velocity_ratio",
        "max_acceleration_ratio",
        "max_jerk_ratio",
        "observation_age_s",
        "action_age_s",
    },
    "srv/ResetEnv.srv": {
        "seed",
        "randomization_enabled",
        "randomization_profile",
        "has_object_pose_override",
        "object_pose_override",
        "task_instruction",
        "success",
        "message",
        "applied_seed",
        "applied_config_json",
    },
    "srv/GenerateRollouts.srv": {
        "num_rollouts",
        "seed",
        "task_instruction",
        "sampling_mode",
        "failure_distribution_ref",
        "include_perturbed",
        "success",
        "num_generated",
        "num_failed",
        "message",
        "result_metadata_json",
    },
}

REQUIRED_CONSTANTS = {
    "msg/TrajectoryStep.msg": {
        "ACTION_JOINT_POSITION_TARGET": "0",
        "ACTION_JOINT_DELTA": "1",
        "ACTION_EE_DELTA": "2",
    },
    "msg/RolloutEpisode.msg": {
        "OUTCOME_UNKNOWN": "0",
        "OUTCOME_SUCCESS": "1",
        "OUTCOME_FAILURE": "2",
        "OUTCOME_TIMEOUT": "3",
        "OUTCOME_ABORTED": "4",
    },
    "msg/FilterMetrics.msg": {
        "DECISION_ACCEPT": "0",
        "DECISION_REJECT": "1",
        "DECISION_INVALID": "2",
    },
    "msg/SafetyStatus.msg": {
        "STATE_NOMINAL": "0",
        "STATE_DEGRADED": "1",
        "STATE_SAFE_STOP": "2",
        "STATE_ESTOP": "3",
    },
    "srv/GenerateRollouts.srv": {
        "SAMPLING_UNIFORM": "0",
        "SAMPLING_FAILURE_TARGETED": "1",
    },
}


@dataclass(frozen=True)
class Field:
    type_string: str
    name: str
    default: str | None
    section: int

    @property
    def is_constant(self) -> bool:
        return self.default is not None


def parse_interface(path: Path) -> list[Field]:
    fields: list[Field] = []
    section = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line == "---":
            section += 1
            continue
        if "=" in line:
            left, default = line.split("=", 1)
            parts = left.split()
            fields.append(Field(parts[0], parts[1], default.strip(), section))
        else:
            parts = line.split()
            fields.append(Field(parts[0], parts[1], None, section))
    return fields


def interface_ids() -> list[str]:
    return [path.relative_to(MSGS_DIR).as_posix() for path in INTERFACES]


def test_all_six_interfaces_exist():
    assert interface_ids() == sorted(REQUIRED_FIELDS)


@pytest.mark.parametrize("relative_path", interface_ids())
def test_required_fields_are_present(relative_path: str):
    path = MSGS_DIR / relative_path
    names = {field.name for field in parse_interface(path)}
    missing = REQUIRED_FIELDS[relative_path] - names
    assert not missing, f"{relative_path} missing fields: {sorted(missing)}"


@pytest.mark.parametrize("relative_path", interface_ids())
def test_field_types_are_valid_ros_types(relative_path: str):
    for field in parse_interface(MSGS_DIR / relative_path):
        match = FIELD_TYPE_RE.match(field.type_string)
        assert match, f"{relative_path}: invalid type {field.type_string!r} for {field.name}"
        package = match.group("package")
        if package is not None:
            assert package in KNOWN_PACKAGES, f"{relative_path}: unknown package {package}"


@pytest.mark.parametrize("relative_path", interface_ids())
def test_no_duplicate_field_names_within_a_section(relative_path: str):
    seen: set[tuple[int, str]] = set()
    for field in parse_interface(MSGS_DIR / relative_path):
        key = (field.section, field.name)
        assert key not in seen, f"{relative_path}: duplicate field {field.name} in {key[0]}"
        seen.add(key)


@pytest.mark.parametrize("relative_path", interface_ids())
def test_names_follow_ros_idl_conventions(relative_path: str):
    for field in parse_interface(MSGS_DIR / relative_path):
        if field.is_constant:
            assert re.fullmatch(r"[A-Z][A-Z0-9_]*", field.name), field.name
        else:
            assert re.fullmatch(r"[a-z][a-z0-9_]*", field.name), field.name


@pytest.mark.parametrize("relative_path", interface_ids())
def test_srv_files_have_exactly_one_separator(relative_path: str):
    path = MSGS_DIR / relative_path
    separators = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip() == "---"
    ]
    expected = 1 if relative_path.endswith(".srv") else 0
    assert len(separators) == expected, f"{relative_path} has {len(separators)} separators"


@pytest.mark.parametrize("relative_path", sorted(REQUIRED_CONSTANTS))
def test_required_constants_exist_with_expected_values(relative_path: str):
    constants = {
        field.name: field.default
        for field in parse_interface(MSGS_DIR / relative_path)
        if field.is_constant
    }
    for name, value in REQUIRED_CONSTANTS[relative_path].items():
        assert constants.get(name) == value, f"{relative_path}: {name} != {value}"
