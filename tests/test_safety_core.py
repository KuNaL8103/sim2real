"""Hardware-free tests for the safety core (no ROS, no torch, no simulator)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sim2real import contracts
from sim2real.mapping.config import WorkspaceBounds
from sim2real.mapping.obstacle_query import NO_DATA, STALE, MapStatus, ObstacleQuery
from sim2real.safety import (
    ACCELERATION_LIMIT,
    CONTINUITY,
    EMPTY_CHUNK,
    FUTURE_STAMP,
    JERK_LIMIT,
    JOINT_NAMES_MISMATCH,
    MAP_LOW_COVERAGE,
    MAP_NO_DATA,
    MAP_STALE,
    MAP_UNAVAILABLE,
    NON_FINITE,
    NON_INCREASING_TIME,
    OBSTACLE_TOO_CLOSE,
    OK,
    OUTSIDE_MAP,
    POSITION_LIMIT,
    STALE_CHUNK,
    TOO_MANY_POINTS,
    UNSAFE,
    VELOCITY_LIMIT,
    WORKSPACE_VIOLATION,
    RawChunk,
    SafetyGate,
    ValidationContext,
    check_joint_limits,
    check_obstacle_distance,
    check_workspace,
    ema_targets,
    end_effector_positions,
    load_safety_config,
    validate_chunk,
)
from sim2real.sim.kinematics import tcp_pose

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "safety" / "default.yaml"
PANDA_PATH = ROOT / "configs" / "sim" / "panda.yaml"
STAMP = 1_000_000_000


@pytest.fixture(scope="module")
def safety():
    config, panda = load_safety_config(CONFIG_PATH, panda_config_path=PANDA_PATH)
    return config, panda


def _home(panda) -> np.ndarray:
    return np.array([*panda.home_arm, panda.home_gripper], dtype=np.float64)


def _map(points, *, status=None, resolution=0.02):
    bounds = WorkspaceBounds((0.2, -0.4, 0.30), (0.8, 0.4, 0.90))
    return ObstacleQuery(
        np.asarray(points, dtype=np.float64).reshape(-1, 3),
        bounds,
        resolution,
        0.6,
        status=status,
    )


def _valid_status(**overrides) -> MapStatus:
    values = {
        "valid": True,
        "age_s": 0.0,
        "num_integrations": 1,
        "coverage_ratio": 1.0,
        "reasons": (),
    }
    values.update(overrides)
    return MapStatus(**values)


def _far_map():
    return _map([[0.20, 0.40, 0.30], [0.25, 0.35, 0.31]], status=_valid_status())


def _context(query=None, status=None, *, now=STAMP, received=100.0, now_monotonic=100.0):
    if query is None:
        query = _far_map()
    if status is None:
        status = _valid_status()
    return ValidationContext(
        now_stamp_ns=now,
        now_monotonic_s=now_monotonic,
        received_monotonic_s=received,
        query=query,
        map_status=status,
    )


def _chunk(positions, *, offsets=None, stamp_ns=STAMP, names=None) -> RawChunk:
    positions = np.asarray(positions, dtype=np.float64)
    if offsets is None:
        offsets = tuple(0.1 * index for index in range(positions.shape[0]))
    return RawChunk(
        stamp_ns=stamp_ns,
        joint_names=tuple(contracts.CONTROLLED_JOINT_NAMES if names is None else names),
        positions=positions,
        time_from_start_s=tuple(offsets),
    )


def _home_chunk(panda, count=3) -> RawChunk:
    return _chunk(np.tile(_home(panda), (count, 1)))


# ---------------------------------------------------------------------------
# 1. Limits flip exactly at the threshold, with the margin applied
# ---------------------------------------------------------------------------


def test_position_limit_flips_at_margin_shrunk_bound(safety):
    config, panda = safety
    joint = 0
    limit = float(config.effective_joint_upper_rad[joint])
    measured = _home(panda)
    measured[joint] = limit
    command = measured.reshape(1, 8).copy()
    at = check_joint_limits(command, (config.command_period_s,), measured, config)
    assert at.ok
    assert at.ratios.position == pytest.approx(1.0)

    command[0, joint] = limit + 1e-6
    above = check_joint_limits(command, (config.command_period_s,), measured, config)
    assert not above.ok
    assert above.reason == POSITION_LIMIT
    assert above.ratios.position > 1.0


def test_velocity_limit_flips_at_margin_shrunk_bound(safety):
    config, panda = safety
    limit = float(config.effective_velocity_limits_rad_s[0])
    dt = config.command_period_s
    measured = _home(panda)
    for scale, expected in ((0.999, True), (1.001, False)):
        speed = limit * scale
        command = np.tile(measured, (2, 1))
        command[0, 0] += speed * dt
        command[1, 0] += 2.0 * speed * dt
        result = check_joint_limits(command, (dt, 2 * dt), measured, config)
        assert result.ok is expected
        if not expected:
            assert result.reason == VELOCITY_LIMIT
        assert result.ratios.velocity == pytest.approx(scale, rel=1e-9)


def test_acceleration_limit_flips_at_margin_shrunk_bound(safety):
    config, panda = safety
    limit = float(config.effective_acceleration_limits_rad_s2[0])
    dt = 0.01
    measured = _home(panda)
    for scale, expected in ((0.999, True), (1.001, False)):
        acceleration = limit * scale
        command = np.tile(measured, (3, 1))
        command[:, 0] += 0.5 * acceleration * np.asarray([dt, 2 * dt, 3 * dt]) ** 2
        result = check_joint_limits(command, (dt, 2 * dt, 3 * dt), measured, config)
        assert result.ok is expected
        if not expected:
            assert result.reason == ACCELERATION_LIMIT
        assert result.ratios.acceleration == pytest.approx(scale, rel=1e-8)


def test_jerk_limit_flips_at_margin_shrunk_bound(safety):
    config, panda = safety
    limit = float(config.effective_jerk_limits_rad_s3[0])
    # Small dt keeps the constant-jerk profile inside the velocity/acceleration
    # limits so the expected failure is specifically JERK_LIMIT.
    dt = 1e-4
    measured = _home(panda)
    for scale, expected in ((0.999, True), (1.001, False)):
        jerk = limit * scale
        times = np.asarray([dt, 2 * dt, 3 * dt, 4 * dt])
        command = np.tile(measured, (4, 1))
        command[:, 0] += jerk * times**3 / 6.0
        result = check_joint_limits(command, tuple(times), measured, config)
        assert result.ok is expected
        if not expected:
            assert result.reason == JERK_LIMIT
        assert result.ratios.jerk == pytest.approx(scale, rel=1e-7)


# ---------------------------------------------------------------------------
# 2. Workspace via the real FK
# ---------------------------------------------------------------------------


def test_workspace_inside_outside_and_boundary_use_real_fk(safety):
    config, panda = safety
    home = _home(panda)
    command = home.reshape(1, 8)
    inside = check_workspace(command, panda, config)
    assert inside.ok
    assert inside.ee_positions[0] == pytest.approx(tcp_pose(home[:7], panda)[:3, 3])

    ee = end_effector_positions(command, panda)[0]
    on_boundary = replace(config, workspace_max_m=np.array([ee[0], ee[1], ee[2]]))
    boundary = check_workspace(command, panda, on_boundary)
    assert boundary.ok
    assert boundary.worst_margin_m == pytest.approx(0.0, abs=1e-12)

    outside_config = replace(config, workspace_max_m=np.array([ee[0] - 1e-6, ee[1], ee[2]]))
    outside = check_workspace(command, panda, outside_config)
    assert not outside.ok
    assert outside.reason == WORKSPACE_VIOLATION
    assert outside.outside_indices == (0,)


# ---------------------------------------------------------------------------
# 3. Obstacle distance and unknown/stale/invalid maps
# ---------------------------------------------------------------------------


def test_obstacle_near_far_and_boundary(safety):
    config, _panda = safety
    point = np.array([[0.55449948, 0.0, 0.52450243]])
    near = _map([point[0] + [0.0, 0.0, 0.02]], status=_valid_status())
    clearance = check_obstacle_distance(point, near, _valid_status(), config)
    assert clearance.ok
    distance = clearance.min_clearance_m

    boundary = check_obstacle_distance(
        point, near, _valid_status(), replace(config, min_obstacle_distance_m=distance)
    )
    assert boundary.ok
    above = check_obstacle_distance(
        point,
        near,
        _valid_status(),
        replace(config, min_obstacle_distance_m=distance + 1e-6),
    )
    assert not above.ok
    assert above.reason == OBSTACLE_TOO_CLOSE

    far_point = np.array([[0.79, 0.39, 0.89]])
    far = check_obstacle_distance(far_point, _far_map(), _valid_status(), config)
    assert far.ok


@pytest.mark.parametrize(
    "query,status,expected",
    [
        (None, None, MAP_UNAVAILABLE),
        (
            _map([], status=_valid_status(valid=False, reasons=(NO_DATA,))),
            _valid_status(valid=False, reasons=(NO_DATA,)),
            MAP_NO_DATA,
        ),
        (
            _map([[0.21, 0.39, 0.31]], status=_valid_status(valid=False, reasons=(STALE,))),
            _valid_status(valid=False, reasons=(STALE,)),
            MAP_STALE,
        ),
        (
            _map([[0.21, 0.39, 0.31]], status=_valid_status(age_s=9.0)),
            _valid_status(age_s=9.0),
            MAP_STALE,
        ),
        (
            _map([[0.21, 0.39, 0.31]], status=_valid_status(coverage_ratio=0.0)),
            _valid_status(coverage_ratio=0.0),
            MAP_LOW_COVERAGE,
        ),
        (_map([], status=_valid_status()), _valid_status(), MAP_NO_DATA),
    ],
)
def test_unknown_missing_stale_and_invalid_maps_follow_map_required(
    safety, query, status, expected
):
    config, _panda = safety
    point = np.array([[0.55, 0.0, 0.52]])
    result = check_obstacle_distance(point, query, status, config)
    assert not result.ok
    assert result.reason == expected

    allowed = check_obstacle_distance(point, query, status, replace(config, map_required=False))
    assert allowed.ok
    assert not allowed.map_valid
    assert np.isnan(allowed.min_clearance_m)


def test_end_effector_outside_the_map_grid_is_not_free_space(safety):
    config, _panda = safety
    result = check_obstacle_distance(
        np.array([[9.0, 9.0, 9.0]]), _far_map(), _valid_status(), config
    )
    assert not result.ok
    assert result.reason == OUTSIDE_MAP


# ---------------------------------------------------------------------------
# 4. EMA
# ---------------------------------------------------------------------------


def test_ema_known_step_response_and_passthrough():
    target = np.ones((3, 8))
    measured = np.zeros(8)
    output = ema_targets(target, measured, 0.5)
    assert output[:, 0] == pytest.approx([0.5, 0.75, 0.875])
    passthrough = ema_targets(target, measured, 1.0)
    assert np.array_equal(passthrough, target)


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.1, float("nan")])
def test_ema_alpha_bounds_are_rejected(alpha):
    with pytest.raises(ValueError):
        ema_targets(np.zeros((1, 8)), np.zeros(8), alpha)


def test_ema_first_sample_is_initialised_from_measured_state():
    measured = np.full(8, 0.2)
    target = np.full((1, 8), 1.0)
    output = ema_targets(target, measured, 0.25)
    assert output[0, 0] == pytest.approx(0.25 * 1.0 + 0.75 * 0.2)


# ---------------------------------------------------------------------------
# 5. Malformed chunks
# ---------------------------------------------------------------------------


def test_valid_chunk_is_accepted_with_final_times(safety):
    config, panda = safety
    chunk = _home_chunk(panda, 3)
    decision = validate_chunk(chunk, _home(panda), config, panda, _context())
    assert decision.accepted, decision.reason
    assert decision.command is not None
    assert decision.command.header_stamp_ns == 0
    # The raw k=0 point is at/before "now" and is dropped, matching the Task
    # 10b latency observation. The remaining points keep their offsets.
    assert decision.command.time_from_start_s == pytest.approx((0.1, 0.2))
    assert decision.command.joint_names == contracts.CONTROLLED_JOINT_NAMES
    assert decision.command.positions == pytest.approx(np.tile(_home(panda), (2, 1)))


@pytest.mark.parametrize(
    "mutator,expected",
    [
        (lambda c: replace(c, positions=np.vstack([c.positions, c.positions[-1]])), None),
        (lambda c: replace(c, joint_names=tuple(reversed(c.joint_names))), JOINT_NAMES_MISMATCH),
        (lambda c: replace(c, time_from_start_s=(0.1, 0.1, 0.3)), NON_INCREASING_TIME),
        (lambda c: replace(c, stamp_ns=STAMP + int(1e9)), FUTURE_STAMP),
        (lambda c: replace(c, positions=c.positions[0:0]), EMPTY_CHUNK),
    ],
)
def test_malformed_chunks_are_rejected(safety, mutator, expected):
    config, panda = safety
    chunk = mutator(_home_chunk(panda, 3))
    decision = validate_chunk(chunk, _home(panda), config, panda, _context())
    if expected is None:
        # Appending a point without extending time_from_start is a length error.
        assert not decision.accepted
        return
    assert not decision.accepted
    assert decision.reason == expected


def test_nan_and_infinity_are_rejected(safety):
    config, panda = safety
    for value in (np.nan, np.inf, -np.inf):
        chunk = _home_chunk(panda, 3)
        positions = np.array(chunk.positions, copy=True)
        positions[1, 2] = value
        decision = validate_chunk(
            replace(chunk, positions=positions), _home(panda), config, panda, _context()
        )
        assert not decision.accepted
        assert decision.reason == NON_FINITE


def test_too_many_points_are_rejected(safety):
    config, panda = safety
    chunk = _chunk(np.tile(_home(panda), (config.max_chunk_points + 1, 1)))
    decision = validate_chunk(chunk, _home(panda), config, panda, _context())
    assert not decision.accepted
    assert decision.reason == TOO_MANY_POINTS


# ---------------------------------------------------------------------------
# 6. Finger clipping
# ---------------------------------------------------------------------------


def test_finger_is_clipped_with_counter_and_flag(safety):
    config, panda = safety
    positions = np.tile(_home(panda), (3, 1))
    positions[0, 7] = -0.02
    positions[1, 7] = 0.10
    decision = validate_chunk(_chunk(positions), _home(panda), config, panda, _context())
    assert decision.accepted, decision.reason
    assert decision.finger_clipped
    assert decision.finger_clipped_points == 2
    assert decision.command is not None
    assert np.all(decision.command.positions[:, 7] >= 0.0)
    assert np.all(decision.command.positions[:, 7] <= 0.04)


# ---------------------------------------------------------------------------
# 7. Continuity gate
# ---------------------------------------------------------------------------


def test_continuity_gate_rejects_a_far_first_point(safety):
    config, panda = safety
    measured = _home(panda)
    positions = np.tile(measured, (3, 1))
    # The raw k=0 point is dropped as past; index 1 is the first command point.
    # EMA halves the step, so make the raw jump large enough that the smoothed
    # first command point still exceeds the gate.
    positions[1, 0] += 2.0 * config.max_first_point_distance_rad + 0.05
    decision = validate_chunk(_chunk(positions), measured, config, panda, _context())
    assert not decision.accepted
    assert decision.reason == CONTINUITY

    near = np.tile(measured, (3, 1))
    near[1, 0] += 0.02
    accepted = validate_chunk(_chunk(near), measured, config, panda, _context())
    assert accepted.accepted, accepted.reason


# ---------------------------------------------------------------------------
# 8-9. Stale policies and retiming
# ---------------------------------------------------------------------------


def test_reject_policy_all_past_and_age_boundary(safety):
    config, panda = safety
    config = replace(config, max_chunk_age_s=0.2)
    measured = _home(panda)
    chunk = _chunk(np.tile(measured, (3, 1)), offsets=(0.2, 0.3, 0.4))

    at_boundary = validate_chunk(
        chunk,
        measured,
        config,
        panda,
        _context(now=STAMP + int(config.max_chunk_age_s * 1e9)),
    )
    # At exactly max age the first point is still current, so it is accepted.
    assert at_boundary.accepted, at_boundary.reason

    too_old = validate_chunk(
        chunk,
        measured,
        config,
        panda,
        _context(now=STAMP + int((config.max_chunk_age_s + 1e-3) * 1e9)),
    )
    assert not too_old.accepted
    assert too_old.reason == STALE_CHUNK

    all_past = validate_chunk(
        _chunk(np.tile(measured, (3, 1)), offsets=(0.0, 0.1, 0.2)),
        measured,
        config,
        panda,
        _context(now=STAMP + int(1.0e9)),
    )
    assert not all_past.accepted
    assert all_past.reason == STALE_CHUNK


def test_reject_policy_trims_a_partially_past_chunk_without_retiming(safety):
    config, panda = safety
    measured = _home(panda)
    decision = validate_chunk(
        _chunk(np.tile(measured, (3, 1)), offsets=(0.0, 0.1, 0.2)),
        measured,
        config,
        panda,
        _context(now=STAMP + int(0.05e9)),
    )
    assert decision.accepted, decision.reason
    assert decision.dropped_points == 1
    assert decision.command is not None
    # Reject keeps the original future offsets; it does not replay from now.
    assert decision.command.time_from_start_s == pytest.approx((0.1, 0.2))


def test_retime_policy_drops_past_points_and_retimes_from_now(safety):
    config, panda = safety
    config = replace(config, stale_chunk_policy="retime_if_continuous")
    measured = _home(panda)
    chunk = _chunk(np.tile(measured, (5, 1)), offsets=(0.0, 0.1, 0.2, 0.3, 0.4))
    decision = validate_chunk(
        chunk,
        measured,
        config,
        panda,
        _context(now=STAMP + int(0.25e9)),
    )
    assert decision.accepted, decision.reason
    assert decision.command is not None
    assert decision.dropped_points == 3
    assert decision.command.time_from_start_s == pytest.approx((0.1, 0.2))
    assert decision.command.header_stamp_ns == 0


# ---------------------------------------------------------------------------
# 10-12. Gate, e-stop, watchdog
# ---------------------------------------------------------------------------


def test_watchdog_recovery_and_estop_latch(safety):
    config, _panda = safety
    gate = SafetyGate(config)
    changed, state, reason = gate.update_inputs(
        joint_state_age_s=0.05, map_age_s=0.1, map_valid=True
    )
    assert changed and state == OK and reason is None

    changed, state, reason = gate.update_inputs(
        joint_state_age_s=1.0, map_age_s=0.1, map_valid=True
    )
    assert changed and state == UNSAFE

    changed, state, _reason = gate.update_inputs(
        joint_state_age_s=0.05, map_age_s=0.1, map_valid=True
    )
    assert changed and state == OK

    assert gate.trigger_estop()
    assert gate.estop_latched
    assert not gate.trigger_estop()  # idempotent
    assert gate.state == "ESTOP"
    ok, message = gate.reset(inputs_fresh=False)
    assert not ok and "refused" in message
    ok, _message = gate.reset(inputs_fresh=True)
    assert ok
    assert not gate.estop_latched


def test_map_unavailable_is_unsafe_when_required(safety):
    config, _panda = safety
    gate = SafetyGate(config)
    _changed, state, reason = gate.update_inputs(
        joint_state_age_s=0.05, map_age_s=None, map_valid=False
    )
    assert state == UNSAFE
    assert reason in {"WATCHDOG_MAP_UNAVAILABLE", "WATCHDOG_MAP_STALE"}


def test_map_stale_later_is_unsafe(safety):
    config, _panda = safety
    gate = SafetyGate(config)
    gate.update_inputs(joint_state_age_s=0.05, map_age_s=0.1, map_valid=True)
    _changed, state, reason = gate.update_inputs(
        joint_state_age_s=0.05, map_age_s=config.max_map_age_s + 1.0, map_valid=True
    )
    assert state == UNSAFE
    assert reason == "WATCHDOG_MAP_STALE"


# ---------------------------------------------------------------------------
# 16. Config validation
# ---------------------------------------------------------------------------


def test_shipped_config_requires_verified_jerk_and_effort(safety):
    config, panda = safety
    assert panda.jerk_limits is not None
    assert panda.effort_limits is not None
    assert panda.torque_rate_limits is not None
    assert config.jerk_limits_rad_s3.tolist() == [
        7500.0,
        3750.0,
        5000.0,
        6250.0,
        7500.0,
        10000.0,
        10000.0,
    ]
    assert config.effort_limits_nm.tolist() == [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]


def test_unknown_config_keys_and_bad_bounds_are_rejected(tmp_path):
    import yaml

    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    path = tmp_path / "safety.yaml"
    raw["safety"]["unknown_key"] = 1
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_safety_config(path, panda_config_path=PANDA_PATH)

    raw["safety"].pop("unknown_key")
    raw["safety"]["ema_alpha"] = 0.0
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="ema_alpha"):
        load_safety_config(path, panda_config_path=PANDA_PATH)

    raw["safety"]["ema_alpha"] = 0.5
    raw["safety"]["stale_chunk_policy"] = "ignore"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="stale_chunk_policy"):
        load_safety_config(path, panda_config_path=PANDA_PATH)
