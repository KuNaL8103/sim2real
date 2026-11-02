# Safety

## Status

**Safety logic is not implemented.** Task 1 defines interfaces only. There is no
safety watchdog, limit checker, obstacle monitor, watchdog timer, or e-stop
handler in this repository yet. Task 6 adds a workspace map and an
obstacle-distance query, but that query is a **helper for Task 11**, not a
safety layer.

The interfaces reserved for Task 11 are:

- `/safety/status` (`sim2real_msgs/msg/SafetyStatus`)
- `/estop` (`std_srvs/srv/Trigger`, latched)
- `/estop_reset` (`std_srvs/srv/Trigger`, defined in Task 11)

## What the Task 6 workspace map can and cannot guarantee

The map is a **surface** TSDF over a bounded workspace:

- It stores measured surface geometry within `sdf_trunc_m` of a depth return.
- It does **not** carve free space, so it cannot prove that a region is clear.
- It does not model unknown space explicitly. Unknown space is expressed only by
  `MapStatus` (`NO_DATA`, `STALE`, `LOW_COVERAGE`).
- A moved object leaves a ghost until the map is reset.
- The robot self-filter is a coarse sphere-per-link approximation. A grasped
  object near the hand can be removed with the hand, and geometry between links
  is not covered.
- Coverage is an xy-column proxy, not a completeness metric.

Therefore:

1. **An invalid `MapStatus` is unsafe.** `valid=False`, a stale age, or low
   coverage must stop or degrade motion; it must never be read as "no obstacle".
2. A distance above `max_query_distance_m` is a **clamp**, not proof of
   clearance. Points outside the grid are flagged by `outside_workspace`.
3. The map is not a collision mesh and must not be treated as one.
4. Task 11 must rebuild its `ObstacleQuery` from `/workspace/pointcloud` and
   must not share mutable map state across processes.


## Non-negotiable rules

1. **Mock hardware is not physical safety.** Passing simulator tests says
   nothing about a real robot, a real person, or a real workspace.
2. **No physical robot is connected.** This project targets simulation only.
3. **Raw VLA output never reaches a controller directly.** It must pass through
   the safety/control adapter (Task 11).
4. **`NaN` means unknown, not safe.** A missing obstacle distance or stale
   observation must never be interpreted as clearance.
5. **E-stop is latched.** It must not clear automatically; recovery is an
   explicit operator action via `/estop_reset` (Task 11).
6. **One `/joint_states` publisher.** Running the sim node and ros2_control
   together is a configuration error.
7. Safety thresholds must live in configuration, not in scattered literals.
8. **An invalid workspace map is unsafe.** `NO_DATA`, `STALE`, and
   `LOW_COVERAGE` block motion; they are not "clear".
9. **No free-space carving.** A missing obstacle point is not evidence that the
   space is free.
10. **The robot self-filter is coarse.** It removes link-centred spheres, not
    the true robot geometry, and can remove a grasped object.


## Limits verification (2026-11-02)

The official Franka Emika Robot (FER) limits are in the FCI page
<https://frankarobotics.github.io/docs/robot_specifications.html> (section
"Limits for Franka Emika Robot (FER)"). Values below are Joint 1..7 in order.
`configs/sim/panda.yaml` uses per-joint lists, not a single value.

| Quantity | `configs/sim/panda.yaml` | Official FER | Match? |
| --- | --- | --- | --- |
| `q_min` | -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973 | same | **yes** |
| `q_max` | 2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973 | same | **yes** |
| `dq_max` | 2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61 | 2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100 | **yes** |
| `ddq_max` | 15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0 | 15, 7.5, 10, 12.5, 15, 20, 20 | **yes** |
| `dddq_max` (jerk) | **absent** | 7500, 3750, 5000, 6250, 7500, 10000, 10000 rad/s^3 | **MISSING** |
| `tau_max` (effort) | **absent** | 87, 87, 87, 87, 12, 12, 12 Nm | **MISSING** |
| `dtau_max` | **absent** | 1000 Nm/s | **MISSING** |

Position, velocity, and acceleration are verified, not invented. Jerk, effort,
and torque rate are absent from the YAML and must be added deliberately before
Task 11. Components that depend on these limits:

- the expert planner uses `velocity_limits` and `acceleration_limits`;
- the filter's limit margins use joint and velocity limits;
- the future safety node needs position, velocity, acceleration, jerk, effort,
  and torque rate.

No YAML or limit code was changed in Task 6b. Recommendation: add the official
jerk/effort/torque-rate values to `configs/sim/panda.yaml` in a separate,
reviewed change before the safety work.



## Planned Task 11 scope

- Joint position, velocity, acceleration, and jerk limit checks.
- Observation and action age checks.
- Obstacle map validity and minimum-distance checks, built from
  `/workspace/pointcloud` with `sim2real.mapping.ObstacleQuery`.
- `MapStatus` gating: `NO_DATA`, `STALE`, and `LOW_COVERAGE` are unsafe.
- Latched e-stop and explicit reset.
- Safe-stop behaviour when any check fails.
- Tests for limit violations and stale data.

## Explicit non-claims

This repository does not claim functional safety, ISO 10218 / ISO/TS 15066
compliance, or readiness for physical deployment.
