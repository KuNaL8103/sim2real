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
