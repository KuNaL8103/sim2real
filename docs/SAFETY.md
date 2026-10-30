# Safety

## Status

**Not yet implemented.** Task 1 defines interfaces only. There is no safety
watchdog, limit checker, obstacle monitor, watchdog timer, or e-stop handler in
this repository yet.

The interfaces reserved for Task 11 are:

- `/safety/status` (`sim2real_msgs/msg/SafetyStatus`)
- `/estop` (`std_srvs/srv/Trigger`, latched)
- `/estop_reset` (`std_srvs/srv/Trigger`, defined in Task 11)

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

## Planned Task 11 scope

- Joint position, velocity, acceleration, and jerk limit checks.
- Observation and action age checks.
- Obstacle map validity and minimum-distance checks.
- Latched e-stop and explicit reset.
- Safe-stop behaviour when any check fails.
- Tests for limit violations and stale data.

## Explicit non-claims

This repository does not claim functional safety, ISO 10218 / ISO/TS 15066
compliance, or readiness for physical deployment.
