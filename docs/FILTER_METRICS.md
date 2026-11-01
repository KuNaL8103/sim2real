# Trajectory filtering metrics (Task 5)

Status: implemented for the fake backend and MuJoCo. All formulas below are
computed from the **recorded executed state**, never from the commanded action
vector. A smooth command does not prove a smooth motion.

## Inputs

| quantity | source | units |
| --- | --- | --- |
| end-effector position | `step.ee_pose[:3, 3]` (recorded) | m |
| joint positions | `step.joint_positions` (7) | rad |
| joint velocities | `step.joint_velocities` (7, recorded) | rad/s |
| gripper state | `step.gripper_position` (measured) | m |
| collision flag | `step.unintended_collision` | bool |
| time | `step.stamp_ns` | ns |
| task outcome | `record.outcome` (measured success rule) | enum |

Actions are used only to validate the 8-value shape (`7 joints + gripper`). They
never feed a metric. Perturbation labels (`is_perturbed`, `perturbation_name`,
`expected_quality`) are never read by the metrics or the decision engine; a test
changes all three and asserts identical score, decision, and reason order.

## Resampling and derivatives

The stamps are resampled onto a uniform grid with the same number of samples,
then differentiated:

- `derivative_method: savgol` (default): Savitzky-Golay, `window=9`,
  `polyorder=5`. At 20 Hz the window spans 0.45 s. Polyorder 5 differentiates a
  quintic minimum-jerk profile exactly. `window <= validation.min_steps`, so a
  too-short episode is rejected before the derivative is attempted.
- `derivative_method: finite_difference`: repeated `numpy.gradient` with
  `edge_order=2`.

Measured empirical trade-off (20 Hz, 1 mm Gaussian position noise, 4 s / 0.5 m
minimum-jerk reach):

| smooth window | clean J_d | noisy J_d | 6 Hz oscillation J_d |
| --- | --- | --- | --- |
| 5 (poly 3) | 697.4 | 6.50e6 | — |
| **9 (poly 5, chosen)** | **721.1** | **3.15e6** | **5.54e6** |
| 17 (poly 5) | 721.1 | 5.58e4 | 1.29e4 |

Window 9 cuts the 20 Hz noise-driven jerk by about half versus window 5 while
preserving a real 6 Hz oscillation. Window 17 suppresses noise further but also
attenuates real motion by more than 400x, which would hide jerky demonstrations.
Finite differences are not the default: on the analytic minimum-jerk profile they
measured 381-614 instead of 720 because three repeated 20 Hz differences amplify
discretisation error.

## Dimensionless squared jerk

For end-effector position `x(t)`, duration `T`, and path length
`L = sum(||x[k+1] - x[k]||)`:

```
J_d = (T^5 / L^2) * integral_0^T ||d^3 x / dt^3||^2 dt
```

The analytic minimum-jerk reach `x0 + (xf-x0)(10u^3 - 15u^4 + 6u^5)` gives
exactly `J_d = 720` for any distance and duration. At 20 Hz the measured values
were 724.5 (T=2 s), 721.1 (T=4 s), and 738.0 (T=1 s); the test tolerance is 5%.

`T` includes dwell time (grasp and release holds). Because `J_d` scales with
`T^5`, pauses increase it strongly. A 4 s episode with a 0.5 s hold is therefore
not comparable to a 2 s reach. Thresholds are empirical, not literature values.

If `L < validation.min_path_length_m` or `T <= 0`, a
`DegenerateTrajectoryError` is raised. The decision engine catches it and rejects
with `DEGENERATE_TRAJECTORY`; it never returns a silent zero.

## SPARC

Speed profile `v(t) = ||dx/dt||`. With zero padding (`padlevel=4`), the FFT
magnitude `V(f)` is normalised by `V(0)`. The adaptive cutoff is the highest
frequency at or below `f_c*` where `V(f) >= amplitude_threshold` (0.05):

```
f_c* = min(f_c, max{f : V(f) >= 0.05})
SPARC = - integral_0^{f_c*} sqrt( (1/f_c*)^2 + (dV/df)^2 ) df
```

More negative means less smooth. At 20 Hz the Nyquist frequency is 10 Hz, so a
default `f_c` of 10 Hz would sit exactly at Nyquist. The effective cutoff is
`min(f_c_hz, nyquist_fraction * Nyquist)` with `nyquist_fraction=0.8`; at 20 Hz
that is **8 Hz**, strictly below Nyquist. The cutoff is derived from the actual
sampling rate, not hard-coded.

All-zero speed raises `DegenerateTrajectoryError`. The signature is invariant to
amplitude and duration scaling: the same shape at 2 s/0.4 m and 4 s/0.8 m both
measured `-1.400`. These reference values are empirical, not literature values.

## Path length, margins, collisions

- **Path length:** executed end-effector path in metres, and joint-space path in
  radians. A straight 1 m line measures 1.0 m; the polyline (0,0)->(1,0)->(1,1)
  measures 2.0 m.
- **Joint-limit margin:** per joint,
  `min(q - q_min, q_max - q) / ((q_max - q_min)/2)`, minimum over joints and
  time. `0` is exactly at a limit, negative is a violation, `1` is mid-range.
- **Velocity-limit margin:** `1 - max|qdot| / qdot_max`, minimum over joints and
  time. `qdot` is the **recorded** joint velocity. The metrics also report
  `velocity_crosscheck_max_abs_diff_rad_s` (recorded vs position-derived); it is
  diagnostic only, because a 50 ms finite difference is not an instantaneous
  velocity. The fake backend reports zero joint velocities, so its velocity
  margin is always 1.0; this is a test-double limitation.
- **Limits:** position and velocity limits come from `configs/sim/panda.yaml`.
  Position limits are cross-checked against the loaded MuJoCo model by an
  existing test. The velocity values (2.175 rad/s for joints 1-4, 2.61 rad/s for
  5-7) are documented in that file as Franka Emika datasheet values; they are
  **not** re-derived or independently verified here.
- **Collision count:** number of maximal runs of consecutive steps with
  `unintended_collision == true`, not the number of true steps. One continuous
  contact is one collision; two separate contacts are two.

## Acceptance engine

### Hard gates (config `hard_gates`)

| gate | default |
| --- | --- |
| `require_success` | true |
| `max_collisions` | 0 (fake) / 3 (MuJoCo, see caveat) |
| `min_joint_margin` | 0.0 |
| `min_velocity_margin` | 0.0 |

### Soft thresholds and score

Soft violations produce `JERK_TOO_HIGH`, `SPARC_TOO_LOW`, or `PATH_TOO_LONG` and
reject the episode. The score is a weighted mean of clipped sub-scores:

- lower-is-better: `1 - 0.5 * value / limit` (1 at zero, 0.5 at the threshold);
- SPARC (higher is better, negative): `0.5 + 0.5 * (value - min_sparc) / (0 - min_sparc)`.

Weights default to `0.4/0.4/0.2` for jerk/SPARC/path. `decision = ACCEPT` only
if every hard gate and soft threshold passes **and** the score reaches
`accept_score_threshold` (0.5). The engine is pure and deterministic; reasons
are emitted in a fixed order and a hard gate always wins over a high score.

### Rejection codes

`MALFORMED_EPISODE`, `MISSING_FIELD`, `SHAPE_MISMATCH`, `ACTION_DIM`,
`NAN_VALUES`, `TOO_SHORT`, `NON_MONOTONIC_TIME`, `DT_JITTER`,
`DUPLICATE_EPISODE`, `DEGENERATE_TRAJECTORY`, `NOT_SUCCESS`, `COLLISION`,
`JOINT_LIMIT_MARGIN`, `VELOCITY_LIMIT_MARGIN`, `JERK_TOO_HIGH`,
`SPARC_TOO_LOW`, `PATH_TOO_LONG`.

## Threshold provenance

**No literature numbers and no hold-out tuning.** Thresholds were chosen only
from clean, successful, non-perturbed **tuning** episodes:

- tuning: 100 episodes per backend, seeds **1000-1099**, perturbed fraction 0.4;
- hold-out: 60 episodes per backend, seeds **2000-2059**, same fraction;
- rule: `p95 * 1.2` for upper bounds, `p5 - 0.3` (more negative) for SPARC.

The fake and MuJoCo distributions differ materially, so each has its own file:
`configs/filtering/default.yaml` (fake) and `configs/filtering/mujoco.yaml`.

### Tuning distributions (clean + successful + non-perturbed)

| backend | n | metric | min | p50 | p95 | max | chosen |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fake | 67 | J_d | 4.96e6 | 6.62e6 | 1.218e7 | 1.919e7 | **1.46116e7** |
| fake | 67 | SPARC | -5.094 | -3.755 | -3.324 | -3.245 | **-5.05682** |
| fake | 67 | path m | 0.8265 | 0.9119 | 1.185 | 1.250 | **1.3032** |
| MuJoCo | 40 | J_d | 2.071e6 | 2.473e6 | 3.386e6 | 3.937e6 | **4.06284e6** |
| MuJoCo | 40 | SPARC | -3.907 | -3.196 | -2.831 | -2.766 | **-4.16341** |
| MuJoCo | 40 | path m | 0.7891 | 0.827 | 1.046 | 1.114 | **1.15026** |

Clean tuning false rejections: fake **1/67**, MuJoCo **0/40**. Clean joint
margins were 0.468 (fake) / 0.474 (MuJoCo); velocity margins 1.0 (fake, zero
reported velocities) / 0.332 (MuJoCo). Clean collision runs were 0 (fake) and
1-3 (MuJoCo).

### Tuning per perturbation (n / successes)

| perturbation | fake | MuJoCo |
| --- | --- | --- |
| clean | 67 / 67 | 67 / 40 |
| bad_grasp | 5 / 0 | 5 / 0 |
| detour | 2 / 2 | 2 / 1 |
| hesitation | 7 / 7 | 7 / 4 |
| jitter | 8 / 8 | 8 / 3 |
| low_sweep | 5 / 5 | 5 / 1 |
| overspeed | 6 / 6 | 6 / 3 |

### Hold-out evaluation (60 episodes per backend, never used for tuning)

| perturbation | fake accepted / n | MuJoCo accepted / n |
| --- | --- | --- |
| clean | **31 / 33** | 9 / 33 |
| bad_grasp | **0 / 3** | **0 / 3** |
| detour | 6 / 6 | 3 / 6 |
| hesitation | 2 / 4 | 2 / 4 |
| jitter | 0 / 2 | 0 / 2 |
| low_sweep | 8 / 8 | 6 / 8 |
| overspeed | 4 / 4 | 0 / 4 |

Clean-successful false rejections on the hold-out: fake **2/33 (6.1%)**,
MuJoCo **2/11 (18.2%)**. The MuJoCo clean row is dominated by the expert, not
the filter: only 11 of 33 clean hold-out episodes succeeded, and 9 of those 11
were accepted. Hold-out reason counts: fake `JERK_TOO_HIGH=6`,
`NOT_SUCCESS=3`, `SPARC_TOO_LOW=1`; MuJoCo `NOT_SUCCESS=35`,
`JERK_TOO_HIGH=8`, `PATH_TOO_LONG=1`, `SPARC_TOO_LOW=1`.

## Known limitations

- **Collision flag over-reports intended grasps (pre-existing Task 4 bug).**
  `_unintended_collision` skips `left_finger<->cube_red` contacts inside its
  loop but ends with `return count > 0`, so any contact sets the flag. Clean
  successful MuJoCo episodes therefore show 1-3 collision runs. Expert code was
  **not changed** in Task 5; `mujoco.yaml` uses `max_collisions: 3` to avoid
  rejecting every episode. The collision gate is not a physical collision test
  until that flag is fixed.
- **20 Hz derivative noise.** Three derivatives at 50 ms spacing are sensitive
  to noise. Savitzky-Golay window 9 reduces but does not eliminate it; J_d is
  comparative, not an absolute physical quantity.
- **J_d depends on dwell time.** `T^5` scaling means grasp/release pauses raise
  the value. Thresholds are per-backend and empirical.
- **The fake backend reports zero joint velocities**, so velocity-limit margins
  are always 1.0 there.
- **Accepted data may skew toward easy object poses** because the MuJoCo expert
  fails at many others (22/40 clean successes in the seed 3000-3039 census).
- **Small samples.** The tables report counts, not just percentages; 2/33 and
  1/67 are noisy estimates.
- **LeRobot-style only.** The Parquet columns follow LeRobot naming but are not
  validated against the LeRobot loader until Task 7.
- **No images.** Staging captures no images, so the curated dataset cannot train
  a VLA until RGB-D frames are rendered later.
