# Datasets

## Status

**Not yet implemented.** Task 1 contains no dataset generation, export,
conversion, or mixing code. The `data/` directory contains only `.gitkeep` and
its contents are ignored by git.

## Planned work

| Item | Task | Status |
| --- | --- | --- |
| `curated_v1` on-disk episode format | Task 5 | planned, schema not defined |
| Filtering and acceptance rules | Task 5 | planned |
| LeRobot dataset export | Task 6 | planned |
| Open-X / LeRobot dataset mixing | Task 6 | planned |
| Dataset statistics and provenance | Task 6 | planned |

## Storage policy

- No dataset, recording, or exported episode is committed to this repository.
- `RolloutEpisode` does not embed images. Raw RGB-D frames stay on disk and are
  referenced from `metadata_json` (bounded to 64 KiB).
- `data/**` is ignored by git; only `data/.gitkeep` is tracked.
- Dataset schemas will be versioned. `curated_v1` is a placeholder name for
  Task 5, not a committed schema.

## Known unknowns

- Storage budget and retention policy.
- Whether depth is stored as float32 metres (`32FC1`) or converted on export.
- LeRobot feature mapping for the overhead and wrist cameras.
- How failure-targeted rollouts are balanced against successes.

These are intentionally unresolved until Task 5 and Task 6.
