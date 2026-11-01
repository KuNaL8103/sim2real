#!/usr/bin/env python3
"""Summarize a staging run directory.

    uv run python scripts/summarize_staging.py data/staging/dev_run

Prints outcome and perturbation counts, disk size, and any corrupt entries that
the reader skipped. Corrupt files are reported, never silently ignored.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sim2real.expert.staging import StagingBuffer

OUTCOME_NAMES = {1: "success", 2: "failure", 3: "timeout", 4: "aborted", 0: "unknown"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="directory containing manifest.jsonl")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        parser.error(f"not a directory: {run_dir}")
    buffer = StagingBuffer(run_dir.parent, run_dir.name)

    outcomes: Counter[str] = Counter()
    perturbations: Counter[str] = Counter()
    qualities: Counter[str] = Counter()
    seeds: list[int] = []
    steps = 0
    for record in buffer.iter_records():
        outcomes[OUTCOME_NAMES.get(record.outcome, "unknown")] += 1
        perturbations[record.perturbation_name or "clean"] += 1
        qualities[record.expected_quality] += 1
        seeds.append(record.seed)
        steps += len(record.steps)

    verification = buffer.verify()
    payload = {
        "staging_dir": str(run_dir),
        "episodes": len(seeds),
        "outcomes": dict(sorted(outcomes.items())),
        "perturbations": dict(sorted(perturbations.items())),
        "expected_quality": dict(sorted(qualities.items())),
        "seeds": seeds,
        "total_steps": steps,
        "disk_bytes": buffer.disk_bytes(),
        "disk_mib": round(buffer.disk_bytes() / (1024.0**2), 3),
        "verification": verification,
        "read_errors": buffer.last_errors,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"staging: {payload['staging_dir']}")
        print(f"episodes: {payload['episodes']}  steps: {payload['total_steps']}")
        print("outcomes:", payload["outcomes"])
        print("perturbations:", payload["perturbations"])
        print("expected_quality:", payload["expected_quality"])
        print(f"disk: {payload['disk_mib']} MiB")
        print(
            "verification: "
            f"valid={verification['valid']} corrupt={verification['corrupt']} "
            f"missing={verification['missing']} duplicates={verification['duplicates']}"
        )
        for problem in verification["problems"]:
            print("  problem:", problem)
    return 0 if verification["corrupt"] == 0 and verification["missing"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
