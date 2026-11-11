#!/usr/bin/env python3
"""Fit a failure distribution from a closed-loop evaluation run directory.

The run directory must contain ``records.jsonl`` (one ``EvalEpisodeRecord`` per
line).  ``pairs.json`` is required for ``failure_definition: policy_only``.

Example:

    uv run python scripts/fit_failure_distribution.py \
        --eval-run data/eval/dev_eval --out data/failure_distributions
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sim2real.evaluation import (  # noqa: E402
    EvalEpisodeRecord,
    FailureDistributionError,
    fit_failure_distribution,
    load_evaluation_config,
)


def _load_records(run_dir: Path) -> list[EvalEpisodeRecord]:
    path = run_dir / "records.jsonl"
    if not path.is_file():
        raise SystemExit(f"error: {path} not found")
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(EvalEpisodeRecord.from_json(line))
        except (ValueError, KeyError, TypeError) as exc:
            raise SystemExit(f"error: {path}:{number}: {exc}") from exc
    return records


def _load_pairs(run_dir: Path) -> list[dict] | None:
    path = run_dir / "pairs.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("pairs")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-run", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--config", default="configs/evaluation/default.yaml")
    parser.add_argument(
        "--failure-definition", default=None, choices=("policy_only", "any_failure")
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.eval_run).expanduser()
    records = _load_records(run_dir)
    pairs = _load_pairs(run_dir)
    config = load_evaluation_config(args.config)
    if args.failure_definition is not None:
        config = replace(
            config,
            failure_analysis=replace(
                config.failure_analysis, failure_definition=args.failure_definition
            ),
        )
    definition = config.failure_analysis.failure_definition
    if definition == "policy_only" and pairs is None:
        print(
            "error: failure_definition=policy_only needs pairs.json; re-run evaluation "
            "with --paired-expert (or use --failure-definition any_failure)",
            file=sys.stderr,
        )
        return 2
    manifest_path = run_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    )
    source_run_id = str(manifest.get("run_id", run_dir.name))
    try:
        distribution = fit_failure_distribution(
            records, config, source_eval_run_id=source_run_id, paired=pairs
        )
    except FailureDistributionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    output_dir = Path(args.out or config.failure_analysis.output_dir)
    path = distribution.save(output_dir)
    summary = {
        "path": str(path),
        "distribution": distribution.to_dict(),
        "n_records": len(records),
        "n_pairs": len(pairs) if pairs is not None else 0,
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
