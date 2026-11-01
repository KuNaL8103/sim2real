#!/usr/bin/env python3
"""Filter staged rollouts into a versioned curated dataset and a rejection log.

    uv run python scripts/filter_staging.py \
        --staging data/staging/hold_fake \
        --config configs/filtering/default.yaml \
        --curated data/curated --rejected data/rejected \
        --dataset-name dev_curated

``--staging`` may be one run directory (containing ``manifest.jsonl``) or a
parent containing several run directories. Accepted episodes are written once,
atomically, as a new minor version; rejected/invalid episodes go to
``data/rejected/<run_id>/rejections.jsonl``. The two sets are disjoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from sim2real.data import CuratedDatasetWriter, DatasetWriterError, RejectionLog
from sim2real.expert.staging import StagingBuffer
from sim2real.filtering.config import load_filter_config
from sim2real.filtering.pipeline import FilterPipeline


def _buffers(path: Path) -> list[StagingBuffer]:
    if (path / "manifest.jsonl").is_file():
        return [StagingBuffer(path.parent, path.name)]
    buffers = [
        StagingBuffer(candidate.parent, candidate.name)
        for candidate in sorted(path.iterdir())
        if (candidate / "manifest.jsonl").is_file()
    ]
    if not buffers:
        raise SystemExit(f"no manifest.jsonl under {path}")
    return buffers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", required=True)
    parser.add_argument("--config", default="configs/filtering/default.yaml")
    parser.add_argument("--curated", default="data/curated")
    parser.add_argument("--rejected", default="data/rejected")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--version", default=None)
    args = parser.parse_args(argv)

    config = load_filter_config(args.config)
    pipeline = FilterPipeline(config)
    staging_root = Path(args.staging).expanduser().resolve()
    buffers = _buffers(staging_root)

    accepted: list[tuple[object, object]] = []
    logs: dict[str, RejectionLog] = {}
    reason_counts: Counter[str] = Counter()
    per_perturbation: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    clean_success = [0, 0]
    total = 0
    for buffer in buffers:
        for record in buffer.iter_records():
            total += 1
            result = pipeline.process(record)
            name = record.perturbation_name or "clean"
            per_perturbation[name][0] += int(result.accepted)
            per_perturbation[name][1] += 1
            if name == "clean" and record.outcome == 1:
                clean_success[0] += int(result.accepted)
                clean_success[1] += 1
            if result.accepted:
                accepted.append((record, result))
                continue
            for reason in result.reasons:
                reason_counts[reason.code] += 1
            run_id = record.run_id or buffer.run_id
            log = logs.get(run_id)
            if log is None:
                log = RejectionLog(args.rejected, run_id)
                logs[run_id] = log
            log.append(
                record,
                result,
                staging_reference=f"{buffer.root.name}/episode_{record.episode_id}.npz",
            )

    summary: dict[str, object] = {
        "staging": str(staging_root),
        "runs": [buffer.root.name for buffer in buffers],
        "episodes": total,
        "accepted": len(accepted),
        "rejected_or_invalid": total - len(accepted),
        "clean_success_accepted": clean_success[0],
        "clean_success_total": clean_success[1],
        "reason_counts": dict(sorted(reason_counts.items())),
        "per_perturbation": {
            name: {"accepted": values[0], "total": values[1]}
            for name, values in sorted(per_perturbation.items())
        },
        "filter_config": config.source_path,
        "filter_config_sha256": config.config_hash(),
    }

    if accepted:
        writer = CuratedDatasetWriter(args.curated, args.dataset_name)
        try:
            written = writer.write(accepted, config=config, requested_version=args.version)
        except DatasetWriterError as exc:
            summary["dataset"] = None
            summary["error"] = str(exc)
            print(json.dumps(summary, indent=2, sort_keys=True))
            print(f"filter_staging: refusing to write: {exc}", file=sys.stderr)
            return 1
        summary.update(
            {
                "dataset": str(written.path),
                "dataset_version": written.version,
                "dataset_episodes": written.num_episodes,
                "dataset_new_episodes": written.num_new_episodes,
            }
        )
    else:
        summary["dataset"] = None

    rejected_ids = {row["episode_id"] for log in logs.values() for row in log.rows()}
    accepted_ids = {str(record.episode_id) for record, _ in accepted}
    overlap = sorted(rejected_ids & accepted_ids)
    if overlap:
        raise SystemExit(f"accepted and rejected sets overlap: {overlap}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
