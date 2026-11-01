#!/usr/bin/env python3
"""Summarize a curated_v1 dataset directory.

    uv run python scripts/summarize_dataset.py data/curated/dev_curated

Prints the latest version's info, decision breakdown, per-perturbation acceptance,
rejection reasons, and the image mode.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _versions(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("v") and len(path.name[1:].split(".")) == 3
    )


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def summarize(root: Path, version: str | None = None) -> dict:
    versions = _versions(root)
    if not versions:
        raise SystemExit(f"no curated versions under {root}")
    selected = root / version if version else versions[-1]
    info = json.loads((selected / "meta" / "info.json").read_text(encoding="utf-8"))
    episodes = _jsonl(selected / "meta" / "episodes.jsonl")
    tasks = _jsonl(selected / "meta" / "tasks.jsonl")
    decisions: Counter[int] = Counter()
    per_perturbation: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    reasons: Counter[str] = Counter()
    for row in episodes:
        decisions[int(row["decision"])] += 1
        name = row.get("perturbation_name") or "clean"
        per_perturbation[name][1] += 1
        if int(row["decision"]) == 0:
            per_perturbation[name][0] += 1
        for code in row.get("reason_codes", []):
            reasons[code] += 1
    return {
        "dataset": str(selected),
        "version": info["version"],
        "schema_version": info["schema_version"],
        "episodes": len(episodes),
        "frames": info.get("num_frames"),
        "fps": info.get("fps"),
        "images": info.get("images"),
        "decisions": dict(sorted(decisions.items())),
        "per_perturbation": {
            name: {"accepted": values[0], "total": values[1]}
            for name, values in sorted(per_perturbation.items())
        },
        "reason_counts": dict(sorted(reasons.items())),
        "tasks": tasks,
        "filter_config_sha256": info.get("filter_config_sha256"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir")
    parser.add_argument("--version", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = summarize(Path(args.dataset_dir).expanduser().resolve(), args.version)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print(f"dataset: {payload['dataset']} ({payload['schema_version']})")
    print(f"episodes: {payload['episodes']}  frames: {payload['frames']}  fps: {payload['fps']}")
    print("images:", payload["images"])
    print("decisions:", payload["decisions"])
    print("per perturbation:", payload["per_perturbation"])
    print("reasons:", payload["reason_counts"])
    for task in payload["tasks"]:
        print(f"  task[{task['task_index']}]: {task['instruction']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
