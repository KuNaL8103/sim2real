#!/usr/bin/env python3
"""Compute LeRobot-compatible statistics for curated episodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sim2real.dataset_builder import (  # noqa: E402
    compute_dataset_stats,
    load_embodiments,
    stats_to_json,
)
from sim2real.dataset_builder.adapters.curated_v1 import CuratedAdapter  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curated", required=True)
    parser.add_argument("--staging", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--embodiments", default="configs/dataset/embodiments.yaml")
    parser.add_argument("--action-representation", default="joint_position")
    parser.add_argument("--decimation", type=int, default=2)
    parser.add_argument("--max-episodes", type=int, default=40)
    parser.add_argument("--renderer", default="llvmpipe")
    parser.add_argument("--allow-imageless", action="store_true")
    parser.add_argument("--include-images", action="store_true")
    parser.add_argument("--quantile-method", choices=("exact", "reservoir"), default="exact")
    parser.add_argument("--reservoir-size", type=int, default=10_000)
    args = parser.parse_args(argv)

    spec = load_embodiments(args.embodiments)["panda_sim"]
    episodes = CuratedAdapter(
        curated_root=args.curated,
        embodiment=spec,
        decimation=args.decimation,
        action_representation=args.action_representation,
        staging_root=args.staging,
        allow_imageless=args.allow_imageless,
        declared_renderer=args.renderer,
    ).load(max_episodes=args.max_episodes)
    stats = compute_dataset_stats(
        episodes,
        include_images=args.include_images,
        quantile_method=args.quantile_method,
        reservoir_size=args.reservoir_size,
    )
    payload = stats_to_json(stats)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out),
                "episodes": len(episodes),
                "features": {
                    key: {
                        "count": int(value["count"][0]),
                        "mean": value["mean"].tolist(),
                        "std": value["std"].tolist(),
                        "q01": value["q01"].tolist(),
                        "q99": value["q99"].tolist(),
                    }
                    for key, value in stats.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
