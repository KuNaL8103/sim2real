#!/usr/bin/env python3
"""Print empirical source proportions for a mixture manifest."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sim2real.dataset_builder import (  # noqa: E402
    MixingSampler,
    indexes_from_config,
    load_mixture_config,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--show", type=int, default=5)
    args = parser.parse_args(argv)
    if args.draws <= 0:
        parser.error("--draws must be positive")
    config = load_mixture_config(args.config)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    indexes = indexes_from_config(config)
    sampler = MixingSampler(config, indexes)
    draws = sampler.draw_n(args.draws)
    counts = {source.source_id: 0 for source in config.sources}
    for ref in draws:
        counts[ref.source_id] += 1
    proportions = {key: value / args.draws for key, value in counts.items()}
    print(
        json.dumps(
            {
                "name": config.name,
                "seed": config.seed,
                "draws": args.draws,
                "weights": dict(
                    zip(
                        [source.source_id for source in config.sources],
                        config.weights,
                        strict=True,
                    )
                ),
                "counts": counts,
                "proportions": proportions,
                "first": [ref.to_dict() for ref in draws[: args.show]],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
