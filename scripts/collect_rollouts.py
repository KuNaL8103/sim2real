#!/usr/bin/env python3
"""Collect scripted-expert rollouts into an atomic staging directory.

Example:

    uv run python scripts/collect_rollouts.py --backend fake --num 20 \
        --seed 0 --perturbed-fraction 0.3 --out data/staging/dev_run

``--backend mujoco`` runs physics without rendering. Images are OFF unless
``--capture-images`` is given; when on, the estimated disk size and wall time are
printed first, and the run refuses to start above the approval limits unless
``--allow-large`` is passed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter

from sim2real.expert import (
    load_expert_config,
    load_perturbations,
    parse_instruction,
    run_episode,
    select_perturbation,
)
from sim2real.expert.staging import StagingBuffer, estimate_storage
from sim2real.sim import create_backend

OUTCOME_NAMES = {1: "success", 2: "failure", 3: "timeout", 4: "aborted", 0: "unknown"}
MAX_UNAPPROVED_GIB = 1.0
MAX_UNAPPROVED_MINUTES = 10.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="fake", choices=("fake", "mujoco"))
    parser.add_argument("--num", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--perturbed-fraction", type=float, default=None)
    parser.add_argument("--out", default="data/staging/dev_run")
    parser.add_argument("--profile", default=None)
    parser.add_argument(
        "--instruction",
        default="pick up the red cube and place it in the left zone",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--config-dir", default="configs/sim")
    parser.add_argument("--expert-config-dir", default="configs/expert")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--capture-images", action="store_true")
    parser.add_argument("--allow-large", action="store_true")
    args = parser.parse_args(argv)

    if args.num <= 0:
        parser.error("--num must be positive")
    config = load_expert_config(args.expert_config_dir)
    perturbations = load_perturbations(config)
    fraction = (
        perturbations.perturbed_fraction
        if args.perturbed_fraction is None
        else args.perturbed_fraction
    )
    if not 0.0 <= fraction <= 1.0:
        parser.error("--perturbed-fraction must be in [0, 1]")

    if args.capture_images:
        estimate = estimate_storage(
            num_steps=config.max_steps,
            num_episodes=args.num,
            resolution=config.staging.resolution,
            decimation=config.staging.image_decimation,
            render_seconds_per_frame=config.staging.desired_render_seconds_per_frame,
        )
        print(
            "image capture estimate: "
            f"{estimate['frames']:.0f} frames, {estimate['gib']:.3f} GiB, "
            f"{estimate['minutes']:.1f} min (upper bound; images currently "
            "captured only at the configured decimation)"
        )
        if not args.allow_large and (
            estimate["gib"] > MAX_UNAPPROVED_GIB or estimate["minutes"] > MAX_UNAPPROVED_MINUTES
        ):
            print(
                "refusing to start: estimate exceeds the unapproved limits "
                f"({MAX_UNAPPROVED_GIB} GiB / {MAX_UNAPPROVED_MINUTES} min); "
                "pass --allow-large after approval",
                file=sys.stderr,
            )
            return 2
        print("note: per-episode image capture is not wired into the runner yet", file=sys.stderr)

    backend = create_backend(args.backend, args.config_dir)
    try:
        task = parse_instruction(args.instruction, backend.configs.scene, config.instructions_file)
        run_id = args.run_id or f"cli_{args.backend}_{args.seed}_{int(time.time())}"
        buffer = StagingBuffer(args.out, run_id)
        profile = args.profile or config.default_profile
        counts: Counter[str] = Counter()
        perturbed = 0
        start = time.perf_counter()
        for index in range(args.num):
            seed = args.seed + index
            perturbation = select_perturbation(
                perturbations, seed, index, perturbed_fraction=fraction
            )
            record = run_episode(
                backend,
                task,
                seed,
                instruction=args.instruction,
                episode_index=index,
                run_id=run_id,
                randomization_profile=profile,
                perturbation=perturbation,
                config=config,
                max_steps=args.max_steps,
            )
            buffer.write(record)
            counts[OUTCOME_NAMES.get(record.outcome, "unknown")] += 1
            perturbed += int(record.is_perturbed)
            print(
                f"{index:03d} seed={seed} outcome={OUTCOME_NAMES.get(record.outcome)} "
                f"perturbation={record.perturbation_name} steps={len(record.steps)} "
                f"reason={record.failure_reason or '-'}",
                flush=True,
            )
        elapsed = time.perf_counter() - start
        summary = {
            "run_id": run_id,
            "staging_dir": str(buffer.root),
            "episodes": args.num,
            "counts": dict(counts),
            "perturbed": perturbed,
            "wall_seconds": round(elapsed, 3),
            "episodes_per_second": round(args.num / elapsed, 3) if elapsed > 0 else None,
            "disk_bytes": buffer.disk_bytes(),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    finally:
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
