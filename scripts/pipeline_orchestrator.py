#!/usr/bin/env python3
"""Run or inspect the explicit sim2real pipeline orchestrator.

The default mode is a **dry run**: it prints the stage plan and writes nothing.
An actual cycle requires ``--execute``; every gate (external checkpoint, GPU
time, large writes) has an explicit flag and fails closed.

Examples:

    # Plan only (writes nothing unless --plan-file is given)
    uv run python scripts/pipeline_orchestrator.py --config configs/pipeline/default.yaml

    # One fake-backend end-to-end cycle on the CPU
    uv run python scripts/pipeline_orchestrator.py --config configs/pipeline/fake_e2e.yaml \
        --execute --root /tmp/opencode/pipeline_run

    # Lineage for a checkpoint produced by an earlier cycle
    uv run python scripts/pipeline_orchestrator.py --config configs/pipeline/default.yaml \
        --lineage outputs/task8/train_v1_lora/checkpoint-00001000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sim2real.orchestrator import (  # noqa: E402
    ApprovalRequiredError,
    CheckpointVerificationError,
    CorruptStageOutputError,
    CycleLimitError,
    PipelineConfigError,
    PipelineOrchestrator,
    StageError,
    load_pipeline_config,
)
from sim2real.orchestrator.seed_ledger import describe_ledger  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/pipeline/default.yaml")
    parser.add_argument("--root", default=None, help="Override pipeline.run_root")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--execute", action="store_true", help="Run cycles (default is a dry run)")
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--plan-file", default=None)
    parser.add_argument("--approve-cycle", action="store_true")
    parser.add_argument("--supply-checkpoint", default="")
    parser.add_argument("--allow-train", action="store_true")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--allow-large-write", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--lineage", default="", help="Print lineage for a checkpoint path or hash")
    parser.add_argument("--ledger", action="store_true", help="Print the seed ledger ranges")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.ledger:
        print(json.dumps(describe_ledger(), indent=2, sort_keys=True))
        return 0

    try:
        config = load_pipeline_config(args.config)
    except PipelineConfigError as exc:
        print(f"pipeline_orchestrator: invalid config: {exc}", file=sys.stderr)
        return 2

    orchestrator = PipelineOrchestrator(
        config,
        repo_root=ROOT,
        root=args.root,
        run_id=args.run_id,
    )

    if args.lineage:
        report = orchestrator.lineage(args.lineage)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    if not args.execute:
        plan = orchestrator.run(dry_run=True, plan_file=args.plan_file)
        if args.json:
            print(
                json.dumps(
                    {
                        "config": config.source_path,
                        "config_hash": config.fingerprint,
                        "run_root": str(orchestrator.run_root),
                        "dry_run": True,
                        "stages": plan,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(f"DRY RUN  config={config.source_path}")
            print(f"config_hash={config.fingerprint}")
            print(f"run_root={orchestrator.run_root}")
            print()
            for row in plan:
                print(f"{row['stage']:<18} executor={row['executor']:<8} env={row['environment']}")
                print(f"  inputs:  {row['inputs']}")
                print(f"  outputs: {row['outputs']}")
                if row["approval"] != "none":
                    print(f"  approval: {row['approval']}")
        return 0

    try:
        manifests = orchestrator.run(
            cycles=args.cycles,
            dry_run=False,
            approve_cycle=args.approve_cycle,
            supplied_checkpoint=args.supply_checkpoint,
            allow_train=args.allow_train,
            allow_gpu=args.allow_gpu,
            allow_large_write=args.allow_large_write,
            allow_download=args.allow_download,
        )
    except ApprovalRequiredError as exc:
        print(f"APPROVAL REQUIRED: {exc}", file=sys.stderr)
        return 2
    except CheckpointVerificationError as exc:
        print(f"CHECKPOINT REFUSED: {exc}", file=sys.stderr)
        return 3
    except CorruptStageOutputError as exc:
        print(f"OUTPUT QUARANTINED: {exc}", file=sys.stderr)
        return 4
    except (CycleLimitError, StageError) as exc:
        print(f"PIPELINE STOPPED: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(manifests, indent=2, sort_keys=True))
    else:
        for manifest in manifests:
            print(
                f"{manifest['cycle_id']}: {manifest['status']} "
                f"wall={manifest['wall_time_s']}s "
                f"checkpoint={(manifest['checkpoint'] or {}).get('metadata_hash', '')[:12]} "
                f"eval={manifest['evaluation_run_id'] or '-'}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
