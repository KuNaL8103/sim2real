#!/usr/bin/env python3
"""Print the flywheel ledger status and every guard value.

This never trains and never requests rollouts; it only reads the append-only
ledger and reports corrupt lines instead of crashing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sim2real.evaluation import FlywheelController, load_evaluation_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation/default.yaml")
    parser.add_argument("--ledger", default=None)
    parser.add_argument("--stop-file", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    config = load_evaluation_config(args.config)
    controller = FlywheelController(config, ledger_path=args.ledger, stop_file=args.stop_file)
    status = controller.status()
    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(f"ledger: {status['ledger_path']}")
        print(f"stop file: {status['stop_file']} present={status['stop_file_present']}")
        print(f"cycles: {status['num_cycles']}")
        print(
            "rollouts requested/generated: "
            f"{status['total_rollouts_requested']}/{status['total_rollouts_generated']}"
        )
        print(f"decisions: {status['decision_counts']}")
        print(f"corrupt lines: {len(status['corrupt_lines'])}")
        print(f"will_train: {status['will_train']}")
        if status["last_cycle"] is not None:
            print("last cycle:")
            print(json.dumps(status["last_cycle"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
