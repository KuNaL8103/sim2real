#!/usr/bin/env python3
"""Offline closed-loop evaluation CLI (fake backend works without a GPU).

Examples:

    uv run python scripts/run_eval.py --backend fake --policy noisy_expert \
        --num 60 --seed 20000 --profile default --out data/eval/dev_eval

    uv run python scripts/run_eval.py --backend fake --policy zero --num 30 \
        --seed 20000 --out data/eval/zero

The run directory receives ``records.jsonl``, ``manifest.json`` and, when the
expert is paired, ``pairs.json``.  Rendering only happens when the policy needs
images (``checkpoint``); the renderer string is recorded in every record.
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
    CheckpointPolicy,
    EvaluationConfigError,
    NoisyExpertPolicy,
    RandomPolicy,
    ScriptedExpertPolicy,
    ZeroPolicy,
    aggregate_records,
    load_evaluation_config,
    run_eval,
    run_paired_eval,
)
from sim2real.evaluation.records import EvalRun  # noqa: E402
from sim2real.expert import load_expert_config  # noqa: E402
from sim2real.filtering.config import load_filter_config  # noqa: E402
from sim2real.sim import create_backend  # noqa: E402


def _make_policy(args: argparse.Namespace, backend_args: tuple[str, str]):
    backend_name, config_dir = backend_args
    if args.policy == "zero":
        return ZeroPolicy()
    if args.policy == "random":
        backend = create_backend(backend_name, config_dir)
        try:
            return RandomPolicy(backend.joint_limits, seed=args.seed)
        finally:
            backend.close()
    factory = lambda: create_backend(backend_name, config_dir)  # noqa: E731
    expert = ScriptedExpertPolicy(
        backend_factory=factory,
        config_dir=config_dir,
        expert_config_dir=args.expert_config_dir,
        randomization_dir=args.randomization_dir,
        chunk_size=50,
    )
    if args.policy == "scripted":
        return expert
    if args.policy == "noisy_expert":
        return NoisyExpertPolicy(
            expert,
            noise_sigma=args.noise_sigma,
            dropout=args.dropout,
            correlation_gain=args.correlation_gain,
            seed=args.seed,
        )
    if args.policy == "checkpoint":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for --policy checkpoint")
        return CheckpointPolicy(
            args.checkpoint,
            fake_model=args.fake_policy_model,
            repo_root=ROOT,
        )
    raise SystemExit(f"unknown policy {args.policy!r}")


def _write_outputs(
    out_dir: Path,
    records,
    run: EvalRun,
    *,
    pairs: list[dict] | None,
    paired_summary: dict | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.to_json() + "\n")
    manifest = run.to_dict()
    if paired_summary is not None:
        manifest["paired"] = paired_summary
        manifest["aggregates"] = aggregate_records(records)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if pairs is not None:
        (out_dir / "pairs.json").write_text(
            json.dumps({"pairs": pairs}, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="fake", choices=("fake", "mujoco"))
    parser.add_argument(
        "--policy",
        default="noisy_expert",
        choices=("scripted", "noisy_expert", "random", "zero", "checkpoint"),
    )
    parser.add_argument("--num", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--config", default="configs/evaluation/default.yaml")
    parser.add_argument("--config-dir", default="configs/sim")
    parser.add_argument("--expert-config-dir", default="configs/expert")
    parser.add_argument("--randomization-dir", default="configs/randomization")
    parser.add_argument("--filter-config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--fake-policy-model", action="store_true")
    parser.add_argument("--noise-sigma", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--correlation-gain", type=float, default=1.0)
    parser.add_argument(
        "--paired-expert",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="evaluate the scripted expert on the same seeds (needed for policy_only failures)",
    )
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    try:
        config = load_evaluation_config(args.config)
    except EvaluationConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    settings = config.settings
    if args.n_action_steps is not None:
        settings = replace(settings, n_action_steps=int(args.n_action_steps))
    if args.num is not None:
        settings = replace(settings, num_episodes=int(args.num))
    if args.seed is not None:
        settings = replace(settings, eval_seed=int(args.seed))
    if args.profile is not None:
        settings = replace(settings, randomization_profile=args.profile)
    if args.out is not None:
        settings = replace(settings, output_dir=str(args.out))
    config = replace(config, settings=settings)
    out_dir = Path(args.out or settings.output_dir)
    if not args.filter_config:
        candidate = ROOT / "configs" / "filtering" / f"{args.backend}.yaml"
        filter_config = load_filter_config(
            candidate if candidate.is_file() else ROOT / "configs" / "filtering" / "default.yaml"
        )
    else:
        filter_config = load_filter_config(args.filter_config)

    backend = create_backend(args.backend, args.config_dir)
    policy = _make_policy(args, (args.backend, args.config_dir))
    expert_config = load_expert_config(args.expert_config_dir)
    run_id = args.run_id or f"eval_{args.policy}_{settings.eval_seed}_{settings.num_episodes}"
    try:
        if args.paired_expert and args.policy != "scripted":
            expert_policy = ScriptedExpertPolicy(
                backend_factory=lambda: create_backend(args.backend, args.config_dir),
                config_dir=args.config_dir,
                expert_config_dir=args.expert_config_dir,
                randomization_dir=args.randomization_dir,
                chunk_size=50,
            )
            try:
                result = run_paired_eval(
                    expert_backend_factory=None,
                    backend=backend,
                    expert_policy=expert_policy,
                    candidate_policy=policy,
                    config=config,
                    num_episodes=settings.num_episodes,
                    eval_seed=settings.eval_seed,
                    run_id=run_id,
                    config_hash=config.config_hash(),
                    randomization_dir=args.randomization_dir,
                    filter_config=filter_config,
                    expert_config=expert_config,
                )
            finally:
                expert_policy.close()
            records = result["policy_records"]
            run = EvalRun(
                run_id=run_id,
                config_hash=config.config_hash(),
                eval_seed=settings.eval_seed,
                seeds=result["seeds"],
                policy_id=str(getattr(policy, "name", type(policy).__name__)),
                policy_info=dict(getattr(policy, "info", {})),
                randomization_profile=settings.randomization_profile,
                n_action_steps=settings.n_action_steps,
                renderer=records[0].renderer if records else settings.renderer,
            )
            run.aggregates = result["policy"]["aggregates"]
            paired_summary = {
                "expert": result["expert"],
                "policy_success_given_expert_success": result[
                    "policy_success_given_expert_success"
                ],
                "policy_only_failures": result["policy_only_failures"],
            }
            _write_outputs(
                out_dir, records, run, pairs=result["pairs"], paired_summary=paired_summary
            )
            summary = {
                "run": run.to_dict(),
                "expert": paired_summary["expert"],
                "policy": result["policy"],
                "policy_success_given_expert_success": paired_summary[
                    "policy_success_given_expert_success"
                ],
                "policy_only_failures": paired_summary["policy_only_failures"],
            }
        else:
            result = run_eval(
                backend,
                policy,
                config,
                num_episodes=settings.num_episodes,
                eval_seed=settings.eval_seed,
                run_id=run_id,
                config_hash=config.config_hash(),
                randomization_dir=args.randomization_dir,
                filter_config=filter_config,
                expert_config=expert_config,
            )
            _write_outputs(out_dir, result.records, result.run, pairs=None, paired_summary=None)
            summary = {"run": result.run.to_dict()}
    finally:
        backend.close()
        close = getattr(policy, "close", None)
        if close is not None:
            close()

    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
