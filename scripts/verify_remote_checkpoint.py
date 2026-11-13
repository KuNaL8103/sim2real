#!/usr/bin/env python3
"""Verify a checkpoint returned from the remote GPU run, locally.

Checks provenance and semantics before any evaluation:

* ``checkpoint_metadata.json`` is complete and records ``git_dirty=false``;
* the recorded commit exists in this clone's history;
* the config hash, dataset stats hash, rename map, action representation,
  chunk size, ``n_action_steps`` and fps match the local config/dataset;
* the adapter directory hash matches the metadata (and RUN_REPORT, if supplied);
* the policy server handshake accepts the checkpoint.

The handshake defaults to CPU (``CUDA_VISIBLE_DEVICES`` is emptied) so a
throttled local GPU is not used. Use ``--handshake cuda`` only for a diagnostic;
any timing from it is labelled throttled, not a performance figure.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sim2real.evaluation import CheckpointPolicy, PolicyError  # noqa: E402
from sim2real.training.checkpoint import (  # noqa: E402
    adapter_tree_sha256,
    dataset_fingerprint,
    file_sha256,
    load_checkpoint,
)
from sim2real.training.config import load_training_config  # noqa: E402

TASK = "pick up the red cube and place it in the left zone"


class VerificationError(RuntimeError):
    """Raised when the returned checkpoint does not match local provenance."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _commit_exists(commit: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-e", f"{commit}^{{commit}}"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _verify_run_report(
    report_path: Path,
    *,
    metadata: dict[str, Any],
    checkpoint: Path,
    metadata_sha256: str,
    adapter_sha256: str,
) -> None:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"{report_path}: unreadable RUN_REPORT.json: {exc}") from exc
    for key in ("git_commit", "config_hash", "stats_file_hash"):
        if key in report:
            _require(
                report[key] == metadata.get(key),
                f"RUN_REPORT {key} does not match checkpoint metadata "
                f"({report[key]!r} != {metadata.get(key)!r})",
            )
    entries = report.get("checkpoints")
    _require(isinstance(entries, list) and entries, "RUN_REPORT has no checkpoints list")
    matches = [entry for entry in entries if int(entry.get("step", -1)) == int(metadata["step"])]
    _require(
        bool(matches),
        f"RUN_REPORT has no checkpoint entry for step {metadata['step']}",
    )
    entry = matches[0]
    _require(
        entry.get("metadata_sha256") == metadata_sha256,
        "RUN_REPORT metadata sha256 does not match the returned checkpoint",
    )
    _require(
        entry.get("adapter_sha256") == adapter_sha256,
        "RUN_REPORT adapter sha256 does not match the returned adapter directory",
    )
    for key in ("path", "name"):
        if key in entry:
            reported = Path(str(entry[key]))
            _require(
                reported.name == checkpoint.name,
                f"RUN_REPORT checkpoint {key} {reported.name!r} != {checkpoint.name!r}",
            )


def verify_checkpoint(
    checkpoint: Path,
    config_path: Path,
    *,
    run_report: Path | None = None,
    dataset_root: Path | None = None,
    handshake: str = "cpu",
    server_python: Path | None = None,
    timeout_s: float = 600.0,
    seed: int = 0,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint).expanduser().resolve()
    config = load_training_config(config_path)
    dataset = Path(dataset_root) if dataset_root is not None else Path(config.data.dataset_root)
    fingerprint = dataset_fingerprint(dataset)
    expected = {
        "expected_stats_hash": fingerprint["stats_file_hash"],
        "expected_action_representation": config.data.action_representation,
        "expected_rename_map": dict(config.data.rename_map),
        "expected_config_hash": config.config_hash,
    }
    try:
        metadata = load_checkpoint(checkpoint, **expected)
    except Exception as exc:  # noqa: BLE001 - normalize every refusal
        raise VerificationError(str(exc)) from exc

    _require(
        metadata.get("git_dirty") is False,
        f"checkpoint git_dirty is {metadata.get('git_dirty')!r}; expected false",
    )
    commit = str(metadata.get("git_commit") or "")
    _require(bool(commit), "checkpoint metadata has no git_commit")
    _require(
        _commit_exists(commit),
        f"commit {commit} is not present in local history; fetch/pull the public "
        "repository before verifying",
    )
    _require(
        metadata.get("config_hash") == config.config_hash,
        "checkpoint config hash does not match the local config",
    )
    _require(
        metadata.get("stats_file_hash") == fingerprint["stats_file_hash"],
        "checkpoint stats hash does not match the local dataset stats",
    )
    _require(
        metadata.get("dataset_hash") == fingerprint["dataset_hash"],
        "checkpoint dataset hash does not match the local dataset metadata",
    )
    _require(
        metadata.get("camera_rename_map") == dict(config.data.rename_map),
        "checkpoint camera rename map does not match the local config",
    )
    _require(
        metadata.get("action_representation") == config.data.action_representation,
        "checkpoint action representation does not match the local config",
    )
    _require(
        int(metadata.get("chunk_size", -1)) == int(config.training.chunk_size),
        f"checkpoint chunk_size {metadata.get('chunk_size')} != {config.training.chunk_size}",
    )
    _require(
        int(metadata.get("n_action_steps", -1)) == int(config.training.n_action_steps),
        f"checkpoint n_action_steps {metadata.get('n_action_steps')} != "
        f"{config.training.n_action_steps}",
    )
    _require(
        float(metadata.get("fps", -1.0)) == float(config.data.fps),
        f"checkpoint fps {metadata.get('fps')} != {config.data.fps}",
    )

    metadata_path = checkpoint / "checkpoint_metadata.json"
    metadata_sha256 = file_sha256(metadata_path)
    adapter_sha256 = adapter_tree_sha256(checkpoint / "adapter")
    recorded_adapter = metadata.get("adapter_hash")
    _require(
        bool(recorded_adapter),
        "checkpoint metadata has no adapter_hash; rebuild the checkpoint with the "
        "current trainer before using it as remote evidence",
    )
    _require(
        recorded_adapter == adapter_sha256,
        "adapter directory hash does not match checkpoint metadata "
        f"({adapter_sha256} != {recorded_adapter})",
    )
    if run_report is not None:
        _verify_run_report(
            Path(run_report),
            metadata=metadata,
            checkpoint=checkpoint,
            metadata_sha256=metadata_sha256,
            adapter_sha256=adapter_sha256,
        )

    handshake_used = handshake
    if handshake != "skip":
        old_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            if handshake == "cpu":
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
            policy = CheckpointPolicy(
                checkpoint,
                repo_root=ROOT,
                fake_model=handshake == "fake",
                timeout_s=float(timeout_s),
                server_python=str(server_python) if server_python else None,
            )
            try:
                policy.reset(TASK, int(seed))
            finally:
                policy.close()
        except (PolicyError, OSError) as exc:
            raise VerificationError(
                f"policy-server handshake refused the checkpoint: {exc}"
            ) from exc
        finally:
            if old_cuda is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda

    if handshake == "cpu":
        handshake_note = "CPU load (CUDA_VISIBLE_DEVICES empty); no timing claim"
    elif handshake == "cuda":
        handshake_note = "local GPU load (throttled, not a performance figure)"
    elif handshake == "fake":
        handshake_note = "fake-model protocol handshake; model weights were not evaluated"
    else:
        handshake_note = "skipped by request"
    return {
        "checkpoint": str(checkpoint),
        "verified": True,
        "git_commit": commit,
        "git_dirty": metadata.get("git_dirty"),
        "config_hash": metadata.get("config_hash"),
        "stats_file_hash": metadata.get("stats_file_hash"),
        "dataset_hash": metadata.get("dataset_hash"),
        "step": metadata.get("step"),
        "adapter_hash": adapter_sha256,
        "metadata_sha256": metadata_sha256,
        "handshake": handshake_used,
        "handshake_note": handshake_note,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-report", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument(
        "--handshake",
        choices=("cpu", "cuda", "fake", "skip"),
        default="cpu",
        help="cpu is the default; cuda is a short throttled diagnostic only",
    )
    parser.add_argument("--server-python", default=None)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        summary = verify_checkpoint(
            Path(args.checkpoint),
            Path(args.config),
            run_report=Path(args.run_report) if args.run_report else None,
            dataset_root=Path(args.dataset_root) if args.dataset_root else None,
            handshake=args.handshake,
            server_python=Path(args.server_python) if args.server_python else None,
            timeout_s=args.timeout,
            seed=args.seed,
        )
    except VerificationError as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 2
    print("checkpoint verified")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
