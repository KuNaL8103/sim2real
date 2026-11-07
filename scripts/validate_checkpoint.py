#!/usr/bin/env python3
"""Verify a checkpoint's provenance and run offline action validation (L1).

The metadata check is the important part: a checkpoint whose stats hash,
action representation, or camera rename map differs from the dataset you are
evaluating is refused.  The L1 block reports per-dimension and horizon-wise
error, excluding padded chunk positions, with two trivial baselines.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sim2real.training.adapters import FakePolicyAdapter  # noqa: E402
from sim2real.training.checkpoint import dataset_fingerprint, load_checkpoint  # noqa: E402
from sim2real.training.config import load_training_config  # noqa: E402
from sim2real.training.validation import evaluate_l1, valid_mask_from_padding  # noqa: E402


def _synthetic_teacher(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.5, size=(dim, dim))


def _fake_validation(checkpoint: Path, metadata: dict, seed: int | None = None):
    state_dim = int(metadata["state_dim"])
    action_dim = int(metadata["action_dim"])
    adapter = FakePolicyAdapter(state_dim, action_dim, seed=int(metadata["seed"]))
    adapter.load_adapter(checkpoint / "adapter")
    teacher = _synthetic_teacher(int(metadata["seed"] if seed is None else seed), state_dim)
    rng = np.random.default_rng(12345 + int(metadata["seed"]))
    inputs = rng.normal(0.0, 1.0, size=(4, state_dim))
    targets = inputs @ teacher + 0.01 * rng.normal(0.0, 1.0, size=(4, action_dim))
    chunk_size = int(metadata["chunk_size"])
    predictions = np.repeat((inputs @ adapter.weights + adapter.bias)[:, None, :], chunk_size, 1)
    repeated_targets = np.repeat(targets[:, None, :], chunk_size, 1)
    mask = np.ones((inputs.shape[0], chunk_size), dtype=bool)
    action_std = np.asarray(metadata.get("action_std", np.ones(action_dim)), dtype=np.float64)
    report = evaluate_l1(
        predictions,
        repeated_targets,
        mask,
        action_std=action_std,
        last_state=inputs,
        mean_action=np.zeros(action_dim),
        horizon_seconds=float(metadata["horizon_seconds"]),
    )
    return report.to_dict()


def _real_validation(checkpoint: Path, metadata: dict, config) -> dict:
    """Best-effort guarded L1 evaluation of a real SmolVLA checkpoint."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from torch.utils.data import DataLoader
        from torch.utils.data._utils.collate import default_collate

        from sim2real.training.adapters import SmolVLAAdapter
    except Exception as exc:  # noqa: BLE001
        return {"status": "NOT RUN", "reason": f"ML dependencies unavailable: {exc}"}

    root = Path(config.data.dataset_root)
    split = json.loads((root / "meta" / "sim2real_split.json").read_text(encoding="utf-8"))
    val_indices = [int(row["index"]) for row in split["val"]]
    dataset = LeRobotDataset(
        repo_id=root.name,
        root=root,
        episodes=val_indices,
        delta_timestamps={
            "action": [i / config.data.fps for i in range(config.training.chunk_size)]
        },
    )
    adapter = SmolVLAAdapter(config, dataset_stats=dataset.meta.stats, dataset_meta=dataset.meta)
    adapter.load_adapter(checkpoint / "adapter")
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=default_collate,
    )
    predictions = []
    targets = []
    masks = []
    states = []
    for batch in loader:
        prepared = adapter.prepare_batch(batch)
        with adapter.torch.no_grad():
            actions = adapter.policy.predict_action_chunk(prepared)
            if adapter.postprocessor is not None:
                # PolicyAction is torch.Tensor in LeRobot 0.6.1, not a dict.
                actions = adapter.postprocessor(actions)
        action_dim = int(metadata["action_dim"])
        predictions.append(np.asarray(actions.detach().cpu())[..., :action_dim])
        targets.append(np.asarray(batch["action"].detach().cpu()))
        # ``action_is_pad`` is True where the chunk is padded; the named
        # helper is the regression guard for the inverted-mask bug.
        padded = batch.get("action_is_pad")
        if padded is None:
            mask = np.ones(targets[-1].shape[:2], dtype=bool)
        else:
            mask = valid_mask_from_padding(padded.detach().cpu())
        masks.append(mask)
        state = np.asarray(batch["observation.state"].detach().cpu())
        states.append(state[:, -1, :] if state.ndim == 3 else state)
    action_dim = int(metadata["action_dim"])
    std = dataset.meta.stats["action"]["std"]
    report = evaluate_l1(
        np.concatenate(predictions),
        np.concatenate(targets),
        np.concatenate(masks),
        action_std=np.asarray(std)[:action_dim],
        last_state=np.concatenate(states)[:, :action_dim],
        mean_action=np.asarray(dataset.meta.stats["action"]["mean"])[:action_dim],
        horizon_seconds=float(metadata["horizon_seconds"]),
    )
    return report.to_dict()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--fake", action="store_true", help="evaluate the fake linear adapter")
    args = parser.parse_args(argv)

    checkpoint = Path(args.checkpoint)
    expected: dict = {}
    config = None
    if args.config:
        config = load_training_config(args.config)
        fingerprint = dataset_fingerprint(config.data.dataset_root)
        expected = {
            "expected_stats_hash": fingerprint["stats_file_hash"],
            "expected_action_representation": config.data.action_representation,
            "expected_rename_map": dict(config.data.rename_map),
        }
    metadata = load_checkpoint(checkpoint, **expected)
    summary = {
        "checkpoint": str(checkpoint),
        "base_model_id": metadata["base_model_id"],
        "base_model_revision": metadata["base_model_revision"],
        "config_hash": metadata["config_hash"],
        "dataset_hash": metadata["dataset_hash"],
        "stats_file_hash": metadata["stats_file_hash"],
        "git_commit": metadata["git_commit"],
        "git_dirty": metadata["git_dirty"],
        "step": metadata["step"],
        "chunk_size": metadata["chunk_size"],
        "n_action_steps": metadata["n_action_steps"],
        "horizon_seconds": metadata["horizon_seconds"],
        "library_versions": metadata["library_versions"],
    }
    print("checkpoint metadata verified")
    print(json.dumps(summary, indent=2))

    adapter_dir = checkpoint / "adapter"
    if args.fake or (adapter_dir / "fake_adapter.json").is_file():
        print(json.dumps({"offline_l1": _fake_validation(checkpoint, metadata)}, indent=2))
        return 0
    if config is None:
        print("real checkpoint requires --config for offline L1; metadata only", file=sys.stderr)
        return 0
    print(json.dumps({"offline_l1": _real_validation(checkpoint, metadata, config)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
