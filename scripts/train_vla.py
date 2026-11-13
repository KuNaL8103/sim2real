#!/usr/bin/env python3
"""Small, reproducible VLA fine-tuning entry point.

``--fake`` runs a NumPy-only linear adapter so the full loop, checkpoint, and
provenance path can be exercised in the ROS-free core environment:

    uv run python scripts/train_vla.py --config configs/training/smolvla_lora.yaml \
        --fake --smoke

Without ``--fake`` this loads SmolVLA from .venv-ml (torch/LeRobot/PEFT) and is
guarded by the ``ml``/``gpu`` environment.  A smoke run proves the pipeline runs
and the loss is finite; it says nothing about policy quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sim2real.training.adapters import FakePolicyAdapter, make_adapter  # noqa: E402
from sim2real.training.checkpoint import dataset_fingerprint  # noqa: E402
from sim2real.training.config import load_training_config  # noqa: E402
from sim2real.training.tracking import make_tracker  # noqa: E402
from sim2real.training.trainer import Trainer, load_split_manifest  # noqa: E402


def _synthetic_task(seed: int, dim: int = 8, batches: int = 8, batch_size: int = 4):
    rng = np.random.default_rng(seed)
    teacher = rng.normal(0.0, 0.5, size=(dim, dim))
    result = []
    for index in range(batches):
        inputs = rng.normal(0.0, 1.0, size=(batch_size, dim))
        targets = inputs @ teacher + 0.01 * rng.normal(0.0, 1.0, size=(batch_size, dim))
        result.append(
            {
                "inputs": inputs,
                "targets": targets,
                "episode_id": f"synthetic_train_{index:02d}",
            }
        )
    return result, teacher


def _fake_dataset_metadata(teacher: np.ndarray) -> dict:
    digest = hashlib.sha256(np.ascontiguousarray(teacher).tobytes()).hexdigest()
    return {
        "dataset_root": "synthetic://linear",
        "dataset_version": "fake-v1",
        "dataset_hash": digest,
        "stats_file_hash": None,
        "state_dim": teacher.shape[0],
        "action_dim": teacher.shape[1],
    }


def _fake_split() -> tuple[list[str], list[str], dict]:
    train = [f"synthetic_train_{index:02d}" for index in range(8)]
    val = ["synthetic_val"]
    manifest = {
        "seed": 0,
        "val_fraction": 0.125,
        "train": [
            {"episode_id": name, "index": index, "source": "synthetic"}
            for index, name in enumerate(train)
        ],
        "val": [{"episode_id": "synthetic_val", "index": 8, "source": "synthetic"}],
        "by_source": {"synthetic": {"train": len(train), "val": 1}},
    }
    return train, val, manifest


def _real_batches(config, max_batches: int | None = None):
    """Load LeRobot data + metadata for the real SmolVLA path (guarded)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from torch.utils.data import DataLoader
    from torch.utils.data._utils.collate import default_collate

    root = Path(config.data.dataset_root)
    if not (root / "meta" / "info.json").is_file():
        raise SystemExit(f"dataset metadata not found under {root}")
    split = json.loads((root / "meta" / "sim2real_split.json").read_text(encoding="utf-8"))
    if config.data.episode_indices is not None:
        train_indices = [int(index) for index in config.data.episode_indices]
    else:
        train_indices = [int(row["index"]) for row in split["train"]]
    dataset = LeRobotDataset(
        repo_id=root.name,
        root=root,
        episodes=train_indices,
        delta_timestamps={
            "action": [index / config.data.fps for index in range(config.training.chunk_size)]
        },
    )
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=default_collate,
    )
    batches = []
    for index, batch in enumerate(loader):
        batches.append(batch)
        if max_batches is not None and index + 1 >= max_batches:
            break
    return dataset, batches, train_indices


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/smolvla_lora.yaml")
    parser.add_argument("--smoke", action="store_true", help="bound the run to a few steps")
    parser.add_argument("--fake", action="store_true", help="NumPy-only linear adapter")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    config = load_training_config(args.config)
    run_kind = "fake" if args.fake else config.model.family
    output_dir = Path(args.output_dir or f"outputs/task8/{run_kind}_smoke")
    tracker = make_tracker(
        config.tracking.backend,
        offline=config.tracking.offline,
        run_name=run_kind + "_smoke",
        output_dir=str(output_dir / "tracking"),
    )

    if args.fake:
        batches, teacher = _synthetic_task(config.training.seed)
        adapter = FakePolicyAdapter(8, 8, seed=config.training.seed)
        dataset_metadata = _fake_dataset_metadata(teacher)
        train_ids, val_ids, split_manifest = _fake_split()
        default_steps = 5 if args.smoke else config.training.max_steps
    else:
        dataset, batches, train_indices = _real_batches(
            config, max_batches=4 if args.smoke else None
        )
        adapter = make_adapter(
            config,
            fake=False,
            dataset_stats=dataset.meta.stats,
            dataset_meta=dataset.meta,
        )
        dataset_metadata = {
            **dataset_fingerprint(config.data.dataset_root),
            "state_dim": int(dataset.meta.features["observation.state"]["shape"][0]),
            "action_dim": int(dataset.meta.features["action"]["shape"][0]),
        }
        split_manifest = json.loads(
            (Path(config.data.dataset_root) / "meta" / "sim2real_split.json").read_text(
                encoding="utf-8"
            )
        )
        if config.data.episode_indices is not None:
            id_by_index = {
                int(row["index"]): str(row["episode_id"])
                for row in split_manifest.get("train", []) + split_manifest.get("val", [])
            }
            train_ids = [id_by_index[int(index)] for index in train_indices]
            val_ids = []
        else:
            train_ids, val_ids = load_split_manifest(
                Path(config.data.dataset_root) / "meta" / "sim2real_split.json"
            )
        default_steps = 2 if args.smoke else config.training.max_steps

    limit = args.max_steps if args.max_steps is not None else default_steps
    trainer = Trainer(
        config,
        adapter,
        batches,
        output_dir=output_dir,
        dataset_metadata=dataset_metadata,
        split_manifest=split_manifest,
        train_episode_ids=train_ids,
        val_episode_ids=val_ids,
        mixture_manifest=config.data.mixture_path,
        tracker=tracker,
        repo_root=ROOT,
    )
    result = trainer.run(max_steps=limit)
    peak_vram_mib = None
    if not args.fake:
        try:
            import torch

            if torch.cuda.is_available():
                peak_vram_mib = torch.cuda.max_memory_allocated() / (1024**2)
        except Exception:  # pragma: no cover - reporting only
            peak_vram_mib = None
    try:
        import resource

        peak_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:  # pragma: no cover - reporting only
        peak_rss_mib = None
    summary = {
        "run_kind": run_kind,
        "config": str(config.source),
        "config_hash": config.config_hash,
        "horizon_seconds": config.horizon_seconds,
        "final_step": result.final_step,
        "last_loss": result.last_loss,
        "stopped": result.stopped,
        "checkpoint_dir": result.checkpoint_dir,
        "trainable_parameters": adapter.trainable_parameters(),
        "total_parameters": adapter.total_parameters(),
        "peak_vram_mib": peak_vram_mib,
        "peak_rss_mib": peak_rss_mib,
        "losses": list(result.losses),
        "step_seconds": list(result.step_seconds),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
