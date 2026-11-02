#!/usr/bin/env python3
"""Build a local LeRobot v3 dataset from curated synthetic episodes.

The writer step requires ``.venv-ml``; ``--dry-run`` only imports the
numpy-only core and is safe in the core environment.  Replay images must be
produced first with ``scripts/render_replay.py``; this script reuses the PNG
references and SHA-256 values that script writes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sim2real.dataset_builder import (  # noqa: E402
    load_embodiments,
    split_episodes,
    validate_image_policy,
)
from sim2real.dataset_builder.adapters.curated_v1 import CuratedAdapter  # noqa: E402


def _load_defaults(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(raw.get("dataset", {}))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curated", required=True)
    parser.add_argument("--staging", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--embodiments", default="configs/dataset/embodiments.yaml")
    parser.add_argument("--config", default="configs/dataset/default.yaml")
    parser.add_argument("--action-representation", default=None)
    parser.add_argument("--decimation", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--renderer", default=None)
    parser.add_argument("--allow-imageless", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    defaults = _load_defaults(Path(args.config))
    decimation = args.decimation or int(defaults.get("decimation", 2))
    representation = args.action_representation or str(
        defaults.get("action_representation", "joint_position")
    )
    max_episodes = args.max_episodes or int(defaults.get("max_episodes", 40))
    renderer = args.renderer or str(defaults.get("renderer", "llvmpipe"))
    allow_imageless = args.allow_imageless or bool(defaults.get("allow_imageless", False))
    if decimation < 1:
        parser.error("--decimation must be >= 1")

    embodiments = load_embodiments(args.embodiments)
    spec = embodiments["panda_sim"]
    adapter = CuratedAdapter(
        curated_root=args.curated,
        embodiment=spec,
        decimation=decimation,
        action_representation=representation,
        staging_root=args.staging,
        allow_imageless=allow_imageless,
        declared_renderer=renderer,
        cameras=tuple(defaults.get("cameras", spec.cameras)),
    )
    episodes = adapter.load(max_episodes=max_episodes)
    if not episodes:
        raise SystemExit("no episodes selected")
    fps = episodes[0].fps
    image_policy = validate_image_policy(
        episodes,
        renderer=renderer,
        decimation=decimation,
        fps=fps,
        allow_imageless=allow_imageless,
        max_joint_error_rad=float(defaults.get("max_replay_joint_error_rad", 1e-6)),
        max_object_error_m=float(defaults.get("max_replay_object_error_m", 1e-6)),
    )
    split = split_episodes(
        episodes,
        val_fraction=(
            float(defaults.get("val_fraction", 0.2))
            if args.val_fraction is None
            else args.val_fraction
        ),
        seed=int(defaults.get("split_seed", 123) if args.seed is None else args.seed),
    )
    frames = sum(episode.num_images for episode in episodes)
    image_bytes = frames * 256 * 256 * 3
    summary = {
        "episodes": len(episodes),
        "frames": sum(episode.num_frames for episode in episodes),
        "image_frames": frames,
        "image_gib": image_bytes / (1024.0**3),
        "estimated_replay_minutes_at_0.2s": frames * 0.2 / 60.0,
        "fps": fps,
        "decimation": decimation,
        "renderer": image_policy["renderer"],
        "image_mode": image_policy["mode"],
        "split": split.manifest(episodes),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.dry_run:
        return 0

    from sim2real.dataset_builder.lerobot_writer import build_lerobot_dataset  # noqa: PLC0415

    out_root = args.out or str(defaults.get("output_root", "data/lerobot"))
    dataset_name = args.name or str(defaults.get("name", "train_v1"))
    result = build_lerobot_dataset(
        episodes,
        spec,
        out_root,
        dataset_name,
        representation=representation,
        use_videos=str(defaults.get("image_mode", "image")) == "video",
        renderer=image_policy["renderer"],
        decimation=decimation,
        split_manifest=split.manifest(episodes),
        max_episodes=max_episodes,
    )
    print(
        json.dumps(
            {
                "path": str(result.path),
                "episodes": result.num_episodes,
                "frames": result.num_frames,
                "fps": result.fps,
                "image_mode": result.image_mode,
                "stats_keys": list(result.stats_keys),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
