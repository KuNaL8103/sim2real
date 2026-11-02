#!/usr/bin/env python3
"""Convert a bounded Bridge fixture (or approved TFDS subset) to LeRobot.

The TFDS path is deliberately gated: without ``--i-approved-download`` it
prints the published size estimate and exits non-zero without touching the
network.  The fixture path is the only one used by CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sim2real.dataset_builder import load_embodiments  # noqa: E402
from sim2real.dataset_builder.adapters.rlds_bridge import (  # noqa: E402
    BridgeAdapterConfig,
    FixtureEpisodeSource,
    TFDSEpisodeSource,
    adapt_rlds_source,
)

BRIDGE_SIZE_ESTIMATE_GB = 387.49
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "bridge" / "tiny_bridge.json"


def _resolution(value: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in value.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("resolution must look like 256x256") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("resolution must be positive")
    return height, width


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("fixture", "tfds"), required=True)
    parser.add_argument("--max-episodes", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=_resolution, default=(256, 256))
    parser.add_argument("--embodiments", default="configs/dataset/embodiments.yaml")
    parser.add_argument("--action-representation", default="bridge_ee_delta")
    parser.add_argument(
        "--on-missing-instruction", choices=("skip", "fill_default"), default="fill_default"
    )
    parser.add_argument("--missing-image", choices=("skip", "zeros"), default="skip")
    parser.add_argument("--name", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--i-approved-download", action="store_true")
    args = parser.parse_args(argv)

    if args.source == "tfds" and not args.i_approved_download:
        print(
            f"refusing TFDS without --i-approved-download: Bridge/Open X is about "
            f"{BRIDGE_SIZE_ESTIMATE_GB:.2f} GB on disk; no network access was made",
            file=sys.stderr,
        )
        return 2
    if args.source == "fixture":
        source = FixtureEpisodeSource(args.fixture)
    else:
        source = TFDSEpisodeSource(
            "bridge",
            data_dir=args.data_dir,
            download=args.i_approved_download,
        )
    spec = load_embodiments(args.embodiments)["widowx_bridge"]
    config = BridgeAdapterConfig(
        max_episodes=args.max_episodes,
        seed=args.seed,
        resolution=args.resolution,
        on_missing_instruction=args.on_missing_instruction,
        missing_image=args.missing_image,
    )
    episodes, report = adapt_rlds_source(
        source,
        spec,
        config,
        action_representation=args.action_representation,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    if not episodes:
        print("no episodes converted; nothing written", file=sys.stderr)
        return 1
    from sim2real.dataset_builder.lerobot_writer import build_lerobot_dataset  # noqa: PLC0415

    out = Path(args.out).expanduser().resolve()
    name = args.name or out.name
    result = build_lerobot_dataset(
        episodes,
        spec,
        out.parent,
        name,
        representation=args.action_representation,
        use_videos=False,
        renderer="bridge_source",
        decimation=spec.decimation,
        split_manifest=None,
    )
    print(
        json.dumps(
            {
                "path": str(result.path),
                "episodes": result.num_episodes,
                "frames": result.num_frames,
                "fps": result.fps,
                "image_mode": result.image_mode,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
