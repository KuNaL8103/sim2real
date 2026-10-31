#!/usr/bin/env python3
"""Render benchmark for the MuJoCo backend (opt-in, never used by CI).

Prints the GL renderer string, milliseconds per frame for both cameras, and
peak RSS. Optionally dumps one overhead RGB-D frame for cross-renderer
comparison:

    uv run --group sim python scripts/bench_render.py --frames 3 \
        --dump log/render_llvmpipe.npz

Environment variables affect the selected driver, e.g.
``GALLIUM_DRIVER=d3d12`` and ``MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA``.
Nothing here is a supported default.
"""

from __future__ import annotations

import argparse
import os
import resource
import time
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=5, help="timed frames per camera")
    parser.add_argument("--config-dir", default="configs/sim")
    parser.add_argument("--dump", type=Path, default=None, help="write overhead RGB-D .npz")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")

    from sim2real.sim import create_backend

    backend = create_backend("mujoco", args.config_dir)
    try:
        backend.reset(seed=0)
        # Warm up EGL and both camera renderers before timing.
        for camera in ("overhead", "wrist"):
            backend.render(camera)

        renderer = "unavailable"
        try:
            from OpenGL import GL

            value = GL.glGetString(GL.GL_RENDERER)
            renderer = value.decode() if value else "unavailable"
        except Exception as exc:  # diagnostic only
            renderer = f"query failed: {type(exc).__name__}: {exc}"

        timings: dict[str, float] = {}
        last_frames: dict[str, object] = {}
        for camera in ("overhead", "wrist"):
            start = time.perf_counter()
            for _ in range(args.frames):
                last_frames[camera] = backend.render(camera)
            timings[camera] = (time.perf_counter() - start) / args.frames * 1000.0

        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL', '<unset>')}")
        print(f"GALLIUM_DRIVER={os.environ.get('GALLIUM_DRIVER', '<unset>')}")
        print(
            "MESA_D3D12_DEFAULT_ADAPTER_NAME="
            f"{os.environ.get('MESA_D3D12_DEFAULT_ADAPTER_NAME', '<unset>')}"
        )
        print(f"renderer={renderer}")
        print(f"overhead_ms={timings['overhead']:.1f}")
        print(f"wrist_ms={timings['wrist']:.1f}")
        print(f"peak_rss_kb={peak_kb}")

        if args.dump is not None:
            frame = last_frames["overhead"]
            args.dump.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.dump, rgb=frame.rgb, depth=frame.depth, stamp_ns=frame.stamp_ns
            )
            print(f"dump={args.dump}")
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
