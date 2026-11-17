#!/usr/bin/env python3
"""Tiny policy server that speaks the real protocol without torch.

Used by inference worker tests to exercise the *real* ``CheckpointPolicy``
client. Behaviour is selected with ``FAKE_POLICY_SERVER_MODE`` so the client's
fixed argv does not need test-only flags.

Modes: ok (default), exit, slow-start, crash, hang, slow-act, bad-nan.
"""

from __future__ import annotations

import os
import sys
import time

from sim2real.evaluation.policy_protocol import (
    PROTOCOL_VERSION,
    read_packet,
    write_packet,
)

MODE = os.environ.get("FAKE_POLICY_SERVER_MODE", "ok")
DELAY = float(os.environ.get("FAKE_POLICY_SERVER_DELAY", "2.0"))


def _actions():
    # Import lazily: the hang/crash/startup modes must not pay numpy's import
    # cost inside the client's short hello timeout.
    import numpy as np

    return (np.arange(4 * 8, dtype=np.float32).reshape(4, 8) * 0.1 + 1.0).astype(np.float32)


def main() -> int:
    out = sys.stdout.buffer
    if MODE == "exit":
        return 1
    hello, _ = read_packet(sys.stdin.buffer)
    if hello.get("type") != "hello":
        return 2
    write_packet(
        out,
        {
            "type": "hello",
            "ok": True,
            "protocol": PROTOCOL_VERSION,
            "checkpoint_hash": hello.get("checkpoint_hash"),
            "state_dim": 8,
            "action_dim": 8,
            "chunk_size": 4,
            "fps": 10.0,
            "image_keys": ["overhead", "wrist"],
        },
    )
    if MODE == "slow-start":
        time.sleep(DELAY)
    write_packet(out, {"type": "ready", "ok": True, "sample_seed": None})

    while True:
        try:
            header, _ = read_packet(sys.stdin.buffer)
        except Exception:
            return 0
        kind = header.get("type")
        if kind == "close":
            return 0
        if kind == "reset":
            write_packet(
                out,
                {"type": "reset", "ok": True, "sample_seed": header.get("sample_seed")},
            )
            continue
        if kind != "act":
            write_packet(out, {"type": "error", "message": f"unknown {kind!r}"})
            continue
        if MODE == "crash":
            return 3
        if MODE == "hang":
            time.sleep(30.0)
            continue
        if MODE == "slow-act":
            time.sleep(DELAY)
        actions = _actions()
        if MODE == "bad-nan":
            actions[0, 0] = float("nan")
        write_packet(
            out,
            {
                "type": "act",
                "actions_shape": [4, 8],
                "dtype": "float32",
                "step_period_s": 0.1,
            },
            [actions.tobytes()],
        )


if __name__ == "__main__":
    raise SystemExit(main())
