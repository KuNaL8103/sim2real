#!/usr/bin/env python3
"""Out-of-process policy server.

Runs in the ML environment (``.venv-ml`` with ``PYTHONPATH=src``) and speaks the
length-prefixed JSON + raw-array protocol from
:mod:`sim2real.evaluation.policy_protocol` over stdin/stdout.  The simulator and
evaluation code stay in the core environment; the two never share a Python
interpreter.

Modes:

* ``--fake-model`` — deterministic, torch-free test double.  Used by the
  hardware-free protocol tests.
* real mode — loads the SmolVLA + PEFT checkpoint, runs LeRobot's official
  preprocessor and postprocessor **inside this process** (so nothing is
  normalised twice), and returns the post-processed action chunk.

The checkpoint metadata content hash is verified against ``--expected-hash``
before any model is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sim2real.evaluation.policy_protocol import (  # noqa: E402
    PROTOCOL_VERSION,
    ProtocolError,
    read_packet,
    write_packet,
)
from sim2real.training.checkpoint import METADATA_NAME, load_checkpoint  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    return value


def _image_keys(metadata: dict[str, Any]) -> list[str]:
    rename = metadata.get("camera_rename_map") or {}
    keys = []
    for source in rename:
        keys.append(source.rsplit(".", 1)[-1])
    if not keys:
        for camera in metadata.get("config", {}).get("model", {}).get("expected_cameras", []):
            keys.append(str(camera).rsplit(".", 1)[-1])
    return keys


def _model_image_name(metadata: dict[str, Any], short_name: str) -> str:
    rename = metadata.get("camera_rename_map") or {}
    for source, target in rename.items():
        if source.rsplit(".", 1)[-1] == short_name:
            return str(target)
    return f"observation.images.{short_name}"


class FakeModel:
    def __init__(self, metadata: dict[str, Any]) -> None:
        self.chunk_size = int(metadata.get("chunk_size", 50))
        self.fps = float(metadata.get("fps", 10.0))

    def act(self, header: dict[str, Any], blobs: list[bytes]) -> np.ndarray:
        del blobs
        state = np.asarray(header.get("state", []), dtype=np.float64)
        if state.shape != (8,) or not np.all(np.isfinite(state)):
            raise ProtocolError("fake model received an invalid state")
        task = str(header.get("task", ""))
        seed = int(hashlib.sha256(task.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng([seed, int(header.get("stamp_ns", 0))])
        offsets = rng.normal(0.0, 1.0e-4, size=(self.chunk_size, 8))
        actions = np.tile(state, (self.chunk_size, 1)) + offsets
        return actions.astype(np.float32)


class RealModel:
    def __init__(self, checkpoint: Path, metadata: dict[str, Any], device: str | None) -> None:
        import torch

        from sim2real.training.adapters import SmolVLAAdapter

        self.torch = torch
        config = _namespace(metadata["config"])
        stats_path = Path(metadata["dataset_root"]) / "meta" / "stats.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else None
        self.adapter = SmolVLAAdapter(config, device=device, dataset_stats=stats)
        self.adapter.load_adapter(checkpoint / "adapter")
        self.adapter.policy.eval()
        self.chunk_size = int(metadata.get("chunk_size", config.training.chunk_size))
        self.fps = float(metadata.get("fps", 10.0))
        self.metadata = metadata

    def act(self, header: dict[str, Any], blobs: list[bytes]) -> np.ndarray:
        torch = self.torch
        key_names = [str(key) for key in header.get("image_keys", [])]
        shapes = [tuple(int(v) for v in shape) for shape in header.get("image_shapes", [])]
        if len(key_names) != len(blobs) or len(shapes) != len(blobs):
            raise ProtocolError("image keys, shapes and blobs disagree")
        batch: dict[str, Any] = {
            "observation.state": torch.tensor([header.get("state", [])], dtype=torch.float32),
            "task": [str(header.get("task", ""))],
        }
        for short_name, shape, blob in zip(key_names, shapes, blobs, strict=True):
            expected = int(np.prod(shape))
            if len(blob) != expected:
                raise ProtocolError(
                    f"image {short_name} has {len(blob)} bytes, expected {expected}"
                )
            image = np.frombuffer(blob, dtype=np.uint8).reshape(shape)
            tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
            batch[_model_image_name(self.metadata, short_name)] = tensor.unsqueeze(0)
        with torch.no_grad():
            prepared = self.adapter.preprocessor(batch)
            actions = self.adapter.policy.predict_action_chunk(prepared)
            if self.adapter.postprocessor is not None:
                actions = self.adapter.postprocessor(actions)
        array = np.asarray(actions.detach().cpu(), dtype=np.float32)
        if array.ndim == 3:
            array = array[0]
        return array


def _handshake(
    metadata: dict[str, Any], checkpoint_hash: str, expected_hash: str | None, *, model: str
) -> dict[str, Any]:
    if expected_hash is not None and checkpoint_hash != expected_hash:
        return {
            "type": "hello",
            "ok": False,
            "protocol": PROTOCOL_VERSION,
            "reason": f"checkpoint hash {checkpoint_hash} != expected {expected_hash}",
        }
    return {
        "type": "hello",
        "ok": True,
        "protocol": PROTOCOL_VERSION,
        "checkpoint_hash": checkpoint_hash,
        "chunk_size": int(metadata.get("chunk_size", 50)),
        "fps": float(metadata.get("fps", 10.0)),
        "state_dim": int(metadata.get("state_dim", 8)),
        "action_dim": int(metadata.get("action_dim", 8)),
        "image_keys": _image_keys(metadata),
        "model": model,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-hash", default=None)
    parser.add_argument("--fake-model", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    metadata_path = checkpoint / METADATA_NAME
    try:
        metadata = load_checkpoint(checkpoint)
    except Exception as exc:  # noqa: BLE001 - the client must see a clear refusal
        write_packet(
            sys.stdout.buffer,
            {"type": "hello", "ok": False, "protocol": PROTOCOL_VERSION, "reason": str(exc)},
        )
        return 2
    checkpoint_hash = _sha256(metadata_path)
    hello, _ = read_packet(sys.stdin.buffer)
    if hello.get("type") != "hello":
        write_packet(
            sys.stdout.buffer,
            {
                "type": "hello",
                "ok": False,
                "protocol": PROTOCOL_VERSION,
                "reason": "expected a hello packet",
            },
        )
        return 2
    if int(hello.get("protocol", -1)) != PROTOCOL_VERSION:
        write_packet(
            sys.stdout.buffer,
            {
                "type": "hello",
                "ok": False,
                "protocol": PROTOCOL_VERSION,
                "reason": f"protocol {hello.get('protocol')} != {PROTOCOL_VERSION}",
            },
        )
        return 2
    reply = _handshake(
        metadata,
        checkpoint_hash,
        args.expected_hash,
        model="fake" if args.fake_model else "smolvla_lora",
    )
    write_packet(sys.stdout.buffer, reply)
    if not reply["ok"]:
        return 3

    try:
        model: Any = (
            FakeModel(metadata) if args.fake_model else RealModel(checkpoint, metadata, args.device)
        )
    except Exception as exc:  # noqa: BLE001 - report through the protocol
        write_packet(
            sys.stdout.buffer,
            {"type": "error", "message": f"model load failed: {type(exc).__name__}: {exc}"},
        )
        return 4

    while True:
        try:
            header, blobs = read_packet(sys.stdin.buffer)
        except ProtocolError as exc:
            write_packet(sys.stdout.buffer, {"type": "error", "message": str(exc)})
            return 5
        kind = header.get("type")
        if kind == "close":
            return 0
        if kind != "act":
            write_packet(
                sys.stdout.buffer, {"type": "error", "message": f"unknown packet {kind!r}"}
            )
            continue
        try:
            actions = np.asarray(model.act(header, blobs), dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != 8 or not np.all(np.isfinite(actions)):
                raise ProtocolError(f"model returned an invalid chunk {actions.shape}")
        except Exception as exc:  # noqa: BLE001 - one bad request must not kill the server
            write_packet(
                sys.stdout.buffer,
                {"type": "error", "message": f"{type(exc).__name__}: {exc}"},
            )
            continue
        write_packet(
            sys.stdout.buffer,
            {
                "type": "act",
                "actions_shape": [int(actions.shape[0]), int(actions.shape[1])],
                "dtype": "float32",
                "step_period_s": 1.0 / float(model.fps),
            },
            [np.ascontiguousarray(actions).tobytes()],
        )


if __name__ == "__main__":
    raise SystemExit(main())
