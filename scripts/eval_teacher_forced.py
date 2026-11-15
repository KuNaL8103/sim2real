#!/usr/bin/env python3
"""Teacher-forced first-step / first-10 error and pre-registered overfit check.

Runs in ``.venv-ml``. It loads the checkpoint through the same policy-server
model wrapper used by evaluation (LeRobot pre/post processors included), feeds
recorded dataset frames, and compares the predicted chunk with the recorded
actions. The training loss is optional and always advisory.

The pure functions (``load_rule``, ``frame_list``, ``compute_frame_metrics``,
``evaluate_criteria``) are CPU-only and mocked in tests; the CLI loads torch,
LeRobot and the checkpoint only when actually run.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

RULE_KEYS = {
    "arm_dims",
    "first10_steps",
    "sample_seeds",
    "first_step_arm_l1_max_rad",
    "first10_beats_hold_fraction_min",
    "gripper_first_step_l1_max_m",
    "gripper_clip_min_m",
    "gripper_clip_max_m",
    "gripper_clip_delta_max_m",
    "loss_advisory_threshold",
}


class TeacherForcedError(RuntimeError):
    """Raised for malformed rules, frames, actions or predictions."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TeacherForcedError(message)


def load_rule(path: str | Path) -> dict[str, Any]:
    rule_path = Path(path)
    if not rule_path.is_file():
        raise TeacherForcedError(f"{rule_path}: rule file not found")
    try:
        data = yaml.safe_load(rule_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise TeacherForcedError(f"{rule_path}: invalid YAML: {exc}") from exc
    _require(isinstance(data, dict), f"{rule_path}: expected a mapping")
    _require(data.get("schema_version") == 1, f"{rule_path}: schema_version must be 1")
    rule = data.get("overfit_rule")
    _require(isinstance(rule, dict), f"{rule_path}: overfit_rule must be a mapping")
    unknown = sorted(set(rule) - RULE_KEYS)
    _require(not unknown, f"{rule_path}: unknown overfit_rule keys {unknown}")
    missing = sorted(RULE_KEYS - set(rule))
    _require(not missing, f"{rule_path}: missing overfit_rule keys {missing}")
    arm_dims = int(rule["arm_dims"])
    first10 = int(rule["first10_steps"])
    _require(arm_dims >= 1, f"{rule_path}: arm_dims must be >= 1")
    _require(first10 >= 1, f"{rule_path}: first10_steps must be >= 1")
    seeds = rule["sample_seeds"]
    _require(isinstance(seeds, list) and seeds, f"{rule_path}: sample_seeds must be non-empty")
    _require(
        all(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0 for seed in seeds),
        f"{rule_path}: sample_seeds must be non-negative integers",
    )
    _require(len(set(seeds)) == len(seeds), f"{rule_path}: sample_seeds contains duplicates")
    for key in (
        "first_step_arm_l1_max_rad",
        "first10_beats_hold_fraction_min",
        "gripper_first_step_l1_max_m",
        "gripper_clip_min_m",
        "gripper_clip_max_m",
        "gripper_clip_delta_max_m",
        "loss_advisory_threshold",
    ):
        value = float(rule[key])
        _require(math.isfinite(value), f"{rule_path}: {key} must be finite")
    _require(
        float(rule["gripper_clip_min_m"]) <= float(rule["gripper_clip_max_m"]),
        f"{rule_path}: gripper clip range is inverted",
    )
    parsed = dict(rule)
    parsed["arm_dims"] = arm_dims
    parsed["first10_steps"] = first10
    parsed["sample_seeds"] = [int(seed) for seed in seeds]
    for key in (
        "first_step_arm_l1_max_rad",
        "first10_beats_hold_fraction_min",
        "gripper_first_step_l1_max_m",
        "gripper_clip_min_m",
        "gripper_clip_max_m",
        "gripper_clip_delta_max_m",
        "loss_advisory_threshold",
    ):
        parsed[key] = float(parsed[key])
    return parsed


def default_frame_list(length: int) -> list[int]:
    """Frames 0-5, 10, 20, 30, then every 10th frame up to the episode end."""
    _require(length >= 1, "episode length must be >= 1")
    base = [*range(0, 6), 10, 20, 30]
    frames = [frame for frame in base if frame < length]
    if length > 40:
        frames.extend(range(40, length, 10))
    return sorted(set(frames))


def parse_frames(text: str | None, length: int) -> list[int]:
    if text is None:
        return default_frame_list(length)
    frames: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            frames.append(int(part))
        except ValueError as exc:
            raise TeacherForcedError(f"invalid frame {part!r}; expected integers") from exc
    _require(bool(frames), "explicit frame list is empty")
    _require(all(frame >= 0 for frame in frames), "frames must be non-negative")
    _require(all(frame < length for frame in frames), f"frames must be < episode length {length}")
    return sorted(set(frames))


def _as_actions(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    _require(array.ndim == 2 and array.shape[1] == 8, f"{name}: expected shape (L, 8)")
    _require(np.all(np.isfinite(array)), f"{name}: contains NaN or infinity")
    return array


def compute_frame_metrics(
    predicted_chunk: Any,
    recorded_actions: Any,
    state: Any,
    frame: int,
    rule: dict[str, Any],
) -> dict[str, Any]:
    """Metrics for one frame; all L1 values are in physical units."""
    predictions = np.asarray(predicted_chunk, dtype=np.float64)
    _require(predictions.ndim == 2 and predictions.shape[1] == 8, "predictions: expected (K, 8)")
    _require(np.all(np.isfinite(predictions)), "predictions: contains NaN or infinity")
    actions = _as_actions(recorded_actions, "recorded_actions")
    state_array = np.asarray(state, dtype=np.float64)
    _require(state_array.shape == (8,), "state: expected shape (8,)")
    _require(np.all(np.isfinite(state_array)), "state: contains NaN or infinity")
    _require(0 <= frame < len(actions), f"frame {frame} outside actions length {len(actions)}")
    arm_dims = int(rule["arm_dims"])
    _require(arm_dims <= 8, "arm_dims must be <= 8")
    remaining = len(actions) - frame
    steps = min(int(rule["first10_steps"]), remaining, len(predictions))
    _require(steps >= 1, "no future actions available for first-10 comparison")

    first_pred = predictions[0]
    first_recorded = actions[frame]
    first_error = np.abs(first_pred - first_recorded)
    first10_pred = predictions[:steps]
    first10_recorded = actions[frame : frame + steps]
    hold = np.repeat(state_array[None, :], steps, axis=0)
    clipped = np.clip(
        predictions[:, 7],
        float(rule["gripper_clip_min_m"]),
        float(rule["gripper_clip_max_m"]),
    )
    return {
        "frame": int(frame),
        "steps_compared": int(steps),
        "predicted_first_step": first_pred.tolist(),
        "recorded_first_step": first_recorded.tolist(),
        "first_step_abs_error_per_dim": first_error.tolist(),
        "first_step_l1_arm": float(first_error[:arm_dims].mean()),
        "first_step_l1_all": float(first_error.mean()),
        "first10_l1": float(np.abs(first10_pred - first10_recorded).mean()),
        "hold_last_first_step_l1": float(np.abs(hold[0] - first_recorded).mean()),
        "hold_last_first10_l1": float(np.abs(hold - first10_recorded).mean()),
        "gripper_first_step_l1": float(first_error[7]),
        "gripper_clip_delta": float(np.abs(clipped - predictions[:, 7]).mean()),
    }


def evaluate_criteria(rows: list[dict[str, Any]], rule: dict[str, Any]) -> dict[str, Any]:
    _require(bool(rows), "no frame metrics to evaluate")
    arm = float(np.mean([row["first_step_l1_arm"] for row in rows]))
    beats = float(
        np.mean([1.0 if row["first10_l1"] < row["hold_last_first10_l1"] else 0.0 for row in rows])
    )
    gripper = float(np.mean([row["gripper_first_step_l1"] for row in rows]))
    clip_delta = float(np.mean([row["gripper_clip_delta"] for row in rows]))
    criteria = {
        "first_step_arm_l1": {
            "value": arm,
            "threshold": float(rule["first_step_arm_l1_max_rad"]),
            "passed": arm <= float(rule["first_step_arm_l1_max_rad"]),
        },
        "first10_beats_hold_fraction": {
            "value": beats,
            "threshold": float(rule["first10_beats_hold_fraction_min"]),
            "passed": beats >= float(rule["first10_beats_hold_fraction_min"]),
        },
        "gripper_first_step_l1": {
            "value": gripper,
            "threshold": float(rule["gripper_first_step_l1_max_m"]),
            "passed": gripper <= float(rule["gripper_first_step_l1_max_m"]),
        },
        "gripper_clip_delta": {
            "value": clip_delta,
            "threshold": float(rule["gripper_clip_delta_max_m"]),
            "passed": clip_delta < float(rule["gripper_clip_delta_max_m"]),
        },
    }
    overall = all(entry["passed"] for entry in criteria.values())
    return {
        "frames_evaluated": len(rows),
        "criteria": criteria,
        "overall_passed": overall,
        "loss_advisory": True,
        "loss_advisory_threshold": float(rule["loss_advisory_threshold"]),
    }


def _image_uint8(tensor: Any) -> np.ndarray:
    import torch

    array = (tensor.detach().cpu().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    return array.permute(1, 2, 0).numpy()


def _predict_episode(model: Any, metadata: dict[str, Any], dataset: Any, frames: list[int]):
    rename = metadata.get("camera_rename_map") or {}
    short_to_model = {source.rsplit(".", 1)[-1]: target for source, target in rename.items()}
    short_names = list(short_to_model)
    predictions: list[np.ndarray] = []
    for frame in frames:
        sample = dataset[frame]
        state = sample["observation.state"].numpy()
        images = {name: _image_uint8(sample[f"observation.images.{name}"]) for name in short_names}
        header = {
            "state": [float(value) for value in state],
            "task": str(sample["task"]),
            "stamp_ns": 0,
            "image_keys": short_names,
            "image_shapes": [list(images[name].shape) for name in short_names],
            "image_dtypes": ["uint8" for _ in short_names],
        }
        blobs = [np.ascontiguousarray(images[name]).tobytes() for name in short_names]
        chunks = []
        for seed in metadata["_sample_seeds"]:
            model.set_sample_seed(seed)
            chunks.append(np.asarray(model.act(header, blobs), dtype=np.float64))
        predictions.append(np.stack(chunks))
    return predictions


def run(
    *,
    checkpoint: Path,
    dataset_root: Path,
    episode_indices: list[int],
    frames: str | None,
    rule_path: Path,
    sample_seeds: list[int] | None,
    loss_log: Path | None,
    device: str | None,
) -> dict[str, Any]:
    rule = load_rule(rule_path)
    if sample_seeds is not None:
        _require(bool(sample_seeds), "sample seeds are empty")
        rule["sample_seeds"] = [int(seed) for seed in sample_seeds]

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from sim2real.training.checkpoint import load_checkpoint

    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    from policy_server import RealModel

    metadata = load_checkpoint(checkpoint)
    metadata = dict(metadata)
    metadata["dataset_root"] = str(dataset_root)
    metadata["_sample_seeds"] = rule["sample_seeds"]
    model = RealModel(checkpoint, metadata, device)

    episodes: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for index in episode_indices:
        dataset = LeRobotDataset(repo_id=dataset_root.name, root=dataset_root, episodes=[index])
        _require(len(dataset) > 0, f"episode {index}: dataset is empty")
        frame_indices = parse_frames(frames, len(dataset))
        recorded = np.stack(
            [dataset[frame]["action"].numpy() for frame in range(len(dataset))]
        ).astype(np.float64)
        states = np.stack(
            [dataset[frame]["observation.state"].numpy() for frame in range(len(dataset))]
        ).astype(np.float64)
        chunks = _predict_episode(model, metadata, dataset, frame_indices)
        rows = []
        for frame, chunk in zip(frame_indices, chunks, strict=True):
            seed_rows = [
                compute_frame_metrics(chunk[seed_index], recorded, states[frame], frame, rule)
                for seed_index in range(chunk.shape[0])
            ]
            mean = {
                "frame": frame,
                "steps_compared": seed_rows[0]["steps_compared"],
                "first_step_l1_arm": float(
                    np.mean([row["first_step_l1_arm"] for row in seed_rows])
                ),
                "first_step_l1_all": float(
                    np.mean([row["first_step_l1_all"] for row in seed_rows])
                ),
                "first10_l1": float(np.mean([row["first10_l1"] for row in seed_rows])),
                "hold_last_first_step_l1": float(
                    np.mean([row["hold_last_first_step_l1"] for row in seed_rows])
                ),
                "hold_last_first10_l1": float(
                    np.mean([row["hold_last_first10_l1"] for row in seed_rows])
                ),
                "gripper_first_step_l1": float(
                    np.mean([row["gripper_first_step_l1"] for row in seed_rows])
                ),
                "gripper_clip_delta": float(
                    np.mean([row["gripper_clip_delta"] for row in seed_rows])
                ),
                "first_step_abs_error_per_dim": np.mean(
                    [row["first_step_abs_error_per_dim"] for row in seed_rows], axis=0
                ).tolist(),
                "per_seed": seed_rows,
            }
            rows.append(mean)
            all_rows.append(mean)
        episodes[str(index)] = {"length": len(dataset), "frames": rows}
        dataset = None

    criteria = evaluate_criteria(all_rows, rule)
    advisory = {"provided": loss_log is not None}
    if loss_log is not None and loss_log.is_file():
        losses = [
            json.loads(line)["loss"]
            for line in loss_log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if losses:
            advisory = {
                "provided": True,
                "last20_mean_loss": float(np.mean(losses[-20:])),
                "below_threshold": bool(np.mean(losses[-20:]) < rule["loss_advisory_threshold"]),
                "note": "advisory only; not a pass criterion",
            }
    return {
        "checkpoint": str(checkpoint),
        "dataset_root": str(dataset_root),
        "episode_indices": episode_indices,
        "rule_path": str(rule_path),
        "rule": rule,
        "episodes": episodes,
        "criteria": criteria,
        "advisory_loss": advisory,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="data/lerobot/train_v1")
    parser.add_argument("--episodes", default="0,1")
    parser.add_argument("--frames", default=None, help="explicit comma list; default is the rule")
    parser.add_argument("--sample-seeds", default=None, help="comma list; default from the rule")
    parser.add_argument("--rule", default="configs/evaluation/overfit_rule.yaml")
    parser.add_argument("--loss-log", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    try:
        episodes = [int(part) for part in args.episodes.split(",") if part.strip()]
        _require(bool(episodes), "no episode indices")
        seeds = (
            [int(part) for part in args.sample_seeds.split(",") if part.strip()]
            if args.sample_seeds
            else None
        )
        result = run(
            checkpoint=Path(args.checkpoint),
            dataset_root=Path(args.dataset),
            episode_indices=episodes,
            frames=args.frames,
            rule_path=Path(args.rule),
            sample_seeds=seeds,
            loss_log=Path(args.loss_log) if args.loss_log else None,
            device=args.device,
        )
    except (TeacherForcedError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    for name, entry in result["criteria"]["criteria"].items():
        status = "PASS" if entry["passed"] else "FAIL"
        print(f"{status} {name}: {entry['value']:.6g} (threshold {entry['threshold']:.6g})")
    print("OVERALL", "PASS" if result["criteria"]["overall_passed"] else "FAIL")
    return 0 if result["criteria"]["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
