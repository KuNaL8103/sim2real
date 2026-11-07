"""Deterministic training loop over a :class:`PolicyAdapter`.

The loop is deliberately small: it owns seeding, epoch order, gradient
accumulation, finite-loss checks, checkpoint cadence, resume, and graceful
interruption.  Dataset loading, preprocessing, and the real model live in the
adapter and the CLI.

The Task 7 episode-level split is enforced here: a validation episode id in the
training set or in any batch is a hard error, not a warning.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import checkpoint as checkpoint_module
from .adapters import PolicyAdapter


class TrainerError(RuntimeError):
    """Raised for NaN/inf loss, a split violation, or an invalid resume."""


TRAINER_STATE_NAME = "trainer_state.json"


def load_split_manifest(path: str | Path) -> tuple[list[str], list[str]]:
    """Read the Task 7 ``meta/sim2real_split.json`` episode ids."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise TrainerError(f"{manifest_path}: split manifest not found")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    train = [str(row["episode_id"]) for row in data.get("train", [])]
    val = [str(row["episode_id"]) for row in data.get("val", [])]
    overlap = sorted(set(train) & set(val))
    if overlap:
        raise TrainerError(f"{manifest_path}: train/val overlap {overlap}")
    return train, val


def select_episode_ids(
    records: Iterable[dict[str, Any]],
    *,
    train_ids: Sequence[str],
    exclude_perturbed: bool,
) -> list[str]:
    """Return training episode ids present in ``records``.

    ``exclude_perturbed`` is opt-in (default false) because the accepted set
    already contains perturbed episodes that passed the filter; hiding that
    would misrepresent the data.
    """
    allowed = set(train_ids)
    selected: list[str] = []
    for record in records:
        episode_id = str(record.get("episode_id"))
        if episode_id not in allowed:
            continue
        if exclude_perturbed and bool(record.get("is_perturbed", False)):
            continue
        selected.append(episode_id)
    return selected


@dataclass(frozen=True)
class RunResult:
    final_step: int
    last_loss: float | None
    checkpoint_dir: str | None
    stopped: bool
    losses: tuple[float, ...]
    step_seconds: tuple[float, ...] = ()


class Trainer:
    def __init__(
        self,
        config: Any,
        adapter: PolicyAdapter,
        batches: Sequence[dict[str, Any]],
        *,
        output_dir: str | Path,
        dataset_metadata: dict[str, Any],
        split_manifest: dict[str, Any],
        train_episode_ids: Sequence[str],
        val_episode_ids: Sequence[str],
        mixture_manifest: str | None = None,
        tracker: Any = None,
        repo_root: str | Path = ".",
    ) -> None:
        if not batches:
            raise TrainerError("batches: expected at least one batch")
        overlap = sorted(set(train_episode_ids) & set(val_episode_ids))
        if overlap:
            raise TrainerError(f"train/val episode overlap: {overlap}")
        self.config = config
        self.adapter = adapter
        self.batches = list(batches)
        self.output_dir = Path(output_dir)
        self.dataset_metadata = dict(dataset_metadata)
        self.split_manifest = dict(split_manifest)
        self.train_episode_ids = list(train_episode_ids)
        self.val_episode_ids = set(val_episode_ids)
        self.mixture_manifest = mixture_manifest
        self.tracker = tracker
        self.repo_root = repo_root
        self.step = 0
        self.epoch = 0
        self.offset = 0
        self._order: list[int] = []
        self._rng = np.random.default_rng(config.training.seed)
        self._stop_requested = False
        self.losses: list[float] = []
        self.step_times: list[float] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loss_log_path = self.output_dir / "losses.jsonl"

    def request_stop(self) -> None:
        self._stop_requested = True

    def _shuffle_order(self) -> None:
        self._order = [int(index) for index in self._rng.permutation(len(self.batches))]

    def _next_batch(self) -> dict[str, Any]:
        if not self._order or self.offset >= len(self._order):
            self._shuffle_order()
            self.offset = 0
            self.epoch += 1
        index = self._order[self.offset]
        self.offset += 1
        batch = self.batches[index]
        episode_id = batch.get("episode_id") if isinstance(batch, dict) else None
        if episode_id is not None and str(episode_id) in self.val_episode_ids:
            raise TrainerError(
                f"split violation: validation episode {episode_id!r} was drawn for training"
            )
        return batch

    def _accumulate(self) -> float:
        accumulated = 0.0
        for _ in range(self.config.training.grad_accum):
            batch = self._next_batch()
            prepared = (
                self.adapter.prepare_batch(batch)
                if hasattr(self.adapter, "prepare_batch")
                else batch
            )
            loss = self.adapter.forward_loss(prepared)
            try:
                value = (
                    float(loss.detach().cpu().item()) if hasattr(loss, "detach") else float(loss)
                )
            except (TypeError, ValueError) as exc:
                raise TrainerError(f"step {self.step}: loss is not a scalar: {loss!r}") from exc
            if not np.isfinite(value):
                raise TrainerError(
                    f"step {self.step}: loss is {value}; aborting instead of writing "
                    "a bad checkpoint"
                )
            self.adapter.backward(loss)
            accumulated += value / self.config.training.grad_accum
        return float(accumulated)

    def _write_trainer_state(self, path: Path) -> None:
        state = {
            "step": self.step,
            "epoch": self.epoch,
            "offset": self.offset,
            "order": self._order,
            "rng_state": self._rng.bit_generator.state,
        }
        (path / TRAINER_STATE_NAME).write_text(
            json.dumps(state, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _save(self, step: int) -> Path:
        metadata = checkpoint_module.build_metadata(
            config=self.config,
            dataset=self.dataset_metadata,
            split=self.split_manifest,
            step=step,
            state_dim=self._state_dim,
            action_dim=self._action_dim,
            mixture_manifest=self.mixture_manifest,
            repo_root=self.repo_root,
        )
        path = self.output_dir / f"checkpoint-{step:08d}"
        checkpoint_module.save_checkpoint(path, self.adapter, metadata)
        self._write_trainer_state(path)
        return path

    def _log(self, step: int, loss: float, seconds: float | None = None) -> None:
        row = {"step": step, "loss": loss}
        if seconds is not None:
            row["step_seconds"] = seconds
        with self.loss_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        if self.tracker is not None:
            self.tracker.log_metrics(step, {"loss": loss})

    def _restore(self, resume_from: str | Path) -> None:
        checkpoint_dir = Path(resume_from)
        metadata = checkpoint_module.load_checkpoint(
            checkpoint_dir, expected_config_hash=self.config.config_hash
        )
        if metadata.get("config_hash") != self.config.config_hash:
            raise TrainerError("resume checkpoint config hash does not match this config")
        self.adapter.load_adapter(checkpoint_dir / "adapter")
        state_path = checkpoint_dir / TRAINER_STATE_NAME
        if not state_path.is_file():
            raise TrainerError(f"{state_path}: trainer state not found; cannot resume exactly")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        self.step = int(state["step"])
        self.epoch = int(state["epoch"])
        self.offset = int(state["offset"])
        if "order" in state:
            self._order = [int(index) for index in state["order"]]
        self._rng.bit_generator.state = state["rng_state"]

    @property
    def _state_dim(self) -> int:
        return int(self.dataset_metadata.get("state_dim", 0))

    @property
    def _action_dim(self) -> int:
        return int(self.dataset_metadata.get("action_dim", 0))

    def run(
        self,
        *,
        max_steps: int | None = None,
        resume_from: str | Path | None = None,
    ) -> RunResult:
        if resume_from is not None:
            self._restore(resume_from)
        limit = self.config.training.max_steps if max_steps is None else int(max_steps)
        if limit < 1:
            raise TrainerError("max_steps: expected >= 1")
        if self.tracker is not None:
            self.tracker.log_config(self.config.to_dict())
        checkpoint_dir: Path | None = None
        stopped = False
        try:
            while self.step < limit:
                if self._stop_requested:
                    stopped = True
                    break
                started = time.perf_counter()
                loss = self._accumulate()
                self.adapter.optimizer_step(self.config.training.lr)
                self.adapter.zero_grad()
                seconds = time.perf_counter() - started
                self.step += 1
                self.losses.append(loss)
                self.step_times.append(seconds)
                self._log(self.step, loss, seconds)
                save_every = self.config.training.save_every
                if save_every > 0 and self.step % save_every == 0:
                    checkpoint_dir = self._save(self.step)
            if self.step > 0:
                checkpoint_dir = self._save(self.step)
        except KeyboardInterrupt:
            stopped = True
            if self.step > 0:
                checkpoint_dir = self._save(self.step)
        finally:
            if self.tracker is not None:
                self.tracker.close()
        return RunResult(
            final_step=self.step,
            last_loss=self.losses[-1] if self.losses else None,
            checkpoint_dir=str(checkpoint_dir) if checkpoint_dir is not None else None,
            stopped=stopped,
            losses=tuple(self.losses),
            step_seconds=tuple(self.step_times),
        )


__all__ = [
    "RunResult",
    "Trainer",
    "TrainerError",
    "TRAINER_STATE_NAME",
    "load_split_manifest",
    "select_episode_ids",
]
