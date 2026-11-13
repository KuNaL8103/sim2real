"""Policy interface and the reference/test policies used by evaluation.

A policy receives an observation (8-D state, optional cameras) and returns an
``ActionChunk`` of absolute joint targets (K x 8, seven arm joints plus the
gripper) plus the policy step period.  The closed-loop runner owns execution and
termination; a policy never steps the backend itself.

``ScriptedExpertPolicy`` is the reference upper bound: it runs the Task 4 expert
on its own backend instance at reset and replays the recorded executed actions
against the evaluation backend.  The evaluation backend is reset from the same
seed, so replay is faithful for both the deterministic fake backend and MuJoCo.

``CheckpointPolicy`` is a *client* of the out-of-process policy server.  It is
guarded: a missing server, a handshake mismatch, a timeout, or a dead child raise
``PolicyError``, which the runner records as ``POLICY_ERROR`` instead of crashing
the batch.
"""

from __future__ import annotations

import json
import math
import os
import select
import subprocess
import sys
import threading
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from sim2real.evaluation.policy_protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    read_packet,
    write_packet,
)
from sim2real.expert import (
    derive_episode_seed,
    load_expert_config,
    parse_instruction,
    run_episode,
)
from sim2real.sim import create_backend
from sim2real.training.checkpoint import METADATA_NAME, file_sha256

POLICY_STEP_PERIOD_S = 0.1
DEFAULT_CHECKPOINT_HASH = "none"


class PolicyError(RuntimeError):
    """Raised when a policy cannot produce a valid action chunk."""


@dataclass(frozen=True)
class PolicyObservation:
    state: np.ndarray
    task_instruction: str
    stamp_ns: int = 0
    images: Mapping[str, np.ndarray] = field(default_factory=dict)
    object_poses: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        state = np.asarray(self.state, dtype=np.float64)
        if state.shape != (8,):
            raise PolicyError(f"observation state must be (8,), got {state.shape}")
        if not np.all(np.isfinite(state)):
            raise PolicyError("observation state must be finite")
        object.__setattr__(self, "state", state)
        for name, image in self.images.items():
            array = np.asarray(image)
            if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
                raise PolicyError(f"image {name!r} must be uint8 HxWx3")


@dataclass(frozen=True)
class ActionChunk:
    actions: np.ndarray
    step_period_s: float = POLICY_STEP_PERIOD_S

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 8 or actions.shape[0] < 1:
            raise PolicyError(f"actions must have shape (K>=1, 8), got {actions.shape}")
        if not np.all(np.isfinite(actions)):
            raise PolicyError("actions must be finite")
        if not math.isfinite(float(self.step_period_s)) or self.step_period_s <= 0:
            raise PolicyError("step_period_s must be finite and > 0")
        object.__setattr__(self, "actions", actions)


@runtime_checkable
class Policy(Protocol):
    name: str
    needs_images: bool
    info: dict[str, Any]

    def reset(self, task_instruction: str, seed: int) -> None: ...

    def act(self, observation: PolicyObservation) -> ActionChunk: ...


def _state_from_observation(observation: Any) -> np.ndarray:
    robot = observation.robot_state
    return np.concatenate(
        [np.asarray(robot.joint_positions, dtype=np.float64), [float(robot.gripper_position)]]
    )


class ZeroPolicy:
    """Hold the current joint positions and gripper opening."""

    name = "zero"
    needs_images = False

    def __init__(self) -> None:
        self.info: dict[str, Any] = {"name": self.name, "checkpoint_hash": DEFAULT_CHECKPOINT_HASH}

    def reset(self, task_instruction: str, seed: int) -> None:
        del task_instruction, seed

    def act(self, observation: PolicyObservation) -> ActionChunk:
        # ``PolicyObservation.state`` is exactly the eight held values.
        return ActionChunk(observation.state.reshape(1, 8))


class RandomPolicy:
    """Uniform random joint targets inside the supplied limits (test policy)."""

    name = "random"
    needs_images = False

    def __init__(
        self,
        joint_limits: np.ndarray | None = None,
        *,
        seed: int = 0,
        chunk_size: int = 1,
        gripper_range: tuple[float, float] = (0.0, 0.04),
    ) -> None:
        if joint_limits is None:
            joint_limits = np.tile(np.array([-0.3, 0.3]), (7, 1))
        limits = np.asarray(joint_limits, dtype=np.float64)
        if limits.shape != (7, 2):
            raise PolicyError("joint_limits must be shape (7, 2)")
        self._limits = limits
        self._seed = int(seed)
        self._chunk_size = int(chunk_size)
        self._gripper_range = gripper_range
        if self._chunk_size < 1:
            raise PolicyError("chunk_size must be >= 1")
        self._rng = np.random.default_rng(self._seed)
        self.info: dict[str, Any] = {"name": self.name, "checkpoint_hash": DEFAULT_CHECKPOINT_HASH}

    def reset(self, task_instruction: str, seed: int) -> None:
        del task_instruction
        self._rng = np.random.default_rng([self._seed, int(seed)])

    def act(self, observation: PolicyObservation) -> ActionChunk:
        del observation
        low, high = self._limits[:, 0], self._limits[:, 1]
        joints = self._rng.uniform(low, high, size=(self._chunk_size, 7))
        gripper = self._rng.uniform(*self._gripper_range, size=(self._chunk_size, 1))
        return ActionChunk(np.concatenate([joints, gripper], axis=1))


class ScriptedExpertPolicy:
    """Replay the Task 4 expert's executed actions as the reference policy."""

    name = "scripted_expert"
    needs_images = False

    def __init__(
        self,
        *,
        backend_factory: Callable[[], Any] | None = None,
        backend: Any | None = None,
        config_dir: str | Path = "configs/sim",
        expert_config_dir: str | Path = "configs/expert",
        randomization_dir: str | Path = "configs/randomization",
        profile: str | None = None,
        chunk_size: int = 50,
        max_steps: int | None = None,
    ) -> None:
        if backend is None and backend_factory is None:
            backend_factory = lambda: create_backend("fake", config_dir)  # noqa: E731
        self._backend_factory = backend_factory
        self._backend = backend
        self._config_dir = str(config_dir)
        self._expert_config_dir = str(expert_config_dir)
        self._randomization_dir = str(randomization_dir)
        self._profile = profile
        self._chunk_size = int(chunk_size)
        self._max_steps = max_steps
        if self._chunk_size < 1:
            raise PolicyError("chunk_size must be >= 1")
        self._episode_index = 0
        self._actions: list[np.ndarray] = []
        self._cursor = 0
        self._context: dict[str, Any] = {}
        self.info: dict[str, Any] = {"name": self.name, "checkpoint_hash": DEFAULT_CHECKPOINT_HASH}

    @property
    def backend(self) -> Any:
        if self._backend is None:
            self._backend = self._backend_factory()
        return self._backend

    def set_episode_index(self, episode_index: int) -> None:
        self._episode_index = int(episode_index)

    def set_n_action_steps(self, n_action_steps: int) -> None:
        if int(n_action_steps) < 1:
            raise PolicyError("n_action_steps must be >= 1")
        self._chunk_size = int(n_action_steps)

    def set_reset_context(self, sample: Mapping[str, Any], observation: Any) -> None:
        self._context = {"sample": dict(sample), "observation": observation}

    def reset(self, task_instruction: str, seed: int) -> None:
        config = load_expert_config(self._expert_config_dir)
        backend = self.backend
        task = parse_instruction(task_instruction, backend.configs.scene, config.instructions_file)
        record = run_episode(
            backend,
            task,
            int(seed),
            instruction=task_instruction,
            episode_index=self._episode_index,
            run_id="policy_expert",
            randomization_profile=self._profile or config.default_profile,
            randomization_dir=self._randomization_dir,
            config=config,
            max_steps=self._max_steps,
        )
        self._actions = [np.asarray(step.action, dtype=np.float64).copy() for step in record.steps]
        self._cursor = 0
        self.info = {
            "name": self.name,
            "checkpoint_hash": DEFAULT_CHECKPOINT_HASH,
            "expert_outcome": int(record.outcome),
            "expert_success": bool(record.succeeded),
            "expert_steps": len(record.steps),
            "expert_failure_reason": record.failure_reason,
            "randomization_seed": derive_episode_seed(int(seed), self._episode_index),
        }

    def act(self, observation: PolicyObservation) -> ActionChunk:
        del observation
        if not self._actions:
            raise PolicyError("scripted expert produced no actions")
        batch = self._actions[self._cursor : self._cursor + self._chunk_size]
        self._cursor += len(batch)
        if not batch:
            # Keep holding the last commanded target; the episode will time out
            # or fail on its own terms rather than crash the run.
            batch = [self._actions[-1]]
        return ActionChunk(np.asarray(batch, dtype=np.float64))

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()
            self._backend = None


class NoisyExpertPolicy:
    """Expert actions plus deterministic noise and step dropout.

    The noise scale can be coupled to the sampled initial object x through
    ``set_reset_context``.  That makes the policy a controllable *proxy* for
    "harder initial states are harder", without pretending it is a trained VLA.
    """

    name = "noisy_expert"
    needs_images = False

    def __init__(
        self,
        inner: Policy,
        *,
        noise_sigma: float = 0.01,
        dropout: float = 0.05,
        correlation_gain: float = 1.0,
        seed: int = 0,
    ) -> None:
        if noise_sigma < 0:
            raise PolicyError("noise_sigma must be >= 0")
        if not 0 <= dropout <= 1:
            raise PolicyError("dropout must be in [0, 1]")
        self._inner = inner
        self._noise_sigma = float(noise_sigma)
        self._dropout = float(dropout)
        self._correlation_gain = float(correlation_gain)
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)
        self._last_action: np.ndarray | None = None
        self._initial_x: float | None = None
        self.info: dict[str, Any] = {
            "name": self.name,
            "checkpoint_hash": DEFAULT_CHECKPOINT_HASH,
            "inner": getattr(inner, "name", type(inner).__name__),
            "noise_sigma": self._noise_sigma,
            "dropout": self._dropout,
            "correlation_gain": self._correlation_gain,
        }

    def set_episode_index(self, episode_index: int) -> None:
        hook = getattr(self._inner, "set_episode_index", None)
        if hook is not None:
            hook(episode_index)

    def set_n_action_steps(self, n_action_steps: int) -> None:
        hook = getattr(self._inner, "set_n_action_steps", None)
        if hook is not None:
            hook(n_action_steps)

    def set_reset_context(self, sample: Mapping[str, Any], observation: Any) -> None:
        hook = getattr(self._inner, "set_reset_context", None)
        if hook is not None:
            hook(sample, observation)
        self._initial_x = None
        try:
            poses = sample.get("object_pose", {}).get("poses", {})
            if poses:
                first = next(iter(poses.values()))
                self._initial_x = float(first[0][3])
        except (AttributeError, KeyError, TypeError, ValueError):
            self._initial_x = None

    def reset(self, task_instruction: str, seed: int) -> None:
        self._inner.reset(task_instruction, seed)
        self._rng = np.random.default_rng([self._seed, int(seed)])
        self._last_action = None

    def _scale(self) -> float:
        scale = 1.0
        if self._initial_x is not None:
            scale += self._correlation_gain * abs(self._initial_x - 0.45) / 0.10
        return max(0.0, scale)

    def act(self, observation: PolicyObservation) -> ActionChunk:
        chunk = self._inner.act(observation)
        actions = chunk.actions.copy()
        scale = self._scale()
        noise = self._rng.normal(0.0, self._noise_sigma * scale, size=actions.shape)
        actions = actions + noise
        if self._dropout > 0:
            drops = self._rng.random(len(actions)) < self._dropout
            if self._last_action is not None:
                actions[drops] = self._last_action
            elif np.any(drops):
                actions[drops] = actions[0]
        self._last_action = actions[-1].copy()
        return ActionChunk(actions, chunk.step_period_s)

    def close(self) -> None:
        hook = getattr(self._inner, "close", None)
        if hook is not None:
            hook()


def _default_policy_python(repo_root: Path) -> str:
    configured = os.environ.get("SIM2REAL_POLICY_PYTHON")
    if configured:
        return configured
    ml_python = repo_root / ".venv-ml" / "bin" / "python"
    return str(ml_python) if ml_python.is_file() else sys.executable


class CheckpointPolicy:
    """Client for ``scripts/policy_server.py`` (guarded, no pickle)."""

    needs_images = True

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        server_python: str | None = None,
        server_script: str | Path | None = None,
        repo_root: str | Path | None = None,
        timeout_s: float = 30.0,
        fake_model: bool = False,
        deterministic_sampling: bool = True,
    ) -> None:
        root = Path(repo_root or Path(__file__).resolve().parents[3]).resolve()
        self._repo_root = root
        self._checkpoint = Path(checkpoint_path).expanduser().resolve()
        self._metadata_path = self._checkpoint / METADATA_NAME
        if not self._metadata_path.is_file():
            raise PolicyError(f"{self._metadata_path}: checkpoint metadata not found")
        self.metadata = json.loads(self._metadata_path.read_text(encoding="utf-8"))
        self.checkpoint_hash = file_sha256(self._metadata_path)
        self._server_python = server_python or _default_policy_python(root)
        self._server_script = Path(server_script or (root / "scripts" / "policy_server.py"))
        self._timeout_s = float(timeout_s)
        self._fake_model = bool(fake_model)
        self._deterministic_sampling = bool(deterministic_sampling)
        self._process: subprocess.Popen[bytes] | None = None
        self._ready = False
        self._stderr_lines: deque[str] = deque(maxlen=200)
        self._chunk_size = int(self.metadata.get("chunk_size", 50))
        self._state_dim = int(self.metadata.get("state_dim", 8))
        self._image_keys: list[str] = []
        self.name = "checkpoint"
        self.info: dict[str, Any] = {
            "name": self.name,
            "checkpoint": str(self._checkpoint),
            "checkpoint_hash": self.checkpoint_hash,
            "fake_model": self._fake_model,
            "deterministic_sampling": self._deterministic_sampling,
        }

    def _start(self) -> None:
        if not self._server_script.is_file():
            raise PolicyError(f"policy server script not found: {self._server_script}")
        command = [
            self._server_python,
            str(self._server_script),
            "--checkpoint",
            str(self._checkpoint),
            "--expected-hash",
            self.checkpoint_hash,
        ]
        if self._fake_model:
            command.append("--fake-model")
        environment = dict(os.environ)
        source = str(self._repo_root / "src")
        current = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = source if not current else f"{source}:{current}"
        try:
            self._process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                env=environment,
            )
        except OSError as exc:
            raise PolicyError(f"could not start policy server: {exc}") from exc
        thread = threading.Thread(
            target=self._drain_stderr, name="policy_server_stderr", daemon=True
        )
        thread.start()

    def _drain_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        for line in iter(process.stderr.readline, b""):
            self._stderr_lines.append(line.decode("utf-8", errors="replace").rstrip())
        process.stderr.close()

    def stderr_tail(self, lines: int = 20) -> str:
        return "\n".join(list(self._stderr_lines)[-lines:])

    def _alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _send(self, header: dict[str, Any], blobs: list[bytes] | None = None) -> None:
        if not self._alive() or self._process is None or self._process.stdin is None:
            raise PolicyError("policy server is not running")
        try:
            write_packet(self._process.stdin, header, blobs)
        except (BrokenPipeError, OSError) as exc:
            raise PolicyError(f"policy server pipe closed: {exc}") from exc

    def _recv(self, timeout_s: float | None = None) -> tuple[dict[str, Any], list[bytes]]:
        if not self._alive() or self._process is None or self._process.stdout is None:
            raise PolicyError("policy server died")
        timeout = self._timeout_s if timeout_s is None else timeout_s
        try:
            ready, _, _ = select.select([self._process.stdout], [], [], timeout)
        except (OSError, ValueError) as exc:
            raise PolicyError(f"policy server select failed: {exc}") from exc
        if not ready:
            raise PolicyError(f"policy server timed out after {timeout:.1f} s")
        try:
            return read_packet(self._process.stdout)
        except ProtocolError as exc:
            raise PolicyError(f"malformed policy-server frame: {exc}") from exc

    def _handshake(self) -> None:
        self._send(
            {
                "type": "hello",
                "protocol": PROTOCOL_VERSION,
                "checkpoint_hash": self.checkpoint_hash,
            }
        )
        header, _ = self._recv()
        if header.get("type") != "hello":
            raise PolicyError(f"expected hello, got {header.get('type')!r}")
        if header.get("protocol") != PROTOCOL_VERSION:
            raise PolicyError(
                f"policy protocol mismatch: server {header.get('protocol')} "
                f"!= client {PROTOCOL_VERSION}"
            )
        if not header.get("ok"):
            raise PolicyError(f"policy server refused the checkpoint: {header.get('reason')}")
        if header.get("checkpoint_hash") != self.checkpoint_hash:
            raise PolicyError(
                "policy server checkpoint hash mismatch: "
                f"{header.get('checkpoint_hash')} != {self.checkpoint_hash}"
            )
        if int(header.get("state_dim", -1)) != self._state_dim:
            raise PolicyError("policy server state dimension mismatch")
        self._chunk_size = int(header.get("chunk_size", self._chunk_size))
        self._image_keys = [str(key) for key in header.get("image_keys", [])]
        self.info.update(
            {
                "server_protocol": header.get("protocol"),
                "chunk_size": self._chunk_size,
                "fps": header.get("fps"),
                "image_keys": list(self._image_keys),
            }
        )
        ready, _ = self._recv()
        if ready.get("type") == "error":
            raise PolicyError(f"policy server model load failed: {ready.get('message')}")
        if ready.get("type") != "ready":
            raise PolicyError(f"expected ready, got {ready.get('type')!r}")

    def reset(self, task_instruction: str, seed: int) -> None:
        if self._process is not None and self._process.poll() is None and self._ready:
            reset: dict[str, Any] = {"type": "reset"}
            if self._deterministic_sampling:
                reset["sample_seed"] = int(seed)
            self._send(reset)
            reply, _ = self._recv()
            if reply.get("type") == "error":
                raise PolicyError(f"policy server reset failed: {reply.get('message')}")
            if not reply.get("ok"):
                raise PolicyError("policy server refused the reset")
            self.info["sample_seed"] = reply.get("sample_seed")
        else:
            if self._process is not None:
                self.close()
            self._start()
            self._handshake()
            self._ready = True
            if self._deterministic_sampling:
                self._send({"type": "reset", "sample_seed": int(seed)})
                reply, _ = self._recv()
                if reply.get("type") == "error":
                    raise PolicyError(f"policy server reset failed: {reply.get('message')}")
                if not reply.get("ok"):
                    raise PolicyError("policy server refused the reset")
                self.info["sample_seed"] = reply.get("sample_seed")
        self._task_instruction = task_instruction

    def act(self, observation: PolicyObservation) -> ActionChunk:
        missing = [key for key in self._image_keys if key not in observation.images]
        if missing:
            raise PolicyError(f"checkpoint policy requires images {missing}")
        header: dict[str, Any] = {
            "type": "act",
            "state": [float(value) for value in observation.state],
            "task": observation.task_instruction or getattr(self, "_task_instruction", ""),
            "stamp_ns": int(observation.stamp_ns),
            "image_keys": list(self._image_keys),
            "image_shapes": [list(observation.images[key].shape) for key in self._image_keys],
            "image_dtypes": [str(observation.images[key].dtype) for key in self._image_keys],
        }
        blobs = [
            np.ascontiguousarray(observation.images[key]).tobytes() for key in self._image_keys
        ]
        self._send(header, blobs)
        reply, reply_blobs = self._recv()
        if reply.get("type") == "error":
            raise PolicyError(f"policy server error: {reply.get('message')}")
        if reply.get("type") != "act":
            raise PolicyError(f"expected act reply, got {reply.get('type')!r}")
        shape = tuple(int(v) for v in reply.get("actions_shape", ()))
        dtype = np.dtype(str(reply.get("dtype", "float64")))
        if len(shape) != 2 or shape[1] != 8 or not reply_blobs:
            raise PolicyError(f"invalid action chunk shape {shape}")
        expected = int(np.prod(shape)) * dtype.itemsize
        if len(reply_blobs[0]) != expected:
            raise PolicyError(
                f"action chunk payload has {len(reply_blobs[0])} bytes, expected {expected}"
            )
        actions = np.frombuffer(reply_blobs[0], dtype=dtype).reshape(shape).astype(np.float64)
        return ActionChunk(actions, float(reply.get("step_period_s", POLICY_STEP_PERIOD_S)))

    def close(self) -> None:
        process = self._process
        self._process = None
        self._ready = False
        if process is None:
            return
        try:
            if process.poll() is None and process.stdin is not None:
                write_packet(process.stdin, {"type": "close"})
        except (BrokenPipeError, OSError, ProtocolError):
            pass
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:  # pragma: no cover - stubborn child
                process.kill()
                process.wait(timeout=2.0)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()


__all__ = [
    "ActionChunk",
    "CheckpointPolicy",
    "NoisyExpertPolicy",
    "POLICY_STEP_PERIOD_S",
    "Policy",
    "PolicyError",
    "PolicyObservation",
    "RandomPolicy",
    "ScriptedExpertPolicy",
    "ZeroPolicy",
    "_state_from_observation",
]
