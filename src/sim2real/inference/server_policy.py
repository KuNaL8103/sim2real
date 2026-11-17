"""Policy factories for the inference node.

The server backend reuses the verified ``CheckpointPolicy`` client; the fake
backend uses :class:`~sim2real.inference.fake_policy.FakeChunkPolicy`. Nothing
here imports torch: the ML environment is reached only through the child
process that ``scripts/policy_server.py`` starts.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sim2real.evaluation.policy_interface import CheckpointPolicy, PolicyError
from sim2real.inference.config import InferenceConfig
from sim2real.inference.fake_policy import FakeChunkPolicy

PolicyFactory = Callable[[], Any]


def resolve_repo_root(start: str | Path | None = None) -> Path:
    """Find the checkout that contains ``src/sim2real``; never assume a path."""
    candidate = Path(start).resolve() if start is not None else Path(__file__).resolve()
    for parent in (candidate, *candidate.parents):
        if (parent / "src" / "sim2real" / "__init__.py").is_file():
            return parent
    raise PolicyError("could not locate the repository root containing src/sim2real")


def resolve_server_python(config: InferenceConfig, repo_root: Path) -> str:
    """Resolve and check the interpreter that will run the policy server.

    Never call ``Path.resolve()`` on the interpreter: a uv venv's
    ``bin/python`` is usually a symlink to the system interpreter, and following
    it discards ``pyvenv.cfg`` (the child then runs without LeRobot/torch).
    ``absolute()`` keeps the venv path while still making it explicit.
    """
    if config.server_python:
        path = Path(config.server_python).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).absolute()
    else:
        configured = os.environ.get("SIM2REAL_POLICY_PYTHON", "").strip()
        if configured:
            path = Path(configured).expanduser()
            if not path.is_absolute():
                path = (Path.cwd() / path).absolute()
        else:
            path = (repo_root / ".venv-ml" / "bin" / "python").absolute()
    if not path.is_file():
        raise PolicyError(
            f"policy server interpreter not found: {path} (set server_python or "
            "SIM2REAL_POLICY_PYTHON)"
        )
    if not os.access(path, os.X_OK):
        raise PolicyError(f"policy server interpreter is not executable: {path}")
    return str(path)


def resolve_server_script(config: InferenceConfig, repo_root: Path) -> Path:
    if config.server_script:
        path = Path(config.server_script).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
    else:
        path = (repo_root / "scripts" / "policy_server.py").resolve()
    if not path.is_file():
        raise PolicyError(f"policy server script not found: {path}")
    return path


def make_server_policy_factory(
    config: InferenceConfig, repo_root: Path | None = None
) -> PolicyFactory:
    """Return a factory that resolves and checks paths when the worker calls it.

    Deferring the checks is deliberate: a missing interpreter or checkpoint must
    leave the node alive in ``MODEL_UNAVAILABLE``/``FAILED``, not abort the
    process during construction.
    """
    root = resolve_repo_root(repo_root)
    deterministic = config.sample_seed >= 0

    def factory() -> CheckpointPolicy:
        server_python = resolve_server_python(config, root)
        server_script = resolve_server_script(config, root)
        checkpoint = Path(config.checkpoint_path).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = (Path.cwd() / checkpoint).resolve()
        metadata = checkpoint / "checkpoint_metadata.json"
        if not metadata.is_file():
            raise PolicyError(f"{metadata}: checkpoint metadata not found")
        return CheckpointPolicy(
            checkpoint,
            server_python=server_python,
            server_script=server_script,
            repo_root=root,
            timeout_s=config.call_timeout_s,
            startup_timeout_s=config.startup_timeout_s,
            fake_model=False,
            deterministic_sampling=deterministic,
        )

    return factory


def make_fake_policy_factory(config: InferenceConfig) -> PolicyFactory:
    chunk_size = max(config.n_action_steps, min(50, config.max_chunk_length))
    chunk_size = min(chunk_size, config.max_chunk_length)

    def factory() -> FakeChunkPolicy:
        return FakeChunkPolicy(
            chunk_size=chunk_size,
            step_period_s=config.policy_period_s,
            mode="hold",
        )

    return factory


def make_policy_factory(config: InferenceConfig, repo_root: Path | None = None) -> PolicyFactory:
    if config.policy_backend == "server":
        return make_server_policy_factory(config, repo_root)
    if config.policy_backend == "fake":
        return make_fake_policy_factory(config)
    raise PolicyError(f"unknown policy_backend {config.policy_backend!r}")


__all__ = [
    "PolicyFactory",
    "make_fake_policy_factory",
    "make_policy_factory",
    "make_server_policy_factory",
    "resolve_repo_root",
    "resolve_server_python",
    "resolve_server_script",
]
