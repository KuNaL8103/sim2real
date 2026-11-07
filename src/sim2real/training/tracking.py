"""Optional offline experiment tracking.

The default is ``none``.  Importing the trainer (or this module) never imports
``wandb`` or ``mlflow``: those imports happen only inside :func:`make_tracker`
when the configured backend is explicitly requested.  Tests use ``none`` and
make no network calls.

Only offline modes are approved.  A non-offline request is rejected before any
library import.
"""

from __future__ import annotations

from typing import Any, Protocol

BACKENDS = ("none", "wandb", "mlflow")


class TrackingError(ValueError):
    """Raised for an unknown backend or a non-offline request."""


class Tracker(Protocol):
    def log_metrics(self, step: int, metrics: dict[str, Any]) -> None: ...

    def log_config(self, config: dict[str, Any]) -> None: ...

    def close(self) -> None: ...


class NullTracker:
    """A no-op tracker; no optional library is imported and no I/O happens."""

    def log_metrics(self, step: int, metrics: dict[str, Any]) -> None:
        return None

    def log_config(self, config: dict[str, Any]) -> None:
        return None

    def close(self) -> None:
        return None


def make_tracker(
    backend: str,
    *,
    offline: bool = True,
    project: str = "sim2real",
    run_name: str | None = None,
    output_dir: str | None = None,
) -> Tracker:
    if backend not in BACKENDS:
        raise TrackingError(f"tracking backend: expected one of {list(BACKENDS)}")
    if backend == "none":
        return NullTracker()
    if not offline:
        raise TrackingError("only offline tracking is approved in Task 8")
    if backend == "wandb":
        import wandb  # imported lazily; never at module import time

        wandb.init(
            project=project,
            name=run_name,
            mode="offline",
            dir=output_dir,
            config=None,
        )
        return _WandbTracker(wandb)
    import mlflow  # imported lazily; never at module import time

    if output_dir:
        mlflow.set_tracking_uri(f"file:{output_dir}")
    mlflow.start_run(run_name=run_name)
    return _MlflowTracker(mlflow)


class _WandbTracker:
    def __init__(self, module: Any) -> None:
        self._module = module

    def log_metrics(self, step: int, metrics: dict[str, Any]) -> None:
        self._module.log({**metrics, "step": step})

    def log_config(self, config: dict[str, Any]) -> None:
        self._module.config.update(config)

    def close(self) -> None:
        self._module.finish()


class _MlflowTracker:
    def __init__(self, module: Any) -> None:
        self._module = module

    def log_metrics(self, step: int, metrics: dict[str, Any]) -> None:
        numeric = {
            key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))
        }
        if numeric:
            self._module.log_metrics(numeric, step=step)

    def log_config(self, config: dict[str, Any]) -> None:
        self._module.log_params({key: str(value) for key, value in config.items()})

    def close(self) -> None:
        self._module.end_run()


__all__ = ["BACKENDS", "NullTracker", "Tracker", "TrackingError", "make_tracker"]
