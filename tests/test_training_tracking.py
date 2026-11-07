"""Tracking is disabled by default and imports nothing optional."""

from __future__ import annotations

import sys

import pytest

from sim2real.training.config import ConfigError, TrackingConfig
from sim2real.training.tracking import NullTracker, TrackingError, make_tracker


def test_none_backend_does_not_import_wandb_or_mlflow(monkeypatch):
    for name in ("wandb", "mlflow"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    tracker = make_tracker("none")
    assert isinstance(tracker, NullTracker)
    assert "wandb" not in sys.modules
    assert "mlflow" not in sys.modules
    tracker.log_metrics(1, {"loss": 0.5})
    tracker.log_config({"seed": 0})
    tracker.close()


def test_unknown_backend_rejected():
    with pytest.raises(TrackingError, match="expected one of"):
        make_tracker("tensorboard")


def test_online_tracking_is_rejected_before_import():
    with pytest.raises(TrackingError, match="offline"):
        make_tracker("wandb", offline=False)


def test_tracking_config_validation():
    TrackingConfig(backend="none", offline=True)
    with pytest.raises(ConfigError, match="tracking.backend"):
        TrackingConfig(backend="tensorboard")
    with pytest.raises(ConfigError, match="tracking.offline"):
        TrackingConfig(backend="mlflow", offline=False)
