"""Isaac Sim adapter must fail closed."""

from __future__ import annotations

import importlib.util

import pytest

from sim2real.sim.backend import BackendUnavailableError
from sim2real.sim.isaac_backend import IsaacBackend


def test_is_available_false_without_isaacsim():
    if importlib.util.find_spec("isaacsim") is not None:  # pragma: no cover
        pytest.skip("isaacsim is installed in this environment; guard test not meaningful")
    available, reason = IsaacBackend.is_available()
    assert available is False
    assert "not installed" in reason.lower()
    assert "UNAVAILABLE" in reason


def test_construction_raises_with_clear_message():
    with pytest.raises(BackendUnavailableError, match="Isaac Sim is not installed"):
        IsaacBackend("configs/sim")
