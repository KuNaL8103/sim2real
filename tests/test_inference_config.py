"""Inference config: strict types, bounds, unknown keys, and the shipped YAML."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sim2real.inference.config import (
    InferenceConfig,
    InferenceConfigError,
    load_inference_config,
)
from sim2real.inference.server_policy import resolve_server_python

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "inference" / "default.yaml"


def test_shipped_default_config_loads():
    config = load_inference_config(CONFIG)
    assert config.policy_backend == "fake"
    assert config.policy_period_s == pytest.approx(0.1)
    assert config.n_action_steps == 10
    assert config.latency_compensation is True
    assert config.max_restarts == 3
    assert config.sample_seed == -1
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert raw["schema_version"] == 1
    assert raw["inference"]["policy_backend"] == config.policy_backend


def test_shipped_startup_timeout_matches_the_class_default():
    """The YAML and the dataclass default must not drift apart."""
    config = load_inference_config(CONFIG)
    assert config.startup_timeout_s == pytest.approx(180.0)
    assert config.startup_timeout_s == pytest.approx(InferenceConfig.startup_timeout_s)
    # The validator still bounds the value; the default is not a special case.
    with pytest.raises(InferenceConfigError, match="startup_timeout_s"):
        InferenceConfig.from_values({"startup_timeout_s": 0.0})


def test_unknown_top_level_and_inference_keys_are_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("schema_version: 1\ninference: {}\nextra: 1\n", encoding="utf-8")
    with pytest.raises(InferenceConfigError, match="unknown keys"):
        load_inference_config(path)

    path.write_text("schema_version: 1\ninference:\n  policy_period: 0.1\n", encoding="utf-8")
    with pytest.raises(InferenceConfigError, match="unknown keys"):
        load_inference_config(path)


@pytest.mark.parametrize(
    "values, match",
    [
        ({"policy_backend": "torch"}, "policy_backend"),
        ({"policy_backend": "server"}, "checkpoint_path"),
        ({"policy_period_s": 0.0}, "policy_period_s"),
        ({"n_action_steps": 0}, "n_action_steps"),
        ({"n_action_steps": 101, "max_chunk_length": 100}, "max_chunk_length"),
        ({"max_restarts": -1}, "max_restarts"),
        ({"max_camera_skew_s": -0.1}, "max_camera_skew_s"),
        ({"sample_seed": -2}, "sample_seed"),
        ({"latency_compensation": "yes"}, "latency_compensation"),
        ({"status_rate_hz": 0.0}, "status_rate_hz"),
        ({"startup_timeout_s": True}, "startup_timeout_s"),
    ],
)
def test_invalid_values_are_rejected(values, match):
    with pytest.raises(InferenceConfigError, match=match):
        InferenceConfig.from_values(values)


def test_server_backend_accepts_a_checkpoint():
    config = InferenceConfig.from_values(
        {"policy_backend": "server", "checkpoint_path": "outputs/some/checkpoint"}
    )
    assert config.policy_backend == "server"
    assert config.checkpoint_path == "outputs/some/checkpoint"


def test_server_python_keeps_the_venv_symlink(tmp_path):
    """A uv venv's bin/python is a symlink; resolving it loses the venv."""
    root = tmp_path
    target = root / "python-real"
    target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    target.chmod(0o755)
    bin_dir = root / ".venv-ml" / "bin"
    bin_dir.mkdir(parents=True)
    interpreter = bin_dir / "python"
    interpreter.symlink_to(target)
    resolved = resolve_server_python(InferenceConfig(), root)
    assert resolved == str(interpreter.absolute())
    assert resolved != str(target.resolve())
