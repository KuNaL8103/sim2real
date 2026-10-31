"""Config validation fails loudly and specifically."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from sim2real.sim.backend import ConfigError
from sim2real.sim.config import load_sim_configs

CONFIG_DIR = Path("configs/sim")


def _copy_valid(tmp_path: Path) -> Path:
    for name in ("scene.yaml", "panda.yaml", "cameras.yaml"):
        shutil.copy(CONFIG_DIR / name, tmp_path / name)
    return tmp_path


def test_valid_configs_load():
    configs = load_sim_configs(CONFIG_DIR)
    assert configs.scene.control_rate_hz == 20.0
    assert configs.scene.physics_dt == 0.002
    assert set(configs.cameras) == {"overhead", "wrist"}
    assert configs.panda.model.is_file()


def test_missing_file_is_named(tmp_path):
    _copy_valid(tmp_path)
    (tmp_path / "cameras.yaml").unlink()
    with pytest.raises(ConfigError, match="cameras.yaml"):
        load_sim_configs(tmp_path)


def test_malformed_yaml_is_reported(tmp_path):
    _copy_valid(tmp_path)
    (tmp_path / "scene.yaml").write_text("schema_version: 1\n  bad: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_sim_configs(tmp_path)


def test_missing_required_key_is_reported(tmp_path):
    _copy_valid(tmp_path)
    data = yaml.safe_load((tmp_path / "scene.yaml").read_text(encoding="utf-8"))
    del data["cubes"]
    (tmp_path / "scene.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="cubes"):
        load_sim_configs(tmp_path)


def test_unordered_joint_limits_are_rejected(tmp_path):
    _copy_valid(tmp_path)
    data = yaml.safe_load((tmp_path / "panda.yaml").read_text(encoding="utf-8"))
    data["joint_limits"][0] = [1.0, -1.0]
    (tmp_path / "panda.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="lower < upper"):
        load_sim_configs(tmp_path)


def test_bad_camera_resolution_is_rejected(tmp_path):
    _copy_valid(tmp_path)
    data = yaml.safe_load((tmp_path / "cameras.yaml").read_text(encoding="utf-8"))
    data["cameras"]["overhead"]["resolution"] = [0, 256]
    (tmp_path / "cameras.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="resolution must be positive"):
        load_sim_configs(tmp_path)
