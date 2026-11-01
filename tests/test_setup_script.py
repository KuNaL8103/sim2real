"""The environment setup script must reproduce the .venv-ros MuJoCo install."""

from __future__ import annotations

from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "setup_envs.sh"


def test_setup_script_installs_mujoco_into_venv_ros():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "mujoco==3.17.0" in text
    assert "--python .venv-ros/bin/python --no-deps" in text
    for package in (
        "glfw==2.11.0",
        "pyopengl==3.1.10",
        "absl-py==2.5.1",
        "etils==1.14.0",
        "fsspec==2026.9.0",
        "typing-extensions==4.16.0",
        "websockets==17.4",
        "zipp==4.1.2",
    ):
        assert package in text


def test_setup_script_syncs_the_sim_group():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "uv sync --group sim" in text


def test_setup_script_installs_numpy126_compatible_scipy_and_pyarrow():
    text = SCRIPT.read_text(encoding="utf-8")
    assert '"scipy==1.16.3"' in text
    assert '"pyarrow==21.0.0"' in text
    assert "--no-deps" in text
    assert "import scipy, pyarrow" in text
