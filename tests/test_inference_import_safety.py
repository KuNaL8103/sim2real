"""The inference core and the ROS node must not import torch."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NODE = ROOT / "ros2_ws" / "src" / "sim2real_vla" / "sim2real_vla" / "vla_inference_node.py"


def _run(code: str) -> subprocess.CompletedProcess[str]:
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(ROOT / "src"),
            str(ROOT / "ros2_ws" / "src"),
            str(ROOT / "ros2_ws" / "src" / "sim2real_vla"),
            env.get("PYTHONPATH", ""),
        ]
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def test_core_inference_import_does_not_load_torch():
    result = _run(
        "import sys; import sim2real.inference; "
        "import sim2real.inference.async_worker; "
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if m == 'torch')"
    )
    assert result.returncode == 0, result.stderr


def test_node_source_never_imports_torch_or_cv_bridge():
    source = NODE.read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "from torch" not in source
    assert "cv_bridge" not in source


def test_node_module_import_does_not_load_torch_when_ros_is_available():
    result = _run(
        "import sys\n"
        "try:\n"
        "    import rclpy  # noqa: F401\n"
        "except ImportError:\n"
        "    print('SKIP: rclpy is not importable')\n"
        "    raise SystemExit(0)\n"
        "import sim2real_vla.vla_inference_node  # noqa: F401\n"
        "assert 'torch' not in sys.modules\n"
    )
    assert result.returncode == 0, result.stderr
