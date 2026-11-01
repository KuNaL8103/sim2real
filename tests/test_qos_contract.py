"""QoS constants must match the table in docs/ARCHITECTURE.md."""

from __future__ import annotations

import re
from pathlib import Path

from sim2real import contracts
from sim2real.sim.qos import (
    CAMERA_QOS,
    FILTER_QOS,
    JOINT_STATES_QOS,
    MAP_QOS,
    TF_QOS,
    TF_STATIC_QOS,
)

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE = ROOT / "docs" / "ARCHITECTURE.md"


def _qos_cell(topic: str) -> str:
    text = ARCHITECTURE.read_text(encoding="utf-8")
    pattern = re.compile(rf"^\| `{re.escape(topic)}` \|.*\|$", re.MULTILINE)
    rows = pattern.findall(text)
    assert rows, f"no contract row for {topic}"
    cells = [cell.strip() for cell in rows[0].strip().strip("|").split("|")]
    # name, type, producer, consumers, rate, qos, safety note
    return cells[5]


def test_joint_state_qos_matches_docs():
    cell = _qos_cell(contracts.JOINT_STATES_TOPIC)
    assert "reliable" in cell and "depth 10" in cell
    assert JOINT_STATES_QOS.reliability == "reliable"
    assert JOINT_STATES_QOS.depth == 10


def test_tf_qos_matches_docs():
    cell = _qos_cell(contracts.TF_TOPIC)
    assert "reliable" in cell and "depth 10" in cell
    assert TF_QOS.reliability == "reliable"
    assert TF_QOS.depth == 10


def test_static_tf_qos_matches_docs():
    cell = _qos_cell(contracts.TF_STATIC_TOPIC)
    assert "reliable" in cell and "transient-local" in cell and "depth 1" in cell
    assert TF_STATIC_QOS.durability == "transient_local"
    assert TF_STATIC_QOS.depth == 1


def test_camera_qos_matches_docs_for_every_camera_topic():
    for camera in contracts.CAMERAS:
        for suffix in ("color/image_raw", "depth/image_raw", "camera_info"):
            cell = _qos_cell(f"/camera/{camera}/{suffix}")
            assert "reliable" in cell and "keep-last 5" in cell
    assert CAMERA_QOS.reliability == "reliable"
    assert CAMERA_QOS.depth == 5


def test_filter_qos_matches_docs():
    for topic in (contracts.FILTERED_EPISODE_TOPIC, contracts.FILTER_METRICS_TOPIC):
        cell = _qos_cell(topic)
        assert "reliable" in cell and "keep-last 50" in cell
    assert FILTER_QOS.reliability == "reliable"
    assert FILTER_QOS.depth == 50


def test_mapping_qos_matches_docs():
    for topic in (
        contracts.WORKSPACE_POINTCLOUD_TOPIC,
        contracts.WORKSPACE_TSDF_MESH_TOPIC,
    ):
        cell = _qos_cell(topic)
        assert "reliable" in cell and "keep-last 1" in cell
    assert MAP_QOS.reliability == "reliable"
    assert MAP_QOS.depth == 1


def test_reset_map_service_is_documented():
    text = ARCHITECTURE.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"^\| `{re.escape(contracts.WORKSPACE_RESET_MAP_SERVICE)}` \|.*\|$",
        re.MULTILINE,
    )
    rows = pattern.findall(text)
    assert rows
    cells = [cell.strip().strip("`") for cell in rows[0].strip().strip("|").split("|")]
    assert cells[1] == contracts.TRIGGER_TYPE
    assert "reliable" in cells[5]
