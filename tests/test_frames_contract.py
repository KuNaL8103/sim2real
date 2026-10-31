"""Frame/topic names and a cycle-free TF tree rooted at ``world``."""

from __future__ import annotations

from pathlib import Path

from sim2real import contracts
from sim2real.sim.config import load_sim_configs
from sim2real.sim.ros_names import camera_topics

CONFIG_DIR = Path("configs/sim")


def test_camera_topics_match_contracts():
    for camera in contracts.CAMERAS:
        color, depth, info = camera_topics(camera)
        assert contracts.TOPICS[color] == contracts.IMAGE_TYPE
        assert contracts.TOPICS[depth] == contracts.IMAGE_TYPE
        assert contracts.TOPICS[info] == contracts.CAMERA_INFO_TYPE


def test_frame_tree_has_no_cycles_and_reaches_world():
    configs = load_sim_configs(CONFIG_DIR)
    parents: dict[str, str] = {contracts.WORLD_FRAME: None}  # type: ignore[dict-item]
    parents[contracts.PANDA_BASE_FRAME] = contracts.WORLD_FRAME
    previous = contracts.PANDA_BASE_FRAME
    for ros_name, _ in configs.panda.body_frames[1:]:
        parents[ros_name] = previous
        previous = ros_name
    for camera in configs.cameras.values():
        parents[camera.frame_id] = camera.parent_frame

    for frame in parents:
        seen = set()
        cursor: str | None = frame
        while cursor is not None:
            assert cursor not in seen, f"cycle in TF tree at {cursor}"
            seen.add(cursor)
            cursor = parents.get(cursor)
        assert contracts.WORLD_FRAME in seen, f"{frame} does not reach world"

    assert set(contracts.FRAMES).issubset(parents)


def test_contract_frames_are_used_by_configs():
    configs = load_sim_configs(CONFIG_DIR)
    ros_frames = [ros for ros, _ in configs.panda.body_frames]
    assert contracts.PANDA_BASE_FRAME in ros_frames
    assert contracts.PANDA_HAND_FRAME in ros_frames
    for camera in configs.cameras.values():
        assert camera.frame_id in contracts.FRAMES
