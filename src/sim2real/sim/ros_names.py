"""Pure naming helpers shared by ROS adapters and tests."""

from __future__ import annotations

from sim2real import contracts


def camera_topics(camera: str) -> tuple[str, str, str]:
    """Return ``(color, depth, camera_info)`` topics for a configured camera."""
    if camera not in contracts.CAMERAS:
        raise ValueError(f"unknown camera {camera!r}; expected one of {contracts.CAMERAS}")
    return (
        f"/camera/{camera}/color/image_raw",
        f"/camera/{camera}/depth/image_raw",
        f"/camera/{camera}/camera_info",
    )
