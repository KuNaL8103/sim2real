"""CLAUDE.md status lines must stay machine-parseable across tasks."""

from __future__ import annotations

from pathlib import Path

import pytest

from sim2real.status import StatusParseError, parse_last_verified, parse_task1_status

ROOT = Path(__file__).resolve().parents[1]
CLAUDE_MD = ROOT / "CLAUDE.md"


def claude_text() -> str:
    return CLAUDE_MD.read_text(encoding="utf-8")


def test_last_verified_line_is_parseable():
    status = parse_last_verified(claude_text())
    assert status.passed >= 0
    assert status.failed >= 0
    assert status.skipped >= 0
    assert status.errors >= 0
    assert len(status.date) == 10
    assert status.base_commit == "none" or 7 <= len(status.base_commit) <= 40


def test_task1_status_line_is_parseable():
    status = parse_task1_status(claude_text())
    if status.complete:
        assert status.pending == ()
    else:
        assert status.pending, "pending checks list must not be empty"


def test_parser_rejects_missing_lines():
    with pytest.raises(StatusParseError):
        parse_last_verified("no status here")
    with pytest.raises(StatusParseError):
        parse_task1_status("no status here")


def test_parser_accepts_complete_marker():
    status = parse_task1_status("- Task 1 status: complete")
    assert status.complete is True
    assert status.pending == ()


def test_parser_splits_pending_checks():
    status = parse_task1_status(
        "- Task 1 status: complete with pending checks: ROS 2 build, CI run"
    )
    assert status.complete is False
    assert status.pending == ("ROS 2 build", "CI run")
