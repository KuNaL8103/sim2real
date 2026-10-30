"""Machine-readable parsing of the status lines in ``CLAUDE.md``.

Later tasks read the verified test counts from here instead of re-parsing
prose. Keep the two line formats in ``CLAUDE.md`` exactly as documented.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

LAST_VERIFIED_RE = re.compile(
    r"^- Last verified test run: passed=(?P<passed>\d+) failed=(?P<failed>\d+) "
    r"skipped=(?P<skipped>\d+) errors=(?P<errors>\d+) \| date=(?P<date>\d{4}-\d{2}-\d{2}) "
    r"\| base_commit=(?P<base_commit>[0-9a-fA-F]{7,40}|none)$",
    re.MULTILINE,
)

TASK1_STATUS_RE = re.compile(
    r"^- Task 1 status: (?P<status>complete|complete with pending checks: (?P<pending>.+))$",
    re.MULTILINE,
)


class StatusParseError(ValueError):
    """Raised when ``CLAUDE.md`` does not contain a parseable status line."""


@dataclass(frozen=True)
class TestRunStatus:
    passed: int
    failed: int
    skipped: int
    errors: int
    date: str
    base_commit: str


@dataclass(frozen=True)
class TaskStatus:
    complete: bool
    pending: tuple[str, ...]


def parse_last_verified(text: str) -> TestRunStatus:
    """Parse the ``Last verified test run`` line from CLAUDE.md text."""
    match = LAST_VERIFIED_RE.search(text)
    if match is None:
        raise StatusParseError("CLAUDE.md is missing a parseable 'Last verified test run' line")
    return TestRunStatus(
        passed=int(match.group("passed")),
        failed=int(match.group("failed")),
        skipped=int(match.group("skipped")),
        errors=int(match.group("errors")),
        date=match.group("date"),
        base_commit=match.group("base_commit"),
    )


def parse_task1_status(text: str) -> TaskStatus:
    """Parse the ``Task 1 status`` line from CLAUDE.md text."""
    match = TASK1_STATUS_RE.search(text)
    if match is None:
        raise StatusParseError("CLAUDE.md is missing a parseable 'Task 1 status' line")
    status = match.group("status")
    if status == "complete":
        return TaskStatus(complete=True, pending=())
    pending = tuple(
        item.strip() for item in (match.group("pending") or "").split(",") if item.strip()
    )
    return TaskStatus(complete=False, pending=pending)
