"""Seed-ledger enforcement for the orchestrator.

The ranges below are the tracked ledger in ``docs/EVALUATION.md``.  The
orchestrator refuses any *drawn* seed that falls inside an already-owned range,
so a pipeline cycle cannot silently reuse a tuning, hold-out, or diagnostic set.

Two carve-outs are inside the 20000-20999 evaluation range:

* 20900-20919: Safety / obstacle-mode expert false-positive check (already used);
* 20950-20969: fake-backend end-to-end tests only (recorded in the ledger doc).

The 20950-20969 carve-out is accepted only when the configured backend is
``fake``.  A MuJoCo run must pick a different range.
"""

from __future__ import annotations

from dataclasses import dataclass

EVALUATION_RANGE = (20000, 20999)
TARGETED_RANGE = (30000, 30999)
PIPELINE_FAKE_E2E_RANGE = (20950, 20969)
TASK11_RANGE = (20900, 20919)


class SeedLedgerError(ValueError):
    """Raised when a drawn seed overlaps an owned or reserved range."""


@dataclass(frozen=True)
class LedgerRange:
    low: int
    high: int
    owner: str


#: Ranges that must never be reused for a new choice.
USED_RANGES: tuple[LedgerRange, ...] = (
    LedgerRange(1000, 1099, "filter tuning"),
    LedgerRange(2000, 2059, "reserved"),
    LedgerRange(3000, 3039, "reserved"),
    LedgerRange(4000, 4129, "reserved"),
    LedgerRange(6000, 6059, "filter hold-out"),
    LedgerRange(7000, 7002, "replay prototype"),
    LedgerRange(8000, 8059, "reserved"),
    LedgerRange(10000, 10199, "training/evaluation diagnosis"),
    LedgerRange(20000, 20999, "closed-loop evaluation"),
    LedgerRange(20900, 20919, "safety expert false-positive check"),
    LedgerRange(30000, 30999, "targeted generation"),
)


def _in(seed: int, low: int, high: int) -> bool:
    return low <= seed <= high


def check_generation_seeds(seeds: list[int], *, backend: str) -> None:
    """Refuse drawn generation seeds that overlap an owned range."""
    fake_e2e = PIPELINE_FAKE_E2E_RANGE
    for seed in seeds:
        if _in(seed, *fake_e2e):
            if backend == "fake":
                continue
            raise SeedLedgerError(
                f"seed {seed} is in the fake-backend end-to-end carve-out "
                f"{fake_e2e[0]}-{fake_e2e[1]}; it may not be used with backend={backend!r}"
            )
        for owned in USED_RANGES:
            if _in(seed, owned.low, owned.high):
                raise SeedLedgerError(
                    f"seed {seed} overlaps the {owned.low}-{owned.high} range owned by "
                    f"{owned.owner!r}; choose an unused range"
                )


def check_evaluation_seed(seed: int, *, backend: str) -> None:
    """Evaluation base seeds must stay in 20000-20999."""
    if not _in(seed, *EVALUATION_RANGE):
        raise SeedLedgerError(
            f"evaluation seed {seed} is outside the owned range "
            f"{EVALUATION_RANGE[0]}-{EVALUATION_RANGE[1]}"
        )
    if _in(seed, *PIPELINE_FAKE_E2E_RANGE) and backend != "fake":
        raise SeedLedgerError(
            f"evaluation seed {seed} is reserved for fake-backend end-to-end tests"
        )


def check_targeted_seeds(seeds: list[int]) -> None:
    """Targeted-generation seeds must stay in 30000-30999."""
    for seed in seeds:
        if not _in(seed, *TARGETED_RANGE):
            raise SeedLedgerError(
                f"targeted seed {seed} is outside the owned range "
                f"{TARGETED_RANGE[0]}-{TARGETED_RANGE[1]}"
            )


def describe_ledger() -> dict[str, object]:
    return {
        "evaluation_range": list(EVALUATION_RANGE),
        "targeted_range": list(TARGETED_RANGE),
        "pipeline_fake_e2e": list(PIPELINE_FAKE_E2E_RANGE),
        "task11_range": list(TASK11_RANGE),
        "used_ranges": [
            {"low": item.low, "high": item.high, "owner": item.owner} for item in USED_RANGES
        ],
    }


__all__ = [
    "EVALUATION_RANGE",
    "PIPELINE_FAKE_E2E_RANGE",
    "SeedLedgerError",
    "TARGETED_RANGE",
    "TASK11_RANGE",
    "USED_RANGES",
    "check_evaluation_seed",
    "check_generation_seeds",
    "check_targeted_seeds",
    "describe_ledger",
]
