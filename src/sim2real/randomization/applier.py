"""Backend-aware application of a randomization sample.

The applier is the only place that decides whether a parameter group was really
applied. It first gates each group on ``BackendCapabilities`` and then merges the
backend's own per-group result. A group is reported ``APPLIED`` only when the
backend method actually executed and returned ``APPLIED`` for it; otherwise the
report says ``SKIPPED`` or ``PARTIAL`` with a reason.

Isaac Sim is deliberately refused: its capabilities are unverified, so claiming
an application there would be dishonest.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from sim2real.randomization.sampler import GROUP_ORDER, RandomizationSample
from sim2real.sim.backend import BackendCapabilities, SimBackend

APPLIED = "APPLIED"
PARTIAL = "PARTIAL"
SKIPPED = "SKIPPED"
_VALID_STATUSES = {APPLIED, PARTIAL, SKIPPED}

# Which capability flags a parameter group depends on. ``physics`` needs all
# three; a backend that supports only some of them yields PARTIAL.
GROUP_CAPABILITIES: dict[str, tuple[str, ...]] = {
    "object_pose": ("object_pose",),
    "texture": ("texture",),
    "lighting": ("lighting",),
    "camera": ("camera_extrinsics",),
    "physics": ("physics_mass", "physics_friction", "joint_damping"),
    "distractors": ("distractors",),
}


class RandomizationUnsupportedError(RuntimeError):
    """Raised when a backend must not be used for randomization."""


@dataclass(frozen=True)
class GroupStatus:
    status: str
    reason: str = ""

    def __post_init__(self) -> None:
        if self.status not in _VALID_STATUSES:
            raise ValueError(f"invalid randomization status {self.status!r}")

    def to_dict(self) -> dict[str, str]:
        return {"status": self.status, "reason": self.reason}


@dataclass(frozen=True)
class AppliedRandomization:
    groups: Mapping[str, GroupStatus] = field(default_factory=dict)
    nominal: bool = False

    def applied_groups(self) -> tuple[str, ...]:
        return tuple(name for name, result in self.groups.items() if result.status == APPLIED)

    def skipped_groups(self) -> tuple[str, ...]:
        return tuple(name for name, result in self.groups.items() if result.status == SKIPPED)

    def to_dict(self) -> dict[str, object]:
        return {
            "nominal": bool(self.nominal),
            "groups": {name: result.to_dict() for name, result in self.groups.items()},
        }


def _capability_status(capabilities: BackendCapabilities, group: str) -> GroupStatus | None:
    flags = GROUP_CAPABILITIES[group]
    supported = [flag for flag in flags if bool(getattr(capabilities, flag, False))]
    if not supported:
        reasons = "; ".join(capabilities.reasons.get(flag, "unsupported") for flag in flags)
        return GroupStatus(SKIPPED, reasons)
    missing = [flag for flag in flags if flag not in supported]
    if missing:
        return GroupStatus(PARTIAL, "missing capability: " + ", ".join(missing))
    return None


def apply_randomization(
    backend: SimBackend,
    sample: RandomizationSample,
    *,
    only_groups: Iterable[str] | None = None,
) -> AppliedRandomization:
    """Apply ``sample`` and return an honest per-group report.

    Args:
        backend: Backend whose model parameters should change.
        sample: Pure sample from :func:`sim2real.randomization.sampler.sample`.
        only_groups: Restrict the report/application to these groups. Used for
            an explicit object-pose override.

    Raises:
        RandomizationUnsupportedError: For the guarded Isaac adapter.
        ValueError: If ``only_groups`` contains an unknown group name.
    """
    if backend.name == "isaac":
        raise RandomizationUnsupportedError(
            "Isaac randomization is unverified: Isaac Sim is not installed and no "
            "behaviour is tested. Refusing to report a randomization as applied."
        )

    groups = tuple(only_groups) if only_groups is not None else GROUP_ORDER
    unknown = sorted(set(groups) - set(GROUP_ORDER))
    if unknown:
        raise ValueError(f"unknown randomization group(s): {unknown}")

    if sample.nominal:
        return AppliedRandomization(
            groups={
                group: GroupStatus(SKIPPED, "nominal sample (randomization disabled)")
                for group in groups
            },
            nominal=True,
        )

    capabilities = backend.capabilities()
    gated = {group: _capability_status(capabilities, group) for group in groups}

    # Ask the backend to execute once. It must check its own capabilities and
    # must not touch groups listed in ``sample.disabled_groups``.
    executed = backend.apply_randomization(sample)
    if not isinstance(executed, Mapping):
        raise TypeError("backend.apply_randomization() must return a mapping")

    report: dict[str, GroupStatus] = {}
    for group in groups:
        if group in sample.disabled_groups:
            report[group] = GroupStatus(SKIPPED, "group disabled in profile")
            continue
        gated_status = gated[group]
        if gated_status is not None:
            report[group] = gated_status
            continue
        result = executed.get(group)
        if result is None:
            report[group] = GroupStatus(SKIPPED, "backend did not execute this group")
            continue
        if not isinstance(result, GroupStatus):
            raise TypeError(f"backend returned an invalid result for group {group!r}")
        report[group] = result

    return AppliedRandomization(groups=report, nominal=False)
