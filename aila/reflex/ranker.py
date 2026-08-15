from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import ConfigDict, field_validator

from aila.contracts import Observation
from aila.reflex.models import Priority, StrictModel


def coerce_priority(value: Any) -> Priority:
    """Coerce ``"P2"``/``"p2"``/``2``/``Priority.P2`` into a :class:`Priority`."""

    if isinstance(value, Priority):
        return value
    if isinstance(value, bool):
        raise ValueError(f"invalid priority: {value!r}")
    if isinstance(value, int):
        return Priority(value)
    if isinstance(value, str):
        name = value.strip().upper()
        if name in Priority.__members__:
            return Priority[name]
        if name.isdigit():
            return Priority(int(name))
    raise ValueError(f"invalid priority: {value!r}")


class RankingRule(StrictModel):
    """A single static ranking rule.

    A rule matches an observation when every specified criterion equals the
    observation's corresponding field. Unspecified criteria (``None``) act as
    wildcards. ``worker``/``kind`` are treated as opaque strings so future,
    not-yet-known sources (e.g. ``telegram``/``email``) can be ranked purely by
    configuration without code changes.
    """

    model_config = ConfigDict(extra="forbid")

    worker: str | None = None
    kind: str | None = None
    severity: str | None = None
    priority: Priority

    @field_validator("priority", mode="before")
    @classmethod
    def _coerce_priority(cls, value: Any) -> Priority:
        return coerce_priority(value)

    def matches(self, observation: Observation) -> bool:
        if self.worker is not None and self.worker != observation.worker:
            return False
        if self.kind is not None and self.kind != observation.kind:
            return False
        if self.severity is not None and self.severity != str(observation.severity.value):
            return False
        return True


class RankingRules(StrictModel):
    """Ordered ruleset plus a default fallback priority.

    Rules are evaluated in order and the **first** matching rule wins. When no
    rule matches, ``default_priority`` applies — this guarantees unknown
    workers/kinds still receive a sensible priority.
    """

    model_config = ConfigDict(extra="forbid")

    rules: tuple[RankingRule, ...] = ()
    default_priority: Priority = Priority.P5

    @field_validator("default_priority", mode="before")
    @classmethod
    def _coerce_default_priority(cls, value: Any) -> Priority:
        return coerce_priority(value)

    def priority_for(self, observation: Observation) -> Priority:
        for rule in self.rules:
            if rule.matches(observation):
                return rule.priority
        return self.default_priority


@runtime_checkable
class Ranker(Protocol):
    """Pluggable ranking interface.

    v2 ships the deterministic, config-driven :class:`RulesRanker`. A future
    model-based ranker (v3) can implement this protocol as a drop-in.
    """

    def initial_rank(self, observation: Observation) -> Priority: ...


class RulesRanker:
    """Deterministic, config-driven reflex-arc ranker."""

    def __init__(self, rules: RankingRules) -> None:
        self._rules = rules

    def initial_rank(self, observation: Observation) -> Priority:
        return initial_rank(observation, self._rules)


def initial_rank(observation: Observation, rules: RankingRules) -> Priority:
    """Assign the initial (reflex-arc) priority to an observation.

    Pure and deterministic: same observation + rules always yields the same
    priority, with no I/O.
    """

    return rules.priority_for(observation)


def clamp_demotion(initial: Priority, proposed: Priority) -> Priority:
    """Return ``proposed`` but never more urgent than ``initial``.

    The digest stage may only *demote* an event (increase the numeric priority,
    e.g. P2 -> P3). Any attempt to promote is clamped back to ``initial``.
    """

    return proposed if int(proposed) >= int(initial) else initial


def redigest_rank(
    event_initial: Priority,
    event_effective: Priority,
    *,
    proposed: Priority | None = None,
) -> Priority:
    """Compute a digest-time effective priority under the demote-only rule.

    ``proposed`` is an optional demotion target (e.g. under budget/staleness
    pressure). The result is clamped so it is never more urgent than the
    event's ``initial_priority`` and never more urgent than the current
    effective value could allow escalation.
    """

    target = proposed if proposed is not None else event_effective
    return clamp_demotion(event_initial, target)
