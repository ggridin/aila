"""Models for the wake session briefing.

An :class:`Episode` is the durable record of a single wake: what happened, what
was decided, and -- most importantly -- what was left unfinished. Episodes are
the unit that is retained to the memory provider at session end and recalled at
the start of the next wake.

These models are deliberately host-independent (no Hermes / Hindsight imports)
so the briefing logic stays pure and testable.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import Field, field_validator

from aila.contracts import CONTRACT_VERSION
from aila.reflex.models import StrictModel

ContractVersion = Literal["1.0"]

# How an episode reached the briefing: deterministic recency vs semantic search.
RecallChannel = Literal["recent", "semantic"]

# Hard caps keep a single misbehaving episode from consuming the whole budget.
MAX_SUMMARY_CHARS = 1200
MAX_ITEM_CHARS = 240
MAX_ITEMS = 10


class Episode(StrictModel):
    """A durable record of one wake session.

    ``open_loops`` is the field that carries intent forward: it is what the
    previous session wanted the next one to pick up. ``entities`` seeds the
    semantic recall query on the next wake.
    """

    v: ContractVersion = CONTRACT_VERSION
    episode_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    started_ts: datetime
    ended_ts: datetime
    summary: str = Field(default="", max_length=MAX_SUMMARY_CHARS)
    decisions: tuple[str, ...] = ()
    open_loops: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()

    @field_validator("started_ts", "ended_ts")
    @classmethod
    def _as_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @field_validator("decisions", "open_loops", "entities")
    @classmethod
    def _bounded_items(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = [item.strip()[:MAX_ITEM_CHARS] for item in value if item and item.strip()]
        return tuple(cleaned[:MAX_ITEMS])

    @property
    def is_empty(self) -> bool:
        """True when the episode carries nothing worth briefing on."""
        return not (self.summary.strip() or self.decisions or self.open_loops)


class BriefingEntry(StrictModel):
    """One episode as it appears in the injected briefing block."""

    v: ContractVersion = CONTRACT_VERSION
    episode_id: str = Field(min_length=1)
    channel: RecallChannel
    started_ts: datetime
    ended_ts: datetime
    summary: str = Field(default="")
    decisions: tuple[str, ...] = ()
    open_loops: tuple[str, ...] = ()


def entry_for(episode: Episode, *, channel: RecallChannel) -> BriefingEntry:
    """Project an :class:`Episode` onto its briefing representation."""

    return BriefingEntry(
        episode_id=episode.episode_id,
        channel=channel,
        started_ts=episode.started_ts,
        ended_ts=episode.ended_ts,
        summary=episode.summary,
        decisions=episode.decisions,
        open_loops=episode.open_loops,
    )


class MemoryFact(StrictModel):
    """A single synthesized fact recalled from the semantic memory store.

    Hindsight does not return what was stored -- it consumes text and emits its
    own LLM-synthesized knowledge (``fact_type`` ``observation`` for the
    consolidated layer, ``world`` / ``experience`` for supporting evidence).
    Facts are therefore prose, not structured episodes, and are surfaced to the
    agent as narrative context rather than parsed back into records.
    """

    v: ContractVersion = CONTRACT_VERSION
    fact_id: str = Field(default="")
    text: str = Field(min_length=1, max_length=MAX_SUMMARY_CHARS)
    fact_type: str = Field(default="")
    ts: datetime | None = None
    # How much evidence supports this fact; higher means more corroborated.
    proof_count: int = Field(default=0, ge=0)

    @field_validator("ts")
    @classmethod
    def _fact_ts_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
