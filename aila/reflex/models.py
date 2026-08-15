from __future__ import annotations

from datetime import datetime
from enum import IntEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Priority(IntEnum):
    """Reflex event priority tiers.

    Lower numeric value == higher urgency. ``P0`` is the most urgent
    (hardwired reflex; v3), ``P5`` the least (storage only). The integer
    ordering makes "more urgent" comparisons and "demote-only" guards
    straightforward: a demotion strictly increases the numeric value.
    """

    P0 = 0
    P1 = 1
    P2 = 2
    P3 = 3
    P4 = 4
    P5 = 5


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Event(StrictModel):
    """A ranked, reduced perception derived from one or more ``Observation``s.

    The reflex layer never mutates the frozen v1 ``Observation`` contract; an
    ``Event`` references its originating observation by ``obs_id`` and carries
    the compact, injectable form plus ranking/seen bookkeeping. Repeated
    observations are collapsed into a single event via ``dedup_key`` with a
    ``count`` and a ``first_ts``/``last_ts`` window.
    """

    event_id: str = Field(min_length=1)
    obs_id: str = Field(min_length=1)
    worker: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    ts: datetime

    dedup_key: str = Field(min_length=1)
    count: int = Field(default=1, ge=1)
    first_ts: datetime
    last_ts: datetime

    initial_priority: Priority
    effective_priority: Priority

    title: str = Field(min_length=1)
    summary: str = ""
    detail_available: bool = False

    seen: bool = False
    seen_ts: datetime | None = None

    supersede_next_tool_call: bool = False

    @model_validator(mode="after")
    def validate_event(self) -> Event:
        if self.last_ts < self.first_ts:
            raise ValueError("last_ts must be greater than or equal to first_ts")
        # Digest may only demote (raise the numeric priority), never promote.
        if int(self.effective_priority) < int(self.initial_priority):
            raise ValueError(
                "effective_priority must not be more urgent than initial_priority"
            )
        if self.seen and self.seen_ts is None:
            raise ValueError("seen events must record seen_ts")
        if not self.seen and self.seen_ts is not None:
            raise ValueError("unseen events must not record seen_ts")
        return self


class ExpandedEvent(StrictModel):
    """Full context returned by the ``reflex_expand`` tool.

    For media-bearing observations this exposes the ``media_ref`` and any
    caption/metadata rather than raw bytes, since the brain is a text model.
    """

    event_id: str = Field(min_length=1)
    obs_id: str = Field(min_length=1)
    worker: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    ts: datetime
    priority: Priority
    payload: Any
    media_ref: str | None = None
